//! GPU-resident creature presentation.
//!
//! Culls the simulation's agent slots straight from the compute buffers into visible instances
//! and their trails, writes the indirect draw args, and draws the four creature layers (glow,
//! trail, body, outline) from those buffers. Nothing crosses back to the CPU per frame.

use crate::gpu::types::{GpuAgentTrail, GpuDartInstance};
use std::collections::HashMap;

/// Per-view uniform shared by the cull and the draw (`RenderParams` in creature_render.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CreatureRenderParams {
    /// World (Bevy coordinates, y up) to clip space.
    pub clip_from_world: [[f32; 4]; 4],
    /// Camera center in simulation coordinates (y down).
    pub camera_pos: [f32; 2],
    pub camera_size: [f32; 2],
    pub world_size: [f32; 2],
    /// Agent slots to consider, starting at slot 0.
    pub slot_count: u32,
    pub _pad: u32,
}

/// The simulation buffers the renderer reads in place. All must live on the renderer's device.
#[derive(Clone)]
pub struct CreatureSource {
    pub states: wgpu::Buffer,
    pub atomics: wgpu::Buffer,
    pub trails: wgpu::Buffer,
    /// Changes whenever the buffers are replaced (a new driver), so bind groups are rebuilt.
    pub generation: u64,
}

/// Render-target properties a draw pipeline is specialized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CreatureDrawKey {
    pub color_format: wgpu::TextureFormat,
    pub samples: u32,
    /// Depth attachment format, when the pass has one (tested, never written).
    pub depth: Option<wgpu::TextureFormat>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CreatureLayerKind {
    Glow = 0,
    Trail = 1,
    Body = 2,
    Outline = 3,
}

impl CreatureLayerKind {
    /// Back to front, as clankolution.html draws them.
    pub const ALL: [CreatureLayerKind; 4] = [
        CreatureLayerKind::Glow,
        CreatureLayerKind::Trail,
        CreatureLayerKind::Body,
        CreatureLayerKind::Outline,
    ];

    /// The z the CPU meshes used for this layer, for sorting among other 2D items.
    pub fn sort_z(self) -> f32 {
        match self {
            CreatureLayerKind::Glow => -2.2,
            CreatureLayerKind::Trail => -2.05,
            CreatureLayerKind::Body => -2.0,
            CreatureLayerKind::Outline => -1.9,
        }
    }

    pub fn vertices_per_instance(self) -> u32 {
        match self {
            CreatureLayerKind::Glow => 60,
            CreatureLayerKind::Trail => 16,
            CreatureLayerKind::Body => 6,
            CreatureLayerKind::Outline => 44,
        }
    }

    fn topology(self) -> wgpu::PrimitiveTopology {
        match self {
            CreatureLayerKind::Glow | CreatureLayerKind::Body => {
                wgpu::PrimitiveTopology::TriangleList
            }
            CreatureLayerKind::Trail | CreatureLayerKind::Outline => {
                wgpu::PrimitiveTopology::LineList
            }
        }
    }

    fn vertex_entry(self) -> &'static str {
        match self {
            CreatureLayerKind::Glow => "vs_glow",
            CreatureLayerKind::Trail => "vs_trail",
            CreatureLayerKind::Body => "vs_body",
            CreatureLayerKind::Outline => "vs_outline",
        }
    }
}

const INSTANCE_BYTES: u64 = std::mem::size_of::<GpuDartInstance>() as u64;
const TRAIL_BYTES: u64 = std::mem::size_of::<GpuAgentTrail>() as u64;

pub struct CreatureRenderer {
    module: wgpu::ShaderModule,
    max_instances: u32,
    params_buf: wgpu::Buffer,
    cull_count_buf: wgpu::Buffer,
    instances_buf: wgpu::Buffer,
    instance_trails_buf: wgpu::Buffer,
    draw_args_buf: wgpu::Buffer,
    cull_layout: wgpu::BindGroupLayout,
    cull_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    cull_bind_group: Option<(u64, wgpu::BindGroup)>,
    draw_pipeline_layout: wgpu::PipelineLayout,
    draw_bind_group: wgpu::BindGroup,
    pipelines: HashMap<CreatureDrawKey, [wgpu::RenderPipeline; 4]>,
}

fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl CreatureRenderer {
    pub fn new(device: &wgpu::Device, max_instances: u32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("creature_render_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../assets/shaders/creature_render.wgsl"
            ))),
        });
        let max_instances = max_instances.max(1);
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let params_buf = buffer(
            "creature_params",
            std::mem::size_of::<CreatureRenderParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let cull_count_buf = buffer("creature_cull_count", 16, U::STORAGE | U::COPY_DST);
        let instances_buf = buffer(
            "creature_instances",
            max_instances as u64 * INSTANCE_BYTES,
            U::STORAGE | U::COPY_SRC,
        );
        let instance_trails_buf = buffer(
            "creature_instance_trails",
            max_instances as u64 * TRAIL_BYTES,
            U::STORAGE | U::COPY_SRC,
        );
        let draw_args_buf = buffer(
            "creature_draw_args",
            64,
            U::STORAGE | U::INDIRECT | U::COPY_SRC,
        );

        use wgpu::BufferBindingType as B;
        use wgpu::ShaderStages as S;
        let storage = |read_only| B::Storage { read_only };
        let cull_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("creature_cull_bgl"),
            entries: &[
                buffer_entry(0, S::COMPUTE, B::Uniform),
                buffer_entry(1, S::COMPUTE, storage(true)),
                buffer_entry(2, S::COMPUTE, storage(true)),
                buffer_entry(3, S::COMPUTE, storage(true)),
                buffer_entry(4, S::COMPUTE, storage(false)),
                buffer_entry(5, S::COMPUTE, storage(false)),
                buffer_entry(6, S::COMPUTE, storage(false)),
                buffer_entry(7, S::COMPUTE, storage(false)),
            ],
        });
        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("creature_cull_pl"),
            bind_group_layouts: &[Some(&cull_layout)],
            immediate_size: 0,
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&cull_pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let cull_pipeline = compute("cull_main");
        let finalize_pipeline = compute("finalize_args_main");

        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("creature_draw_bgl"),
            entries: &[
                buffer_entry(8, S::VERTEX_FRAGMENT, B::Uniform),
                buffer_entry(9, S::VERTEX_FRAGMENT, storage(true)),
                buffer_entry(10, S::VERTEX_FRAGMENT, storage(true)),
            ],
        });
        let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("creature_draw_bg"),
            layout: &draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: instance_trails_buf.as_entire_binding(),
                },
            ],
        });
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("creature_draw_pl"),
            bind_group_layouts: &[Some(&draw_layout)],
            immediate_size: 0,
        });

        Self {
            module,
            max_instances,
            params_buf,
            cull_count_buf,
            instances_buf,
            instance_trails_buf,
            draw_args_buf,
            cull_layout,
            cull_pipeline,
            finalize_pipeline,
            cull_bind_group: None,
            draw_pipeline_layout,
            draw_bind_group,
            pipelines: HashMap::new(),
        }
    }

    pub fn max_instances(&self) -> u32 {
        self.max_instances
    }

    /// Records the cull of `rp.slot_count` slots of `source` and the indirect-args update.
    pub fn encode_cull(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: &CreatureSource,
        rp: &CreatureRenderParams,
    ) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(rp));
        if self.cull_bind_group.as_ref().map(|(g, _)| *g) != Some(source.generation) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("creature_cull_bg"),
                layout: &self.cull_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: source.states.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: source.atomics.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: source.trails.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.cull_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.instances_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.instance_trails_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.draw_args_buf.as_entire_binding(),
                    },
                ],
            });
            self.cull_bind_group = Some((source.generation, group));
        }
        let (_, group) = self.cull_bind_group.as_ref().unwrap();

        encoder.clear_buffer(&self.cull_count_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("creature_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, group, &[]);
            if rp.slot_count > 0 {
                pass.set_pipeline(&self.cull_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        // The args need the final count, so they are written in a pass of their own.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("creature_args_pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&self.finalize_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Builds the four layer pipelines for a render target, once per target kind.
    pub fn ensure_draw_pipelines(&mut self, device: &wgpu::Device, key: CreatureDrawKey) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let build = |layer: CreatureLayerKind| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(layer.vertex_entry()),
                layout: Some(&self.draw_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some(layer.vertex_entry()),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_color"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.color_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: layer.topology(),
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: key.depth.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: key.samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipelines = CreatureLayerKind::ALL.map(build);
        self.pipelines.insert(key, pipelines);
    }

    /// Draws one layer of every visible creature. `encode_cull` must have run earlier in the
    /// same submission, and `ensure_draw_pipelines(key)` before that.
    pub fn draw_layer(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        key: CreatureDrawKey,
        layer: CreatureLayerKind,
    ) {
        let Some(pipelines) = self.pipelines.get(&key) else {
            return;
        };
        pass.set_pipeline(&pipelines[layer as usize]);
        pass.set_bind_group(0, &self.draw_bind_group, &[]);
        pass.draw_indirect(&self.draw_args_buf, layer as u64 * 16);
    }

    /// Diagnostics/tests: the visible instances and their trails, in draw order. Blocks.
    pub fn readback_visible(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (Vec<GpuDartInstance>, Vec<GpuAgentTrail>) {
        let args = self.readback_draw_args(device, queue);
        let n = args[0][1] as u64;
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let instances = read_buffer(device, queue, &self.instances_buf, n * INSTANCE_BYTES);
        let trails = read_buffer(device, queue, &self.instance_trails_buf, n * TRAIL_BYTES);
        (
            bytemuck::pod_collect_to_vec(&instances),
            bytemuck::pod_collect_to_vec(&trails),
        )
    }

    /// Diagnostics/tests: the indirect args of the four layers. Blocks.
    pub fn readback_draw_args(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [[u32; 4]; 4] {
        let bytes = read_buffer(device, queue, &self.draw_args_buf, 64);
        bytemuck::pod_read_unaligned(&bytes)
    }
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    len: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("creature_readback"),
        size: len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("creature_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, len);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    out
}
