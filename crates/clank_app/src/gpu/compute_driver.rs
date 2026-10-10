//! GPU Compute Simulation Driver
//!
//! Owns the WGPU compute pipelines and VRAM storage buffers. Dispatches multi-tick
//! simulation passes directly onto GPU silicon via `dispatch_workgroups`.

use crate::gpu::frame_readback::{FrameReadback, ReadbackRing, ReadbackSources};
use crate::gpu::types::{
    AudioVoice, GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuAgentTrail, GpuSimParams,
    GpuSoilCell, GpuTelemetry,
};
use std::sync::Arc;

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DartVertex {
    pub position: [f32; 2],
    pub edge_flag: f32,
    pub _pad: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
    pub view_proj: [f32; 16],
    pub world_size: [f32; 2],
    pub _pad: [f32; 2],
}

pub struct GpuComputeDriver {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,

    // Storage Buffers
    pub agent_states_buf: wgpu::Buffer,
    pub agent_genomes_buf: wgpu::Buffer,
    pub agent_atomics_buf: wgpu::Buffer,
    pub soil_buf: wgpu::Buffer,
    pub bloom_table_buf: wgpu::Buffer,
    pub spatial_keys_buf: wgpu::Buffer,
    pub freelist_buf: wgpu::Buffer,
    pub queue_buffer: wgpu::Buffer,
    pub visible_instances_buf: wgpu::Buffer,
    pub cull_output_buf: wgpu::Buffer,
    pub dart_instances_buf: wgpu::Buffer,
    pub agent_trails_buf: wgpu::Buffer,
    pub dart_trails_buf: wgpu::Buffer,
    pub dart_template_buf: wgpu::Buffer,
    pub soil_display_tex: wgpu::Texture,

    // Uniform Buffers
    pub sim_params_buf: wgpu::Buffer,
    pub soil_params_buf: wgpu::Buffer,
    pub view_uniform_buf: wgpu::Buffer,

    // Staging Buffers for readbacks
    pub staging_telemetry: wgpu::Buffer,
    pub staging_agent_states: wgpu::Buffer,
    pub staging_genomes: wgpu::Buffer,
    pub staging_soil: wgpu::Buffer,
    pub staging_soil_display: wgpu::Buffer,
    pub staging_atomics: wgpu::Buffer,
    pub staging_cull_buf: wgpu::Buffer,
    pub staging_visible_instances: wgpu::Buffer,
    pub staging_dart_instances: wgpu::Buffer,
    pub staging_dart_trails: wgpu::Buffer,
    pub staging_freelist: wgpu::Buffer,
    pub staging_audio: wgpu::Buffer,

    // Pipelines
    pub soil_pipeline: wgpu::ComputePipeline,
    pub eclipse_pipeline: wgpu::ComputePipeline,
    pub preamble_pipeline: wgpu::ComputePipeline,
    pub morton_encode_pipeline: wgpu::ComputePipeline,
    pub spatial_sort: crate::gpu::radix_sort::SpatialSort,
    pub agent_pipeline: wgpu::ComputePipeline,
    pub birth_pipeline: wgpu::ComputePipeline,
    pub birth_dispatch: crate::gpu::birth_pipeline::BirthDispatch,
    census: crate::gpu::census_pipeline::CensusPass,
    pub merge_pending_free_pipeline: wgpu::ComputePipeline,
    pub spatial_query_pipeline: wgpu::ComputePipeline,
    pub cull_clear_pipeline: wgpu::ComputePipeline,
    pub cull_pipeline: wgpu::ComputePipeline,
    pub trail_record_pipeline: wgpu::ComputePipeline,
    pub trail_gather_pipeline: wgpu::ComputePipeline,
    pub capacity_pipeline: wgpu::ComputePipeline,
    pub dart_render_pipeline: wgpu::RenderPipeline,

    // Bind Groups
    pub soil_bind_group: wgpu::BindGroup,
    pub preamble_bg0: wgpu::BindGroup,
    pub preamble_bg1: wgpu::BindGroup,
    pub morton_bind_group: wgpu::BindGroup,
    pub agent_group0: wgpu::BindGroup,
    pub agent_group1: wgpu::BindGroup,
    pub birth_group0: wgpu::BindGroup,
    pub birth_group1: wgpu::BindGroup,
    pub spatial_query_bind_group: wgpu::BindGroup,
    pub trail_bind_group: wgpu::BindGroup,
    pub capacity_bind_group: wgpu::BindGroup,
    pub dart_render_bind_group: wgpu::BindGroup,

    pub max_agents: u32,
    pub soil_cols: u32,
    pub soil_rows: u32,
    pub initialized: std::sync::atomic::AtomicBool,
    /// The carrying capacity the freelist holds slots for; a new cap rebuilds it.
    freelist_capacity: std::sync::atomic::AtomicU32,
    /// Creatures in the last upload (slots `0..n`); with the cap, bounds the live slots.
    uploaded_slots: std::sync::atomic::AtomicU32,
    pub next_agent_id: std::sync::atomic::AtomicU32,
    pub next_agent_root: std::sync::atomic::AtomicU32,
    /// True when built on Bevy's render device, so the renderer can bind these buffers.
    pub on_render_device: bool,
    /// Async per-frame telemetry/audio/selection readbacks.
    frame_ring: std::sync::Mutex<ReadbackRing>,
    /// Times the CPU stalled until the GPU went idle (each synchronous readback is one).
    blocking_waits: std::sync::atomic::AtomicU64,
    /// Unique per driver, so renderers rebuild bind groups when the buffers are replaced.
    pub generation: u64,
}

static NEXT_DRIVER_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Whether `device` is on a card that drives no display (Metal's `isHeadless`; false elsewhere).
pub fn device_is_headless(device: &wgpu::Device) -> bool {
    #[cfg(target_os = "macos")]
    {
        use objc2_metal::MTLDevice;
        // SAFETY: only reads a property of the raw Metal device.
        if let Some(raw) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            return raw.raw_device().isHeadless();
        }
    }
    let _ = device;
    false
}

impl GpuComputeDriver {
    pub fn create_default() -> Option<Self> {
        Self::create_for_world(75, 50, 65536)
    }

    /// A device of our own, for headless use (tests, tools). The app builds the driver on
    /// Bevy's render device instead (`from_render_device`) so the renderer can read its buffers.
    /// Prefers a card that drives no display (a Mac Pro's second FirePro): long compute on the
    /// display's card freezes the whole screen until it finishes.
    pub fn request_standalone_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let request = |adapter: &wgpu::Adapter| {
            bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("clank_gpu_compute_device"),
                ..Default::default()
            }))
            .ok()
        };
        for adapter in bevy::tasks::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
            if let Some((device, queue)) = request(&adapter) {
                if device_is_headless(&device) {
                    return Some((device, queue));
                }
            }
        }

        let adapter =
            bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()?;
        request(&adapter)
    }

    /// The app's driver: built on Bevy's render device so the renderer can bind the
    /// simulation buffers directly (zero-copy presentation).
    pub fn from_render_device(
        device: &bevy::render::renderer::RenderDevice,
        queue: &bevy::render::renderer::RenderQueue,
        soil_cols: u32,
        soil_rows: u32,
        max_agents: u32,
    ) -> Self {
        let device = device.wgpu_device().clone();
        let queue = wgpu::Queue::clone(&queue.0);
        let mut driver = Self::new_with_grid(
            Arc::new(device),
            Arc::new(queue),
            max_agents,
            soil_cols,
            soil_rows,
        );
        driver.on_render_device = true;
        driver
    }

    pub fn create_for_world(soil_cols: u32, soil_rows: u32, max_agents: u32) -> Option<Self> {
        let (device, queue) = Self::request_standalone_device()?;
        Some(Self::new_with_grid(
            Arc::new(device),
            Arc::new(queue),
            max_agents,
            soil_cols,
            soil_rows,
        ))
    }

    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, max_agents: u32) -> Self {
        Self::new_with_grid(device, queue, max_agents, 75, 50)
    }

    pub fn new_with_grid(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        max_agents: u32,
        soil_cols: u32,
        soil_rows: u32,
    ) -> Self {
        use wgpu::BufferUsages;

        let total_soil_cells = (soil_cols * soil_rows) as u64;
        let agent_states_size = (max_agents as u64) * 128;
        let agent_genomes_size = (max_agents as u64) * 352;
        let agent_atomics_size = (max_agents as u64) * 16;
        let soil_size = total_soil_cells * 16;
        let bloom_size = total_soil_cells * 4;
        let spatial_keys_size = crate::gpu::radix_sort::spatial_keys_size(max_agents);
        let freelist_size = (max_agents as u64) * 4;
        // ConsolidatedQueue: 128B telemetry + 65536*16B births + 256*16B audio + 64*16B chronicle
        let queue_size = std::mem::size_of::<crate::gpu::types::ConsolidatedQueue>() as u64;

        let agent_states_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_states_buf"),
            size: agent_states_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let agent_genomes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_genomes_buf"),
            size: agent_genomes_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let agent_atomics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_atomics_buf"),
            size: agent_atomics_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let soil_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil_buf"),
            size: soil_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bloom_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_table_buf"),
            size: bloom_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bloom_table = vec![0.0f32; total_soil_cells as usize];
        for y in 0..soil_rows {
            let y_f = y as f32;
            for x in 0..soil_cols {
                let x_f = x as f32;
                let i = (y * soil_cols + x) as usize;
                bloom_table[i] = 0.0008
                    + 0.0022
                        * (0.5
                            + 0.5 * (x_f * 0.13 + (y_f * 0.19).sin()).sin() * (y_f * 0.11).cos());
            }
        }
        queue.write_buffer(&bloom_table_buf, 0, bytemuck::cast_slice(&bloom_table));

        let spatial_keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial_keys_buf"),
            size: spatial_keys_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The stack in [0, max_agents), the pending-free list in [max_agents, 2·max_agents).
        let freelist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("freelist_buf"),
            size: freelist_size * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("queue_buffer"),
            size: queue_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_params_buf"),
            size: std::mem::size_of::<GpuSimParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let soil_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil_params_buf"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_telemetry = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_telemetry"),
            size: 128,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_agent_states = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_agent_states"),
            size: agent_states_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_genomes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_genomes"),
            size: agent_genomes_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_soil = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_soil"),
            size: soil_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let unpadded_bytes_per_row = soil_cols * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
        let staging_soil_display_size = (padded_bytes_per_row * soil_rows) as u64;

        let staging_soil_display = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_soil_display"),
            size: staging_soil_display_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_atomics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_atomics"),
            size: agent_atomics_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_freelist = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_freelist"),
            size: freelist_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_audio = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_audio"),
            size: 4096, // 256 AudioVoice * 16 bytes
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_instances_size = (max_agents as u64) * 4;
        let visible_instances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible_instances_buf"),
            size: visible_instances_size,
            usage: BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let cull_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_output_buf"),
            size: 16,
            usage: BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let staging_cull_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_cull_buf"),
            size: 16,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_visible_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_visible_instances"),
            size: visible_instances_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dart_template: [DartVertex; 12] = [
            // Body fill (2 triangles):
            DartVertex {
                position: [0.0, 1.3],
                edge_flag: 0.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [-1.0, -1.0],
                edge_flag: 0.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [0.0, -0.4],
                edge_flag: 0.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [0.0, 1.3],
                edge_flag: 0.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [0.0, -0.4],
                edge_flag: 0.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [1.0, -1.0],
                edge_flag: 0.0,
                _pad: 0.0,
            },
            // Outline border (2 triangles):
            DartVertex {
                position: [0.0, 1.4],
                edge_flag: 1.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [-1.1, -1.1],
                edge_flag: 1.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [0.0, -0.5],
                edge_flag: 1.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [0.0, 1.4],
                edge_flag: 1.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [0.0, -0.5],
                edge_flag: 1.0,
                _pad: 0.0,
            },
            DartVertex {
                position: [1.1, -1.1],
                edge_flag: 1.0,
                _pad: 0.0,
            },
        ];

        let dart_template_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dart_template_buf"),
            size: 12 * 16,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dart_template_buf, 0, bytemuck::cast_slice(&dart_template));

        let dart_instances_size = (max_agents as u64) * 32;
        let dart_instances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dart_instances_buf"),
            size: dart_instances_size,
            usage: BufferUsages::STORAGE
                | BufferUsages::VERTEX
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_dart_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_dart_instances"),
            size: dart_instances_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let trails_size = (max_agents as u64) * std::mem::size_of::<GpuAgentTrail>() as u64;
        let agent_trails_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_trails_buf"),
            size: trails_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dart_trails_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dart_trails_buf"),
            size: trails_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_dart_trails = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_dart_trails"),
            size: trails_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view_uniform_buf"),
            size: 80,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Soil Textures
        let soil_display_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("soil_display_tex"),
            size: wgpu::Extent3d {
                width: soil_cols,
                height: soil_rows,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let soil_display_view =
            soil_display_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Compile Shader Modules
        let soil_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/soil_step.wgsl").into(),
            ),
        });
        let morton_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morton_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/morton_grid.wgsl").into(),
            ),
        });
        let agent_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agent_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/agent_step.wgsl").into(),
            ),
        });
        let birth_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("birth_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/birth_step.wgsl").into(),
            ),
        });

        // 1. Soil Pipeline & Bind Group
        let soil_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let soil_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil_bg"),
            layout: &soil_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: soil_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&soil_display_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bloom_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: soil_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sim_params_buf.as_entire_binding(),
                },
            ],
        });
        let soil_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil_pl"),
            bind_group_layouts: &[Some(&soil_bgl)],
            immediate_size: 0,
        });
        let soil_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil_pipeline"),
            layout: Some(&soil_pl),
            module: &soil_sm,
            entry_point: Some("soil_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let eclipse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("eclipse_pipeline"),
            layout: Some(&soil_pl),
            module: &soil_sm,
            entry_point: Some("eclipse_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 2. Morton Grid Pipelines & Bind Group
        let morton_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("morton_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let morton_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton_bg"),
            layout: &morton_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_keys_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sim_params_buf.as_entire_binding(),
                },
            ],
        });
        let morton_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morton_pl"),
            bind_group_layouts: &[Some(&morton_bgl)],
            immediate_size: 0,
        });
        let morton_encode_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("morton_encode"),
                layout: Some(&morton_pl),
                module: &morton_sm,
                entry_point: Some("morton_encode"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let spatial_sort = crate::gpu::radix_sort::SpatialSort::new(
            &device,
            &agent_states_buf,
            &spatial_keys_buf,
            &queue_buffer,
            &sim_params_buf,
            max_agents,
        );
        let birth_dispatch = crate::gpu::birth_pipeline::BirthDispatch::new(&device, &queue_buffer);
        let census = crate::gpu::census_pipeline::CensusPass::new(
            &device,
            &queue_buffer,
            &agent_states_buf,
            &agent_atomics_buf,
            max_agents,
        );

        // 4. Agent Pipeline & Bind Groups (Group 0: 8 storage buffers; Group 1: uniform + texture + sampler)
        let agent_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let agent_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_bg0"),
            layout: &agent_bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_genomes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: agent_atomics_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: soil_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spatial_keys_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: freelist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: queue_buffer.as_entire_binding(),
                },
            ],
        });

        let agent_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let agent_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_bg1"),
            layout: &agent_bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sim_params_buf.as_entire_binding(),
            }],
        });

        let agent_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_pl"),
            bind_group_layouts: &[Some(&agent_bgl0), Some(&agent_bgl1)],
            immediate_size: 0,
        });
        let agent_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("agent_pipeline"),
            layout: Some(&agent_pl),
            module: &agent_sm,
            entry_point: Some("agent_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 5. Birth Pipeline & Bind Groups
        let birth_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("birth_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let birth_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("birth_bg0"),
            layout: &birth_bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_genomes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: agent_atomics_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: freelist_buf.as_entire_binding(),
                },
            ],
        });

        let birth_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("birth_bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let birth_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("birth_bg1"),
            layout: &birth_bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sim_params_buf.as_entire_binding(),
            }],
        });

        let birth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("birth_pl"),
            bind_group_layouts: &[Some(&birth_bgl0), Some(&birth_bgl1)],
            immediate_size: 0,
        });
        let birth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("birth_pipeline"),
            layout: Some(&birth_pl),
            module: &birth_sm,
            entry_point: Some("birth_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let merge_pending_free_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("merge_pending_free_pipeline"),
                layout: Some(&birth_pl),
                module: &birth_sm,
                entry_point: Some("merge_pending_free_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let spatial_query_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spatial_query_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../assets/shaders/spatial_query.wgsl"
            ))),
        });

        let spatial_query_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_query_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let spatial_query_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_query_bg"),
            layout: &spatial_query_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_keys_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sim_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visible_instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cull_output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dart_instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: agent_atomics_buf.as_entire_binding(),
                },
            ],
        });

        let spatial_query_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_query_pl"),
            bind_group_layouts: &[Some(&spatial_query_bgl)],
            immediate_size: 0,
        });

        let spatial_query_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spatial_query_pipeline"),
                layout: Some(&spatial_query_pl),
                module: &spatial_query_sm,
                entry_point: Some("spatial_query_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let cull_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cull_clear_pipeline"),
                layout: Some(&spatial_query_pl),
                module: &spatial_query_sm,
                entry_point: Some("frustum_cull_clear"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull_pipeline"),
            layout: Some(&spatial_query_pl),
            module: &spatial_query_sm,
            entry_point: Some("frustum_cull_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let trail_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trail_step_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../assets/shaders/trail_step.wgsl"
            ))),
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let trail_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trail_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });

        let trail_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trail_bg"),
            layout: &trail_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_trails_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible_instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cull_output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dart_trails_buf.as_entire_binding(),
                },
            ],
        });

        let trail_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trail_pl"),
            bind_group_layouts: &[Some(&trail_bgl)],
            immediate_size: 0,
        });

        let trail_record_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("trail_record_pipeline"),
                layout: Some(&trail_pl),
                module: &trail_sm,
                entry_point: Some("trail_record_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let trail_gather_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("trail_gather_pipeline"),
                layout: Some(&trail_pl),
                module: &trail_sm,
                entry_point: Some("trail_gather_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let capacity_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("capacity_step_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../assets/shaders/capacity_step.wgsl"
            ))),
        });
        let capacity_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("capacity_bgl"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let capacity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("capacity_bg"),
            layout: &capacity_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_atomics_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: freelist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sim_params_buf.as_entire_binding(),
                },
            ],
        });
        let capacity_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("capacity_pl"),
            bind_group_layouts: &[Some(&capacity_bgl)],
            immediate_size: 0,
        });
        let capacity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("capacity_pipeline"),
            layout: Some(&capacity_pl),
            module: &capacity_sm,
            entry_point: Some("rebuild_freelist_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let preamble_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preamble_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/preamble_clear.wgsl").into(),
            ),
        });

        let preamble_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preamble_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let preamble_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preamble_bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let preamble_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preamble_bg0"),
            layout: &preamble_bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_atomics_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queue_buffer.as_entire_binding(),
                },
            ],
        });
        let preamble_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preamble_bg1"),
            layout: &preamble_bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sim_params_buf.as_entire_binding(),
            }],
        });
        let preamble_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preamble_pl"),
            bind_group_layouts: &[Some(&preamble_bgl0), Some(&preamble_bgl1)],
            immediate_size: 0,
        });
        let preamble_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preamble_pipeline"),
            layout: Some(&preamble_pl),
            module: &preamble_sm,
            entry_point: Some("preamble_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dart_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dart_instanced_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../../assets/shaders/dart_instanced.wgsl"
            ))),
        });

        let dart_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dart_render_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let dart_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dart_render_bg"),
            layout: &dart_render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_uniform_buf.as_entire_binding(),
            }],
        });

        let dart_render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dart_render_pl"),
            bind_group_layouts: &[Some(&dart_render_bgl)],
            immediate_size: 0,
        });

        let dart_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dart_render_pipeline"),
            layout: Some(&dart_render_pl),
            vertex: wgpu::VertexState {
                module: &dart_sm,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 8,
                                shader_location: 1,
                            },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32x2,
                                offset: 16,
                                shader_location: 3,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dart_sm,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let frame_ring = ReadbackRing::new(&device);

        Self {
            device,
            queue,
            agent_states_buf,
            agent_genomes_buf,
            agent_atomics_buf,
            soil_buf,
            bloom_table_buf,
            spatial_keys_buf,
            freelist_buf,
            queue_buffer,
            visible_instances_buf,
            cull_output_buf,
            dart_instances_buf,
            agent_trails_buf,
            dart_trails_buf,
            dart_template_buf,
            soil_display_tex,
            sim_params_buf,
            soil_params_buf,
            view_uniform_buf,
            staging_telemetry,
            staging_agent_states,
            staging_genomes,
            staging_soil,
            staging_soil_display,
            staging_atomics,
            staging_cull_buf,
            staging_visible_instances,
            staging_dart_instances,
            staging_dart_trails,
            staging_freelist,
            staging_audio,
            soil_pipeline,
            eclipse_pipeline,
            preamble_pipeline,
            morton_encode_pipeline,
            spatial_sort,
            agent_pipeline,
            birth_pipeline,
            birth_dispatch,
            census,
            merge_pending_free_pipeline,
            spatial_query_pipeline,
            cull_clear_pipeline,
            cull_pipeline,
            trail_record_pipeline,
            trail_gather_pipeline,
            capacity_pipeline,
            dart_render_pipeline,
            soil_bind_group,
            preamble_bg0,
            preamble_bg1,
            morton_bind_group,
            agent_group0,
            agent_group1,
            birth_group0,
            birth_group1,
            spatial_query_bind_group,
            trail_bind_group,
            capacity_bind_group,
            dart_render_bind_group,
            max_agents,
            soil_cols,
            soil_rows,
            initialized: std::sync::atomic::AtomicBool::new(false),
            freelist_capacity: std::sync::atomic::AtomicU32::new(0),
            uploaded_slots: std::sync::atomic::AtomicU32::new(0),
            next_agent_id: std::sync::atomic::AtomicU32::new(10000),
            next_agent_root: std::sync::atomic::AtomicU32::new(0),
            on_render_device: false,
            frame_ring: std::sync::Mutex::new(frame_ring),
            blocking_waits: std::sync::atomic::AtomicU64::new(0),
            generation: NEXT_DRIVER_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// The buffers a `CreatureRenderer` culls and draws from.
    pub fn creature_source(&self) -> crate::gpu::creature_render::CreatureSource {
        crate::gpu::creature_render::CreatureSource {
            states: self.agent_states_buf.clone(),
            atomics: self.agent_atomics_buf.clone(),
            trails: self.agent_trails_buf.clone(),
            generation: self.generation,
        }
    }

    /// Writes the Rust engine's creatures (`bridge::render_snapshot`) into the slots the
    /// renderer reads, so both engines are drawn by the same GPU path. Only for when the GPU
    /// engine holds no live state (`!is_initialized()`): it overwrites slots `0..n`.
    /// Returns `n`, the slot count to cull.
    pub fn upload_render_snapshot(
        &self,
        states: &[GpuAgentState],
        trails: &[GpuAgentTrail],
    ) -> u32 {
        let n = states.len().min(trails.len()).min(self.max_agents as usize);
        if n == 0 {
            return 0;
        }
        self.queue.write_buffer(
            &self.agent_states_buf,
            0,
            bytemuck::cast_slice(&states[..n]),
        );
        self.queue.write_buffer(
            &self.agent_trails_buf,
            0,
            bytemuck::cast_slice(&trails[..n]),
        );
        // Stale death claims from an earlier GPU session would hide live creatures.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_snapshot_encoder"),
            });
        encoder.clear_buffer(&self.agent_atomics_buf, 0, Some(n as u64 * 16));
        self.queue.submit([encoder.finish()]);
        n as u32
    }

    /// Stalls the CPU until every submitted GPU command has finished.
    fn wait_blocking(&self) {
        self.blocking_waits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    /// The slots a frame must cover for carrying capacity `max_cap`: every live creature
    /// sits below this. Creatures are uploaded into slots `0..n`, births take free slots
    /// under the cap, and the capacity pass removes every creature at or above a new cap.
    pub fn active_slots(&self, max_cap: u32) -> u32 {
        let uploaded = self
            .uploaded_slots
            .load(std::sync::atomic::Ordering::Relaxed);
        max_cap.max(uploaded).min(self.max_agents)
    }

    /// How many times the CPU has stalled on the GPU so far.
    pub fn blocking_readback_waits(&self) -> u64 {
        self.blocking_waits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Queues this frame's telemetry, audio and (optionally) one creature slot for async
    /// readback. Waits only when `RING_DEPTH` frames are already in flight.
    pub fn request_frame_readback(&self, selected_slot: Option<u32>, picking: bool) {
        let src = ReadbackSources {
            queue_buffer: &self.queue_buffer,
            agent_states: &self.agent_states_buf,
            agent_genomes: &self.agent_genomes_buf,
            agent_atomics: &self.agent_atomics_buf,
            agent_trails: &self.agent_trails_buf,
            max_agents: self.max_agents,
        };
        let mut ring = self.frame_ring.lock().unwrap();
        while !ring.try_request(&self.device, &self.queue, &src, selected_slot, picking) {
            if ring.oldest_ready() == Some(false) {
                self.wait_blocking();
            }
            ring.collect_ready();
        }
    }

    /// Frames whose readback has landed, oldest first. Never blocks.
    pub fn poll_frame_readbacks(&self) -> Vec<FrameReadback> {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let mut ring = self.frame_ring.lock().unwrap();
        ring.collect_ready();
        ring.take_completed()
    }

    /// Frames requested but not yet returned by `poll_frame_readbacks`.
    pub fn pending_frame_readbacks(&self) -> usize {
        self.frame_ring.lock().unwrap().in_flight()
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_initialized(&self, val: bool) {
        self.initialized
            .store(val, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn update_params(&self, params: &GpuSimParams) {
        self.queue
            .write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(params));
        let sp = crate::gpu::soil_pipeline::SoilParams {
            renewal: params.renewal,
            width: self.soil_cols,
            height: self.soil_rows,
            decay_rate: 0.006,
        };
        self.queue
            .write_buffer(&self.soil_params_buf, 0, bytemuck::bytes_of(&sp));
    }

    pub fn upload_state(
        &self,
        states: &[GpuAgentState],
        genomes: &[GpuAgentGenome],
        atomics: &[GpuAgentAtomic],
        soil: &[GpuSoilCell],
        params: &GpuSimParams,
    ) {
        // The CPU engine can hold one creature over its cap (original seed tool), so never
        // write more agents than the GPU has slots.
        let total_agents = self.max_agents as usize;
        let states = &states[..states.len().min(total_agents)];
        let genomes = &genomes[..genomes.len().min(total_agents)];
        let atomics = &atomics[..atomics.len().min(total_agents)];
        if !states.is_empty() {
            self.queue
                .write_buffer(&self.agent_states_buf, 0, bytemuck::cast_slice(states));
        }
        if total_agents > states.len() {
            let chunk_size = 2048;
            let tombstone = GpuAgentState {
                pos_vel: [0.0; 4],
                angle_energy: [0.0; 4],
                traits: [0.0; 8],
                hidden: [0.0; 10],
                id: 0,
                meta_flags: 1 << 13,
                age_gen: 0,
                morton_code: 0,
                root: 0,
                visual_cache: 0,
            };
            let dead_atomic = GpuAgentAtomic {
                energy_milli: 0,
                mate_claim: 0,
                mate_energy_milli: 0,
                dead_claimed: 1,
            };
            let tombstone_chunk = vec![tombstone; chunk_size];
            let atomic_chunk = vec![dead_atomic; chunk_size];
            let mut curr = states.len();
            while curr < total_agents {
                let this_chunk = (total_agents - curr).min(chunk_size);
                self.queue.write_buffer(
                    &self.agent_states_buf,
                    (curr * 128) as u64,
                    bytemuck::cast_slice(&tombstone_chunk[..this_chunk]),
                );
                self.queue.write_buffer(
                    &self.agent_atomics_buf,
                    (curr * 16) as u64,
                    bytemuck::cast_slice(&atomic_chunk[..this_chunk]),
                );
                curr += this_chunk;
            }
        }
        if !genomes.is_empty() {
            self.queue
                .write_buffer(&self.agent_genomes_buf, 0, bytemuck::cast_slice(genomes));
        }
        if !atomics.is_empty() {
            self.queue
                .write_buffer(&self.agent_atomics_buf, 0, bytemuck::cast_slice(atomics));
        }
        if !soil.is_empty() {
            self.queue
                .write_buffer(&self.soil_buf, 0, bytemuck::cast_slice(soil));
        }
        self.queue
            .write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(params));

        // Initialize tombstone freelist strictly up to params.max_capacity (lowest slots on top of stack)
        let cap = (params.max_capacity as usize).min(self.max_agents as usize);
        let mut free_slots: Vec<u32> = Vec::with_capacity(cap);
        if cap > states.len() {
            for i in (states.len()..cap).rev() {
                free_slots.push(i as u32);
            }
        }
        for (i, s) in states.iter().take(cap).enumerate().rev() {
            if (s.meta_flags & (1 << 13)) != 0 {
                free_slots.push(i as u32);
            }
        }
        let freelist_top = free_slots.len() as u32;
        self.freelist_capacity
            .store(cap as u32, std::sync::atomic::Ordering::Relaxed);
        self.uploaded_slots
            .store(states.len() as u32, std::sync::atomic::Ordering::Relaxed);
        if !free_slots.is_empty() {
            self.queue
                .write_buffer(&self.freelist_buf, 0, bytemuck::cast_slice(&free_slots));
        }
        // Lifetime counters (births, deaths) describe the uploaded world only.
        self.queue.write_buffer(
            &self.queue_buffer,
            0,
            bytemuck::bytes_of(&GpuTelemetry::default()),
        );
        self.queue
            .write_buffer(&self.queue_buffer, 28, bytemuck::bytes_of(&freelist_top));

        // Ids continue after every uploaded one (queue_buffer offset 24, apex_agent_id); the
        // world's own counter raises this further (`raise_next_agent_id`).
        let next_id = states.iter().map(|s| s.id + 1).max().unwrap_or(1);
        self.next_agent_id
            .store(next_id, std::sync::atomic::Ordering::Relaxed);
        self.queue
            .write_buffer(&self.queue_buffer, 24, bytemuck::bytes_of(&next_id));
        // Founders start new lineages after every uploaded one (the reference's `roots`).
        let next_root = states
            .iter()
            .filter(|s| (s.meta_flags & (1 << 13)) == 0)
            .map(|s| s.root + 1)
            .max()
            .unwrap_or(0);
        self.next_agent_root
            .store(next_root, std::sync::atomic::Ordering::Relaxed);

        let sp = crate::gpu::soil_pipeline::SoilParams {
            renewal: params.renewal,
            width: self.soil_cols,
            height: self.soil_rows,
            decay_rate: 0.006,
        };
        self.queue
            .write_buffer(&self.soil_params_buf, 0, bytemuck::bytes_of(&sp));

        // Uploaded creatures carry no GPU motion history (ids may repeat after NEW WORLD).
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("upload_clear_trails_encoder"),
            });
        encoder.clear_buffer(&self.agent_trails_buf, 0, None);
        self.queue.submit([encoder.finish()]);
        // Frames still in flight describe the replaced world.
        self.frame_ring.lock().unwrap().discard_all();
        self.initialized
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Follows a changed carrying capacity: rebuilds the freelist for `params.max_capacity`,
    /// freeing dead slots under it and removing creatures at or above it.
    fn apply_capacity(&self, params: &GpuSimParams) {
        let cap = params.max_capacity.min(self.max_agents);
        let previous = self
            .freelist_capacity
            .swap(cap, std::sync::atomic::Ordering::Relaxed);
        if previous == cap {
            return;
        }
        let zero = 0u32;
        self.queue
            .write_buffer(&self.queue_buffer, 28, bytemuck::bytes_of(&zero));
        // Every slot: creatures born under the old cap may sit above this frame's slots.
        let all_slots = GpuSimParams {
            max_agents: self.max_agents,
            ..*params
        };
        self.queue
            .write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(&all_slots));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_capacity_encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_capacity_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.capacity_pipeline);
            cpass.set_bind_group(0, &self.capacity_bind_group, &[]);
            cpass.dispatch_workgroups(self.max_agents.div_ceil(64), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
    }

    pub fn dispatch_sub_ticks(&self, sub_ticks: u32, params: &GpuSimParams) {
        let mut cur_params = *params;
        self.apply_capacity(params);

        for step in 0..sub_ticks {
            cur_params.sub_tick = step;
            cur_params.tick = params.tick + step;
            // The eclipse countdown as the reference sees it in this tick, before its decrement.
            cur_params.eclipse = params.eclipse.saturating_sub(step);
            self.queue
                .write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(&cur_params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu_sim_sub_tick_encoder"),
                });
            // One compute pass for the whole sub-tick. Each dispatch is its own WebGPU
            // synchronization scope, so every step sees the previous steps' storage writes as
            // it would across passes, and a pass costs the CPU about 0.2 ms to record in a
            // debug build against about 0.04 ms for a dispatch.
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_sim_sub_tick_pass"),
                    timestamp_writes: None,
                });

                // 1. Soil decay and renewal
                cpass.set_pipeline(&self.soil_pipeline);
                cpass.set_bind_group(0, &self.soil_bind_group, &[]);
                cpass.dispatch_workgroups((self.soil_cols + 7) / 8, (self.soil_rows + 7) / 8, 1);

                // 1b. Eclipse damage to the soil (160 cells on the reference's even ticks)
                if cur_params.eclipse > 0 {
                    cpass.set_pipeline(&self.eclipse_pipeline);
                    cpass.set_bind_group(0, &self.soil_bind_group, &[]);
                    cpass.dispatch_workgroups(3, 1, 1);
                }

                let active_slots = cur_params.max_agents.max(cur_params.agent_count);
                let agent_workgroups = (active_slots + 63) / 64;

                if agent_workgroups > 0 {
                    // 2. Preamble clear (resets atomics and sub-tick telemetry counters)
                    cpass.set_pipeline(&self.preamble_pipeline);
                    cpass.set_bind_group(0, &self.preamble_bg0, &[]);
                    cpass.set_bind_group(1, &self.preamble_bg1, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    // 2b. Morton keys
                    cpass.set_pipeline(&self.morton_encode_pipeline);
                    cpass.set_bind_group(0, &self.morton_bind_group, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    // 2c. Sort the keys, count the live ones, build the grid cell table
                    self.spatial_sort.encode(&mut cpass, cur_params.max_agents);

                    // 5. Unified Spatial Query (Mouse Picking & AoE tools) on sub_tick 0
                    if cur_params.tool_type != 0xFFFFFFFF && step == 0 {
                        cpass.set_pipeline(&self.spatial_query_pipeline);
                        cpass.set_bind_group(0, &self.spatial_query_bind_group, &[]);
                        if cur_params.tool_type == 0 {
                            cpass.dispatch_workgroups(1, 1, 1);
                        } else {
                            cpass.dispatch_workgroups(agent_workgroups, 1, 1);
                        }
                    }

                    // 6. Agent forward pass, kinematics & combat
                    cpass.set_pipeline(&self.agent_pipeline);
                    cpass.set_bind_group(0, &self.agent_group0, &[]);
                    cpass.set_bind_group(1, &self.agent_group1, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    // 6b. Motion trails: one point per tick from the post-movement positions
                    cpass.set_pipeline(&self.trail_record_pipeline);
                    cpass.set_bind_group(0, &self.trail_bind_group, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    // 7. Decoupled birth step: one workgroup per queued birth
                    self.birth_dispatch.encode_args(&mut cpass);
                    cpass.set_pipeline(&self.birth_pipeline);
                    cpass.set_bind_group(0, &self.birth_group0, &[]);
                    cpass.set_bind_group(1, &self.birth_group1, &[]);
                    cpass.dispatch_workgroups_indirect(&self.birth_dispatch.args, 0);

                    // 8. This sub-tick's deaths become reusable for the next one
                    cpass.set_pipeline(&self.merge_pending_free_pipeline);
                    cpass.set_bind_group(0, &self.birth_group0, &[]);
                    cpass.set_bind_group(1, &self.birth_group1, &[]);
                    cpass.dispatch_workgroups(1, 1, 1);

                    // 9. On the frame's final sub-tick, the living census after the births
                    if step + 1 == sub_ticks {
                        self.census.encode(&mut cpass, agent_workgroups);
                    }
                }
            }

            self.queue.submit([encoder.finish()]);
        }
    }

    pub fn readback_telemetry(&self) -> GpuTelemetry {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_telemetry_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.queue_buffer, 0, &self.staging_telemetry, 0, 128);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_telemetry.slice(0..128);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let telemetry = *bytemuck::from_bytes::<GpuTelemetry>(&data[0..128]);
        drop(data);
        self.staging_telemetry.unmap();

        telemetry
    }

    pub fn readback_agent_states(&self, count: usize) -> Vec<GpuAgentState> {
        let byte_len = (count * 128) as u64;
        if byte_len == 0 {
            return Vec::new();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_states_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.agent_states_buf,
            0,
            &self.staging_agent_states,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_agent_states.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let states =
            bytemuck::cast_slice::<u8, GpuAgentState>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_agent_states.unmap();

        states
    }

    pub fn readback_agent_genomes(&self, count: usize) -> Vec<GpuAgentGenome> {
        let byte_len = (count * 352) as u64;
        if byte_len == 0 {
            return Vec::new();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_genomes_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.agent_genomes_buf,
            0,
            &self.staging_genomes,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_genomes.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let genomes =
            bytemuck::cast_slice::<u8, GpuAgentGenome>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_genomes.unmap();

        genomes
    }

    pub fn readback_soil(&self) -> Vec<GpuSoilCell> {
        let total_cells = (self.soil_cols * self.soil_rows) as usize;
        let byte_len = (total_cells * 16) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_soil_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.soil_buf, 0, &self.staging_soil, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_soil.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let soil = bytemuck::cast_slice::<u8, GpuSoilCell>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_soil.unmap();

        soil
    }

    pub fn copy_soil_display_rgba(&self, out: &mut [u8]) {
        let expected_len = (self.soil_cols * self.soil_rows * 4) as usize;
        assert!(out.len() >= expected_len, "out buffer too small");

        let unpadded_bytes_per_row = self.soil_cols * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
        let total_bytes = (padded_bytes_per_row * self.soil_rows) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_soil_display_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.soil_display_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_soil_display,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.soil_rows),
                },
            },
            wgpu::Extent3d {
                width: self.soil_cols,
                height: self.soil_rows,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);

        let slice = self.staging_soil_display.slice(0..total_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let row = (self.soil_cols * 4) as usize;
        for y in 0..self.soil_rows as usize {
            let src = y * padded_bytes_per_row as usize;
            out[y * row..(y + 1) * row].copy_from_slice(&data[src..src + row]);
        }
        drop(data);
        self.staging_soil_display.unmap();
    }

    /// Replaces the soil colormap (`soil_cols × soil_rows` RGBA8 texels), e.g. with the CPU
    /// colormap of an uploaded world, so it is shown before the first GPU tick rewrites it.
    pub fn write_soil_display(&self, rgba: &[u8]) {
        self.queue.write_texture(
            self.soil_display_tex.as_image_copy(),
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.soil_cols * 4),
                rows_per_image: Some(self.soil_rows),
            },
            self.soil_display_tex.size(),
        );
    }

    /// Writes per-tick trails into slots `0..trails.len()` (the uploaded creatures' order).
    pub fn write_trails(&self, trails: &[GpuAgentTrail]) {
        let n = trails.len().min(self.max_agents as usize);
        if n > 0 {
            self.queue.write_buffer(
                &self.agent_trails_buf,
                0,
                bytemuck::cast_slice(&trails[..n]),
            );
        }
    }

    /// The soil colormap (`rgba8unorm`), for copying straight into the soil sprite.
    pub fn soil_display_texture(&self) -> &wgpu::Texture {
        &self.soil_display_tex
    }

    pub fn readback_soil_display_rgba(&self) -> Vec<u8> {
        let mut out = vec![0u8; (self.soil_cols * self.soil_rows * 4) as usize];
        self.copy_soil_display_rgba(&mut out);
        out
    }

    pub fn readback_atomics(&self, count: usize) -> Vec<GpuAgentAtomic> {
        let byte_len = (count * 16) as u64;
        if byte_len == 0 {
            return Vec::new();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_atomics_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.agent_atomics_buf,
            0,
            &self.staging_atomics,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_atomics.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let atomics =
            bytemuck::cast_slice::<u8, GpuAgentAtomic>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_atomics.unmap();

        atomics
    }

    pub fn dispatch_culling(&self, params: &GpuSimParams) -> u32 {
        self.queue
            .write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cull_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull_pass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.spatial_query_bind_group, &[]);

            // Clear cull counter to 0
            cpass.set_pipeline(&self.cull_clear_pipeline);
            cpass.dispatch_workgroups(1, 1, 1);

            // Frustum cull active agents
            let active_slots = params.max_agents.max(params.agent_count);
            let workgroups = (active_slots + 63) / 64;
            if workgroups > 0 {
                cpass.set_pipeline(&self.cull_pipeline);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // Gather the trails of the visible darts (needs the final cull count, so a new pass)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trail_gather_pass"),
                timestamp_writes: None,
            });
            let workgroups = (params.max_agents + 63) / 64;
            if workgroups > 0 {
                cpass.set_pipeline(&self.trail_gather_pipeline);
                cpass.set_bind_group(0, &self.trail_bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        encoder.copy_buffer_to_buffer(&self.cull_output_buf, 0, &self.staging_cull_buf, 0, 16);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_cull_buf.slice(0..16);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let visible_count = {
            let data = slice.get_mapped_range();
            let count = u32::from_le_bytes(data[4..8].try_into().unwrap());
            drop(data);
            count
        };
        self.staging_cull_buf.unmap();

        visible_count
            .min(params.max_capacity)
            .min(params.max_agents)
    }

    pub fn readback_visible_instances(&self, count: usize) -> Vec<u32> {
        let count = count.min(self.max_agents as usize);
        if count == 0 {
            return Vec::new();
        }
        let byte_len = (count * 4) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_visible_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.visible_instances_buf,
            0,
            &self.staging_visible_instances,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_visible_instances.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let result = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        self.staging_visible_instances.unmap();
        result
    }

    pub fn readback_dart_instances(&self, count: usize) -> Vec<crate::gpu::types::GpuDartInstance> {
        if count == 0 {
            return Vec::new();
        }
        let byte_len = (count * 32) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_dart_instances_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.dart_instances_buf,
            0,
            &self.staging_dart_instances,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_dart_instances.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let result = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, crate::gpu::types::GpuDartInstance>(
                &data[0..byte_len as usize],
            )
            .to_vec()
        };
        self.staging_dart_instances.unmap();
        result
    }

    /// Trails of the darts from the last `dispatch_culling`, index-aligned with `readback_dart_instances`.
    pub fn readback_dart_trails(&self, count: usize) -> Vec<GpuAgentTrail> {
        self.readback_trails_from(
            &self.dart_trails_buf,
            0,
            count.min(self.max_agents as usize),
        )
    }

    /// Trail of the creature in `slot`.
    pub fn readback_agent_trail(&self, slot: u32) -> Option<GpuAgentTrail> {
        if slot >= self.max_agents {
            return None;
        }
        self.readback_trails_from(&self.agent_trails_buf, slot as usize, 1)
            .pop()
    }

    fn readback_trails_from(
        &self,
        src: &wgpu::Buffer,
        first: usize,
        count: usize,
    ) -> Vec<GpuAgentTrail> {
        if count == 0 {
            return Vec::new();
        }
        let stride = std::mem::size_of::<GpuAgentTrail>();
        let byte_len = (count * stride) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_trails_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            src,
            (first * stride) as u64,
            &self.staging_dart_trails,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_dart_trails.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let result = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, GpuAgentTrail>(&data[0..byte_len as usize]).to_vec()
        };
        self.staging_dart_trails.unmap();
        result
    }

    pub fn readback_cull_args(&self) -> [u32; 4] {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_cull_args_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.cull_output_buf, 0, &self.staging_cull_buf, 0, 16);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_cull_buf.slice(0..16);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let args = {
            let data = slice.get_mapped_range();
            let mut arr = [0u32; 4];
            arr[0] = u32::from_le_bytes(data[0..4].try_into().unwrap());
            arr[1] = u32::from_le_bytes(data[4..8].try_into().unwrap());
            arr[2] = u32::from_le_bytes(data[8..12].try_into().unwrap());
            arr[3] = u32::from_le_bytes(data[12..16].try_into().unwrap());
            drop(data);
            arr
        };
        self.staging_cull_buf.unmap();
        args
    }

    pub fn render_darts_instanced(&self, target_view: &wgpu::TextureView, params: &GpuSimParams) {
        // Update view uniform with orthographic projection matching dart_instanced.wgsl inverted Y
        let cam_y = params.world_size[1] - params.camera_pos[1];
        let left = params.camera_pos[0] - params.camera_size[0] * 0.5;
        let right = params.camera_pos[0] + params.camera_size[0] * 0.5;
        let bottom = cam_y - params.camera_size[1] * 0.5;
        let top = cam_y + params.camera_size[1] * 0.5;

        let sx = 2.0 / (right - left);
        let sy = 2.0 / (top - bottom);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);

        let mut view_proj = [0.0f32; 16];
        view_proj[0] = sx;
        view_proj[5] = sy;
        view_proj[10] = 0.0;
        view_proj[12] = tx;
        view_proj[13] = ty;
        view_proj[14] = 0.5;
        view_proj[15] = 1.0;

        let vu = ViewUniform {
            view_proj,
            world_size: params.world_size,
            _pad: [0.0; 2],
        };
        self.queue
            .write_buffer(&self.view_uniform_buf, 0, bytemuck::bytes_of(&vu));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dart_render_encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dart_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.dart_render_pipeline);
            rpass.set_bind_group(0, &self.dart_render_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.dart_template_buf.slice(..));
            rpass.set_vertex_buffer(1, self.dart_instances_buf.slice(..));
            rpass.draw_indirect(&self.cull_output_buf, 0);
        }

        self.queue.submit([encoder.finish()]);
    }

    pub fn dart_template_vertex_count(&self) -> u32 {
        12
    }

    /// Newborns and founders take ids from at least `next_id` on (the world's `next_id`, which
    /// is past every id it ever issued, dead creatures' included).
    pub fn raise_next_agent_id(&self, next_id: u32) {
        let next = self
            .next_agent_id
            .fetch_max(next_id, std::sync::atomic::Ordering::Relaxed)
            .max(next_id);
        self.queue
            .write_buffer(&self.queue_buffer, 24, bytemuck::bytes_of(&next));
    }

    /// Agent replenishment: one founder at each position.
    pub fn replenish_agents_gpu(&self, agents: &[(f32, f32)]) -> Vec<u32> {
        self.place_founders_gpu(agents)
    }

    /// Seed Life (`World::seed_life_at`): two founders per click, each within ±11 px of it,
    /// while free slots under the cap remain.
    pub fn seed_agents_gpu(&self, clicks: &[(f32, f32)]) -> Vec<u32> {
        let world_w = (self.soil_cols as f32) * 12.0;
        let world_h = (self.soil_rows as f32) * 12.0;
        let first_id = self
            .next_agent_id
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut positions = Vec::with_capacity(clicks.len() * 2);
        for (c, &(x, y)) in clicks.iter().enumerate() {
            for k in 0..2u32 {
                let n = first_id.wrapping_add(c as u32 * 2 + k);
                let x = x + seed_jitter(n, 0);
                let y = y + seed_jitter(n, 1);
                positions.push((x.rem_euclid(world_w), y.rem_euclid(world_h)));
            }
        }
        self.place_founders_gpu(&positions)
    }

    fn place_founders_gpu(&self, agents: &[(f32, f32)]) -> Vec<u32> {
        if agents.is_empty() {
            return Vec::new();
        }

        // 1. Read back current telemetry to check available freelist slots
        let telem = self.readback_telemetry();
        // Newborns take ids from the GPU's counter; founders continue after them.
        self.next_agent_id
            .fetch_max(telem.apex_agent_id, std::sync::atomic::Ordering::Relaxed);
        let cur_freelist_top = telem.freelist_top.min(self.max_agents);
        if cur_freelist_top == 0 {
            return Vec::new();
        }

        let count = agents.len().min(cur_freelist_top as usize);
        if count == 0 {
            return Vec::new();
        }

        // 2. Read back top `count` slots from freelist_buf
        let start_slot = cur_freelist_top - count as u32;
        let start_byte = (start_slot as u64) * 4;
        let byte_len = (count as u64) * 4;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_freelist_slots_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.freelist_buf,
            start_byte,
            &self.staging_freelist,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_freelist.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let slots: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_freelist.unmap();

        let world_w = (self.soil_cols as f32) * 12.0;
        let world_h = (self.soil_rows as f32) * 12.0;

        let mut states = Vec::with_capacity(slots.len());
        let mut genomes = Vec::with_capacity(slots.len());
        let mut atomics = Vec::with_capacity(slots.len());

        // 3. For each slot, initialize agent data
        for (i, &_slot) in slots.iter().enumerate() {
            let (x, y) = agents[i];
            let id = self
                .next_agent_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let root = self
                .next_agent_root
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Reference founders (`World::create_agent` with no parent): angle in [0, TAU),
            // energy in [34, 52), each trait uniform in its own range, genes js_round of
            // uniform [-58, 58). Draws are keyed by the founder's id so each one differs.
            let rand = |stream: u32, lo: f32, hi: f32| lo + (hi - lo) * founder_uniform(id, stream);
            let angle = rand(0, 0.0, std::f32::consts::TAU);
            let energy = rand(1, 34.0, 52.0);
            const TRAIT_RANGES: [(f32, f32); 6] = [
                (0.25, 0.8),
                (0.3, 0.8),
                (0.3, 0.8),
                (0.25, 0.75),
                (0.2, 0.8),
                (0.2, 0.8),
            ];
            let mut traits = [0.0f32; 8];
            for (c, &(lo, hi)) in TRAIT_RANGES.iter().enumerate() {
                traits[c] = rand(2 + c as u32, lo, hi);
            }

            let r_u8 = (traits[0].clamp(0.0, 1.0) * 255.0) as u32;
            let e_u8 = ((energy / 100.0).clamp(0.0, 1.0) * 255.0) as u32;
            let visual_cache = r_u8 | (e_u8 << 16);

            let morton =
                crate::gpu::spatial_index::compute_morton_32_with_size([x, y], [world_w, world_h]);

            states.push(GpuAgentState {
                pos_vel: [x, y, 0.0, 0.0],
                angle_energy: [angle, energy, 0.0, 0.0],
                traits,
                hidden: [0.0; 10],
                id,
                meta_flags: 0, // living: bit 13 is 0
                age_gen: 0,
                morton_code: morton,
                root,
                visual_cache,
            });

            let mut packed_genes = [0u32; 88];
            for g in 0..clank_core::agent::GENES {
                let gene = (rand(8 + g as u32, -58.0, 58.0) + 0.5).floor() as i8;
                packed_genes[g >> 2] |= (gene as u8 as u32) << ((g & 3) * 8);
            }
            genomes.push(GpuAgentGenome { packed_genes });

            atomics.push(GpuAgentAtomic {
                energy_milli: (energy * 1000.0) as i32,
                mate_claim: 0,
                mate_energy_milli: 0,
                dead_claimed: 0,
            });
        }

        // Group into contiguous slot runs to minimize write_buffer overhead
        let mut run_start = 0;
        while run_start < slots.len() {
            let mut run_end = run_start + 1;
            while run_end < slots.len() && slots[run_end] == slots[run_end - 1] + 1 {
                run_end += 1;
            }

            let first_slot = slots[run_start];
            let run_states: Vec<GpuAgentState> =
                (run_start..run_end).map(|idx| states[idx]).collect();
            let run_genomes: Vec<GpuAgentGenome> =
                (run_start..run_end).map(|idx| genomes[idx]).collect();
            let run_atomics: Vec<GpuAgentAtomic> =
                (run_start..run_end).map(|idx| atomics[idx]).collect();

            self.queue.write_buffer(
                &self.agent_states_buf,
                (first_slot as u64) * 128,
                bytemuck::cast_slice(&run_states),
            );
            self.queue.write_buffer(
                &self.agent_genomes_buf,
                (first_slot as u64) * 352,
                bytemuck::cast_slice(&run_genomes),
            );
            self.queue.write_buffer(
                &self.agent_atomics_buf,
                (first_slot as u64) * 16,
                bytemuck::cast_slice(&run_atomics),
            );

            run_start = run_end;
        }

        // 4. Update freelist_top and apex_agent_id in queue_buffer
        let new_freelist_top = start_slot;
        self.queue.write_buffer(
            &self.queue_buffer,
            28,
            bytemuck::bytes_of(&new_freelist_top),
        );
        let next_id = self
            .next_agent_id
            .load(std::sync::atomic::Ordering::Relaxed);
        self.queue
            .write_buffer(&self.queue_buffer, 24, bytemuck::bytes_of(&next_id));

        slots
    }

    pub fn readback_audio_voices(&self, count: usize) -> Vec<AudioVoice> {
        let count = count.min(256);
        if count == 0 {
            return Vec::new();
        }
        let byte_len = (count * 16) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_audio_voices_encoder"),
            });
        // Offset 1,048,704 in queue_buffer
        encoder.copy_buffer_to_buffer(
            &self.queue_buffer,
            1_048_704,
            &self.staging_audio,
            0,
            byte_len,
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_audio.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.wait_blocking();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let voices = bytemuck::cast_slice::<u8, AudioVoice>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_audio.unmap();

        voices
    }

    pub fn readback_agent_slot(
        &self,
        slot: u32,
    ) -> Option<(GpuAgentState, GpuAgentGenome, GpuAgentAtomic)> {
        if slot >= self.max_agents {
            return None;
        }
        let state_offset = (slot as u64) * 128;
        let genome_offset = (slot as u64) * 352;
        let atomic_offset = (slot as u64) * 16;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_agent_slot_encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.agent_states_buf,
            state_offset,
            &self.staging_agent_states,
            0,
            128,
        );
        encoder.copy_buffer_to_buffer(
            &self.agent_genomes_buf,
            genome_offset,
            &self.staging_genomes,
            0,
            352,
        );
        encoder.copy_buffer_to_buffer(
            &self.agent_atomics_buf,
            atomic_offset,
            &self.staging_atomics,
            0,
            16,
        );
        self.queue.submit([encoder.finish()]);

        // Map state
        let s_slice = self.staging_agent_states.slice(0..128);
        let (s_tx, s_rx) = std::sync::mpsc::channel();
        s_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = s_tx.send(res);
        });
        self.wait_blocking();
        s_rx.recv().unwrap().unwrap();
        let state = *bytemuck::from_bytes::<GpuAgentState>(&s_slice.get_mapped_range()[0..128]);
        self.staging_agent_states.unmap();

        // Map genome
        let g_slice = self.staging_genomes.slice(0..352);
        let (g_tx, g_rx) = std::sync::mpsc::channel();
        g_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = g_tx.send(res);
        });
        self.wait_blocking();
        g_rx.recv().unwrap().unwrap();
        let genome = *bytemuck::from_bytes::<GpuAgentGenome>(&g_slice.get_mapped_range()[0..352]);
        self.staging_genomes.unmap();

        // Map atomic
        let a_slice = self.staging_atomics.slice(0..16);
        let (a_tx, a_rx) = std::sync::mpsc::channel();
        a_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = a_tx.send(res);
        });
        self.wait_blocking();
        a_rx.recv().unwrap().unwrap();
        let atomic = *bytemuck::from_bytes::<GpuAgentAtomic>(&a_slice.get_mapped_range()[0..16]);
        self.staging_atomics.unmap();

        Some((state, genome, atomic))
    }

    pub fn sync_soil_cells_gpu(
        &self,
        soil: &clank_core::soil::SoilGrid,
        cx: i32,
        cy: i32,
        radius: i32,
    ) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = (cx + dx).rem_euclid(self.soil_cols as i32) as u32;
                    let y = (cy + dy).rem_euclid(self.soil_rows as i32) as u32;
                    let idx = (y * self.soil_cols + x) as usize;
                    let food = soil.food.get(idx).copied().unwrap_or(0.0);
                    let taint = soil.taint.get(idx).copied().unwrap_or(0.0);
                    let scent = soil.scent.get(idx).copied().unwrap_or(0.0);
                    let cell = crate::gpu::soil_pipeline::GpuSoilPipeline::from_physics(
                        food, taint, scent,
                    );
                    self.queue.write_buffer(
                        &self.soil_buf,
                        (idx * 16) as u64,
                        bytemuck::bytes_of(&cell),
                    );
                }
            }
        }
    }
}

/// Uniform [0, 1) draw for founder `id`'s value number `stream` (splitmix64 finalizer).
fn founder_uniform(id: u32, stream: u32) -> f32 {
    let mut h = ((id as u64) << 32 | stream as u64) ^ 0x6a09e667f3bcc909;
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// A jitter in [-11, 11) for Seed Life, hashed from a founder number and an axis (the driver
/// has no random generator of its own).
fn seed_jitter(n: u32, axis: u32) -> f32 {
    let mut h = (n as u64) << 1 | axis as u64;
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    (h >> 40) as f32 / (1u64 << 24) as f32 * 22.0 - 11.0
}
