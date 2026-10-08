#[path = "support/cpu_creature_mesh.rs"]
mod cpu_creature_mesh;

use clank_app::gpu::bridge::render_snapshot;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::creature_render::*;
use clank_app::gpu::types::{
    GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuAgentTrail, GpuSimParams, GpuSoilCell,
};
use clank_app::sim::SimWorld;

const W: u32 = 900;
const H: u32 = 600;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.03,
    g: 0.07,
    b: 0.08,
    a: 1.0,
};

fn ortho(world: [f32; 2]) -> [[f32; 4]; 4] {
    bevy::math::Mat4::orthographic_rh(0.0, world[0], 0.0, world[1], -1000.0, 1000.0)
        .to_cols_array_2d()
}

fn target(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn begin_clear_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    view: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("test_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(CLEAR),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn read_rgba(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let row = W * 4;
    let padded = row.div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_readback"),
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((row * H) as usize);
    for y in 0..H {
        let start = (y * padded) as usize;
        out.extend_from_slice(&data[start..start + row as usize]);
    }
    out
}

const REFERENCE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> clip_from_world: mat4x4f;
struct VOut { @builtin(position) clip: vec4f, @location(0) color: vec4f }
@vertex fn vs(@location(0) pos: vec3f, @location(1) color: vec4f) -> VOut {
    return VOut(clip_from_world * vec4f(pos, 1.0), color);
}
@fragment fn fs(in: VOut) -> @location(0) vec4f { return in.color; }
"#;

type Layer = (wgpu::PrimitiveTopology, Vec<[f32; 3]>, Vec<[f32; 4]>);

/// Draws CPU-built vertex lists, in order, with the blend Bevy's ColorMaterial meshes used
/// (vertex color × white, `BlendState::ALPHA_BLENDING`).
fn draw_reference(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    clip: [[f32; 4]; 4],
    layers: &[Layer],
) {
    use wgpu::util::DeviceExt;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reference_sm"),
        source: wgpu::ShaderSource::Wgsl(REFERENCE_WGSL.into()),
    });
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("reference_clip"),
        contents: bytemuck::cast_slice(&clip),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = |topology| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reference_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let buffers: Vec<_> = layers
        .iter()
        .map(|(topology, pos, col)| {
            let p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(pos),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(col),
                usage: wgpu::BufferUsages::VERTEX,
            });
            (pipeline(*topology), p, c, pos.len() as u32)
        })
        .collect();
    let view = tex.create_view(&Default::default());
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = begin_clear_pass(&mut enc, &view);
        pass.set_bind_group(0, &bg, &[]);
        for (pipeline, p, c, n) in &buffers {
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, p.slice(..));
            pass.set_vertex_buffer(1, c.slice(..));
            pass.draw(0..*n, 0..1);
        }
    }
    queue.submit([enc.finish()]);
}

fn differing_pixel_fraction(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() / 4;
    let diff = (0..n)
        .filter(|i| (0..4).any(|c| (a[i * 4 + c] as i32 - b[i * 4 + c] as i32).abs() > 2))
        .count();
    diff as f64 / n as f64
}

fn full_view_params(slot_count: u32) -> CreatureRenderParams {
    let world = [W as f32, H as f32];
    CreatureRenderParams {
        clip_from_world: ortho(world),
        camera_pos: [W as f32 * 0.5, H as f32 * 0.5],
        camera_size: world,
        world_size: world,
        slot_count,
        _pad: 0,
    }
}

const KEY: CreatureDrawKey = CreatureDrawKey {
    color_format: FORMAT,
    samples: 1,
    depth: None,
};

/// The GPU renderer draws the same pixels the CPU creature meshes did: glows, per-tick trails,
/// dart bodies, outlines, antennae and birth halos, layered back to front.
#[test]
fn test_gpu_layers_match_the_cpu_meshes_pixel_for_pixel() {
    let Some(driver) = GpuComputeDriver::create_for_world(75, 50, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let mut sim = SimWorld::new(42);
    sim.force_step(30); // trails and motion
                        // Detail that only some creatures show: halos, red borders, long antennae, strong glow.
    for (k, a) in sim.world.agents.iter_mut().enumerate().take(12) {
        a.birth = 20 + 6 * k as u32;
        a.attack = if k % 2 == 0 { 0.8 } else { 0.3 };
        a.signal = 0.1 * (k % 10) as f64;
        a.tr[2] = 0.9;
    }
    let (states, trails) = render_snapshot(&sim);
    let slot_count = driver.upload_render_snapshot(&states, &trails);
    assert_eq!(slot_count as usize, sim.world.agents.len());

    let (device, queue) = (driver.device.as_ref(), driver.queue.as_ref());
    let mut renderer = CreatureRenderer::new(device, driver.max_agents);
    renderer.ensure_draw_pipelines(device, KEY);
    let rp = full_view_params(slot_count);
    let source = driver.creature_source();

    let gpu_tex = target(device);
    let view = gpu_tex.create_view(&Default::default());
    let mut enc = device.create_command_encoder(&Default::default());
    renderer.encode_cull(device, queue, &mut enc, &source, &rp);
    {
        let mut pass = begin_clear_pass(&mut enc, &view);
        for layer in CreatureLayerKind::ALL {
            renderer.draw_layer(&mut pass, KEY, layer);
        }
    }
    queue.submit([enc.finish()]);

    let (instances, gathered) = renderer.readback_visible(device, queue);
    let living = sim.world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(
        instances.len(),
        living,
        "whole world in view: every creature visible"
    );
    for (inst, t) in instances.iter().zip(&gathered) {
        assert_eq!(
            t.owner_id & 0x00FF_FFFF,
            inst.pad1[0] & 0x00FF_FFFF,
            "gathered trail belongs to the dart"
        );
    }
    assert!(gathered.iter().any(|t| t.count >= 2), "trails present");

    let (h, w) = (H as f32, W as f32);
    let (mut gp, mut gc, mut tp, mut tc, mut bp, mut bc, mut op, mut oc) = (
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
    );
    // The frozen CPU meshes read a packed colour where the GPU instance carries the root.
    let cpu_instances: Vec<_> = instances
        .iter()
        .map(|i| {
            let mut c = *i;
            c.vis_data[0] = clank_app::gpu::bridge::lineage_packed_color(c.vis_data[0]);
            c
        })
        .collect();
    cpu_creature_mesh::generate_glow_mesh_from_instances(&cpu_instances, h, &mut gp, &mut gc);
    cpu_creature_mesh::generate_swarm_trails_from_gpu(
        &cpu_instances,
        &gathered,
        w,
        h,
        &mut tp,
        &mut tc,
    );
    cpu_creature_mesh::generate_dart_mesh_from_instances(&cpu_instances, h, &mut bp, &mut bc);
    cpu_creature_mesh::generate_outline_mesh_from_instances(&cpu_instances, h, &mut op, &mut oc);
    let ref_tex = target(device);
    use wgpu::PrimitiveTopology::{LineList, TriangleList};
    draw_reference(
        device,
        queue,
        &ref_tex,
        rp.clip_from_world,
        &[
            (TriangleList, gp, gc),
            (LineList, tp, tc),
            (TriangleList, bp, bc),
            (LineList, op, oc),
        ],
    );

    let gpu_px = read_rgba(device, queue, &gpu_tex);
    let ref_px = read_rgba(device, queue, &ref_tex);
    let clear_px: Vec<u8> = ref_px[..4].repeat(ref_px.len() / 4);
    let drawn = differing_pixel_fraction(&ref_px, &clear_px);
    assert!(
        drawn > 0.01,
        "the reference drew creatures ({:.2}% of pixels)",
        drawn * 100.0
    );
    // Identical on the reference machine; the margin (about 54 pixels) only absorbs GPU trig
    // differences. The smallest real defect measured (border colour) changes about 200 pixels.
    let frac = differing_pixel_fraction(&gpu_px, &ref_px);
    assert!(
        frac < 0.0001,
        "GPU layers differ from the CPU meshes on {:.4}% of pixels",
        frac * 100.0
    );
}

fn agent(id: u32, x: f32, y: f32) -> GpuAgentState {
    GpuAgentState {
        pos_vel: [x, y, 0.0, 0.0],
        angle_energy: [0.0, 50.0, 0.0, 0.0],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id,
        meta_flags: 0,
        age_gen: 0,
        morton_code: 0,
        root: 0,
        visual_cache: 0x00FF_00FF,
    }
}

/// The cull keeps exactly the living creatures inside the (toroidal) camera rectangle, within
/// `slot_count`, and writes each layer's indirect args for that many instances.
#[test]
fn test_cull_respects_camera_dead_flags_and_writes_indirect_args() {
    let Some(driver) = GpuComputeDriver::create_for_world(75, 50, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let live = GpuAgentAtomic {
        energy_milli: 50_000,
        mate_claim: 0,
        mate_energy_milli: 0,
        dead_claimed: 0,
    };
    let mut states = vec![
        agent(1, 30.0, 300.0),  // in view
        agent(2, 890.0, 310.0), // in view across the x seam
        agent(3, 450.0, 300.0), // outside the camera
        agent(4, 30.0, 300.0),  // dead flag
        agent(5, 30.0, 300.0),  // no visual (just died this tick)
        agent(6, 30.0, 300.0),  // death already claimed
        agent(7, 30.0, 300.0),  // beyond slot_count
    ];
    states[3].meta_flags = 1 << 13;
    states[3].visual_cache = 0;
    states[4].visual_cache = 0;
    let mut atomics = vec![live; states.len()];
    atomics[5].dead_claimed = 1;
    let genomes = vec![
        GpuAgentGenome {
            packed_genes: [0; 88]
        };
        states.len()
    ];
    let soil = vec![
        GpuSoilCell {
            food_micro: 0,
            taint_micro: 0,
            scent_micro: 0,
            fertility_micro: 1_000_000,
        };
        75 * 50
    ];
    let params = GpuSimParams {
        max_agents: driver.max_agents,
        world_size: [900.0, 600.0],
        soil_grid: [75, 50],
        ..Default::default()
    };
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let (device, queue) = (driver.device.as_ref(), driver.queue.as_ref());
    let mut renderer = CreatureRenderer::new(device, driver.max_agents);
    let rp = CreatureRenderParams {
        camera_pos: [20.0, 300.0],
        camera_size: [200.0, 200.0],
        slot_count: 6,
        ..full_view_params(6)
    };
    let mut enc = device.create_command_encoder(&Default::default());
    renderer.encode_cull(device, queue, &mut enc, &driver.creature_source(), &rp);
    queue.submit([enc.finish()]);

    let (instances, _) = renderer.readback_visible(device, queue);
    let mut ids: Vec<u32> = instances.iter().map(|i| i.pad1[0] & 0x00FF_FFFF).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2]);
    assert_eq!(
        renderer.readback_draw_args(device, queue),
        [[60, 2, 0, 0], [16, 2, 0, 0], [6, 2, 0, 0], [44, 2, 0, 0]],
        "glow, trail, body and outline args for the two visible creatures"
    );

    // A second cull with nothing in view resets the count rather than accumulating.
    let empty = CreatureRenderParams {
        camera_pos: [450.0, 100.0],
        camera_size: [10.0, 10.0],
        ..rp
    };
    let mut enc = device.create_command_encoder(&Default::default());
    renderer.encode_cull(device, queue, &mut enc, &driver.creature_source(), &empty);
    queue.submit([enc.finish()]);
    assert_eq!(renderer.readback_draw_args(device, queue)[2], [6, 0, 0, 0]);
}

/// A slot's trail is drawn only for the creature that wrote it; a reused slot starts empty.
#[test]
fn test_cull_drops_trails_of_previous_slot_owners() {
    let Some(driver) = GpuComputeDriver::create_for_world(75, 50, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let states = vec![agent(10, 100.0, 100.0), agent(11, 200.0, 200.0)];
    let mut trails = vec![GpuAgentTrail::default(); 2];
    trails[0].owner_id = 10;
    trails[0].count = 3;
    trails[1].owner_id = 99; // written by an earlier occupant
    trails[1].count = 5;
    let n = driver.upload_render_snapshot(&states, &trails);

    let (device, queue) = (driver.device.as_ref(), driver.queue.as_ref());
    let mut renderer = CreatureRenderer::new(device, driver.max_agents);
    let mut enc = device.create_command_encoder(&Default::default());
    renderer.encode_cull(
        device,
        queue,
        &mut enc,
        &driver.creature_source(),
        &full_view_params(n),
    );
    queue.submit([enc.finish()]);
    let (instances, gathered) = renderer.readback_visible(device, queue);
    let count_of = |id: u32| {
        let i = instances
            .iter()
            .position(|inst| inst.pad1[0] & 0x00FF_FFFF == id)
            .unwrap();
        gathered[i].count
    };
    assert_eq!(count_of(10), 3);
    assert_eq!(count_of(11), 0);
}
