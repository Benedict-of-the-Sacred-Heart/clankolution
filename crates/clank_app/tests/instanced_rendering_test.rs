//! Integration tests for GPU instanced creature rendering pipeline (`CreatureRenderer`)

use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::creature_render::{
    CreatureDrawKey, CreatureLayerKind, CreatureRenderParams, CreatureRenderer,
};
use clank_app::sim::SimWorld;

#[test]
fn test_gpu_instanced_dart_pipeline_compiles_and_renders() {
    let sim = SimWorld::new(42);
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;

    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let mut renderer = CreatureRenderer::new(&driver.device, driver.max_agents);

    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.camera_pos = [450.0, 300.0];
    params.camera_size = [900.0, 600.0];
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let clip_from_world =
        bevy::math::Mat4::orthographic_rh(0.0, 900.0, 0.0, 600.0, -1000.0, 1000.0)
            .to_cols_array_2d();
    let rp = CreatureRenderParams {
        clip_from_world,
        camera_pos: params.camera_pos,
        camera_size: params.camera_size,
        world_size: params.world_size,
        slot_count: params.max_agents.max(params.agent_count),
        _pad: 0,
    };

    // Create offscreen render target texture (256x256, 256-byte row aligned)
    let width = 256u32;
    let height = 256u32;
    let target_tex = driver.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_dart_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let draw_key = CreatureDrawKey {
        color_format: wgpu::TextureFormat::Rgba8Unorm,
        samples: 1,
        depth: None,
    };
    renderer.ensure_draw_pipelines(&driver.device, draw_key);

    let mut enc = driver.device.create_command_encoder(&Default::default());
    renderer.encode_cull(
        &driver.device,
        &driver.queue,
        &mut enc,
        &driver.creature_source(),
        &rp,
    );
    {
        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("instanced_dart_test_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for layer in CreatureLayerKind::ALL {
            renderer.draw_layer(&mut rpass, draw_key, layer);
        }
    }
    driver.queue.submit([enc.finish()]);

    let args = renderer.readback_draw_args(&driver.device, &driver.queue);
    assert_eq!(args[CreatureLayerKind::Body as usize][1], 72);

    // Readback offscreen texture to verify non-zero pixels were rendered
    let staging = driver.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen_staging"),
        size: (width * height * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = driver
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_offscreen"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    driver.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = sender.send(res);
    });
    driver
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let non_zero_pixels = data.chunks_exact(4).filter(|px| px[3] > 0).count();
    drop(data);
    staging.unmap();

    assert!(
        non_zero_pixels > 0,
        "Instanced dart render pass must produce non-empty pixels in offscreen texture"
    );
    assert_eq!(
        CreatureLayerKind::Body.vertices_per_instance(),
        6,
        "Dart body template must have 6 vertices (2 triangles for concave dart polygon)"
    );
    assert_eq!(
        CreatureLayerKind::Outline.vertices_per_instance(),
        44,
        "Dart outline layer must have 44 vertices (8 body edges + 4 antennae + 32 halo)"
    );
}
