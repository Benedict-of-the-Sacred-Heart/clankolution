use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::rendering::generate_soil_rgba;
use clank_app::sim::SimWorld;

/// The soil sprite's texture format.
const SPRITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let (w, h) = (tex.width(), tex.height());
    let row = w * 4;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_readback"),
        size: (padded * h) as u64,
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
                rows_per_image: Some(h),
            },
        },
        tex.size(),
    );
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range();
    (0..h)
        .flat_map(|y| {
            let start = (y * padded) as usize;
            data[start..start + row as usize].to_vec()
        })
        .collect()
}

/// The GPU soil colormap is copied texture-to-texture into the soil sprite (no CPU round trip),
/// and shows exactly what the CPU colormap (`generate_soil_rgba`) makes of the same soil.
#[test]
fn test_soil_display_copies_into_the_sprite_and_matches_the_cpu_colormap() {
    let mut sim = SimWorld::new(42);
    sim.world.agents.clear(); // soil-only ticks
    let cells = sim.world.soil.food.len();
    for i in (0..cells).step_by(7) {
        sim.world.soil.taint[i] = 0.2 + (i % 5) as f32 * 0.3;
    }
    for i in (0..cells).step_by(11) {
        sim.world.soil.scent[i] = 0.1 + (i % 4) as f32 * 0.4;
    }
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.max_agents = driver.max_agents;
    params.sub_ticks_per_frame = 50;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    driver.dispatch_sub_ticks(50, &params);

    // CPU colormap of the soil the GPU ended with.
    let gpu_soil = driver.readback_soil();
    sync_gpu_to_rust(&[], &[], &[], &gpu_soil, &params, &mut sim);
    let mut cpu_rgba = vec![0u8; cells * 4];
    generate_soil_rgba(&sim.world.soil, &mut cpu_rgba);

    // GPU colormap, copied straight into a sprite-format texture.
    let (device, queue) = (driver.device.as_ref(), driver.queue.as_ref());
    let sprite = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sprite"),
        size: driver.soil_display_texture().size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SPRITE_FORMAT,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_texture(
        driver.soil_display_texture().as_image_copy(),
        sprite.as_image_copy(),
        sprite.size(),
    );
    queue.submit([enc.finish()]);
    let error = bevy::tasks::block_on(scope.pop());
    assert!(
        error.is_none(),
        "soil display copies into the sprite: {error:?}"
    );
    let gpu_rgba = read_texture(device, queue, &sprite);

    assert_eq!(gpu_rgba.len(), cpu_rgba.len());
    let mut differing = 0;
    for (i, (g, c)) in gpu_rgba.chunks(4).zip(cpu_rgba.chunks(4)).enumerate() {
        let max_diff = (0..4)
            .map(|k| (g[k] as i32 - c[k] as i32).abs())
            .max()
            .unwrap();
        assert!(max_diff <= 1, "texel {i}: GPU {g:?} vs CPU {c:?}");
        differing += (max_diff > 0) as usize;
    }
    assert!(
        differing * 500 <= cells,
        "{differing} of {cells} texels differ by one"
    );
}
