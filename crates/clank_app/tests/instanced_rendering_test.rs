use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::{GpuAgentState, GpuAgentGenome, GpuAgentAtomic, GpuSoilCell, GpuSimParams};

#[test]
fn test_instanced_dart_rendering_zero_cpu_vertices() {
    let driver = GpuComputeDriver::create_for_world(75, 50, 65536);
    if driver.is_none() {
        eprintln!("No WebGPU adapter found, skipping instanced rendering test");
        return;
    }
    let driver = driver.unwrap();

    let mut states = Vec::new();
    let mut atomics = Vec::new();

    // Living visible agent: centered at [450, 300], angle 0.0, color cyan (0x00FFFF)
    let a0 = GpuAgentState {
        pos_vel: [450.0, 300.0, 0.0, 0.0],
        angle_energy: [0.0, 50.0, 0.0, 0.0],
        traits: [0.5; 8],
        hidden: [0.0; 10],
        id: 1,
        meta_flags: 0,
        age_gen: 0,
        morton_code: 0,
        packed_color: 0xFFFFFF00, // RGBA cyan/yellow
        visual_cache: 0x00FF00FF, // alive, radius 128
    };
    states.push(a0);
    atomics.push(GpuAgentAtomic { energy_milli: 50000, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 0 });

    let genomes = vec![GpuAgentGenome { packed_genes: [0; 88] }; 1];
    let soil = vec![GpuSoilCell { food_milli: 0, taint_milli: 0, scent_milli: 0, fertility_milli: 1000 }; 75 * 50];

    let params = GpuSimParams {
        tick: 1,
        agent_count: 1,
        max_agents: 1,
        max_capacity: 1,
        camera_pos: [450.0, 300.0],
        camera_size: [900.0, 600.0],
        world_size: [900.0, 600.0],
        soil_grid: [75, 50],
        ..Default::default()
    };

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // 1. Dispatch culling pass on GPU
    let visible_count = driver.dispatch_culling(&params);
    assert_eq!(visible_count, 1);

    // 2. Execute GPU instanced rendering pass into target texture directly from VRAM
    let target_tex = driver.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_render_target"),
        size: wgpu::Extent3d { width: 900, height: 600, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Render directly using GPU instance stream and unit dart template (12 vertices)
    driver.render_darts_instanced(&target_view, &params, visible_count);

    // Verify template mesh contains exactly 12 vertices (unit dart), zero CPU vertex bloat
    assert_eq!(driver.dart_template_vertex_count(), 12);
}
