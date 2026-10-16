//! Integration tests for GPU frustum culling and indirect draw buffer generation (`CreatureRenderer`)

use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::creature_render::{CreatureLayerKind, CreatureRenderParams, CreatureRenderer};
use clank_app::gpu::types::{GpuDartInstance, GpuSimParams};
use clank_app::sim::SimWorld;

fn run_cull(
    driver: &GpuComputeDriver,
    renderer: &mut CreatureRenderer,
    params: &GpuSimParams,
) -> (u32, Vec<GpuDartInstance>, [[u32; 4]; 4]) {
    let rp = CreatureRenderParams {
        clip_from_world: [[0.0; 4]; 4],
        camera_pos: params.camera_pos,
        camera_size: params.camera_size,
        world_size: params.world_size,
        slot_count: params.max_agents.max(params.agent_count),
        _pad: 0,
    };
    let mut enc = driver.device.create_command_encoder(&Default::default());
    renderer.encode_cull(
        &driver.device,
        &driver.queue,
        &mut enc,
        &driver.creature_source(),
        &rp,
    );
    driver.queue.submit([enc.finish()]);
    let args = renderer.readback_draw_args(&driver.device, &driver.queue);
    let (instances, _) = renderer.readback_visible(&driver.device, &driver.queue);
    (args[0][1], instances, args)
}

#[test]
fn test_gpu_frustum_culling_indirection() {
    let sim = SimWorld::new(42);
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;

    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let mut renderer = CreatureRenderer::new(&driver.device, driver.max_agents);

    let (mut states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    assert_eq!(params.agent_count, 72);

    // 1. Full arena view: all 72 living agents should be visible
    params.camera_pos = [450.0, 300.0];
    params.camera_size = [900.0, 600.0];
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let (visible_full, instances, args) = run_cull(&driver, &mut renderer, &params);
    assert_eq!(
        visible_full, 72,
        "Full view must include all 72 living agents"
    );
    assert_eq!(instances.len(), 72);
    for (layer_idx, layer) in CreatureLayerKind::ALL.iter().enumerate() {
        assert_eq!(
            args[layer_idx],
            [layer.vertices_per_instance(), 72, 0, 0],
            "Indirect args for {layer:?} must match vertex count and visible instance count"
        );
    }

    for inst in &instances {
        let id = inst.pad1[0] & 0x00FF_FFFF;
        assert!(
            states.iter().any(|s| s.id == id && s.visual_cache != 0),
            "Visible instance id {id} must correspond to a live uploaded agent"
        );
    }

    // 2. Tightly zoomed view: fewer agents should be visible
    params.camera_pos = [450.0, 300.0];
    params.camera_size = [100.0, 100.0];
    let (visible_zoomed, _, _) = run_cull(&driver, &mut renderer, &params);
    assert!(
        visible_zoomed < 72,
        "Zoomed view (100x100) must cull agents outside frustum (got {})",
        visible_zoomed
    );

    // 3. Toroidal corner wrapping: place agent 0 at (895, 595) and camera at (5, 5) with size (40, 40)
    // Toroidal distance dx = min(890, 900 - 890) = 10 <= 20 + 24, dy = 10 <= 20 + 24
    states[0].pos_vel[0] = 895.0;
    states[0].pos_vel[1] = 595.0;
    for i in 1..72 {
        states[i].pos_vel[0] = 450.0;
        states[i].pos_vel[1] = 300.0;
    }
    params.camera_pos = [5.0, 5.0];
    params.camera_size = [40.0, 40.0];
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let (visible_wrap, wrap_instances, _) = run_cull(&driver, &mut renderer, &params);
    assert_eq!(
        visible_wrap, 1,
        "Toroidal frustum culling must include agent wrapped across corner"
    );
    assert_eq!(
        wrap_instances[0].pad1[0] & 0x00FF_FFFF,
        states[0].id,
        "Wrapped agent must be agent 0"
    );

    // 4. Dead agent exclusion: mark agent 0 as dead via meta_flags (bit 13) and visual_cache = 0
    states[0].meta_flags |= 1 << 13;
    states[0].visual_cache = 0;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let (visible_dead, _, _) = run_cull(&driver, &mut renderer, &params);
    assert_eq!(visible_dead, 0, "Dead agents must be excluded from culling");
}
