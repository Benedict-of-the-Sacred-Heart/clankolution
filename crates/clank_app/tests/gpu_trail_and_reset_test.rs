use bevy::prelude::*;
use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::creature_render::{CreatureRenderParams, CreatureRenderer};
use clank_app::gpu::types::{GpuAgentTrail, GpuDartInstance, GpuSimParams};
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};

fn toroidal_step(a: [f32; 2], b: [f32; 2], w: f32, h: f32) -> f32 {
    let dx = (a[0] - b[0]).abs();
    let dy = (a[1] - b[1]).abs();
    dx.min(w - dx).hypot(dy.min(h - dy))
}

fn cull_visible(
    driver: &GpuComputeDriver,
    renderer: &mut CreatureRenderer,
    params: &GpuSimParams,
) -> (Vec<GpuDartInstance>, Vec<GpuAgentTrail>) {
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
    renderer.readback_visible(&driver.device, &driver.queue)
}

/// Trails are the last nine per-tick positions (clankolution.html `a.trail.push([a.x,a.y])`
/// once per tick, Rust `world.rs` likewise). Running twelve sub-ticks in one frame must not
/// stretch them: every segment is one tick of movement, never one frame's worth.
#[test]
fn test_gpu_trails_hold_per_tick_positions_at_any_speed() {
    let sim = SimWorld::new(42);
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let mut renderer = CreatureRenderer::new(&driver.device, driver.max_agents);
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.max_agents = driver.max_agents;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // A fresh upload carries no motion history.
    let (instances, trails) = cull_visible(&driver, &mut renderer, &params);
    assert_eq!(instances.len(), 72);
    assert!(
        trails.iter().all(|t| t.count == 0),
        "uploaded agents start with empty trails"
    );

    // One frame at speed 12.
    params.sub_ticks_per_frame = 12;
    driver.dispatch_sub_ticks(12, &params);
    params.agent_count = driver.readback_telemetry().population;

    let (instances, trails) = cull_visible(&driver, &mut renderer, &params);
    let visible = instances.len();
    assert!(visible > 0);
    assert_eq!(trails.len(), instances.len());

    let (w, h) = (sim.world_width as f32, sim.world_height as f32);
    let mut full_trails = 0;
    for (inst, trail) in instances.iter().zip(&trails) {
        let id = inst.pad1[0] & 0x00FF_FFFF;
        assert_eq!(
            trail.owner_id & 0x00FF_FFFF,
            id,
            "trail belongs to the instance's creature"
        );
        let n = trail.count as usize;
        assert!(
            (1..=9).contains(&n),
            "agent {id}: trail length {n} outside 1..=9"
        );
        assert_eq!(
            trail.pts[n - 1],
            [inst.pos_angle[0], inst.pos_angle[1]],
            "agent {id}: newest trail point is the current position"
        );
        for i in 1..n {
            let step = toroidal_step(trail.pts[i - 1], trail.pts[i], w, h);
            assert!(
                step < 4.0,
                "agent {id}: trail segment {i} spans {step:.2}px, more than one tick of motion"
            );
        }
        if n == 9 {
            full_trails += 1;
        }
    }
    assert!(
        full_trails > visible / 2,
        "survivors of 12 ticks carry full 9-point trails ({full_trails}/{visible})"
    );
}

fn gpu_slot_positions(app: &App, n: usize) -> Option<Vec<(u32, [f32; 2])>> {
    let driver = app
        .world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()?;
    Some(
        driver
            .readback_agent_states(n)
            .iter()
            .map(|s| (s.id, [s.pos_vel[0], s.pos_vel[1]]))
            .collect(),
    )
}

/// NEW WORLD in GPU mode must replace the creatures on the GPU, not just the CPU copy,
/// including while paused (the renderer draws straight from GPU buffers).
#[test]
fn test_new_world_resyncs_gpu_engine() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    app.world_mut().resource_mut::<SimWorld>().active_engine = ActiveEngine::Gpu;
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    for _ in 0..30 {
        app.update();
    }

    // Paused NEW WORLD: the GPU must hold the fresh creatures exactly.
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = true;
        sim.reset();
    }
    app.update();
    let fresh: Vec<(u32, [f32; 2])> = {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(sim.world.tick, 0);
        assert_eq!(sim.gpu_population, 72);
        sim.world
            .agents
            .iter()
            .map(|a| (a.id, [a.x as f32, a.y as f32]))
            .collect()
    };
    let gpu = gpu_slot_positions(&app, fresh.len()).unwrap();
    assert_eq!(
        gpu, fresh,
        "paused NEW WORLD must upload the fresh world to the GPU"
    );

    // Running NEW WORLD: one tick later every creature is within one tick of its spawn point.
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = false;
        sim.reset();
    }
    let fresh: Vec<(u32, [f32; 2])> = {
        let sim = app.world().resource::<SimWorld>();
        sim.world
            .agents
            .iter()
            .map(|a| (a.id, [a.x as f32, a.y as f32]))
            .collect()
    };
    app.update();
    let gpu = gpu_slot_positions(&app, fresh.len()).unwrap();
    for ((cpu_id, cpu_pos), (gpu_id, gpu_pos)) in fresh.iter().zip(&gpu) {
        assert_eq!(cpu_id, gpu_id);
        let step = toroidal_step(*cpu_pos, *gpu_pos, 900.0, 600.0);
        assert!(
            step < 4.0,
            "agent {cpu_id} is {step:.1}px from its new-world spawn point after one tick"
        );
    }
    let sim = app.world().resource::<SimWorld>();
    assert_eq!(sim.world.tick, 1);
    let telemetry = app
        .world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap()
        .readback_telemetry();
    assert_eq!(
        sim.world.births,
        72 + telemetry.total_births,
        "birth count restarts from the new world's 72 founders"
    );
}
