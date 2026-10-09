use bevy::prelude::*;
use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};

fn blocking_waits(app: &App) -> u64 {
    app.world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap()
        .blocking_readback_waits()
}

/// A running GPU-mode frame never stalls the CPU on a readback: telemetry, audio and the
/// selected creature arrive through the async ring a frame or two later.
#[test]
fn test_gpu_frames_make_no_blocking_readbacks() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    app.world_mut().resource_mut::<SimWorld>().active_engine = ActiveEngine::Gpu;
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    for _ in 0..5 {
        app.update();
    }

    let first_id = app.world().resource::<SimWorld>().world.agents[0].id;
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.selected_agent_slot = Some(0);
        sim.selected_agent_id = Some(first_id);
    }
    let before = blocking_waits(&app);
    for _ in 0..40 {
        app.update();
    }
    assert_eq!(
        blocking_waits(&app) - before,
        0,
        "GPU-mode frames must not block on readbacks"
    );

    let sim = app.world().resource::<SimWorld>();
    assert!(
        sim.gpu_population > 0,
        "population arrives through the ring"
    );
    assert!(
        sim.world.births >= 72,
        "birth totals arrive through the ring"
    );
    let cache = sim
        .selected_agent_cache
        .as_ref()
        .expect("selected creature arrives through the ring");
    assert_eq!(cache.id, first_id);
}

/// Per-frame counters (kills) are applied exactly once per frame: the CPU running total equals
/// the sum the GPU reported, with nothing dropped or double-counted by the ring.
#[test]
fn test_ring_applies_every_frame_exactly_once() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Gpu;
        sim.world.hostility = 100.0;
        sim.speed = 8;
    }
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    for _ in 0..60 {
        app.update();
    }
    // Stop simulating and let every in-flight frame drain.
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    for _ in 0..10 {
        app.update();
    }
    let sim = app.world().resource::<SimWorld>();
    let driver = app
        .world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap();
    let telemetry = driver.readback_telemetry();
    assert_eq!(
        sim.world.births,
        sim.gpu_births_base + telemetry.total_births,
        "every frame's birth total was applied"
    );
    assert_eq!(
        sim.gpu_population, telemetry.population,
        "the last frame's population was applied"
    );
    assert_eq!(
        driver.pending_frame_readbacks(),
        0,
        "nothing left in flight after draining"
    );
}

/// The ring hands back every requested frame exactly once, oldest first, each holding what the
/// GPU reported at the end of that frame. Per-frame counters (kills) are summed by the app, so a
/// dropped or repeated frame would corrupt them.
#[test]
fn test_ring_returns_each_frame_once_in_order() {
    let mut sim = SimWorld::new(7);
    sim.world.hostility = 100.0;
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 4096) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.max_agents = driver.max_agents;
    params.sub_ticks_per_frame = 4;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    const FRAMES: usize = 30;
    let mut frames = Vec::new();
    for _ in 0..FRAMES {
        driver.dispatch_sub_ticks(4, &params);
        driver.request_frame_readback(Some(0), false);
        frames.extend(driver.poll_frame_readbacks());
    }
    // A synchronous read waits for the GPU to go idle, so every frame has landed.
    let telemetry = driver.readback_telemetry();
    frames.extend(driver.poll_frame_readbacks());

    assert_eq!(frames.len(), FRAMES, "each requested frame returned once");
    assert_eq!(driver.pending_frame_readbacks(), 0);
    assert!(
        frames
            .windows(2)
            .all(|w| w[0].telemetry.total_births <= w[1].telemetry.total_births),
        "frames arrive in request order"
    );
    let last = frames.last().unwrap();
    assert_eq!(last.telemetry.total_births, telemetry.total_births);
    assert_eq!(last.telemetry.population, telemetry.population);
    assert_eq!(last.telemetry.kills, telemetry.kills);
    let sel = last.selected.expect("slot 0 was requested");
    assert_eq!(sel.slot, 0);
    assert_eq!(sel.state, driver.readback_agent_states(1)[0]);
    assert_eq!(Some(sel.trail), driver.readback_agent_trail(0));
}

/// Agent replenishment (HTML line 1464) tops a collapsed population up to 15 once. Readbacks
/// requested before the seeding still report the old count and must not trigger it again.
#[test]
fn test_agent_replenishment_seeds_once_despite_lagged_frames() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Gpu;
        sim.speed = 32;
        sim.world.agents.truncate(3);
    }
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    for _ in 0..4 {
        app.update();
    }
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    for _ in 0..4 {
        app.update();
    }
    let sim = app.world().resource::<SimWorld>();
    let driver = app
        .world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap();
    let telemetry = driver.readback_telemetry();
    let born = telemetry.total_births;
    assert!(
        telemetry.population <= 15 + born,
        "population {} overshot 15 (+{born} births): agents seeded more than once",
        telemetry.population
    );
    assert_eq!(sim.gpu_population, telemetry.population);
}
