use bevy::prelude::*;
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};

fn app_with_driver(engine: ActiveEngine) -> Option<App> {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    app.world_mut().resource_mut::<SimWorld>().active_engine = engine;
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return None;
    }
    Some(app)
}

fn gpu_initialized(app: &App) -> bool {
    let gpu = app.world().resource::<GpuDriverResource>();
    gpu.driver.as_ref().unwrap().is_initialized()
}

/// Switching to the GPU while paused hands the world over at once, so the panel shows the
/// world's own population and lineages rather than numbers left from before.
#[test]
fn test_switching_to_gpu_while_paused_shows_the_worlds_numbers() {
    let Some(mut app) = app_with_driver(ActiveEngine::Rust) else {
        return;
    };
    for _ in 0..10 {
        app.update();
    }
    let (population, lineages, generation) = {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = true;
        let numbers = (
            sim.active_population(),
            sim.living_lineages(),
            sim.max_generation(),
        );
        sim.active_engine = ActiveEngine::Gpu;
        numbers
    };
    app.update();

    let sim = app.world().resource::<SimWorld>();
    assert!(
        gpu_initialized(&app),
        "the GPU holds the world after the switch"
    );
    assert_eq!(
        (
            sim.active_population(),
            sim.living_lineages(),
            sim.max_generation()
        ),
        (population, lineages, generation)
    );
}

/// Switching back to Rust while paused copies the GPU's world back at once, so picking and
/// the panel use the world as it is now, not as it was when it went to the GPU.
#[test]
fn test_switching_to_rust_while_paused_brings_the_gpu_world_back() {
    let Some(mut app) = app_with_driver(ActiveEngine::Gpu) else {
        return;
    };
    app.world_mut().resource_mut::<SimWorld>().speed = 8;
    for _ in 0..30 {
        app.update();
    }
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    // Let the frames still in flight land, so the GPU numbers are final.
    for _ in 0..3 {
        app.update();
    }
    let (population, lineages) = {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        let numbers = (sim.active_population(), sim.living_lineages());
        sim.active_engine = ActiveEngine::Rust;
        numbers
    };
    app.update();

    let sim = app.world().resource::<SimWorld>();
    assert!(
        !gpu_initialized(&app),
        "the Rust engine owns the world again"
    );
    assert_eq!(
        (sim.active_population(), sim.living_lineages()),
        (population, lineages)
    );
}

fn gpu_next_id(app: &App) -> u32 {
    let gpu = app.world().resource::<GpuDriverResource>();
    gpu.driver
        .as_ref()
        .unwrap()
        .readback_telemetry()
        .apex_agent_id
}

/// The GPU numbers newborns on from the world's own counter, as the reference does (1, 2, 3...),
/// not from a floor or a counter left over from an earlier world.
#[test]
fn test_gpu_numbers_creatures_on_from_the_worlds_counter() {
    let Some(mut app) = app_with_driver(ActiveEngine::Rust) else {
        return;
    };
    let next_id = {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = true;
        // The newest creature has died and gone: its id must still never be reused.
        let newest = sim.world.agents.iter().map(|a| a.id).max().unwrap();
        sim.world.agents.retain(|a| a.id != newest);
        sim.active_engine = ActiveEngine::Gpu;
        sim.world.next_id
    };
    app.update();
    assert_eq!(gpu_next_id(&app), next_id, "after switching to the GPU");

    // Run the GPU a while so its counter moves on, then start a NEW WORLD.
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = false;
        sim.speed = 8;
    }
    for _ in 0..30 {
        app.update();
    }
    let next_id = {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = true;
        sim.reset();
        sim.world.next_id
    };
    app.update();
    assert_eq!(gpu_next_id(&app), next_id, "after a NEW WORLD");
}

/// Ids the GPU issued stay spent when the world comes back to Rust, even when those creatures
/// are already gone (born and died in frames whose readbacks had not landed yet).
#[test]
fn test_switching_to_rust_keeps_the_gpus_spent_ids() {
    let Some(mut app) = app_with_driver(ActiveEngine::Gpu) else {
        return;
    };
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    app.update();
    // The GPU numbered creatures up to 4,999 that are no longer alive.
    app.world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap()
        .raise_next_agent_id(5000);
    app.world_mut().resource_mut::<SimWorld>().active_engine = ActiveEngine::Rust;
    app.update();

    assert_eq!(app.world().resource::<SimWorld>().world.next_id, 5000);
}
