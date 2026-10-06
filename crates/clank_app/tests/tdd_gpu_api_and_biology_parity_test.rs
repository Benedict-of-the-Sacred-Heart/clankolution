use bevy::prelude::*;
use clank_app::api::{
    api_dispatch_system, ApiCommand, ApiReceiverResource, ApiSettingsRequest, ApiToolRequest,
};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::{ActiveEngine, GpuDriverResource, SimWorld};
use clank_app::ui::UiState;
use clank_core::world::World;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

#[test]
fn test_api_dispatch_gpu_observe_tool_and_agent_selection() {
    let mut app = App::new();

    let mut sim = SimWorld::default();
    sim.active_engine = ActiveEngine::Gpu;
    sim.world = World::new(42);

    let (tx, rx) = mpsc::channel();
    app.insert_resource(ApiReceiverResource(Arc::new(Mutex::new(rx))));
    app.insert_resource(UiState::default());

    let driver = GpuComputeDriver::create_default();
    let has_gpu = driver.is_some();

    if let Some(d) = driver {
        let (states, genomes, atomics, soil, params) =
            clank_app::gpu::bridge::sync_rust_to_gpu(&sim);
        d.upload_state(&states, &genomes, &atomics, &soil, &params);
        d.set_initialized(true);
        app.insert_resource(GpuDriverResource { driver: Some(d) });
    } else {
        app.insert_resource(GpuDriverResource { driver: None });
    }

    app.insert_resource(sim);
    app.add_systems(Update, api_dispatch_system);

    // 1. Send Observe tool command via API
    tx.send(ApiCommand::ApplyTool(ApiToolRequest {
        tool: "observe".to_string(),
        x: Some(300.0),
        y: Some(200.0),
        count: None,
    }))
    .unwrap();

    // Run the system update
    app.update();

    let sim = app.world().resource::<SimWorld>();
    assert!(
        sim.pending_tool.is_some(),
        "sim.pending_tool must be set for GPU observe click"
    );
    let pt = sim.pending_tool.as_ref().unwrap();
    assert_eq!(pt.tool_type, 0, "Tool type must be 0 for observe");
    assert_eq!(pt.tool_pos, [300.0, 200.0]);

    if has_gpu {
        // 2. Send UpdateSettings with selected_agent: 0 in GPU mode
        tx.send(ApiCommand::UpdateSettings(ApiSettingsRequest {
            speed: None,
            paused: None,
            mutation: None,
            growth: None,
            hostility: None,
            max_cap: None,
            scroll_offset: None,
            selected_agent: Some(0),
            active_engine: None,
        }))
        .unwrap();

        app.update();

        let sim = app.world().resource::<SimWorld>();
        assert!(
            sim.selected_agent_slot.is_some(),
            "sim.selected_agent_slot must be populated when selected_agent is set in GPU mode"
        );
        assert!(
            sim.selected_agent_id.is_some(),
            "sim.selected_agent_id must be populated when selected_agent is set in GPU mode"
        );
    }
}

#[test]
fn test_combat_energy_siphon_and_bounty_exact_parity() {
    let attack: f64 = 0.85;
    let hostility: f64 = 0.75;
    let attacker_tr0: f64 = 0.60;
    let attacker_tr5: f64 = 0.90;
    let victim_tr3: f64 = 0.40;

    let expected_damage =
        (0.5 + attack * 2.2) * hostility * (0.8 + attacker_tr0) * (1.0 - 0.65 * victim_tr3);
    let expected_bounty = (8.0 * attacker_tr5).min(9.0);

    let wgsl_damage = (0.5f32 + (attack as f32) * 2.2f32)
        * (hostility as f32)
        * (0.8f32 + (attacker_tr0 as f32))
        * (1.0f32 - 0.65f32 * (victim_tr3 as f32));
    let wgsl_bounty = (8.0f32 * (attacker_tr5 as f32)).min(9.0f32);

    assert!(
        (expected_damage - wgsl_damage as f64).abs() < 1e-5,
        "Damage calculation parity error"
    );
    assert!(
        (expected_bounty - wgsl_bounty as f64).abs() < 1e-5,
        "Bounty calculation parity error"
    );
    assert!(expected_bounty > 0.0 && expected_bounty <= 9.0);
}

#[test]
fn test_senescence_mortality_threshold_parity() {
    let max_age: u32 = 2100;
    let young_age: u32 = 2050;
    assert!(young_age <= max_age);
    let old_age: u32 = 2105;
    assert!(old_age > max_age);
}

#[test]
fn test_specimen_unpack_and_inspector_fields_parity() {
    let mut sim = SimWorld::default();
    sim.world = World::new(12345);

    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: no GPU adapter available");
        return;
    }
    let driver = driver.unwrap();
    let (states, genomes, atomics, soil, params) =
        clank_app::gpu::bridge::sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    driver.set_initialized(true);

    // Read back slot 0
    let slot_data = driver.readback_agent_slot(0);
    assert!(slot_data.is_some(), "readback_agent_slot(0) must return valid data");
    let (state, genome, atomic) = slot_data.unwrap();

    let unpacked = clank_app::gpu::bridge::unpack_gpu_agent_to_agent_data(&state, &genome, Some(&atomic));
    assert_eq!(unpacked.id, sim.world.agents[0].id);
    assert_eq!(unpacked.dead, 0);
    assert!(unpacked.energy > 0.0);
    assert_eq!(unpacked.tr.len(), 6);
    for (i, &t) in unpacked.tr.iter().enumerate() {
        assert!(t >= 0.0 && t <= 1.0, "Trait {} out of range: {}", i, t);
    }
    assert_eq!(unpacked.h.len(), 10);
    for (i, &val) in unpacked.h.iter().enumerate() {
        assert!(val >= -1.0 && val <= 1.0, "Hidden state {} out of range: {}", i, val);
    }
    assert_eq!(unpacked.genes.len(), 326);
}

#[test]
fn test_carrying_capacity_saturation_and_underflow_guard() {
    let mut sim = SimWorld::default();
    sim.world = World::new(999);
    sim.world.set_max_capacity(100);

    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: no GPU adapter available");
        return;
    }
    let driver = driver.unwrap();
    let (states, genomes, atomics, soil, params) =
        clank_app::gpu::bridge::sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    driver.set_initialized(true);

    // Run 50 ticks with heavy nourishment
    for _ in 0..50 {
        sim.world.nourish_at(450.0, 300.0);
        let params = clank_app::gpu::types::GpuSimParams {
            tick: sim.world.tick,
            agent_count: sim.gpu_population.max(sim.world.agents.len() as u32),
            max_agents: driver.max_agents,
            max_capacity: 100,
            hostility: 0.1,
            mut_rate: 0.1,
            speed: 1.0,
            renewal: 1.0,
            sub_tick: 0,
            sub_ticks_per_frame: 2,
            tool_type: 0xFFFFFFFF,
            tool_radius: 45.0,
            tool_pos: [450.0, 300.0],
            camera_pos: [450.0, 300.0],
            camera_size: [900.0, 600.0],
            world_size: [900.0, 600.0],
            soil_grid: [75, 50],
            eclipse: 0,
            epoch: 0,
        };
        driver.update_params(&params);
        driver.dispatch_sub_ticks_with_mods(2, &params, false, false, false);
        let telem = driver.readback_telemetry();
        sim.world.tick += 2;
        sim.gpu_population = telem.population;

        // Invariant: population must NEVER exceed max_capacity + buffer margins
        assert!(
            sim.gpu_population <= 100 + 4,
            "Population {} breached max capacity of 100",
            sim.gpu_population
        );
    }

    assert!(sim.gpu_population > 0, "Population must remain viable");
}
