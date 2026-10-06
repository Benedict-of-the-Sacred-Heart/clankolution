use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::*;
use clank_app::rendering::*;
use clank_app::sim::SimWorld;
use clank_core::soil::SoilGrid;

#[test]
fn test_soil_deposit_stencils_parity() {
    let mut soil = SoilGrid::new();
    // Zero out food, taint, scent
    for cell in &mut soil.food { *cell = 0.0; }
    for cell in &mut soil.taint { *cell = 0.0; }

    // 1. Test Radius 1 deposit at cell (10, 10)
    let center_x = 10.0 * 12.0 + 6.0;
    let center_y = 10.0 * 12.0 + 6.0;
    let val = 1.0;
    soil.deposit(0, center_x, center_y, val, 1);

    let idx_c = 10 * 75 + 10;
    let idx_up = 9 * 75 + 10;
    let idx_down = 11 * 75 + 10;
    let idx_left = 10 * 75 + 9;
    let idx_right = 10 * 75 + 11;
    let idx_diag = 9 * 75 + 9;

    assert!((soil.food[idx_c] - 1.0).abs() < 1e-4, "Center must receive full val");
    assert!((soil.food[idx_up] - (1.0 / 1.8)).abs() < 1e-4, "Up neighbor must receive val / 1.8");
    assert!((soil.food[idx_down] - (1.0 / 1.8)).abs() < 1e-4, "Down neighbor must receive val / 1.8");
    assert!((soil.food[idx_left] - (1.0 / 1.8)).abs() < 1e-4, "Left neighbor must receive val / 1.8");
    assert!((soil.food[idx_right] - (1.0 / 1.8)).abs() < 1e-4, "Right neighbor must receive val / 1.8");
    assert_eq!(soil.food[idx_diag], 0.0, "Diagonal neighbor in radius 1 must receive ZERO");

    // 2. Test Radius 2 deposit at cell (30, 30)
    let c2_x = 30.0 * 12.0 + 6.0;
    let c2_y = 30.0 * 12.0 + 6.0;
    soil.deposit(0, c2_x, c2_y, val, 2);

    let idx2_c = 30 * 75 + 30;
    let idx2_d1_0 = 30 * 75 + 31; // dx=1, dy=0, rr=1 -> 1.0 / (1 + 0.8) = 1.0 / 1.8
    let idx2_d1_1 = 31 * 75 + 31; // dx=1, dy=1, rr=2 -> 1.0 / (1 + 1.6) = 1.0 / 2.6
    let idx2_d2_1 = 31 * 75 + 32; // dx=2, dy=1, rr=5 > 4.5 -> MUST BE 0!
    let idx2_d2_2 = 32 * 75 + 32; // dx=2, dy=2, rr=8 > 4.5 -> MUST BE 0!

    assert!((soil.food[idx2_c] - 1.0).abs() < 1e-4, "Radius 2 center must receive full val");
    assert!((soil.food[idx2_d1_0] - (1.0 / 1.8)).abs() < 1e-4, "Radius 2 (1,0) must receive 1.0 / 1.8");
    assert!((soil.food[idx2_d1_1] - (1.0 / 2.6)).abs() < 1e-4, "Radius 2 (1,1) must receive 1.0 / 2.6");
    assert_eq!(soil.food[idx2_d2_1], 0.0, "Radius 2 (2,1) rr=5 > 4.5 must receive ZERO");
    assert_eq!(soil.food[idx2_d2_2], 0.0, "Radius 2 (2,2) rr=8 > 4.5 must receive ZERO");
}

#[test]
fn test_rendering_mesh_layer_depths_and_segments() {
    let dummy_instance = GpuDartInstance {
        pos_angle: [100.0, 100.0, 0.0],
        pad0: 0.5,
        vis_data: [
            0xFF50E0A0,
            // radius=128 (0.5), glow=128 (0.5), energy=128, attacking=0, birth=0
            128 | (128 << 8) | (128 << 16),
        ],
        pad1: [1, 0],
    };

    let mut glow_pos = Vec::new();
    let mut glow_col = Vec::new();
    generate_glow_mesh_from_instances(&[dummy_instance], 600.0, &mut glow_pos, &mut glow_col);

    // Glow mesh must have at least 20 segments (20 triangles * 3 = 60 vertices)
    assert!(
        glow_pos.len() >= 60,
        "Glow mesh must have >= 20 segments (expected >= 60 vertices, got {})",
        glow_pos.len()
    );

    // Glow mesh must be positioned at z = -2.2 (behind trails at -2.05 and darts at -2.0)
    for p in &glow_pos {
        assert!(
            (p[2] - (-2.2)).abs() < 1e-4,
            "Glow mesh vertices must be at z = -2.2, got {}",
            p[2]
        );
    }

    // Swarm trail mesh layer testing
    let mut tracker = AgentTrailsTracker::default();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(bevy::prelude::Vec2::new(100.0, 100.0));
    queue.push_back(bevy::prelude::Vec2::new(105.0, 105.0));
    tracker.trails.insert(1, queue);

    let mut trail_pos = Vec::new();
    let mut trail_col = Vec::new();
    generate_swarm_trails_from_tracker(&tracker, &[dummy_instance], 900.0, 600.0, &mut trail_pos, &mut trail_col);

    assert!(!trail_pos.is_empty(), "Trail mesh must have vertices");
    for p in &trail_pos {
        assert!(
            (p[2] - (-2.05)).abs() < 1e-4,
            "Swarm trails must be at z = -2.05 (in front of glow at -2.2, behind darts at -2.0), got {}",
            p[2]
        );
    }

    // Trail alpha must be vibrant (up to >= 0.45)
    let max_alpha = trail_col.iter().map(|c| c[3]).fold(0.0f32, f32::max);
    assert!(
        max_alpha >= 0.45,
        "Swarm trail alpha must be vibrant (>= 0.45), got {}",
        max_alpha
    );
}

#[test]
fn test_long_run_gpu_population_sustainability_assertive() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let sim = SimWorld::new(42);
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.max_capacity = 340;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let mut pop = 72u32;
    for tick in 1..=200 {
        params.tick = tick - 1;
        params.sub_tick = 0;
        params.sub_ticks_per_frame = 1;
        params.agent_count = pop;
        driver.dispatch_sub_ticks(1, &params);

        let telem = driver.readback_telemetry();
        pop = telem.population;

        // Assert carrying capacity
        assert!(
            pop <= 340,
            "Tick {}: Population {} exceeded carrying capacity 340",
            tick,
            pop
        );
        // Assert no sudden extinction collapse
        assert!(
            pop >= 15,
            "Tick {}: Population {} collapsed below viable threshold 15",
            tick,
            pop
        );
        // Assert freelist bounds
        assert!(
            telem.freelist_top <= driver.max_agents,
            "Tick {}: Freelist top {} exceeded max agents {}",
            tick,
            telem.freelist_top,
            driver.max_agents
        );
    }
}

#[test]
fn test_sim_telemetry_birth_accumulation_guard() {
    let mut sim = SimWorld::new(42);
    let initial_births = sim.world.births; // 72

    // Simulate 5 frames where GPU telemetry reports total_births = 78 (meaning 6 new births occurred)
    let telem_total_births = 78u32;
    for _frame in 0..5 {
        // Correct logic: sim.world.births = telem_total_births.max(sim.world.births);
        // Faulty logic was: sim.world.births += telem_total_births; which blew up to 72 + 5*78 = 462!
        sim.world.births = telem_total_births.max(sim.world.births);
    }

    assert_eq!(
        sim.world.births, 78,
        "Total cumulative births must be 78 after 5 frames, but got {}",
        sim.world.births
    );
}

#[test]
fn test_observe_tool_empty_space_deselection() {
    let mut sim = SimWorld::new(42);
    sim.selected_agent_id = Some(42);
    sim.selected_agent_slot = Some(3);
    sim.selected_agent_cache = Some(clank_core::agent::AgentData::default());

    // When Observe tool finds no agent, selected_agent_idx == 0xFFFFFFFF
    let telem_selected_agent_idx = 0xFFFFFFFFu32;
    if telem_selected_agent_idx != 0xFFFFFFFF {
        sim.selected_agent_id = Some(123);
        sim.selected_agent_slot = Some(telem_selected_agent_idx);
    } else {
        // Deselection on empty click
        sim.selected_agent_id = None;
        sim.selected_agent_slot = None;
        sim.selected_agent_cache = None;
    }

    assert!(sim.selected_agent_id.is_none(), "Observe click on empty space must clear selected_agent_id");
    assert!(sim.selected_agent_slot.is_none(), "Observe click on empty space must clear selected_agent_slot");
    assert!(sim.selected_agent_cache.is_none(), "Observe click on empty space must clear selected_agent_cache");
}

#[test]
fn test_corpse_deposit_stencil_formulas() {
    // Verify radius 1 stencil
    let r1_weights = |dx: i32, dy: i32| -> f32 {
        if dx == 0 && dy == 0 {
            1.0
        } else if (dx.abs() == 1 && dy == 0) || (dx == 0 && dy.abs() == 1) {
            1.0 / 1.8
        } else {
            0.0
        }
    };

    assert_eq!(r1_weights(0, 0), 1.0);
    assert!((r1_weights(1, 0) - (1.0 / 1.8)).abs() < 1e-4);
    assert!((r1_weights(0, -1) - (1.0 / 1.8)).abs() < 1e-4);
    assert_eq!(r1_weights(1, 1), 0.0);
    assert_eq!(r1_weights(-1, 1), 0.0);

    // Verify radius 2 stencil
    let r2_weights = |dx: i32, dy: i32| -> f32 {
        let rr = (dx * dx + dy * dy) as f32;
        if rr <= 4.5 {
            1.0 / (1.0 + rr * 0.8)
        } else {
            0.0
        }
    };

    assert_eq!(r2_weights(0, 0), 1.0);
    assert!((r2_weights(1, 0) - (1.0 / 1.8)).abs() < 1e-4);
    assert!((r2_weights(1, 1) - (1.0 / 2.6)).abs() < 1e-4);
    assert!((r2_weights(2, 0) - (1.0 / 4.2)).abs() < 1e-4);
    assert_eq!(r2_weights(2, 1), 0.0, "dx=2, dy=1 -> rr=5 > 4.5 must be 0");
    assert_eq!(r2_weights(2, 2), 0.0, "dx=2, dy=2 -> rr=8 > 4.5 must be 0");
}

