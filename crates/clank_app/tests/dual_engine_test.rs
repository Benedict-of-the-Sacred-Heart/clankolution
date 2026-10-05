use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu, GpuSimBridge};
use clank_app::gpu::lbvh::LbvhTree;
use clank_app::gpu::types::{GpuAgentState, GpuTelemetry};
use clank_app::sim::SimWorld;

#[test]
fn test_dual_engine_live_hotswap_parity() {
    let mut sim = SimWorld::new(12345);
    sim.world.tick = 420;
    let initial_agent_count = sim.world.agents.iter().filter(|a| a.dead == 0).count();
    assert!(initial_agent_count > 0);

    // 1. Rust -> GPU
    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    assert_eq!(params.tick, 420);
    assert_eq!(params.agent_count as usize, initial_agent_count);
    assert_eq!(states.len(), sim.world.agents.len());
    assert_eq!(genomes.len(), sim.world.agents.len());
    assert_eq!(atomics.len(), sim.world.agents.len());
    assert_eq!(soil.len(), sim.world.soil.grid_size);

    // Verify visual_cache synthesized correctly for living agents
    for (i, a) in sim.world.agents.iter().enumerate() {
        if a.dead == 0 {
            assert_ne!(states[i].visual_cache, 0);
            assert_eq!(states[i].id, a.id);
        } else {
            assert_eq!(states[i].visual_cache, 0);
        }
    }

    // 2. GPU -> Rust
    let mut sim2 = SimWorld::new(99999);
    sync_gpu_to_rust(&states, &genomes, &atomics, &soil, &params, &mut sim2);

    assert_eq!(sim2.world.tick, 420);
    let restored_count = sim2.world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(restored_count, initial_agent_count);

    // 3. Roots (0..15), generations, IDs preserved
    for (a1, a2) in sim.world.agents.iter().zip(sim2.world.agents.iter()) {
        assert_eq!(a1.id, a2.id);
        assert_eq!(a1.dead, a2.dead);
        if a1.dead == 0 {
            assert_eq!(a1.root % 16, a2.root);
            assert_eq!(a1.gen, a2.gen);
            assert!((a1.energy - a2.energy).abs() < 1e-2);
        }
    }
}

#[test]
fn test_gpu_telemetry_and_extinction_detection() {
    let telemetry = GpuTelemetry {
        population: 150,
        kills: 12,
        starvations: 5,
        apex_record_milli: 85_400, // 85.4 energy
        food_grazed_milli: 120_000,
        sub_ticks_elapsed: 32,
        apex_agent_id: 101,
        freelist_top: 874,
        birth_count: 8,
        audio_voice_count: 4,
        selected_agent_idx: 42,
        selected_agent_id: 1002,
        _reserved0: [0; 4],
        lineage_counts: [10, 0, 15, 20, 0, 5, 8, 12, 14, 0, 11, 9, 13, 16, 7, 10],
    };

    assert_eq!(std::mem::size_of::<GpuTelemetry>(), 128);
    assert_eq!(telemetry.apex_record_milli, 85_400);

    // Check extinction detection via zero-search:
    let extinct_roots = GpuSimBridge::detect_extinct_lineages(&telemetry.lineage_counts);
    assert_eq!(extinct_roots, vec![1, 4, 9]);
}

#[test]
fn test_two_tier_uncapped_picking_disambiguation() {
    let agent_a = GpuAgentState {
        pos_vel: [200.0, 200.0, 0.0, 0.0],
        angle_energy: [0.0, 50.0, 0.0, 0.0],
        traits: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // visual_r = 2.0 + 3.0 * 0.5 = 3.5
        hidden: [0.0; 10],
        id: 777,
        meta_flags: 0,
        age_gen: 0,
        morton_code: 0,
        packed_color: 0,
        visual_cache: 1,
    };

    let agent_b = GpuAgentState {
        pos_vel: [203.0, 200.0, 0.0, 0.0], // 3.0 units away
        angle_energy: [0.0, 50.0, 0.0, 0.0],
        traits: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // visual_r = 2.0
        hidden: [0.0; 10],
        id: 888,
        meta_flags: 0,
        age_gen: 0,
        morton_code: 0,
        packed_color: 0,
        visual_cache: 1,
    };

    let agents = vec![agent_a, agent_b];
    let tree = LbvhTree::build(&agents);

    // Cursor at (201.0, 200.0):
    // Dist to A: 1.0 (<= 3.5 -> Priority 0 body hit)
    // Dist to B: 2.0 (<= 2.0 -> Priority 0 body hit)
    // Tie-break: Dist to A (1.0) < Dist to B (2.0) -> selects A!
    let (best_idx, best_id) = tree.pick_agent([201.0, 200.0], &agents, 16.0);
    assert_eq!(best_idx, Some(0));
    assert_eq!(best_id, Some(777));

    // Identity guard: if slot 0 is recycled and ID changes, picking flags death
    let is_alive = GpuSimBridge::verify_picked_identity(0, 777, &agents);
    assert!(is_alive);
    let is_stale = GpuSimBridge::verify_picked_identity(0, 999, &agents);
    assert!(!is_stale);
}

#[test]
fn test_mating_canonical_symmetry_and_sexual_selection_mod() {
    // 1. Canonical Symmetry: suitor ID must be < partner ID (or partner_id > agent_id)
    assert!(GpuSimBridge::should_propose_mating(10, 20)); // 10 proposes to 20
    assert!(!GpuSimBridge::should_propose_mating(20, 10)); // 20 does NOT propose to 10 (cuts bus contention 50%)

    // 2. Mod 3 Sexual Selection Tournament: suitor bid comparison
    let current_bid = 50_000u32; // 50.0 energy
    let lower_bid = 40_000u32;
    let higher_bid = 65_000u32;

    assert!(!GpuSimBridge::tournament_bid_wins(lower_bid, current_bid));
    assert!(GpuSimBridge::tournament_bid_wins(higher_bid, current_bid));
}
