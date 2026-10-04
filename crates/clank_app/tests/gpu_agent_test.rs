use clank_app::gpu::agent_pipeline::{
    forward_pass_packed, toroidal_wrap, unpack4x8snorm, GpuCombatResolution,
};
use clank_app::gpu::types::{GpuAgentAtomic, GpuAgentGenome};

#[test]
fn test_unpack4x8snorm_forward_pass_parity() {
    // 1. unpack4x8snorm unpacking validation
    // Encodes 4 signed i8 values into a u32: [127, -128, 64, -64]
    let b0 = 127u8;
    let b1 = 0x80u8; // -128
    let b2 = 64u8;
    let b3 = 0xC0u8; // -64
    let word = (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16) | ((b3 as u32) << 24);
    let unpacked = unpack4x8snorm(word);
    assert!((unpacked[0] - 1.0).abs() < 1e-4);
    assert!((unpacked[1] - (-1.0)).abs() < 1e-4);
    assert!((unpacked[2] - 0.5039).abs() < 1e-2);
    assert!((unpacked[3] - (-0.5039)).abs() < 1e-2);

    // Forward pass with dummy inputs
    let genome = GpuAgentGenome {
        packed_genes: [word; 88],
    };
    let ins_hidden = [[0.5f32; 4]; 7];
    let ins_output = [[0.5f32; 4]; 3];
    let (hidden, output) = forward_pass_packed(&genome, &ins_hidden, &ins_output);
    assert_eq!(hidden.len(), 10);
    assert_eq!(output.len(), 6);
    for h in hidden {
        assert!(h >= -1.0 && h <= 1.0);
    }
    for o in output {
        assert!(o >= -1.0 && o <= 1.0);
    }

    // 2. Hardware toroidal wrapping: x - W * floor(x / W) handles negative/positive wraps
    let p_neg = toroidal_wrap([-50.0, -100.0]);
    assert!((p_neg[0] - 850.0).abs() < 1e-4);
    assert!((p_neg[1] - 500.0).abs() < 1e-4);

    let p_pos = toroidal_wrap([950.0, 650.0]);
    assert!((p_pos[0] - 50.0).abs() < 1e-4);
    assert!((p_pos[1] - 50.0).abs() < 1e-4);

    let p_in = toroidal_wrap([450.0, 300.0]);
    assert_eq!(p_in, [450.0, 300.0]);

    // 3. Combat damage formula, decisive killer attribution, and dead_claimed CAS protection
    let attacker_tr0 = 0.5f32;
    let victim_tr3 = 0.2f32;
    let attack_val = 0.8f32;
    let hostility = 1.0f32;

    let damage = GpuCombatResolution::calculate_damage(attack_val, hostility, attacker_tr0, victim_tr3);
    assert!(damage > 0.0);

    let mut victim_atomic = GpuAgentAtomic {
        energy_milli: 1000, // 1.0 energy unit
        mate_claim: 0,
        mate_energy_milli: 0,
        dead_claimed: 0,
    };

    // Sub-lethal attack:
    let lethal = GpuCombatResolution::apply_damage(&mut victim_atomic, 500);
    assert!(!lethal);
    assert_eq!(victim_atomic.energy_milli, 500);

    // Lethal attack:
    let lethal2 = GpuCombatResolution::apply_damage(&mut victim_atomic, 600);
    assert!(lethal2);
    assert_eq!(victim_atomic.energy_milli, -100);
    // dead_claimed CAS must succeed on first claim:
    assert_eq!(victim_atomic.dead_claimed, 1);

    // Subsequent damage should not double-claim:
    let lethal3 = GpuCombatResolution::apply_damage(&mut victim_atomic, 200);
    assert!(!lethal3); // already claimed!
}

#[test]
fn test_zombie_agent_guard_with_dead_claimed() {
    use clank_app::gpu::agent_pipeline::check_dead_guard;
    use clank_app::gpu::types::GpuAgentState;

    let mut state = GpuAgentState {
        pos_vel: [100.0, 100.0, 0.0, 0.0],
        angle_energy: [0.0, 50.0, 0.0, 0.0],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 1,
        meta_flags: 0, // looks alive in meta_flags
        age_gen: 0,
        morton_code: 0,
        packed_color: 0,
        visual_cache: 0xFFFFFFFF,
    };
    let atomic = GpuAgentAtomic {
        energy_milli: 0,
        mate_claim: 0,
        mate_energy_milli: 0,
        dead_claimed: 1, // but was claimed as dead!
    };

    let should_exit = check_dead_guard(&mut state, &atomic);
    assert!(should_exit);
    // Guard must have updated meta_flags with dead bit (bit 13) and cleared visual_cache
    assert_ne!(state.meta_flags & (1 << 13), 0);
    assert_eq!(state.visual_cache, 0);
    assert_eq!(state.angle_energy[1], 0.0);
}

#[test]
fn test_agent_combat_and_neighbor_sensory_bearing() {
    use clank_app::gpu::agent_pipeline::calculate_neighbor_sensory;

    // Agent at [100.0, 100.0] facing East (angle = 0.0)
    // Neighbor at [100.0, 110.0] (directly South / down, dy = 10.0, dx = 0.0 -> bearing = +pi/2)
    let agent_pos = [100.0f32, 100.0f32];
    let agent_angle = 0.0f32;
    let neighbor_pos = [100.0f32, 110.0f32];
    let sight_radius = 120.0f32;

    let (bearing_norm, dist_norm) = calculate_neighbor_sensory(agent_pos, agent_angle, neighbor_pos, sight_radius);
    // bearing is +pi/2 normalized by pi -> ~0.5
    assert!((bearing_norm - 0.5).abs() < 1e-2);
    // dist is 10.0 / 120.0 -> dist_norm = 1.0 - 10.0/120.0 ~ 0.9167
    assert!((dist_norm - (1.0 - 10.0 / 120.0)).abs() < 1e-2);
}

#[test]
fn test_experimental_simulation_mods_logic() {
    use clank_app::gpu::agent_pipeline::{
        evaluate_barnes_hut_flocking, evaluate_expanded_cortex_sensory, tournament_bid_mating,
    };
    use clank_app::gpu::types::GpuAgentAtomic;

    // 1. Mod 2: Expanded Cortex sensory inputs
    let food_gradient = [0.4f32, -0.2f32];
    let swarm_bearing = 0.75f32;

    let baseline_sensory = evaluate_expanded_cortex_sensory(food_gradient, swarm_bearing, false);
    assert_eq!(baseline_sensory, [0.0, 0.0, 0.0, 0.0]);

    let expanded_sensory = evaluate_expanded_cortex_sensory(food_gradient, swarm_bearing, true);
    assert_eq!(expanded_sensory, [0.4, -0.2, 0.75, 1.0]);

    // 2. Mod 1: Barnes-Hut macro-flocking force
    let agent_pos = [100.0f32, 100.0f32];
    let distant_center = [200.0f32, 100.0f32]; // dx = 100.0, dy = 0.0
    let node_count = 50u32;
    let node_size = 20.0f32; // theta = 20 / 100 = 0.2 < 0.6 threshold

    let flock_force = evaluate_barnes_hut_flocking(agent_pos, distant_center, node_count, node_size);
    assert!(flock_force[0] > 0.0); // Attracted towards +x
    assert_eq!(flock_force[1], 0.0);

    // 3. Mod 3: Sexual Selection Tournament
    let mut partner = GpuAgentAtomic {
        energy_milli: 60_000,
        mate_claim: 0,
        mate_energy_milli: 0,
        dead_claimed: 0,
    };

    // First suitor submits bid
    let won1 = tournament_bid_mating(&mut partner, 5, 40_000, true);
    assert!(won1);
    assert_eq!(partner.mate_claim, 6); // 5 + 1
    assert_eq!(partner.mate_energy_milli, 40_000);

    // Weaker suitor submits bid -> rejected
    let won2 = tournament_bid_mating(&mut partner, 8, 30_000, true);
    assert!(!won2);
    assert_eq!(partner.mate_claim, 6);

    // Stronger suitor submits bid -> replaces previous suitor
    let won3 = tournament_bid_mating(&mut partner, 12, 70_000, true);
    assert!(won3);
    assert_eq!(partner.mate_claim, 13); // 12 + 1
    assert_eq!(partner.mate_energy_milli, 70_000);
}


