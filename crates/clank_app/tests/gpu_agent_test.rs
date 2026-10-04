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
