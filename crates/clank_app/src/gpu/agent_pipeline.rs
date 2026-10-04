//! GPU Agent Compute Pipeline: Forward Pass, Kinematics & Combat
//!
//! Provides the mathematical implementation for vectorized 88-word RNN evaluation,
//! toroidal coordinate wrapping, and atomic millijoule combat resolution.

use super::types::{GpuAgentAtomic, GpuAgentGenome};

/// Hardware-equivalent unpack4x8snorm implementation matching WebGPU specification.
/// Interprets a 32-bit word as four 8-bit signed integers and normalizes to [-1.0, 1.0].
#[inline]
pub fn unpack4x8snorm(word: u32) -> [f32; 4] {
    let b0 = (word & 0xFF) as u8 as i8;
    let b1 = ((word >> 8) & 0xFF) as u8 as i8;
    let b2 = ((word >> 16) & 0xFF) as u8 as i8;
    let b3 = ((word >> 24) & 0xFF) as u8 as i8;

    [
        (b0 as f32 / 127.0).max(-1.0),
        (b1 as f32 / 127.0).max(-1.0),
        (b2 as f32 / 127.0).max(-1.0),
        (b3 as f32 / 127.0).max(-1.0),
    ]
}

/// Evaluates 88-word packed neural RNN forward pass matching `agent_step.wgsl`.
pub fn forward_pass_packed(
    genome: &GpuAgentGenome,
    ins_hidden: &[[f32; 4]; 7],
    ins_output: &[[f32; 4]; 3],
) -> ([f32; 10], [f32; 6]) {
    // 10 hidden neurons, each 7 vec4 weights (70 words total)
    let mut hidden = [0.0f32; 10];
    for j in 0..10 {
        let mut s = 0.0f32;
        let base_w = j * 7;
        for k in 0..7 {
            let weights = unpack4x8snorm(genome.packed_genes[base_w + k]);
            let inputs = ins_hidden[k];
            s += weights[0] * inputs[0]
                + weights[1] * inputs[1]
                + weights[2] * inputs[2]
                + weights[3] * inputs[3];
        }
        hidden[j] = (s * 0.61).tanh();
    }

    // 6 output neurons, each 3 vec4 weights (18 words total)
    let mut output = [0.0f32; 6];
    for j in 0..6 {
        let mut s = 0.0f32;
        let base_w = 70 + j * 3;
        for k in 0..3 {
            let weights = unpack4x8snorm(genome.packed_genes[base_w + k]);
            let inputs = ins_output[k];
            s += weights[0] * inputs[0]
                + weights[1] * inputs[1]
                + weights[2] * inputs[2]
                + weights[3] * inputs[3];
        }
        output[j] = (s * 0.66).tanh();
    }

    (hidden, output)
}

/// Hardware toroidal coordinate wrapping matching `p - W * floor(p / W)`.
#[inline]
pub fn toroidal_wrap(p: [f32; 2]) -> [f32; 2] {
    [
        p[0] - 900.0 * (p[0] / 900.0).floor(),
        p[1] - 600.0 * (p[1] / 600.0).floor(),
    ]
}

/// Helper for atomic millijoule combat resolution and CAS death ownership.
pub struct GpuCombatResolution;

impl GpuCombatResolution {
    /// Evaluates predatory combat damage matching `agent_step.wgsl` and HTML reference.
    #[inline]
    pub fn calculate_damage(
        attack: f32,
        hostility: f32,
        attacker_tr0: f32,
        victim_tr3: f32,
    ) -> f32 {
        (0.5 + attack * 2.2) * hostility * (0.8 + attacker_tr0) * (1.0 - 0.65 * victim_tr3)
    }

    /// Applies combat damage to atomic victim state.
    /// Returns true if this hit was the decisive lethal kill and won the CAS death claim.
    pub fn apply_damage(victim: &mut GpuAgentAtomic, damage_milli: i32) -> bool {
        let old_energy = victim.energy_milli;
        victim.energy_milli -= damage_milli;

        if old_energy > 0 && victim.energy_milli <= 0 {
            if victim.dead_claimed == 0 {
                victim.dead_claimed = 1;
                return true;
            }
        }
        false
    }
}

/// Top-of-shader dead-check guard with single-writer dead_claimed synchronization.
/// Returns true if the agent is dead and should exit compute immediately.
pub fn check_dead_guard(
    state: &mut super::types::GpuAgentState,
    atomic: &super::types::GpuAgentAtomic,
) -> bool {
    let is_dead = (state.meta_flags & (1 << 13)) != 0;
    let dead_claimed = atomic.dead_claimed != 0;
    if is_dead || dead_claimed {
        if !is_dead {
            state.meta_flags |= 1 << 13;
            state.angle_energy[1] = 0.0;
            state.visual_cache = 0;
        }
        return true;
    }
    false
}

/// Evaluates sensory bearing [-1.0, 1.0] and normalized proximity [0.0, 1.0] for the closest neighbor.
pub fn calculate_neighbor_sensory(
    agent_pos: [f32; 2],
    agent_angle: f32,
    neighbor_pos: [f32; 2],
    sight_radius: f32,
) -> (f32, f32) {
    let mut dx = neighbor_pos[0] - agent_pos[0];
    if dx > 450.0 {
        dx -= 900.0;
    } else if dx < -450.0 {
        dx += 900.0;
    }

    let mut dy = neighbor_pos[1] - agent_pos[1];
    if dy > 300.0 {
        dy -= 600.0;
    } else if dy < -300.0 {
        dy += 600.0;
    }

    let dist = (dx * dx + dy * dy).sqrt();
    let angle_to_neighbor = dy.atan2(dx);
    let mut bearing = angle_to_neighbor - agent_angle;

    // Normalize bearing to [-PI, PI]
    while bearing > std::f32::consts::PI {
        bearing -= 2.0 * std::f32::consts::PI;
    }
    while bearing < -std::f32::consts::PI {
        bearing += 2.0 * std::f32::consts::PI;
    }

    let bearing_norm = (bearing / std::f32::consts::PI).clamp(-1.0, 1.0);
    let dist_norm = (1.0 - dist / sight_radius.max(1.0)).clamp(0.0, 1.0);

    (bearing_norm, dist_norm)
}

