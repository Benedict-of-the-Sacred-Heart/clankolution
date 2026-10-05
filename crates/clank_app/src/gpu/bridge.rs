//! Dual-Engine State Hot-Swap Bridge
//!
//! Provides bi-directional state synchronization between Bevy CPU `SimWorld`
//! and GPU compute storage buffers, enabling seamless live hot-swapping between
//! [ENGINE: RUST] and [ENGINE: GPU] without losing lineages or resetting the timeline.

use clank_core::agent::AgentData;
use crate::gpu::types::{
    GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuSimParams, GpuSoilCell,
};
use crate::sim::SimWorld;
use crate::theme::PALETTE;

pub struct GpuSimBridge;

impl GpuSimBridge {
    /// Detects extinct lineages from the 16-lineage head count table using zero-searching.
    pub fn detect_extinct_lineages(counts: &[u32; 16]) -> Vec<u32> {
        let mut extinct = Vec::new();
        for (root, &count) in counts.iter().enumerate() {
            if count == 0 {
                extinct.push(root as u32);
            }
        }
        extinct
    }

    /// Verifies specimen picking identity guard:
    /// Returns true if slot is alive and matches the unique agent ID.
    pub fn verify_picked_identity(
        slot_idx: usize,
        agent_id: u32,
        agents: &[GpuAgentState],
    ) -> bool {
        if let Some(a) = agents.get(slot_idx) {
            let is_dead = (a.meta_flags & (1 << 13)) != 0;
            !is_dead && a.id == agent_id
        } else {
            false
        }
    }

    /// Canonical mating symmetry breaking: partner_id > agent_id halves atomic bus traffic by 50%.
    #[inline]
    pub fn should_propose_mating(my_id: u32, partner_id: u32) -> bool {
        partner_id > my_id
    }

    /// Mod 3 Sexual Selection Tournament bid evaluation:
    /// Returns true if the suitor's bid beats the existing highest bid.
    #[inline]
    pub fn tournament_bid_wins(my_bid: u32, current_bid: u32) -> bool {
        my_bid > current_bid
    }
}

/// Packs 4 signed i8 weights into a 32-bit word for GPU unpacked SIMD operations.
#[inline]
fn pack_weights(w: &[i8]) -> u32 {
    let b0 = w.get(0).copied().unwrap_or(0) as u8 as u32;
    let b1 = w.get(1).copied().unwrap_or(0) as u8 as u32;
    let b2 = w.get(2).copied().unwrap_or(0) as u8 as u32;
    let b3 = w.get(3).copied().unwrap_or(0) as u8 as u32;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

/// Transfers CPU Bevy SimWorld state to GPU compute storage buffers.
pub fn sync_rust_to_gpu(
    sim: &SimWorld,
) -> (
    Vec<GpuAgentState>,
    Vec<GpuAgentGenome>,
    Vec<GpuAgentAtomic>,
    Vec<GpuSoilCell>,
    GpuSimParams,
) {
    let mut states = Vec::with_capacity(sim.world.agents.len());
    let mut genomes = Vec::with_capacity(sim.world.agents.len());
    let mut atomics = Vec::with_capacity(sim.world.agents.len());

    let mut living_count = 0u32;

    for a in &sim.world.agents {
        let is_dead = a.dead != 0;
        if !is_dead {
            living_count += 1;
        }

        let pos = [a.x as f32, a.y as f32];
        let morton = crate::gpu::spatial_index::compute_morton_32_with_size(
            pos,
            [sim.world_width as f32, sim.world_height as f32],
        );
        let pal_color = PALETTE[(a.root as usize) % PALETTE.len()];
        let packed_color = (pal_color.r() as u32)
            | ((pal_color.g() as u32) << 8)
            | ((pal_color.b() as u32) << 16)
            | (0xFF << 24);

        let visual_cache = if is_dead {
            0u32
        } else {
            let r_u8 = ((a.tr[0] as f32).clamp(0.0, 1.0) * 255.0) as u32;
            let glow_u8 = ((a.attack as f32)
                .max(if a.birth > 0 { 1.0 } else { 0.0 })
                .clamp(0.0, 1.0)
                * 255.0) as u32;
            let e_u8 = ((a.energy as f32 / 100.0).clamp(0.0, 1.0) * 255.0) as u32;
            let flags = (if a.attack > 0.25 { 1u32 << 24 } else { 0 })
                | (if a.birth > 0 { 1u32 << 25 } else { 0 });
            r_u8 | (glow_u8 << 8) | (e_u8 << 16) | flags
        };

        let state = GpuAgentState {
            pos_vel: [a.x as f32, a.y as f32, a.vx as f32, a.vy as f32],
            angle_energy: [a.angle as f32, a.energy as f32, a.feeding as f32, a.attack as f32],
            traits: [
                a.tr[0] as f32,
                a.tr[1] as f32,
                a.tr[2] as f32,
                a.tr[3] as f32,
                a.tr[4] as f32,
                a.tr[5] as f32,
                a.signal as f32,
                a.last_victim as f32,
            ],
            hidden: [
                a.h[0] as f32, a.h[1] as f32, a.h[2] as f32, a.h[3] as f32, a.h[4] as f32,
                a.h[5] as f32, a.h[6] as f32, a.h[7] as f32, a.h[8] as f32, a.h[9] as f32,
            ],
            id: a.id,
            meta_flags: (a.root & 0x0F)
                | ((a.cooldown as u32 & 0x03) << 4)
                | ((a.birth as u32 & 0x7F) << 6)
                | ((a.dead as u32 & 0x01) << 13)
                | ((a.kills as u32 & 0x3FFFF) << 14),
            age_gen: (a.age as u32 & 0xFFFF) | ((a.gen as u32 & 0xFFFF) << 16),
            morton_code: morton,
            packed_color,
            visual_cache,
        };
        states.push(state);

        // Pack 88 words
        let mut packed_genes = [0u32; 88];
        let mut gene_idx = 0;
        // 10 hidden neurons * 7 words
        for h in 0..10 {
            for w in 0..7 {
                let slice = if gene_idx + 4 <= a.genes.len() {
                    &a.genes[gene_idx..gene_idx + 4]
                } else if gene_idx < a.genes.len() {
                    &a.genes[gene_idx..]
                } else {
                    &[]
                };
                packed_genes[h * 7 + w] = pack_weights(slice);
                gene_idx += 4;
            }
        }
        // 6 output neurons * 3 words
        for o in 0..6 {
            for w in 0..3 {
                let slice = if gene_idx + 4 <= a.genes.len() {
                    &a.genes[gene_idx..gene_idx + 4]
                } else if gene_idx < a.genes.len() {
                    &a.genes[gene_idx..]
                } else {
                    &[]
                };
                packed_genes[70 + o * 3 + w] = pack_weights(slice);
                gene_idx += 4;
            }
        }
        genomes.push(GpuAgentGenome { packed_genes });

        atomics.push(GpuAgentAtomic {
            energy_milli: (a.energy * 1000.0).round() as i32,
            mate_claim: 0,
            mate_energy_milli: 0,
            dead_claimed: a.dead as u32,
        });
    }

    // Convert soil
    let grid_size = sim.world.soil.grid_size;
    let mut soil = Vec::with_capacity(grid_size);
    for i in 0..grid_size {
        let f = sim.world.soil.food[i];
        let t = sim.world.soil.taint[i];
        let s = sim.world.soil.scent[i];
        soil.push(GpuSoilCell {
            food_milli: (f * 1000.0).round() as i32,
            taint_milli: (t * 1000.0).round() as i32,
            scent_milli: (s * 1000.0).round() as i32,
            pad: 0,
        });
    }

    let params = GpuSimParams {
        tick: sim.world.tick,
        agent_count: living_count,
        max_agents: states.len() as u32,
        max_capacity: sim.world.max_cap as u32,
        hostility: (sim.world.hostility / 100.0) as f32,
        mut_rate: (sim.world.mutation / 100.0) as f32,
        speed: sim.speed as f32,
        renewal: (sim.world.growth / 100.0) as f32,
        sub_tick: 0,
        sub_ticks_per_frame: 1,
        tool_type: 0xFFFFFFFF,
        _pad0: 0,
        tool_pos: [0.0, 0.0],
        camera_pos: [(sim.world_width * 0.5) as f32, (sim.world_height * 0.5) as f32],
        camera_size: [sim.world_width as f32, sim.world_height as f32],
        world_size: [sim.world_width as f32, sim.world_height as f32],
        soil_grid: [sim.world.soil.cols as u32, sim.world.soil.rows as u32],
        _pad1: [0, 0],
    };

    (states, genomes, atomics, soil, params)
}

/// Reads back GPU compute storage buffers into CPU Bevy SimWorld state.
pub fn sync_gpu_to_rust(
    states: &[GpuAgentState],
    _genomes: &[GpuAgentGenome],
    atomics: &[GpuAgentAtomic],
    soil: &[GpuSoilCell],
    params: &GpuSimParams,
    sim: &mut SimWorld,
) {
    sim.world.tick = params.tick;

    // Synchronize agents
    for (i, state) in states.iter().enumerate() {
        if state.id == 0 {
            // Uninitialized or empty slot
            continue;
        }

        let is_dead = (state.meta_flags & (1 << 13)) != 0;
        let atomic_energy = atomics.get(i).map(|at| at.energy_milli as f64 * 0.001).unwrap_or(state.angle_energy[1] as f64);

        if is_dead && atomic_energy <= 0.0 {
            if i < sim.world.agents.len() {
                sim.world.agents[i].dead = 1;
            }
            continue;
        }

        let a = if i < sim.world.agents.len() {
            &mut sim.world.agents[i]
        } else {
            if is_dead {
                continue;
            }
            if sim.world.agents.iter().filter(|ag| ag.dead == 0).count() >= sim.world.max_cap {
                continue;
            }
            sim.world.agents.push(AgentData::default());
            sim.world.agents.last_mut().unwrap()
        };
        a.id = state.id;
        a.x = state.pos_vel[0] as f64;
        a.y = state.pos_vel[1] as f64;
        a.vx = state.pos_vel[2] as f64;
        a.vy = state.pos_vel[3] as f64;

        a.angle = state.angle_energy[0] as f64;
        // Prioritize atomic energy ground truth
        a.energy = atomic_energy.max(0.0);
        a.feeding = state.angle_energy[2] as f64;
        a.attack = state.angle_energy[3] as f64;

        for (k, val) in state.traits.iter().enumerate() {
            if k < 6 {
                a.tr[k] = *val as f64;
            } else if k == 6 {
                a.signal = *val as f64;
            } else if k == 7 {
                a.last_victim = *val as u32;
            }
        }

        for (k, val) in state.hidden.iter().enumerate() {
            a.h[k] = *val;
        }

        let meta = state.meta_flags;
        a.root = meta & 0x0F;
        a.cooldown = (meta >> 4) & 0x03;
        a.birth = (meta >> 6) & 0x7F;
        a.dead = (meta >> 13) & 0x01;
        a.kills = meta >> 14;

        a.age = state.age_gen & 0xFFFF;
        a.gen = state.age_gen >> 16;
    }

    // Filter dead agents in-place (matching CPU World::evolve stable compaction)
    sim.world.agents.retain(|a| a.dead == 0 && a.id != 0);
    if sim.world.agents.len() > sim.world.max_cap {
        sim.world.agents.truncate(sim.world.max_cap);
    }
    sim.world.sync_pos_cache();

    // Synchronize soil
    for (i, cell) in soil.iter().enumerate() {
        if i < sim.world.soil.food.len() {
            sim.world.soil.food[i] = cell.food_milli as f32 * 0.001;
            sim.world.soil.taint[i] = cell.taint_milli as f32 * 0.001;
            sim.world.soil.scent[i] = cell.scent_milli as f32 * 0.001;
        }
    }
}
