use bytemuck::Zeroable;
use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu};
use clank_app::gpu::types::*;
use clank_app::sim::SimWorld;
use clank_core::World;

#[test]
fn test_core_seed_life_at_follows_original_cap_rule() {
    // Original seed tool (clankolution.html @ dfe8dc6): `if (agents.length < CAP)` spawn two.
    let mut world = World::new(12345);
    world.set_max_capacity(15);
    assert_eq!(world.max_cap, 15);
    assert_eq!(world.agents.len(), 15);

    // At the cap: nothing is seeded.
    world.seed_life_at(100.0, 100.0);
    assert_eq!(world.agents.len(), 15);

    // One below the cap: two are seeded, ending at CAP + 1.
    world.agents.truncate(14);
    world.sync_pos_cache();
    world.seed_life_at(100.0, 100.0);
    assert_eq!(world.agents.len(), 16);

    // Above the cap: nothing is seeded.
    world.seed_life_at(100.0, 100.0);
    assert_eq!(world.agents.len(), 16);
}

#[test]
fn test_bridge_sync_gpu_to_rust_clamps_to_max_cap() {
    let mut sim = SimWorld::default();
    sim.world.set_max_capacity(20);
    assert_eq!(sim.world.max_cap, 20);

    // Create 30 mock living GPU agent states
    let mut states = Vec::new();
    let mut atomics = Vec::new();
    for i in 0..30 {
        let mut s = GpuAgentState::zeroed();
        s.id = (i + 1) as u32;
        s.pos_vel = [10.0, 10.0, 0.0, 0.0];
        s.angle_energy = [0.0, 50.0, 0.0, 0.0];
        states.push(s);

        let at = GpuAgentAtomic {
            energy_milli: 50_000,
            mate_claim: 0,
            mate_energy_milli: 0,
            dead_claimed: 0,
        };
        atomics.push(at);
    }

    let genomes = vec![GpuAgentGenome::zeroed(); 30];
    let soil = vec![GpuSoilCell::zeroed(); (sim.world.soil.cols * sim.world.soil.rows) as usize];
    let params = GpuSimParams {
        max_capacity: 20,
        ..Default::default()
    };

    sync_gpu_to_rust(&states, &genomes, &atomics, &soil, &params, &mut sim);

    let living = sim.world.agents.iter().filter(|a| a.dead == 0).count();
    assert!(
        living <= sim.world.max_cap,
        "living agents ({}) must not exceed max_cap ({})",
        living,
        sim.world.max_cap
    );
}

#[test]
fn test_bridge_sync_rust_to_gpu_populates_max_capacity() {
    let mut sim = SimWorld::default();
    sim.world.set_max_capacity(400);

    let (_, _, _, _, params) = sync_rust_to_gpu(&sim);
    assert_eq!(params.max_capacity, 400);
}
