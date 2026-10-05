use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu};
use clank_app::gpu::types::*;
use clank_app::sim::SimWorld;
use clank_core::World;
use bytemuck::Zeroable;

#[test]
fn test_core_seed_life_at_strict_capacity() {
    let mut world = World::new(12345);
    world.set_max_capacity(15);
    assert_eq!(world.max_cap, 15);
    assert!(world.agents.len() <= 15);

    // Living agents is 15. Calling seed_life_at should NOT increase agents above 15
    world.seed_life_at(100.0, 100.0);
    let living = world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(living, 15);

    // Kill 2 agents
    world.agents[0].dead = 1;
    world.agents[1].dead = 1;
    let living_before = world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(living_before, 13);

    // Now seed_life_at (which tries to seed 2) should increase to exactly 15, not exceed
    world.seed_life_at(100.0, 100.0);
    let living_after = world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(living_after, 15);

    // Call seed_life_at again when already at 15
    world.seed_life_at(100.0, 100.0);
    let living_capped = world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(living_capped, 15);
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
