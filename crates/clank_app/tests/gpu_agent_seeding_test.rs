use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::{
    GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuSimParams, GpuSoilCell,
};

#[test]
fn test_direct_gpu_agent_seeding_in_vram() {
    let driver = GpuComputeDriver::create_for_world(75, 50, 65536);
    if driver.is_none() {
        eprintln!("No WebGPU adapter found, skipping agent seeding test");
        return;
    }
    let driver = driver.unwrap();

    let states: Vec<GpuAgentState> = Vec::new();
    let genomes: Vec<GpuAgentGenome> = Vec::new();
    let atomics: Vec<GpuAgentAtomic> = Vec::new();
    let soil = vec![
        GpuSoilCell {
            food_micro: 1_000_000,
            taint_micro: 0,
            scent_micro: 0,
            fertility_micro: 1_000_000
        };
        75 * 50
    ];

    let params = GpuSimParams {
        tick: 1,
        agent_count: 0,
        max_agents: 65536,
        max_capacity: 100,
        renewal: 0.05,
        world_size: [900.0, 600.0],
        soil_grid: [75, 50],
        ..Default::default()
    };

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Initial population is 0
    let telem = driver.readback_telemetry();
    assert_eq!(telem.population, 0);
    assert_eq!(telem.freelist_top, 100);

    // Seed 3 agents directly into VRAM
    let positions = vec![
        (150.0f32, 200.0f32),
        (300.0f32, 400.0f32),
        (450.0f32, 100.0f32),
    ];
    let spawned_slots = driver.replenish_agents_gpu(&positions);
    assert_eq!(spawned_slots.len(), 3);

    // Check freelist was decremented
    let telem_after_seed = driver.readback_telemetry();
    assert_eq!(telem_after_seed.freelist_top, 97);

    // Dispatch 1 sub-tick so the GPU simulation processes the living agents
    driver.dispatch_sub_ticks(1, &params);

    // Read back states at the spawned slots
    let states_readback = driver.readback_agent_states(100);
    for &slot in &spawned_slots {
        let agent = &states_readback[slot as usize];
        let is_dead = (agent.meta_flags & (1 << 13)) != 0;
        assert!(!is_dead, "Agent at slot {} should be living", slot);
        assert!(
            agent.angle_energy[1] > 20.0,
            "Agent should have positive energy: {}",
            agent.angle_energy[1]
        );
        assert!(
            agent.id > 0,
            "Agent should have valid non-zero ID: {}",
            agent.id
        );
    }
}
