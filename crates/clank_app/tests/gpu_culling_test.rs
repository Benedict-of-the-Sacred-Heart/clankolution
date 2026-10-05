use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::{GpuAgentState, GpuAgentGenome, GpuAgentAtomic, GpuSoilCell, GpuSimParams};

#[test]
fn test_gpu_frustum_culling_visibility() {
    let driver = GpuComputeDriver::create_for_world(75, 50, 65536);
    if driver.is_none() {
        eprintln!("No WebGPU adapter found, skipping GPU culling test");
        return;
    }
    let driver = driver.unwrap();

    let mut states = Vec::new();
    let mut atomics = Vec::new();

    // Agent 0: Inside camera frustum, alive -> should be VISIBLE
    let a0 = GpuAgentState {
        pos_vel: [450.0, 300.0, 0.0, 0.0],
        angle_energy: [0.0, 50.0, 0.0, 0.0],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 1,
        meta_flags: 0,
        age_gen: 0,
        morton_code: 0,
        packed_color: 0xFFFFFFFF,
        visual_cache: 0x00FF00FF, // alive
    };
    states.push(a0);
    atomics.push(GpuAgentAtomic { energy_milli: 50000, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 0 });

    // Agent 1: Far outside camera frustum, alive -> should be CULLED
    let mut a1 = a0;
    a1.pos_vel = [50.0, 50.0, 0.0, 0.0];
    a1.id = 2;
    states.push(a1);
    atomics.push(GpuAgentAtomic { energy_milli: 50000, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 0 });

    // Agent 2: Inside camera frustum, but DEAD (visual_cache == 0, meta_flags dead bit set) -> should be CULLED
    let mut a2 = a0;
    a2.id = 3;
    a2.meta_flags = 1 << 13;
    a2.visual_cache = 0;
    states.push(a2);
    atomics.push(GpuAgentAtomic { energy_milli: 0, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 1 });

    // Agent 3: Inside camera frustum, alive -> should be VISIBLE
    let mut a3 = a0;
    a3.pos_vel = [460.0, 310.0, 0.0, 0.0];
    a3.id = 4;
    states.push(a3);
    atomics.push(GpuAgentAtomic { energy_milli: 50000, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 0 });

    let genomes = vec![GpuAgentGenome { packed_genes: [0; 88] }; 4];
    let soil = vec![GpuSoilCell { food_milli: 0, taint_milli: 0, scent_milli: 0, fertility_milli: 1000 }; 75 * 50];

    // Camera centered at [450, 300], viewport size [200, 200]
    let params = GpuSimParams {
        tick: 1,
        agent_count: 4,
        max_agents: 4,
        max_capacity: 4,
        camera_pos: [450.0, 300.0],
        camera_size: [200.0, 200.0],
        world_size: [900.0, 600.0],
        soil_grid: [75, 50],
        ..Default::default()
    };

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Dispatch GPU culling pass
    let visible_count = driver.dispatch_culling(&params);
    assert_eq!(visible_count, 2, "Expected exactly 2 visible agents");

    let visible_indices = driver.readback_visible_instances(visible_count as usize);
    assert_eq!(visible_indices.len(), 2);
    assert!(visible_indices.contains(&0), "Agent 0 should be visible");
    assert!(visible_indices.contains(&3), "Agent 3 should be visible");
    assert!(!visible_indices.contains(&1), "Agent 1 should be culled (out of frustum)");
    assert!(!visible_indices.contains(&2), "Agent 2 should be culled (dead)");

    let dart_instances = driver.readback_dart_instances(visible_count as usize);
    assert_eq!(dart_instances.len(), 2);
    assert_eq!(dart_instances[0].pos_angle[0], 450.0);
    assert_eq!(dart_instances[0].pos_angle[1], 300.0);
    assert_eq!(dart_instances[1].pos_angle[0], 460.0);
    assert_eq!(dart_instances[1].pos_angle[1], 310.0);
}
