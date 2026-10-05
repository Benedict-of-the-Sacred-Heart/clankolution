use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::sim::SimWorld;

#[test]
fn test_gpu_compute_driver_initialization_and_dispatch() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let mut sim = SimWorld::new(42);
    // Cluster low-energy victims around attacker to ensure combat kill attribution in 16 ticks
    for i in 1..15 {
        sim.world.agents[i].energy = 0.05;
        sim.world.agents[i].x = sim.world.agents[0].x + (i as f64 * 0.2);
        sim.world.agents[i].y = sim.world.agents[0].y + (i as f64 * 0.2);
    }
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.hostility = 10.0;

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Dispatch 16 sub-ticks on GPU silicon
    driver.dispatch_sub_ticks(16, &params);

    // Read back telemetry to confirm GPU execution
    let telemetry = driver.readback_telemetry();
    assert!(telemetry.kills >= 1);
}

#[test]
fn test_gpu_compute_clears_dead_and_spawns_births() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let mut sim = SimWorld::new(1234);
    // Give all agents high energy and mature age so they reproduce
    for a in &mut sim.world.agents {
        a.energy = 150.0;
        a.age = 100;
        a.cooldown = 0;
        a.birth = 0;
    }

    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Dispatch 16 sub-ticks on GPU silicon
    driver.dispatch_sub_ticks(16, &params);

    let telemetry = driver.readback_telemetry();
    assert!(telemetry.birth_count >= 1, "Expected birth_count >= 1 on GPU silicon, got {}", telemetry.birth_count);

    let updated_states = driver.readback_agent_states(driver.max_agents as usize);
    let updated_atomics = driver.readback_atomics(driver.max_agents as usize);
    let updated_soil = driver.readback_soil();
    let mut updated_params = params;
    updated_params.tick += 16;

    clank_app::gpu::bridge::sync_gpu_to_rust(
        &updated_states,
        &genomes,
        &updated_atomics,
        &updated_soil,
        &updated_params,
        &mut sim,
    );

    // Verify dead agents are cleared from sim.world.agents
    assert!(sim.world.agents.iter().all(|a| a.dead == 0), "Dead agents must be cleared from world");
    // Verify new births exist in sim.world.agents
    assert!(sim.world.agents.iter().any(|a| a.age < 16), "Expected newborn creatures with age < 16");
}
