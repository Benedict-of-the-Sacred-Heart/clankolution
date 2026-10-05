use clank_app::gpu::agent_pipeline::forward_pass_canonical;
use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::sim::SimWorld;
use clank_core::agent::{AgentData, GENES};
use clank_core::World;

#[test]
fn test_rust_core_vs_gpu_pipeline_exact_brain_parity() {
    let mut agent = AgentData::default();
    // Fill genes with a deterministic pseudorandom pattern
    for i in 0..GENES {
        let val = ((i as i32 * 31 + 47) % 251) - 125;
        agent.genes[i] = val as i8;
    }
    // Fill previous hidden states with values in [-1, 1]
    for k in 0..10 {
        agent.h[k] = ((k as f32 * 0.2) - 0.9).clamp(-1.0, 1.0);
    }

    // 15 sensory inputs matching simulation ranges
    let sensory_inputs: [f64; 15] = [
        0.25,   // in0: energy
        0.80,   // in1: here food
        -0.30,  // in2: forward - left
        0.45,   // in3: forward - right
        0.10,   // in4: forward - here
        0.05,   // in5: taint
        -0.12,  // in6: scent diff
        0.707,  // in7: sin bearing
        0.707,  // in8: cos bearing
        0.85,   // in9: distance perception
        0.33,   // in10: density
        0.15,   // in11: age
        -0.42,  // in12: temporal sine
        0.55,   // in13: forward velocity
        -0.20,  // in14: neighbor energy
    ];

    let sensory_f32: [f32; 15] = sensory_inputs.map(|v| v as f32);

    // Run reference CPU brain
    let mut agent_cpu = agent;
    let cpu_out = World::brain(&mut agent_cpu, &sensory_inputs);
    let cpu_h = agent_cpu.h;

    // Pack into GpuAgentGenome
    let mut sim = SimWorld::default();
    sim.world.agents.clear();
    sim.world.agents.push(agent);
    let (_, genomes, _, _, _) = sync_rust_to_gpu(&sim);

    // Run GPU pipeline canonical brain
    let (gpu_h, gpu_out) = forward_pass_canonical(&genomes[0], &sensory_f32, &agent.h);

    // Verify bit-for-bit mathematical equivalence within 1e-4 tolerance
    for j in 0..10 {
        let diff = (cpu_h[j] - gpu_h[j]).abs();
        assert!(
            diff < 1e-4,
            "Hidden unit {} differs: CPU {}, GPU {} (diff {})",
            j,
            cpu_h[j],
            gpu_h[j],
            diff
        );
    }

    for j in 0..6 {
        let diff = (cpu_out[j] - gpu_out[j]).abs();
        assert!(
            diff < 1e-4,
            "Output actuator {} differs: CPU {}, GPU {} (diff {})",
            j,
            cpu_out[j],
            gpu_out[j],
            diff
        );
    }
}

#[test]
fn test_gpu_compute_driver_brain_dispatch_matches_cpu() {
    let driver = clank_app::gpu::compute_driver::GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let mut sim = SimWorld::new(123);
    // Keep only 1 agent so best_neighbor is None
    sim.world.agents.truncate(1);
    sim.world.growth = 0.0; // Disable soil growth during test
    sim.world.soil.food.fill(1.0);
    sim.world.soil.taint.fill(0.0);
    sim.world.soil.scent.fill(0.0);
    sim.world.sync_pos_cache();

    // Fill genes deterministically
    for i in 0..GENES {
        let val = ((i as i32 * 31 + 47) % 251) - 125;
        sim.world.agents[0].genes[i] = val as i8;
    }
    for k in 0..10 {
        sim.world.agents[0].h[k] = ((k as f32 * 0.2) - 0.9).clamp(-1.0, 1.0);
    }
    // Set position and energy
    sim.world.agents[0].x = 450.0;
    sim.world.agents[0].y = 300.0;
    sim.world.agents[0].vx = 0.5;
    sim.world.agents[0].vy = -0.3;
    sim.world.agents[0].angle = 1.25;
    sim.world.agents[0].energy = 55.0;

    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Run 1 sub-tick on GPU
    driver.dispatch_sub_ticks(1, &params);

    let updated_states = driver.readback_agent_states(1);
    assert_eq!(updated_states.len(), 1);

    // Also run 1 step on CPU with identical setup
    let mut cpu_sim = SimWorld::new(123);
    cpu_sim.world.agents.truncate(1);
    cpu_sim.world.growth = 0.0;
    cpu_sim.world.soil.food.fill(1.0);
    cpu_sim.world.soil.taint.fill(0.0);
    cpu_sim.world.soil.scent.fill(0.0);
    cpu_sim.world.agents[0] = sim.world.agents[0];
    cpu_sim.world.sync_pos_cache();
    cpu_sim.step(1);

    let gpu_h = updated_states[0].hidden;
    let cpu_h = cpu_sim.world.agents[0].h;

    // Verify hidden states match within 0.05 tolerance
    for j in 0..10 {
        let diff = (cpu_h[j] - gpu_h[j]).abs();
        assert!(
            diff < 0.05,
            "Hidden unit {} differs: CPU {}, GPU {} (diff {})",
            j, cpu_h[j], gpu_h[j], diff
        );
    }
}
