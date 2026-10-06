use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;

#[test]
fn test_multi_tick_lockstep_parity() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let seed = 42;
    let mut sim_rust = SimWorld::new(seed);
    let mut sim_gpu = SimWorld::new(seed);

    // Disable random soil growth and hostility variation so baseline kinematics and grazing can be tracked
    sim_rust.world.growth = 0.0;
    sim_gpu.world.growth = 0.0;
    sim_rust.world.hostility = 0.0;
    sim_gpu.world.hostility = 0.0;

    // Start with 50 agents
    let n = 50;
    sim_rust.world.agents.truncate(n);
    sim_gpu.world.agents.truncate(n);
    sim_rust.world.sync_pos_cache();
    sim_gpu.world.sync_pos_cache();

    // Verify initial states are identical
    for i in 0..n {
        assert_eq!(sim_rust.world.agents[i].id, sim_gpu.world.agents[i].id);
        assert_eq!(sim_rust.world.agents[i].x, sim_gpu.world.agents[i].x);
        assert_eq!(sim_rust.world.agents[i].y, sim_gpu.world.agents[i].y);
    }

    // Upload sim_gpu to GPU
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim_gpu);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    println!("\n=======================================================");
    println!("MULTI-TICK LOCKSTEP PARITY: RUST CPU VS GPU (50 AGENTS)");
    println!("=======================================================");

    for tick in 1..=50 {
        // Step Rust 1 tick
        sim_rust.world.evolve();

        // Step GPU 1 tick
        params.tick = tick - 1;
        params.sub_tick = 0;
        params.sub_ticks_per_frame = 1;
        params.agent_count = n as u32;
        driver.dispatch_sub_ticks(1, &params);

        // Read back GPU states
        let g_states = driver.readback_agent_states(n);
        let g_atomics = driver.readback_atomics(n);

        let mut max_pos_diff: f64 = 0.0;
        let mut max_angle_diff: f64 = 0.0;
        let mut max_energy_diff: f64 = 0.0;
        let mut total_pos_diff: f64 = 0.0;
        let mut alive_rust = 0;
        let mut alive_gpu = 0;

        for i in 0..n {
            let r_agent = &sim_rust.world.agents[i];
            let g_agent = &g_states[i];
            let g_dead = (g_agent.meta_flags & (1 << 13)) != 0;
            let r_dead = r_agent.dead != 0;

            if !r_dead { alive_rust += 1; }
            if !g_dead { alive_gpu += 1; }

            if !r_dead && !g_dead {
                let gx = g_agent.pos_vel[0] as f64;
                let gy = g_agent.pos_vel[1] as f64;
                let g_angle = g_agent.angle_energy[0] as f64;
                let g_energy = (g_atomics[i].energy_milli as f64) * 0.001;

                let mut dx = (r_agent.x - gx).abs();
                if dx > 450.0 { dx = 900.0 - dx; }
                let mut dy = (r_agent.y - gy).abs();
                if dy > 300.0 { dy = 600.0 - dy; }
                let pos_diff = (dx * dx + dy * dy).sqrt();
                let angle_diff = (r_agent.angle - g_angle).abs();
                let energy_diff = (r_agent.energy - g_energy).abs();

                if pos_diff > max_pos_diff { max_pos_diff = pos_diff; }
                if angle_diff > max_angle_diff { max_angle_diff = angle_diff; }
                if energy_diff > max_energy_diff { max_energy_diff = energy_diff; }
                total_pos_diff += pos_diff;
            }
        }

        let mean_pos_diff = if alive_gpu > 0 { total_pos_diff / alive_gpu as f64 } else { 0.0 };
        println!(
            "Tick {:2}: alive R={} G={} | pos diff max={:.4} mean={:.4} | angle diff max={:.4} | energy diff max={:.4}",
            tick, alive_rust, alive_gpu, max_pos_diff, mean_pos_diff, max_angle_diff, max_energy_diff
        );
    }
}

#[test]
fn test_multi_tick_dynamic_simulation_parity() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let seed = 42;
    let mut sim_rust = SimWorld::new(seed);
    let mut sim_gpu = SimWorld::new(seed);

    // Default simulation settings matching HTML production
    sim_rust.world.growth = 100.0;
    sim_gpu.world.growth = 100.0;
    sim_rust.world.hostility = 100.0;
    sim_gpu.world.hostility = 100.0;
    sim_rust.world.mutation = 16.0;
    sim_gpu.world.mutation = 16.0;

    let n = sim_rust.world.agents.len();
    assert_eq!(n, sim_gpu.world.agents.len());

    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim_gpu);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    println!("\n=======================================================");
    println!("MULTI-TICK DYNAMIC SIMULATION: RUST VS GPU (DEFAULT 100 AGENTS - 200 TICKS)");
    println!("=======================================================");

    for tick in 1..=200 {
        sim_rust.world.evolve();

        params.tick = tick - 1;
        params.sub_tick = 0;
        params.sub_ticks_per_frame = 1;
        params.agent_count = sim_gpu.world.agents.len() as u32;
        driver.dispatch_sub_ticks(1, &params);

        let telem = driver.readback_telemetry();
        if tick % 10 == 0 || tick >= 65 && tick <= 75 {
            println!(
                "Tick {:3}: Rust Pop={:3} Kills={:2} Births={:3} | GPU Pop={:3} Kills={:2} Births={:3} Starvations={:2}",
                tick,
                sim_rust.world.agents.len(),
                sim_rust.world.kills,
                sim_rust.world.births,
                telem.population,
                telem.kills,
                telem.total_births.max(telem.birth_count),
                telem.starvations,
            );
        }
    }
}
