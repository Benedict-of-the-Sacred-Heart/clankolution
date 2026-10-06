use clank_app::sim::SimWorld;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::bridge::sync_rust_to_gpu;

#[test]
fn test_behavior_divergence_debug() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let mut sim_rust = SimWorld::new(42);
    let mut sim_gpu = SimWorld::new(42);
    let n = 50;
    sim_rust.world.agents.resize(n, clank_core::agent::AgentData::default());
    sim_rust.world.max_cap = 340;
    sim_gpu.world.agents.resize(n, clank_core::agent::AgentData::default());
    sim_gpu.world.max_cap = 340;

    for i in 0..n {
        let x = 100.0 + (i as f64 * 17.0) % 700.0;
        let y = 100.0 + (i as f64 * 23.0) % 500.0;
        let angle = (i as f64) * 0.1;
        let tr = [0.5; 6];

        for sim in [&mut sim_rust, &mut sim_gpu] {
            let a = &mut sim.world.agents[i];
            a.id = (i + 1) as u32;
            a.energy = 50.0;
            a.x = x;
            a.y = y;
            a.angle = angle;
            a.tr = tr;
            a.dead = 0;
            a.age = 10;
        }
    }

    // Specifically place agent 0 and agent 1 close together
    for sim in [&mut sim_rust, &mut sim_gpu] {
        sim.world.agents[0].x = 100.0;
        sim.world.agents[0].y = 100.0;
        sim.world.agents[0].angle = 0.0;
        sim.world.agents[0].energy = 50.0;
        sim.world.agents[1].x = 105.0;
        sim.world.agents[1].y = 100.0;
        sim.world.agents[1].angle = 0.0;
        sim.world.agents[1].energy = 45.0;
    }

    // Upload sim_gpu to GPU
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim_gpu);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    println!("\n=== NEIGHBOR STEP COMPARISON ===");
    // Step Rust 1 tick
    sim_rust.world.evolve();
    let r0 = &sim_rust.world.agents[0];
    let r1 = &sim_rust.world.agents[1];
    println!("Post-Tick 1 Rust A0: pos=({:.3}, {:.3}) energy={:.3} cooldown={} attack={:.3}", r0.x, r0.y, r0.energy, r0.cooldown, r0.attack);
    println!("Post-Tick 1 Rust A1: pos=({:.3}, {:.3}) energy={:.3}", r1.x, r1.y, r1.energy);

    // Step GPU 1 tick
    params.tick = 0;
    params.sub_tick = 0;
    params.sub_ticks_per_frame = 1;
    params.agent_count = 50;
    driver.dispatch_sub_ticks(1, &params);

    // Test Combat: Give A0 genes that produce strong attack (o[3] > 0.25)
    // Output neuron 3 weights: bias word is at slot 88, let's set bias gene high
    // Or set A1 energy very low (e.g. 1.0) so damage kills A1
    sim_rust.world.hostility = 100.0;
    sim_gpu.world.hostility = 100.0;

    for sim in [&mut sim_rust, &mut sim_gpu] {
        sim.world.agents[0].x = 100.0;
        sim.world.agents[0].y = 100.0;
        sim.world.agents[0].angle = 0.0;
        sim.world.agents[0].energy = 50.0;
        // set bias of output neuron 3 (output 3 = attack) to +100
        // Output neuron 3 bias is at gene: 10 * 26 + 3 * 11 + 10 = 260 + 33 + 10 = 303
        sim.world.agents[0].genes[303] = 120;
        sim.world.agents[0].cooldown = 0;

        sim.world.agents[1].x = 104.0;
        sim.world.agents[1].y = 100.0;
        sim.world.agents[1].energy = 1.0; // weak, will die from 1 hit
        sim.world.agents[1].angle = 0.0;
    }

    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim_gpu);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    println!("\n=== COMBAT RESOLUTION COMPARISON ===");
    sim_rust.world.evolve();
    let r0 = &sim_rust.world.agents[0];
    let r1 = &sim_rust.world.agents[1];
    println!("Post-Combat Rust A0: energy={:.3} kills={} cooldown={} attack={:.3}", r0.energy, r0.kills, r0.cooldown, r0.attack);
    println!("Post-Combat Rust A1: dead={} energy={:.3}", r1.dead, r1.energy);

    params.tick = 1;
    params.sub_tick = 0;
    params.sub_ticks_per_frame = 1;
    params.agent_count = 50;
    driver.dispatch_sub_ticks(1, &params);

    let g_states = driver.readback_agent_states(50);
    let g_atomics = driver.readback_atomics(50);
    let telem = driver.readback_telemetry();
    let g0 = &g_states[0];
    let g1 = &g_states[1];
    let g0_energy = (g_atomics[0].energy_milli as f32) / 1000.0;
    let g1_energy = (g_atomics[1].energy_milli as f32) / 1000.0;
    let g0_kills = (g0.meta_flags >> 14) & 0x3FFFF;
    let g0_cooldown = (g0.meta_flags >> 4) & 0x03;
    let g1_dead = (g1.meta_flags & (1 << 13)) != 0;
    println!("Post-Combat GPU  A0: energy={:.3} kills={} cooldown={} attack={:.3} telem_kills={}", g0_energy, g0_kills, g0_cooldown, g0.angle_energy[3], telem.kills);
    println!("Post-Combat GPU  A1: dead={} energy={:.3}", g1_dead, g1_energy);
}
