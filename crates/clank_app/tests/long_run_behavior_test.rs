use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;

#[test]
fn test_long_run_gpu_behavior() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let mut sim_rust = SimWorld::new(42);
    let mut sim_gpu = SimWorld::new(42);

    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim_gpu);
    params.max_capacity = 340;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    println!("\n=======================================================");
    println!("LONG RUN SIMULATION BEHAVIOR: RUST CPU VS GPU (1000 TICKS)");
    println!("=======================================================");

    let mut prev_gpu_pop = 72;
    let mut total_gpu_kills = 0;
    let mut total_gpu_starvations = 0;

    for tick in 1..=1000 {
        // Step Rust
        sim_rust.world.evolve();

        // Step GPU
        params.tick = tick - 1;
        params.sub_tick = 0;
        params.sub_ticks_per_frame = 1;
        params.agent_count = prev_gpu_pop;
        driver.dispatch_sub_ticks(1, &params);

        let telem = driver.readback_telemetry();
        prev_gpu_pop = telem.population;
        total_gpu_kills += telem.kills;
        total_gpu_starvations += telem.starvations;

        if tick % 100 == 0 {
            let rust_pop = sim_rust.world.agents.len();
            let rust_kills = sim_rust.world.kills;
            let rust_births = sim_rust.world.births;
            let gpu_births = telem.total_births.max(telem.birth_count);
            println!(
                "Tick {:4}: Rust[Pop={:3}, Kills={:3}, Births={:3}] | GPU[Pop={:3}, Kills={:3}, Births={:3}, Starvations={:3}, Top={:4}]",
                tick,
                rust_pop, rust_kills, rust_births,
                telem.population, total_gpu_kills, gpu_births, total_gpu_starvations, telem.freelist_top
            );

            // Assertions at checkpoints
            assert!(
                telem.population <= 340,
                "Tick {}: Population {} exceeded carrying capacity 340",
                tick, telem.population
            );
            assert!(
                telem.population >= 15,
                "Tick {}: Population {} collapsed below threshold 15",
                tick, telem.population
            );
            assert!(
                telem.freelist_top <= driver.max_agents,
                "Tick {}: Freelist top {} exceeded max agents {}",
                tick, telem.freelist_top, driver.max_agents
            );
        }
    }

    // Final long-run health assertions at Tick 1000:
    let final_telem = driver.readback_telemetry();
    assert!(
        final_telem.population >= 100,
        "After 1000 ticks, population should be flourishing (> 100), got {}",
        final_telem.population
    );
    assert!(
        total_gpu_kills >= 5,
        "After 1000 ticks, combat kills should have occurred (>= 5), got {}",
        total_gpu_kills
    );
    assert!(
        final_telem.total_births >= 50,
        "After 1000 ticks, births should have occurred (>= 50), got {}",
        final_telem.total_births
    );
}
