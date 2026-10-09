//! Rust engine vs GPU engine throughput, on the same worlds (no rendering, no picking).
//! Ignored by default; run with
//! `cargo test -p clank_app --test engine_benchmark -- --ignored --nocapture`
//! (add `--release` for an optimised build).

use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;
use std::time::Instant;

const GPU_SLOTS: u32 = 65536;

/// The seed-42 world with `n` reference founders (`World::create_agent`) at random spots.
fn world_of(n: usize) -> SimWorld {
    let mut sim = SimWorld::new(42);
    sim.world.agents.clear();
    sim.world.max_cap = n;
    for _ in 0..n {
        let x = sim.world.prng.rand(0.0, 900.0);
        let y = sim.world.prng.rand(0.0, 600.0);
        sim.world.create_agent(x, y, None, None);
    }
    sim.world.sync_pos_cache();
    sim
}

fn live(sim: &SimWorld) -> usize {
    sim.world.agents.iter().filter(|a| a.dead == 0).count()
}

/// Milliseconds per tick of `World::evolve` over the world's first `ticks` ticks.
fn rust_ms_per_tick(sim: &mut SimWorld, ticks: u32) -> f64 {
    let t0 = Instant::now();
    for _ in 0..ticks {
        sim.world.evolve();
    }
    t0.elapsed().as_secs_f64() * 1000.0 / ticks as f64
}

/// Milliseconds per tick on the GPU in frames of `per_frame` sub-ticks, each frame waited on
/// (as the app's speed setting runs them), over the world's first `ticks` ticks. Returns
/// (ms/tick, population).
fn gpu_ms_per_tick(
    driver: &GpuComputeDriver,
    sim: &SimWorld,
    per_frame: u32,
    ticks: u32,
) -> (f64, u32) {
    let upload = || {
        let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(sim);
        driver.upload_state(&states, &genomes, &atomics, &soil, &params);
        params.max_agents = driver.active_slots(sim.world.max_cap as u32);
        params.tool_type = 0xFFFF_FFFF;
        params.sub_ticks_per_frame = per_frame;
        params
    };
    let frame = |params: &mut clank_app::gpu::types::GpuSimParams| {
        params.sub_tick = 0;
        driver.dispatch_sub_ticks(per_frame, params);
        let t = driver.readback_telemetry(); // waits for the GPU
        params.tick += per_frame;
        params.agent_count = t.population;
        t
    };
    // Warm up, then start again from the same world: every engine and speed times the same
    // ticks (later ticks cost less, so a 32-tick warm-up frame would flatter speed 32).
    frame(&mut upload());
    let mut params = upload();
    let frames = ticks.div_ceil(per_frame);
    let t0 = Instant::now();
    let mut pop = 0;
    for _ in 0..frames {
        pop = frame(&mut params).population;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / (frames * per_frame) as f64;
    (ms, pop)
}

#[test]
#[ignore]
fn bench_rust_vs_gpu_engines() {
    let probe = SimWorld::new(42);
    let (cols, rows) = (probe.world.soil.cols as u32, probe.world.soil.rows as u32);
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, GPU_SLOTS) else {
        eprintln!("Skipping benchmark: no GPU adapter");
        return;
    };
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!("\nRust vs GPU engine, {profile} build, ms per simulated tick (lower is better)");
    println!(
        "{:>8} | {:>10} {:>7} | {:>12} {:>7} | {:>12} {:>7} | {:>9}",
        "start", "Rust", "pop", "GPU speed 32", "pop", "GPU speed 1", "pop", "Rust/GPU32"
    );
    for n in [340usize, 1_000, 4_000, 16_000, 65_000] {
        let base = world_of(n);
        // Every engine runs the same ticks (populations drift, so equal spans keep the work
        // comparable); fewer where the Rust engine is slow, so the run stays short.
        let ticks = if n >= 16_000 { 32 } else { 64 };
        let mut rust = world_of(n); // the same world: world_of is deterministic
        let rust_ms = rust_ms_per_tick(&mut rust, ticks);
        let (gpu32, pop32) = gpu_ms_per_tick(&driver, &base, 32, ticks);
        let (gpu1, pop1) = gpu_ms_per_tick(&driver, &base, 1, ticks);
        println!(
            "{:>8} | {:>10.3} {:>7} | {:>12.3} {:>7} | {:>12.3} {:>7} | {:>8.1}x",
            n,
            rust_ms,
            live(&rust),
            gpu32,
            pop32,
            gpu1,
            pop1,
            rust_ms / gpu32
        );
    }
}
