use std::time::Instant;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::bridge::{sync_rust_to_gpu, sync_gpu_to_rust};
use clank_app::rendering::{generate_dart_mesh_from_gpu_states, generate_outline_mesh_from_gpu_states};
use clank_app::sim::SimWorld;

#[test]
fn bench_actual_pipeline_bottlenecks() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping benchmark: No GPU adapter");
        return;
    }
    let driver = driver.unwrap();

    let n = 5_000;
    let mut sim = SimWorld::new(42);
    sim.world.agents.resize(n, clank_core::agent::AgentData::default());
    sim.world.max_cap = n;
    for (i, a) in sim.world.agents.iter_mut().enumerate() {
        a.id = (i + 1) as u32;
        a.energy = 50.0;
        a.x = (i as f64 * 13.0) % 950.0;
        a.y = (i as f64 * 17.0) % 747.0;
        a.dead = 0;
    }

    println!("\n==========================================================================");
    println!("     EMPIRICAL PIPELINE TIMING BENCHMARK (5,000 AGENTS)");
    println!("==========================================================================");

    // 1. Serialization: sync_rust_to_gpu
    let iters = 20;
    let t0 = Instant::now();
    let mut last_data = None;
    for _ in 0..iters {
        last_data = Some(sync_rust_to_gpu(&sim));
    }
    let dur_sync_rust_to_gpu = t0.elapsed() / iters;
    println!("1. sync_rust_to_gpu (CPU serialize 5k agents):      {:.3} ms", dur_sync_rust_to_gpu.as_secs_f64() * 1000.0);

    let (states, genomes, atomics, soil, params) = last_data.unwrap();

    // 2. Upload state to GPU
    let t0 = Instant::now();
    for _ in 0..iters {
        driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    }
    let dur_upload_state = t0.elapsed() / iters;
    println!("2. driver.upload_state (PCIe CPU -> GPU write):     {:.3} ms", dur_upload_state.as_secs_f64() * 1000.0);

    // 3. GPU Compute Dispatch (1 sub-tick vs 4 sub-ticks)
    let t0 = Instant::now();
    for _ in 0..iters {
        driver.dispatch_sub_ticks(4, &params);
    }
    // ensure completed
    let _ = driver.readback_telemetry();
    let dur_dispatch = t0.elapsed() / iters;
    println!("3. driver.dispatch_sub_ticks (4 compute sub-ticks): {:.3} ms", dur_dispatch.as_secs_f64() * 1000.0);

    // 4. Current 5 separate blocking readbacks
    let t0 = Instant::now();
    for _ in 0..iters {
        let _s = driver.readback_agent_states(n);
        let _g = driver.readback_agent_genomes(n);
        let _a = driver.readback_atomics(n);
        let _so = driver.readback_soil();
        let _te = driver.readback_telemetry();
    }
    let dur_5_readbacks = t0.elapsed() / iters;
    println!("4. Current 5x separate blocking readbacks:          {:.3} ms  <--- CRITICAL BOTTLENECK", dur_5_readbacks.as_secs_f64() * 1000.0);

    // 5. Individual readback timings
    let t0 = Instant::now();
    for _ in 0..iters { let _ = driver.readback_agent_states(n); }
    let dur_read_states = t0.elapsed() / iters;
    println!("   - readback_agent_states only:                    {:.3} ms", dur_read_states.as_secs_f64() * 1000.0);

    let t0 = Instant::now();
    for _ in 0..iters { let _ = driver.readback_agent_genomes(n); }
    let dur_read_genomes = t0.elapsed() / iters;
    println!("   - readback_agent_genomes only (1.76 MB):         {:.3} ms", dur_read_genomes.as_secs_f64() * 1000.0);

    let t0 = Instant::now();
    for _ in 0..iters { let _ = driver.readback_soil(); }
    let dur_read_soil = t0.elapsed() / iters;
    println!("   - readback_soil only:                            {:.3} ms", dur_read_soil.as_secs_f64() * 1000.0);

    let t0 = Instant::now();
    for _ in 0..iters { let _ = driver.readback_telemetry(); }
    let dur_read_telem = t0.elapsed() / iters;
    println!("   - readback_telemetry only (128 bytes):           {:.3} ms", dur_read_telem.as_secs_f64() * 1000.0);

    // 6. Deserialization: sync_gpu_to_rust
    let t0 = Instant::now();
    for _ in 0..iters {
        sync_gpu_to_rust(&states, &genomes, &atomics, &soil, &params, &mut sim);
    }
    let dur_sync_gpu_to_rust = t0.elapsed() / iters;
    println!("5. sync_gpu_to_rust (CPU deserialize 5k agents):    {:.3} ms", dur_sync_gpu_to_rust.as_secs_f64() * 1000.0);

    // 7. CPU Mesh Generation: Allocating vs Reusable
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        generate_dart_mesh_from_gpu_states(&states, 747.0, &mut positions, &mut colors);
        let mut out_pos = Vec::new();
        let mut out_col = Vec::new();
        generate_outline_mesh_from_gpu_states(&states, 747.0, &mut out_pos, &mut out_col);
    }
    let dur_mesh_alloc = t0.elapsed() / iters;
    println!("6. CPU Mesh Gen (Current: Dynamic Vec realloc):     {:.3} ms", dur_mesh_alloc.as_secs_f64() * 1000.0);

    let mut pre_pos = Vec::with_capacity(n * 6);
    let mut pre_col = Vec::with_capacity(n * 6);
    let mut pre_out_pos = Vec::with_capacity(n * 16);
    let mut pre_out_col = Vec::with_capacity(n * 16);
    let t0 = Instant::now();
    for _ in 0..iters {
        pre_pos.clear();
        pre_col.clear();
        generate_dart_mesh_from_gpu_states(&states, 747.0, &mut pre_pos, &mut pre_col);
        pre_out_pos.clear();
        pre_out_col.clear();
        generate_outline_mesh_from_gpu_states(&states, 747.0, &mut pre_out_pos, &mut pre_out_col);
    }
    let dur_mesh_reuse = t0.elapsed() / iters;
    println!("7. CPU Mesh Gen (Optimized: Pre-allocated buffers): {:.3} ms", dur_mesh_reuse.as_secs_f64() * 1000.0);

    // 8. Parameter update only
    let t0 = Instant::now();
    for _ in 0..iters {
        driver.queue.write_buffer(&driver.sim_params_buf, 0, bytemuck::bytes_of(&params));
    }
    let dur_param_update = t0.elapsed() / iters;
    println!("8. Update params only (Persistent GPU mode):        {:.6} ms", dur_param_update.as_secs_f64() * 1000.0);

    println!("==========================================================================");
    let current_total = dur_sync_rust_to_gpu + dur_upload_state + dur_dispatch + dur_5_readbacks + dur_sync_gpu_to_rust + dur_mesh_alloc;
    let persistent_states_render = dur_param_update + dur_dispatch + dur_read_states + dur_read_telem + dur_mesh_reuse;
    let pure_gpu_vram_telemetry_only = dur_param_update + dur_dispatch + dur_read_telem;

    println!("CURRENT FRAME TIME (5k agents):              {:.2} ms ({:.1} FPS)",
        current_total.as_secs_f64() * 1000.0, 1.0 / current_total.as_secs_f64());
    println!("PERSISTENT GPU + 1x STATE READ FOR MESH:     {:.2} ms ({:.1} FPS)",
        persistent_states_render.as_secs_f64() * 1000.0, 1.0 / persistent_states_render.as_secs_f64());
    println!("PURE GPU-RESIDENT SIMULATION (TELEMETRY):    {:.2} ms ({:.1} FPS)",
        pure_gpu_vram_telemetry_only.as_secs_f64() * 1000.0, 1.0 / pure_gpu_vram_telemetry_only.as_secs_f64());
    println!("==========================================================================");
}
