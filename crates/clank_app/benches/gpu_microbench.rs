//! High-Scale GPU Simulation Microbenchmarks
//!
//! Evaluates isolated hot-path algorithmic components across scales
//! from Tier 1 (1,000 minimum) up to Tier 6 (1,000,000 agents).
//!
//! Includes:
//! 1. Vectorized 88-Word unpack4x8snorm RNN Forward Pass
//! 2. Morton 32-bit Coordinate Encoding & 8-byte Radix Key Sorting
//! 3. Karras 2012 LBVH Hierarchy Construction (Phase 1 & Phase 2)
//! 4. Uncapped Mouse Picking with Dynamic Radius Shrinking
//! 5. Camera Frustum Culling
//! 6. Hierarchical Minimap LOD Cluster Extraction
//! 7. Decoupled Soil Diffusion & Atomic Grazing
//! 8. Bi-directional Dual-Engine Hot-Swap State Bridge

use std::hint::black_box;
use std::time::Instant;

use clank_app::gpu::agent_pipeline::forward_pass_packed;
use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu};
use clank_app::gpu::lbvh::LbvhTree;
use clank_app::gpu::soil_pipeline::GpuSoilPipeline;
use clank_app::gpu::spatial_index::compute_morton_32;
use clank_app::gpu::types::{GpuAgentGenome, GpuAgentState, GpuSoilCell};
use clank_app::sim::SimWorld;

fn create_mock_agents(n: usize) -> (Vec<GpuAgentState>, Vec<GpuAgentGenome>) {
    let mut states = Vec::with_capacity(n);
    let mut genomes = Vec::with_capacity(n);

    for i in 0..n {
        let x = (i as f32 * 17.3) % 900.0;
        let y = (i as f32 * 23.7) % 600.0;
        let morton = compute_morton_32([x, y]);

        states.push(GpuAgentState {
            pos_vel: [x, y, 0.5, -0.5],
            angle_energy: [1.2, 55.0, 0.4, 0.1],
            traits: [0.35, 0.45, 0.60, 0.20, 0.50, 0.15, 0.70, 0.0],
            hidden: [0.1; 10],
            id: i as u32 + 1,
            meta_flags: (i as u32 % 16),
            age_gen: ((i as u32 % 50) << 16) | (i as u32 % 1000),
            morton_code: morton,
            packed_color: 0xFF223344,
            visual_cache: 0x01020304,
        });

        // 88 words packed genes: 4 signed i8 weights each
        let mut words = [0u32; 88];
        for w in 0..88 {
            let b0 = ((i + w) % 255) as u8;
            let b1 = ((i + w * 2) % 255) as u8;
            let b2 = ((i + w * 3) % 255) as u8;
            let b3 = ((i + w * 4) % 255) as u8;
            words[w] = (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16) | ((b3 as u32) << 24);
        }
        genomes.push(GpuAgentGenome {
            packed_genes: words,
        });
    }

    (states, genomes)
}

fn bench_rnn_forward_pass(scales: &[usize]) {
    println!("\n### 1. Vectorized 88-Word RNN Forward Pass (`unpack4x8snorm`)");
    println!("| Scale (Agents) | Total Time | Per-Agent Latency | Throughput |");
    println!("| :--- | :--- | :--- | :--- |");

    let ins_hidden = [
        [0.5, 0.2, -0.1, 0.8],
        [0.1, -0.3, 0.4, 0.9],
        [0.2, -0.1, 0.55, 0.3],
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let ins_output = [
        [0.2, -0.4, 0.6, 0.1],
        [0.5, 0.3, -0.2, 0.8],
        [0.1, 0.7, 0.0, 0.0],
    ];

    for &n in scales {
        let (_, genomes) = create_mock_agents(n);

        // Warmup
        for g in genomes.iter().take(100.min(n)) {
            let _ = black_box(forward_pass_packed(g, &ins_hidden, &ins_output));
        }

        let iters = if n >= 100_000 {
            5
        } else if n >= 10_000 {
            30
        } else {
            100
        };

        let start = Instant::now();
        for _ in 0..iters {
            for g in &genomes {
                let _ = black_box(forward_pass_packed(g, &ins_hidden, &ins_output));
            }
        }
        let elapsed = start.elapsed();
        let total_per_run = elapsed / iters;
        let per_agent_ns = total_per_run.as_nanos() as f64 / n as f64;
        let throughput_mops = (n as f64 / total_per_run.as_secs_f64()) / 1_000_000.0;

        let total_str = if total_per_run.as_millis() > 0 {
            format!("{:.2} ms", total_per_run.as_secs_f64() * 1000.0)
        } else {
            format!("{:.1} µs", total_per_run.as_nanos() as f64 / 1000.0)
        };

        println!(
            "| {:>14} | {:>10} | {:>14.2} ns | {:>10.2} M/s |",
            n, total_str, per_agent_ns, throughput_mops
        );
    }
}

fn bench_morton_encoding_and_sorting(scales: &[usize]) {
    println!("\n### 2. Morton 32-Bit Coordinate Encoding & Radix Key Sorting");
    println!("| Scale (Agents) | Encode Total | Encode / Agent | Sort Total | Throughput |");
    println!("| :--- | :--- | :--- | :--- | :--- |");

    for &n in scales {
        let (agents, _) = create_mock_agents(n);

        let iters = if n >= 100_000 {
            10
        } else if n >= 10_000 {
            50
        } else {
            200
        };

        // Benchmark Morton Encoding
        let start_enc = Instant::now();
        for _ in 0..iters {
            for a in &agents {
                let _ = black_box(compute_morton_32([a.pos_vel[0], a.pos_vel[1]]));
            }
        }
        let elapsed_enc = start_enc.elapsed() / iters;
        let enc_per_agent_ns = elapsed_enc.as_nanos() as f64 / n as f64;
        let enc_total_str = if elapsed_enc.as_millis() > 0 {
            format!("{:.2} ms", elapsed_enc.as_secs_f64() * 1000.0)
        } else {
            format!("{:.1} µs", elapsed_enc.as_nanos() as f64 / 1000.0)
        };

        // Benchmark Sorting (morton_key, slot_idx)
        let keys: Vec<[u32; 2]> = agents
            .iter()
            .enumerate()
            .map(|(i, a)| [a.morton_code, i as u32])
            .collect();

        let iters_sort = if n >= 100_000 { 5 } else { 30 };
        let start_sort = Instant::now();
        for _ in 0..iters_sort {
            let mut k = keys.clone();
            k.sort_unstable_by(|a, b| a[0].cmp(&b[0]).then_with(|| a[1].cmp(&b[1])));
            black_box(&k);
        }
        let elapsed_sort = start_sort.elapsed() / iters_sort;
        let sort_total_str = if elapsed_sort.as_millis() > 0 {
            format!("{:.2} ms", elapsed_sort.as_secs_f64() * 1000.0)
        } else {
            format!("{:.1} µs", elapsed_sort.as_nanos() as f64 / 1000.0)
        };

        let total_throughput_mops = (n as f64 / (elapsed_enc + elapsed_sort).as_secs_f64()) / 1_000_000.0;

        println!(
            "| {:>14} | {:>12} | {:>11.2} ns | {:>10} | {:>10.2} M/s |",
            n, enc_total_str, enc_per_agent_ns, sort_total_str, total_throughput_mops
        );
    }
}

fn bench_lbvh_construction(scales: &[usize]) {
    println!("\n### 3. Karras 2012 LBVH Hierarchy Construction (Two-Phase)");
    println!("| Scale (Agents) | Total Build Time | Per-Node Latency | Build Rate | Tree Nodes |");
    println!("| :--- | :--- | :--- | :--- | :--- |");

    for &n in scales {
        if n > 100_000 {
            continue; // Skip ultra-high memory scales for full BVH building in single-threaded microbench
        }
        let (agents, _) = create_mock_agents(n);
        let mut keys: Vec<[u32; 2]> = agents
            .iter()
            .enumerate()
            .map(|(i, a)| [a.morton_code, i as u32])
            .collect();
        keys.sort_unstable_by(|a, b| a[0].cmp(&b[0]).then_with(|| a[1].cmp(&b[1])));

        // Warmup
        let _ = black_box(LbvhTree::build_from_keys(&keys, &agents));

        let iters = if n >= 32_768 {
            5
        } else if n >= 10_000 {
            15
        } else {
            40
        };

        let start = Instant::now();
        for _ in 0..iters {
            let tree = LbvhTree::build_from_keys(&keys, &agents);
            black_box(&tree);
        }
        let elapsed = start.elapsed() / iters;
        let total_nodes = (2 * n - 1) as f64;
        let per_node_ns = elapsed.as_nanos() as f64 / total_nodes;
        let build_rate_mops = (n as f64 / elapsed.as_secs_f64()) / 1_000_000.0;

        let total_str = if elapsed.as_millis() > 0 {
            format!("{:.2} ms", elapsed.as_secs_f64() * 1000.0)
        } else {
            format!("{:.1} µs", elapsed.as_nanos() as f64 / 1000.0)
        };

        println!(
            "| {:>14} | {:>16} | {:>13.2} ns | {:>8.2} M/s | {:>10} |",
            n, total_str, per_node_ns, build_rate_mops, 2 * n - 1
        );
    }
}

fn bench_uncapped_picking(scales: &[usize]) {
    println!("\n### 4. Uncapped Mouse Picking with Dynamic Radius Shrinking (<40ns Target)");
    println!("| Scale (Agents) | Direct Body Hit (Priority 0) | Halo Hit (Priority 1) | Empty Miss Query |");
    println!("| :--- | :--- | :--- | :--- |");

    for &n in scales {
        if n > 100_000 {
            continue;
        }
        let (agents, _) = create_mock_agents(n);
        let tree = LbvhTree::build(&agents);

        let target_agent = &agents[n / 2];
        let body_pos = [target_agent.pos_vel[0], target_agent.pos_vel[1]];
        let halo_pos = [target_agent.pos_vel[0] + 5.0, target_agent.pos_vel[1] + 5.0];
        let miss_pos = [899.9, 599.9]; // Out in empty corner

        let iters = 10_000;

        // Priority 0: Direct body hit
        let start_body = Instant::now();
        for _ in 0..iters {
            let res = tree.pick_agent(body_pos, &agents, 16.0);
            black_box(res);
        }
        let body_ns = start_body.elapsed().as_nanos() as f64 / iters as f64;

        // Priority 1: Halo hit
        let start_halo = Instant::now();
        for _ in 0..iters {
            let res = tree.pick_agent(halo_pos, &agents, 16.0);
            black_box(res);
        }
        let halo_ns = start_halo.elapsed().as_nanos() as f64 / iters as f64;

        // Miss query
        let start_miss = Instant::now();
        for _ in 0..iters {
            let res = tree.pick_agent(miss_pos, &agents, 16.0);
            black_box(res);
        }
        let miss_ns = start_miss.elapsed().as_nanos() as f64 / iters as f64;

        println!(
            "| {:>14} | {:>25.2} ns | {:>18.2} ns | {:>13.2} ns |",
            n, body_ns, halo_ns, miss_ns
        );
    }
}

fn bench_frustum_culling_and_minimap(scales: &[usize]) {
    println!("\n### 5. Camera Frustum Culling & Hierarchical Minimap LOD Clusters");
    println!("| Scale (Agents) | Frustum Cull Viewport | Minimap LOD Clusters | Visible Agents |");
    println!("| :--- | :--- | :--- | :--- |");

    for &n in scales {
        if n > 100_000 {
            continue;
        }
        let (agents, _) = create_mock_agents(n);
        let tree = LbvhTree::build(&agents);

        let view_min = [300.0, 200.0];
        let view_max = [600.0, 400.0];

        let iters = 2_000;

        let start_cull = Instant::now();
        let mut visible_count = 0;
        for _ in 0..iters {
            let visible = tree.cull_frustum(view_min, view_max, &agents);
            visible_count = visible.len();
            black_box(visible);
        }
        let cull_time = start_cull.elapsed() / iters;
        let cull_str = if cull_time.as_millis() > 0 {
            format!("{:.2} ms", cull_time.as_secs_f64() * 1000.0)
        } else {
            format!("{:.2} µs", cull_time.as_nanos() as f64 / 1000.0)
        };

        let start_minimap = Instant::now();
        for _ in 0..iters {
            let clusters = tree.extract_minimap_clusters(4);
            black_box(clusters);
        }
        let minimap_time = start_minimap.elapsed() / iters;
        let minimap_str = if minimap_time.as_millis() > 0 {
            format!("{:.2} ms", minimap_time.as_secs_f64() * 1000.0)
        } else {
            format!("{:.2} µs", minimap_time.as_nanos() as f64 / 1000.0)
        };

        println!(
            "| {:>14} | {:>21} | {:>20} | {:>14} |",
            n, cull_str, minimap_str, visible_count
        );
    }
}

fn bench_soil_diffusion_and_grazing() {
    println!("\n### 6. Decoupled Soil Diffusion & Atomic Grazing (3,750 Cells)");
    println!("| Component | Total Time | Per-Cell Latency | Throughput |");
    println!("| :--- | :--- | :--- | :--- |");

    let mut cells = vec![GpuSoilCell { food_milli: 1000, taint_milli: 0, scent_milli: 0, pad: 0 }; 3750];
    let bloom_table = vec![1.0f32; 3750];
    let iters = 5_000;

    let start = Instant::now();
    for _ in 0..iters {
        GpuSoilPipeline::step_soil(&mut cells, &bloom_table, 1.0);
        black_box(&cells);
    }
    let elapsed = start.elapsed() / iters;
    let per_cell_ns = elapsed.as_nanos() as f64 / 3750.0;
    let throughput = (3750.0 / elapsed.as_secs_f64()) / 1_000_000.0;

    println!(
        "| Soil Decay & Diffusion | {:>7.2} µs | {:>13.2} ns | {:>8.2} M/s |",
        elapsed.as_nanos() as f64 / 1000.0, per_cell_ns, throughput
    );
}

fn bench_state_hotswap_bridge(scales: &[usize]) {
    println!("\n### 7. Dual-Engine Live Hot-Swap Bridge Latency");
    println!("| Scale (Agents) | Rust -> GPU Upload | GPU -> Rust Readback | Transfer Bandwidth |");
    println!("| :--- | :--- | :--- | :--- |");

    for &n in scales {
        if n > 65_536 {
            continue; // Live sim bridge max capacity
        }
        let mut sim = SimWorld::new(42);
        sim.world.agents.resize(n, clank_core::agent::AgentData::default());
        sim.world.max_cap = n;

        let iters = if n >= 32_768 { 5 } else { 30 };

        // 1. Rust -> GPU
        let start_up = Instant::now();
        let mut last_states = Vec::new();
        let mut last_genomes = Vec::new();
        let mut last_atomics = Vec::new();
        let mut last_soil = Vec::new();
        let mut last_params = clank_app::gpu::types::GpuSimParams::default();

        for _ in 0..iters {
            let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
            last_states = states;
            last_genomes = genomes;
            last_atomics = atomics;
            last_soil = soil;
            last_params = params;
            black_box(&last_states);
        }
        let upload_time = start_up.elapsed() / iters;
        let up_str = format!("{:.2} ms", upload_time.as_secs_f64() * 1000.0);

        // 2. GPU -> Rust
        let start_down = Instant::now();
        for _ in 0..iters {
            sync_gpu_to_rust(
                &last_states,
                &last_genomes,
                &last_atomics,
                &last_soil,
                &last_params,
                &mut sim,
            );
            black_box(&sim.world.agents);
        }
        let readback_time = start_down.elapsed() / iters;
        let down_str = format!("{:.2} ms", readback_time.as_secs_f64() * 1000.0);

        let total_bytes = n * (128 + 352 + 16) + 3750 * 16;
        let bandwidth_gbps = (total_bytes as f64 / upload_time.as_secs_f64()) / 1_000_000_000.0;

        println!(
            "| {:>14} | {:>18} | {:>20} | {:>15.2} GB/s |",
            n, up_str, down_str, bandwidth_gbps
        );
    }
}

fn main() {
    println!("==========================================================================");
    println!("     CLANKOLUTION 2.0: HIGH-SCALE GPU SIMULATION BENCHMARK SUITE");
    println!("     Minimum Tier: 1,000 Agents | Maximum Tier: 1,000,000 Agents");
    println!("==========================================================================");

    let scales = [1_000, 5_000, 10_000, 32_768, 65_536, 100_000, 1_000_000];

    bench_rnn_forward_pass(&scales);
    bench_morton_encoding_and_sorting(&scales);
    bench_lbvh_construction(&scales);
    bench_uncapped_picking(&scales);
    bench_frustum_culling_and_minimap(&scales);
    bench_soil_diffusion_and_grazing();
    bench_state_hotswap_bridge(&scales);

    println!("\n==========================================================================");
    println!("                       BENCHMARK RUN COMPLETE");
    println!("==========================================================================");
}
