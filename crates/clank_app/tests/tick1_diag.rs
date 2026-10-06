use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;

#[test]
fn test_tick_1_diagnostics() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let seed = 42;
    let mut sim_rust = SimWorld::new(seed);
    let mut sim_gpu = SimWorld::new(seed);

    sim_rust.world.growth = 0.0;
    sim_gpu.world.growth = 0.0;
    sim_rust.world.hostility = 0.0;
    sim_gpu.world.hostility = 0.0;

    let n = 50;
    sim_rust.world.agents.truncate(n);
    sim_gpu.world.agents.truncate(n);
    sim_rust.world.sync_pos_cache();
    sim_gpu.world.sync_pos_cache();

    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim_gpu);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Initial state of agent 0
    let a0_init = &sim_rust.world.agents[0];
    println!("INITIAL AGENT 0:");
    println!("  pos: ({}, {}), vel: ({}, {})", a0_init.x, a0_init.y, a0_init.vx, a0_init.vy);
    println!("  angle: {}, energy: {}", a0_init.angle, a0_init.energy);
    println!("  traits: {:?}", a0_init.tr);

    // Step Rust and capture sensory inputs
    let ax = sim_rust.world.agents[0].x;
    let ay = sim_rust.world.agents[0].y;
    let angle = sim_rust.world.agents[0].angle;
    let tr = sim_rust.world.agents[0].tr;
    let fwd_x = angle.cos();
    let fwd_y = angle.sin();
    let reach = 19.0 + 38.0 * tr[2];
    let fwd_reach_x_term = fwd_x * reach;
    let fwd_reach_y_term = fwd_y * reach;
    let idx_here = sim_rust.world.soil.idx_fast(ax, ay);
    let here = sim_rust.world.soil.food[idx_here] as f64;
    println!("CELL idx_here = {}, food in sim_rust = {}", idx_here, here);
    println!("soil vector passed to upload has food_milli = {}", soil[idx_here].food_milli);
    let idx_fwd = sim_rust.world.soil.idx(ax + fwd_reach_x_term, ay + fwd_reach_y_term);
    let forward = sim_rust.world.soil.food[idx_fwd] as f64;
    let mid_x = ax + fwd_reach_x_term * 0.7;
    let mid_y = ay + fwd_reach_y_term * 0.7;
    let off_x = -fwd_reach_y_term * 0.6;
    let off_y = fwd_reach_x_term * 0.6;
    let left = sim_rust.world.soil.food[sim_rust.world.soil.idx(mid_x + off_x, mid_y + off_y)] as f64;
    let right = sim_rust.world.soil.food[sim_rust.world.soil.idx(mid_x - off_x, mid_y - off_y)] as f64;
    sim_rust.world.build_grid();
    let near_res = sim_rust.world.near(0);

    let mut r_ins = [0.0f64; 15];
    r_ins[0] = (sim_rust.world.agents[0].energy / 75.0 - 1.0).clamp(-1.0, 1.0);
    r_ins[1] = (here - 1.0).clamp(-1.0, 1.0);
    r_ins[2] = (forward - left).clamp(-1.0, 1.0);
    r_ins[3] = (forward - right).clamp(-1.0, 1.0);
    r_ins[4] = (forward - here).clamp(-1.0, 1.0);
    r_ins[5] = (sim_rust.world.soil.taint[idx_here] as f64).clamp(0.0, 1.0);
    r_ins[6] = ((sim_rust.world.soil.scent[idx_fwd] - sim_rust.world.soil.scent[idx_here]) as f64).clamp(-1.0, 1.0);
    if near_res.best_idx.is_some() {
        let bearing = near_res.best_dy.atan2(near_res.best_dx) - angle;
        r_ins[7] = bearing.sin();
        r_ins[8] = bearing.cos();
        r_ins[9] = (1.0 - near_res.best_d / (100.0 + 70.0 * tr[2])).clamp(-1.0, 1.0);
        let b = near_res.best_idx.unwrap();
        r_ins[14] = (sim_rust.world.agents[b].energy / 70.0 - 1.0).clamp(-1.0, 1.0);
    } else {
        r_ins[7] = 0.0;
        r_ins[8] = 1.0;
        r_ins[9] = -1.0;
        r_ins[14] = 0.0;
    }
    r_ins[10] = (near_res.density / 9.0).clamp(0.0, 1.0);
    r_ins[11] = ((sim_rust.world.agents[0].age + 1) as f64 / 600.0).clamp(0.0, 1.0);
    r_ins[12] = (0.0 * 0.08 + sim_rust.world.agents[0].id as f64).sin();
    r_ins[13] = (sim_rust.world.agents[0].vx * fwd_x + sim_rust.world.agents[0].vy * fwd_y).clamp(-1.0, 1.0);

    println!("\nRUST SENSORY INPUTS AGENT 0:");
    for (k, v) in r_ins.iter().enumerate() {
        println!("  ins[{:2}]: {:.6}", k, v);
    }
    println!("  near: best_idx={:?}, best_d={:.4}, density={}", near_res.best_idx, near_res.best_d, near_res.density);
    println!("  food: here={:.4}, fwd={:.4}, left={:.4}, right={:.4}", here, forward, left, right);

    // Manual brain calculation for neuron 0
    let g = &sim_rust.world.agents[0].genes;
    let mut s0 = 0.0f64;
    println!("\nNEURON 0 WEIGHTS & PRODUCTS:");
    for k in 0..15 {
        let term = (g[k] as f64) * r_ins[k];
        s0 += term;
        println!("  k={:2}: gene={:4}, in={:8.4}, term={:8.4}", k, g[k], r_ins[k], term);
    }
    let bias = g[25] as f64;
    s0 += bias;
    println!("  bias (gene 25): {}", bias);
    let h0_manual = (s0 * (0.61 / 127.0)).tanh();
    println!("  s0 total = {:.6}, h0 = {:.6}", s0, h0_manual);

    sim_rust.world.evolve();
    let a0_rust = &sim_rust.world.agents[0];

    // Step GPU
    params.tick = 0;
    params.sub_tick = 0;
    params.sub_ticks_per_frame = 1;
    params.agent_count = n as u32;
    driver.dispatch_sub_ticks(1, &params);

    let g_states = driver.readback_agent_states(n);
    let g_atomics = driver.readback_atomics(n);
    let g_genomes = driver.readback_agent_genomes(n);
    let g_soil = driver.readback_soil();
    let a0_gpu = &g_states[0];

    println!("AFTER GPU DISPATCH: g_soil[{}] food_milli = {}", idx_here, g_soil[idx_here].food_milli);

    let gpu_bytes: [u8; 352] = bytemuck::cast(g_genomes[0].packed_genes);
    println!("\nGPU FIRST 26 GENES VS RUST:");
    for k in 0..26 {
        let r_gene = sim_rust.world.agents[0].genes[k];
        let g_gene = gpu_bytes[k] as i8;
        if r_gene != g_gene {
            println!("  MISMATCH at gene {:2}: Rust={}, GPU={}", k, r_gene, g_gene);
        }
    }
    println!("  Gene 0: Rust={}, GPU={}", sim_rust.world.agents[0].genes[0], gpu_bytes[0] as i8);
    println!("  Gene 25: Rust={}, GPU={}", sim_rust.world.agents[0].genes[25], gpu_bytes[25] as i8);

    println!("\nTICK 1 AGENT 0 COMPARISON:");
    println!("  Rust pos: ({:.6}, {:.6})", a0_rust.x, a0_rust.y);
    println!("  GPU  pos: ({:.6}, {:.6})", a0_gpu.pos_vel[0], a0_gpu.pos_vel[1]);
    println!("  Rust vel: ({:.6}, {:.6})", a0_rust.vx, a0_rust.vy);
    println!("  GPU  vel: ({:.6}, {:.6})", a0_gpu.pos_vel[2], a0_gpu.pos_vel[3]);
    println!("  Rust angle: {:.6}", a0_rust.angle);
    println!("  GPU  angle: {:.6}", a0_gpu.angle_energy[0]);
    println!("  Rust energy: {:.6}", a0_rust.energy);
    println!("  GPU  energy: {:.6}", (g_atomics[0].energy_milli as f64) * 0.001);
    println!("  Rust feeding: {:.6}, attack: {:.6}", a0_rust.feeding, a0_rust.attack);
    println!("  GPU  feeding: {:.6}, attack: {:.6}", a0_gpu.angle_energy[2], a0_gpu.angle_energy[3]);
    println!("  Rust hidden: {:?}", a0_rust.h);
    println!("  GPU  hidden: {:?}", a0_gpu.hidden);
}
