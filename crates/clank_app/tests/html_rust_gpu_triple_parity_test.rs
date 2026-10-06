use std::fs;
use serde::Deserialize;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::*;
use clank_core::agent::AgentData;
use clank_core::world::World;

#[derive(Deserialize)]
struct ExportAgent {
    id: u32,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    angle: f64,
    energy: f64,
    root: u32,
    gen: u32,
    age: u32,
    cooldown: u32,
    birth: u32,
    kills: u32,
    dead: u32,
    #[serde(rename = "lastVictim")]
    last_victim: u32,
    tr: Vec<f64>,
    h: Vec<f32>,
    #[serde(default)]
    feeding: f64,
    #[serde(default)]
    attack: f64,
    #[serde(default)]
    genes: Vec<i8>,
}

#[derive(Deserialize)]
struct ExportTick {
    tick: u32,
    pop: usize,
    kills: u32,
    births: u32,
    agents: Vec<ExportAgent>,
}

#[derive(Deserialize)]
struct ExportRoot {
    seed: u32,
    #[serde(rename = "initialAgents")]
    initial_agents: Vec<ExportAgent>,
    #[serde(rename = "initialFood")]
    initial_food: Vec<f32>,
    #[serde(rename = "initialTaint")]
    initial_taint: Vec<f32>,
    #[serde(rename = "initialScent")]
    initial_scent: Vec<f32>,
    ticks: Vec<ExportTick>,
}

#[test]
fn test_html_vs_rust_vs_gpu_triple_parity() {
    let export_path = "../../scratch/html_sim_export.json";
    if !std::path::Path::new(export_path).exists() {
        eprintln!("Skipping test: {} does not exist", export_path);
        return;
    }

    let json_str = fs::read_to_string(export_path).expect("Failed to read export JSON");
    let export: ExportRoot = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    let driver = driver.unwrap();

    let n = export.initial_agents.len();
    assert_eq!(n, 72);

    // Initialize Rust CPU World
    let mut rust_world = World::new(export.seed);
    rust_world.agents.clear();
    for a in &export.initial_agents {
        let mut ad = AgentData::default();
        ad.id = a.id;
        ad.x = a.x;
        ad.y = a.y;
        ad.vx = a.vx;
        ad.vy = a.vy;
        ad.angle = a.angle;
        ad.energy = a.energy;
        ad.root = a.root;
        ad.gen = a.gen;
        ad.age = a.age;
        ad.cooldown = a.cooldown;
        ad.birth = a.birth;
        ad.kills = a.kills;
        ad.dead = a.dead;
        ad.last_victim = a.last_victim;
        for i in 0..6 { ad.tr[i] = a.tr[i]; }
        for i in 0..10 { ad.h[i] = a.h[i]; }
        for i in 0..326 { ad.genes[i] = a.genes[i]; }
        rust_world.agents.push(ad);
    }
    rust_world.sync_pos_cache();
    rust_world.soil.food.copy_from_slice(&export.initial_food);
    rust_world.soil.taint.copy_from_slice(&export.initial_taint);
    rust_world.soil.scent.copy_from_slice(&export.initial_scent);
    rust_world.growth = 100.0;
    rust_world.hostility = 100.0;
    rust_world.mutation = 16.0;

    // Initialize GPU buffers
    let mut gpu_states = Vec::with_capacity(10000);
    let mut gpu_genomes = Vec::with_capacity(10000);
    let mut gpu_atomics = Vec::with_capacity(10000);

    for (slot, a) in export.initial_agents.iter().enumerate() {
        let mut st = GpuAgentState {
            pos_vel: [a.x as f32, a.y as f32, a.vx as f32, a.vy as f32],
            angle_energy: [a.angle as f32, a.energy as f32, 0.0, 0.0],
            traits: [a.tr[0] as f32, a.tr[1] as f32, a.tr[2] as f32, a.tr[3] as f32, a.tr[4] as f32, a.tr[5] as f32, 0.0, a.last_victim as f32],
            hidden: [a.h[0], a.h[1], a.h[2], a.h[3], a.h[4], a.h[5], a.h[6], a.h[7], a.h[8], a.h[9]],
            id: a.id,
            meta_flags: (a.root & 0xF) | (a.cooldown << 4) | (a.birth << 6) | (a.kills << 14),
            age_gen: (a.age & 0xFFFF) | (a.gen << 16),
            morton_code: 0,
            packed_color: 0,
            visual_cache: 0,
        };
        let r_u8 = ((a.tr[0].clamp(0.0, 1.0)) * 255.0) as u32;
        let e_u8 = ((a.energy / 100.0).clamp(0.0, 1.0) * 255.0) as u32;
        st.visual_cache = r_u8 | (e_u8 << 16);
        gpu_states.push(st);

        let mut gen = GpuAgentGenome { packed_genes: [0u32; 88] };
        for w in 0..82 {
            let mut word = 0u32;
            for b in 0..4 {
                let gi = w * 4 + b;
                if gi < 326 {
                    let byte_val = a.genes[gi] as u8;
                    word |= (byte_val as u32) << (b * 8);
                }
            }
            gen.packed_genes[w] = word;
        }
        gpu_genomes.push(gen);

        gpu_atomics.push(GpuAgentAtomic {
            energy_milli: (a.energy * 1000.0) as i32,
            mate_claim: 0,
            mate_energy_milli: 0,
            dead_claimed: 0,
        });
    }

    // Pad to 10,000 for freelist
    for slot in n..10000 {
        let mut st = GpuAgentState {
            pos_vel: [0.0; 4],
            angle_energy: [0.0; 4],
            traits: [0.0; 8],
            hidden: [0.0; 10],
            id: 0,
            meta_flags: 1 << 13, // dead
            age_gen: 0,
            morton_code: 0,
            packed_color: 0,
            visual_cache: 0,
        };
        gpu_states.push(st);
        gpu_genomes.push(GpuAgentGenome { packed_genes: [0; 88] });
        gpu_atomics.push(GpuAgentAtomic {
            energy_milli: 0,
            mate_claim: 0,
            mate_energy_milli: 0,
            dead_claimed: 1,
        });
    }

    let mut soil_cells = Vec::with_capacity(3750);
    for i in 0..3750 {
        soil_cells.push(GpuSoilCell {
            food_milli: (export.initial_food[i] * 1000.0) as i32,
            taint_milli: (export.initial_taint[i] * 1000.0) as i32,
            scent_milli: (export.initial_scent[i] * 1000.0) as i32,
            fertility_milli: 0,
        });
    }

    let mut gpu_params = GpuSimParams {
        tick: 0,
        agent_count: n as u32,
        max_agents: 10000,
        max_capacity: 340,
        hostility: 1.0,
        mut_rate: 0.16,
        speed: 1.0,
        renewal: 1.0,
        sub_tick: 0,
        sub_ticks_per_frame: 1,
        tool_type: 0xFFFFFFFF,
        tool_radius: 45.0,
        tool_pos: [0.0, 0.0],
        camera_pos: [450.0, 300.0],
        camera_size: [900.0, 600.0],
        world_size: [900.0, 600.0],
        soil_grid: [75, 50],
        eclipse: 0,
        epoch: 72,
    };

    driver.upload_state(&gpu_states, &gpu_genomes, &gpu_atomics, &soil_cells, &gpu_params);

    println!("\n==========================================================================");
    println!("TRIPLE PARITY TEST: HTML (GROUND TRUTH) VS RUST CPU VS GPU COMPUTE");
    println!("==========================================================================");

    for (tick_idx, expected_tick) in export.ticks.iter().enumerate() {
        let t = tick_idx + 1;

        // Step Rust CPU
        rust_world.evolve();

        // Step GPU
        gpu_params.tick = (t - 1) as u32;
        gpu_params.sub_tick = 0;
        gpu_params.sub_ticks_per_frame = 1;
        gpu_params.agent_count = n as u32;
        driver.dispatch_sub_ticks(1, &gpu_params);

        let g_states = driver.readback_agent_states(n);
        let g_atomics = driver.readback_atomics(n);
        let telem = driver.readback_telemetry();

        // Compare HTML vs Rust vs GPU for the first 10 agents
        let mut html_rust_max_pos = 0.0f64;
        let mut html_gpu_max_pos = 0.0f64;
        let mut rust_gpu_max_pos = 0.0f64;

        let mut html_rust_max_energy = 0.0f64;
        let mut html_gpu_max_energy = 0.0f64;
        let mut rust_gpu_max_energy = 0.0f64;

        for (i, h_agent) in expected_tick.agents.iter().take(n).enumerate() {
            let r_agent = &rust_world.agents[i];
            let g_agent = &g_states[i];
            let g_energy = (g_atomics[i].energy_milli as f64) * 0.001;
            let gx = g_agent.pos_vel[0] as f64;
            let gy = g_agent.pos_vel[1] as f64;

            let hr_dx = (h_agent.x - r_agent.x).abs().min(900.0 - (h_agent.x - r_agent.x).abs());
            let hr_dy = (h_agent.y - r_agent.y).abs().min(600.0 - (h_agent.y - r_agent.y).abs());
            let hr_pos = (hr_dx * hr_dx + hr_dy * hr_dy).sqrt();

            let hg_dx = (h_agent.x - gx).abs().min(900.0 - (h_agent.x - gx).abs());
            let hg_dy = (h_agent.y - gy).abs().min(600.0 - (h_agent.y - gy).abs());
            let hg_pos = (hg_dx * hg_dx + hg_dy * hg_dy).sqrt();

            let rg_dx = (r_agent.x - gx).abs().min(900.0 - (r_agent.x - gx).abs());
            let rg_dy = (r_agent.y - gy).abs().min(600.0 - (r_agent.y - gy).abs());
            let rg_pos = (rg_dx * rg_dx + rg_dy * rg_dy).sqrt();

            html_rust_max_pos = html_rust_max_pos.max(hr_pos);
            html_gpu_max_pos = html_gpu_max_pos.max(hg_pos);
            rust_gpu_max_pos = rust_gpu_max_pos.max(rg_pos);

            html_rust_max_energy = html_rust_max_energy.max((h_agent.energy - r_agent.energy).abs());
            html_gpu_max_energy = html_gpu_max_energy.max((h_agent.energy - g_energy).abs());
            rust_gpu_max_energy = rust_gpu_max_energy.max((r_agent.energy - g_energy).abs());
        }

        println!(
            "Tick {:2}: Pos Diff [HTML-Rust: {:.4}px, HTML-GPU: {:.4}px, Rust-GPU: {:.4}px] | Energy Diff [HTML-Rust: {:.4}, HTML-GPU: {:.4}, Rust-GPU: {:.4}]",
            t, html_rust_max_pos, html_gpu_max_pos, rust_gpu_max_pos, html_rust_max_energy, html_gpu_max_energy, rust_gpu_max_energy
        );

        // Strict assertions:
        if t == 1 {
            assert!(
                html_rust_max_pos < 1e-4,
                "Tick 1: HTML vs Rust pos diff must be ~0, got {}",
                html_rust_max_pos
            );
            assert!(
                html_rust_max_energy < 1e-4,
                "Tick 1: HTML vs Rust energy diff must be ~0, got {}",
                html_rust_max_energy
            );
            assert!(
                html_gpu_max_pos < 0.05,
                "Tick 1: HTML vs GPU pos diff must be < 0.05px, got {}",
                html_gpu_max_pos
            );
            assert!(
                html_gpu_max_energy < 0.10,
                "Tick 1: HTML vs GPU energy diff must be < 0.10, got {}",
                html_gpu_max_energy
            );
        }

        // Bounded trajectory tolerance across 50 chaotic simulation steps
        assert!(
            html_rust_max_pos < 2.0,
            "Tick {}: HTML vs Rust pos divergence must stay < 2.0px, got {}",
            t, html_rust_max_pos
        );
        assert!(
            html_gpu_max_pos < 5.0,
            "Tick {}: HTML vs GPU pos divergence must stay < 5.0px, got {}",
            t, html_gpu_max_pos
        );
        assert!(
            html_gpu_max_energy < 8.0,
            "Tick {}: HTML vs GPU energy divergence must stay < 8.0, got {}",
            t, html_gpu_max_energy
        );
    }
}

#[test]
fn test_tick1_agent_breakdown() {
    let export_path = "../../scratch/html_sim_export.json";
    if !std::path::Path::new(export_path).exists() { return; }
    let json_str = fs::read_to_string(export_path).unwrap();
    let export: ExportRoot = serde_json::from_str(&json_str).unwrap();
    let driver = GpuComputeDriver::create_default().unwrap();
    let n = export.initial_agents.len();

    let mut rust_world = World::new(export.seed);
    rust_world.agents.clear();
    for a in &export.initial_agents {
        let mut ad = AgentData::default();
        ad.id = a.id; ad.x = a.x; ad.y = a.y; ad.vx = a.vx; ad.vy = a.vy;
        ad.angle = a.angle; ad.energy = a.energy; ad.root = a.root; ad.gen = a.gen;
        ad.age = a.age; ad.cooldown = a.cooldown; ad.birth = a.birth; ad.kills = a.kills;
        ad.dead = a.dead; ad.last_victim = a.last_victim;
        for i in 0..6 { ad.tr[i] = a.tr[i]; }
        for i in 0..10 { ad.h[i] = a.h[i]; }
        for i in 0..326 { ad.genes[i] = a.genes[i]; }
        rust_world.agents.push(ad);
    }
    rust_world.sync_pos_cache();
    rust_world.soil.food.copy_from_slice(&export.initial_food);
    rust_world.soil.taint.copy_from_slice(&export.initial_taint);
    rust_world.soil.scent.copy_from_slice(&export.initial_scent);
    rust_world.growth = 100.0; rust_world.hostility = 100.0; rust_world.mutation = 16.0;

    let mut gpu_states = Vec::new();
    let mut gpu_genomes = Vec::new();
    let mut gpu_atomics = Vec::new();

    for a in &export.initial_agents {
        let mut st = GpuAgentState {
            pos_vel: [a.x as f32, a.y as f32, a.vx as f32, a.vy as f32],
            angle_energy: [a.angle as f32, a.energy as f32, 0.0, 0.0],
            traits: [a.tr[0] as f32, a.tr[1] as f32, a.tr[2] as f32, a.tr[3] as f32, a.tr[4] as f32, a.tr[5] as f32, 0.0, a.last_victim as f32],
            hidden: [a.h[0], a.h[1], a.h[2], a.h[3], a.h[4], a.h[5], a.h[6], a.h[7], a.h[8], a.h[9]],
            id: a.id,
            meta_flags: (a.root & 0xF) | (a.cooldown << 4) | (a.birth << 6) | (a.kills << 14),
            age_gen: (a.age & 0xFFFF) | (a.gen << 16),
            morton_code: 0, packed_color: 0, visual_cache: 0,
        };
        let r_u8 = ((a.tr[0].clamp(0.0, 1.0)) * 255.0) as u32;
        let e_u8 = ((a.energy / 100.0).clamp(0.0, 1.0) * 255.0) as u32;
        st.visual_cache = r_u8 | (e_u8 << 16);
        gpu_states.push(st);

        let mut gen = GpuAgentGenome { packed_genes: [0u32; 88] };
        for w in 0..82 {
            let mut word = 0u32;
            for b in 0..4 {
                let gi = w * 4 + b;
                if gi < 326 {
                    let byte_val = a.genes[gi] as u8;
                    word |= (byte_val as u32) << (b * 8);
                }
            }
            gen.packed_genes[w] = word;
        }
        gpu_genomes.push(gen);
        gpu_atomics.push(GpuAgentAtomic {
            energy_milli: (a.energy * 1000.0) as i32,
            mate_claim: 0, mate_energy_milli: 0, dead_claimed: 0,
        });
    }

    for _ in n..10000 {
        gpu_states.push(GpuAgentState {
            pos_vel: [0.0; 4], angle_energy: [0.0; 4], traits: [0.0; 8], hidden: [0.0; 10],
            id: 0, meta_flags: 1 << 13, age_gen: 0, morton_code: 0, packed_color: 0, visual_cache: 0,
        });
        gpu_genomes.push(GpuAgentGenome { packed_genes: [0; 88] });
        gpu_atomics.push(GpuAgentAtomic { energy_milli: 0, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 1 });
    }

    let mut soil_cells = Vec::new();
    for i in 0..3750 {
        soil_cells.push(GpuSoilCell {
            food_milli: (export.initial_food[i] * 1000.0) as i32,
            taint_milli: (export.initial_taint[i] * 1000.0) as i32,
            scent_milli: (export.initial_scent[i] * 1000.0) as i32,
            fertility_milli: 0,
        });
    }

    let params = GpuSimParams {
        tick: 0, agent_count: n as u32, max_agents: 10000, max_capacity: 340,
        hostility: 1.0, mut_rate: 0.16, speed: 1.0, renewal: 1.0,
        sub_tick: 0, sub_ticks_per_frame: 1, tool_type: 0xFFFFFFFF, tool_radius: 45.0,
        tool_pos: [0.0, 0.0], camera_pos: [450.0, 300.0], camera_size: [900.0, 600.0],
        world_size: [900.0, 600.0], soil_grid: [75, 50], eclipse: 0, epoch: 72,
    };

    driver.upload_state(&gpu_states, &gpu_genomes, &gpu_atomics, &soil_cells, &params);

    rust_world.evolve();
    driver.dispatch_sub_ticks(1, &params);

    let g_states = driver.readback_agent_states(n);
    let g_atomics = driver.readback_atomics(n);
    let h_tick1 = &export.ticks[0];

    println!("\n=== TICK 1 DETAILED AGENT COMPARISON (FIRST 5 AGENTS) ===");
    for i in 0..5 {
        let h = &h_tick1.agents[i];
        let r = &rust_world.agents[i];
        let g = &g_states[i];
        let ge = (g_atomics[i].energy_milli as f64) * 0.001;

        println!("Agent {}:", i);
        println!("  X:     HTML={:.6}, Rust={:.6}, GPU={:.6}", h.x, r.x, g.pos_vel[0]);
        println!("  Y:     HTML={:.6}, Rust={:.6}, GPU={:.6}", h.y, r.y, g.pos_vel[1]);
        println!("  VX:    HTML={:.6}, Rust={:.6}, GPU={:.6}", h.vx, r.vx, g.pos_vel[2]);
        println!("  VY:    HTML={:.6}, Rust={:.6}, GPU={:.6}", h.vy, r.vy, g.pos_vel[3]);
        println!("  Angle: HTML={:.6}, Rust={:.6}, GPU={:.6}", h.angle, r.angle, g.angle_energy[0]);
        println!("  E:     HTML={:.6}, Rust={:.6}, GPU={:.6} (diff={:.6})", h.energy, r.energy, ge, (r.energy - ge).abs());
        println!("  Feed:  HTML={:.6}, Rust={:.6}, GPU={:.6}", h.feeding, r.feeding, g.angle_energy[2]);
        println!("  Atk:   HTML={:.6}, Rust={:.6}, GPU={:.6}", h.attack, r.attack, g.angle_energy[3]);
        if i == 0 {
            println!("  Hidden states for Agent 0:");
            for k in 0..10 {
                println!("    h[{}]: HTML={:.6}, Rust={:.6}, GPU={:.6} (diff={:.6})", k, h.h[k], r.h[k], g.hidden[k], (r.h[k] - g.hidden[k]).abs());
            }
        }
    }
}
