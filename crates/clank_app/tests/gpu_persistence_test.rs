use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::sim::{SimWorld, ActiveEngine};

#[test]
fn test_gpu_persistence_across_frames() {
    let mut sim = SimWorld::new(42);
    sim.active_engine = ActiveEngine::Gpu;

    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let driver = GpuComputeDriver::create_for_world(cols, rows, 65536);
    if driver.is_none() {
        eprintln!("No WebGPU adapter found, skipping GPU persistence test");
        return;
    }
    let driver = driver.unwrap();

    // 1. Initial one-time upload
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    let initial_count = states.len();
    assert!(initial_count > 0);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    assert!(driver.is_initialized());

    // Initial telemetry check: freelist has available slots up to capacity
    let telem0 = driver.readback_telemetry();
    assert!(telem0.freelist_top > 0);

    // 2. Run 15 simulation frames strictly without calling upload_state or full readbacks
    for frame in 1..=15 {
        params.tick += 1;
        driver.update_params(&params);
        driver.dispatch_sub_ticks(1, &params);

        // Telemetry-only readback (128 bytes)
        let telem = driver.readback_telemetry();
        assert!(telem.population > 0, "Population should remain alive in VRAM at frame {}", frame);
    }

    // 3. Verify final VRAM state has evolved dynamically without CPU state re-upload
    let final_states = driver.readback_agent_states(initial_count);
    assert_eq!(final_states.len(), initial_count);

    // Living agents should have updated energy and positions
    let mut moved_count = 0;
    for (i, s) in final_states.iter().enumerate() {
        if (s.meta_flags & (1 << 13)) == 0 {
            if s.pos_vel[0] != states[i].pos_vel[0] || s.pos_vel[1] != states[i].pos_vel[1] {
                moved_count += 1;
            }
        }
    }
    assert!(moved_count > 5, "Agents in VRAM must move dynamically without CPU re-upload; moved: {}", moved_count);
}
