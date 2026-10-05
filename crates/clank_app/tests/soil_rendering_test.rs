use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::{GpuAgentState, GpuAgentGenome, GpuAgentAtomic, GpuSoilCell, GpuSimParams};

#[test]
fn test_direct_gpu_soil_display_output() {
    let driver = GpuComputeDriver::create_for_world(75, 50, 65536);
    if driver.is_none() {
        eprintln!("No WebGPU adapter found, skipping soil rendering test");
        return;
    }
    let driver = driver.unwrap();

    let states: Vec<GpuAgentState> = Vec::new();
    let genomes: Vec<GpuAgentGenome> = Vec::new();
    let atomics: Vec<GpuAgentAtomic> = Vec::new();

    // Fill soil with test pattern: cell 0 rich in food, cell 1 tainted, cell 2 scented
    let mut soil = vec![GpuSoilCell { food_milli: 0, taint_milli: 0, scent_milli: 0, fertility_milli: 1000 }; 75 * 50];
    soil[0].food_milli = 1800;  // 1.8 food
    soil[1].taint_milli = 1000; // 1.0 taint
    soil[2].scent_milli = 1000; // 1.0 scent

    let params = GpuSimParams {
        tick: 1,
        agent_count: 0,
        max_agents: 0,
        max_capacity: 100,
        renewal: 0.05,
        world_size: [900.0, 600.0],
        soil_grid: [75, 50],
        ..Default::default()
    };

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Dispatch 1 sub-tick of soil simulation and direct texture rasterization
    driver.dispatch_sub_ticks(1, &params);

    // Read back soil display RGBA output from the GPU soil_display storage texture
    let rgba_data = driver.readback_soil_display_rgba();
    assert_eq!(rgba_data.len(), 75 * 50 * 4);

    // Cell 0 should have elevated green (food), cell 1 elevated red (taint), cell 2 elevated blue/green (scent)
    let p0 = [rgba_data[0], rgba_data[1], rgba_data[2], rgba_data[3]];
    let p1 = [rgba_data[4], rgba_data[5], rgba_data[6], rgba_data[7]];
    let p2 = [rgba_data[8], rgba_data[9], rgba_data[10], rgba_data[11]];

    assert!(p0[1] > 30, "Food cell should have high green channel: {:?}", p0);
    assert!(p1[0] > 50, "Taint cell should have high red channel: {:?}", p1);
    assert!(p2[2] > 30, "Scent cell should have high blue channel: {:?}", p2);
}
