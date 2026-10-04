use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::sim::SimWorld;

#[test]
fn test_gpu_compute_driver_initialization_and_dispatch() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let sim = SimWorld::new(42);
    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Dispatch 16 sub-ticks on GPU silicon
    driver.dispatch_sub_ticks(16, &params);

    // Read back telemetry to confirm GPU execution
    let telemetry = driver.readback_telemetry();
    assert!(telemetry.kills >= 1);
}
