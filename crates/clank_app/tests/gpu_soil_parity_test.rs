use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;

/// Steps soil alone (no agents) on the real `soil_step.wgsl` pipeline and on the
/// reference `SoilGrid::step_soil`, comparing total food and spawn activity.
#[test]
fn test_gpu_soil_regrowth_and_spawns_match_reference() {
    let Some(driver) = GpuComputeDriver::create_default() else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };

    let mut reference = SimWorld::new(42);
    reference.world.agents.clear();
    reference.world.sync_pos_cache();

    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&reference);
    params.agent_count = 0;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let initial_max = reference
        .world
        .soil
        .food
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    let renewal = reference.world.growth / 100.0;
    let mut gpu_view = SimWorld::new(42);

    for tick in 1..=600u32 {
        reference
            .world
            .soil
            .step_soil(&mut reference.world.prng, renewal);

        params.tick = tick - 1;
        params.sub_tick = 0;
        params.sub_ticks_per_frame = 1;
        params.agent_count = 0;
        driver.dispatch_sub_ticks(1, &params);

        if tick % 100 == 0 {
            let gpu_soil = driver.readback_soil();
            sync_gpu_to_rust(&[], &[], &[], &gpu_soil, &params, &mut gpu_view);

            let ref_total: f64 = reference.world.soil.food.iter().map(|&f| f as f64).sum();
            let gpu_total: f64 = gpu_view.world.soil.food.iter().map(|&f| f as f64).sum();
            let rel = (gpu_total - ref_total).abs() / ref_total;
            println!("tick {tick:3}: reference food {ref_total:.1}, gpu food {gpu_total:.1}, rel diff {rel:.4}");
            assert!(
                rel < 0.03,
                "tick {tick}: GPU total food {gpu_total:.1} deviates {:.1}% from reference {ref_total:.1}",
                rel * 100.0
            );
        }
    }

    let gpu_max = gpu_view
        .world
        .soil
        .food
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    assert!(
        gpu_max > initial_max + 0.1,
        "random food spawns never fired on the GPU: max cell {gpu_max:.3}, initial max {initial_max:.3}"
    );
}
