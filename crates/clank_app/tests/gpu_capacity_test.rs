use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;

const DEAD: u32 = 1 << 13;

/// The seed-42 world (72 creatures in slots 0..72) uploaded with carrying capacity `cap`.
fn uploaded(cap: u32) -> Option<(GpuComputeDriver, clank_app::gpu::types::GpuSimParams)> {
    let mut sim = SimWorld::new(42);
    sim.world.set_max_capacity(cap);
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let driver = GpuComputeDriver::create_for_world(cols, rows, 4096)?;
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.max_agents = driver.max_agents;
    params.sub_ticks_per_frame = 1;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    Some((driver, params))
}

/// Raising the creature capacity while the GPU engine runs frees the slots up to the new cap,
/// so the population can grow past the cap the world was uploaded with (clankolution.html
/// births whenever `agents.length < cap`).
#[test]
fn test_raising_the_cap_frees_slots_up_to_the_new_cap() {
    let Some((driver, mut params)) = uploaded(80) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    assert!(driver.readback_telemetry().freelist_top <= 8);

    params.max_capacity = 500;
    driver.dispatch_sub_ticks(1, &params);
    let telemetry = driver.readback_telemetry();
    assert!(
        telemetry.freelist_top > 400,
        "free slots after raising the cap to 500 with {} alive: {}",
        telemetry.population,
        telemetry.freelist_top
    );
}

/// Lowering the capacity below the population removes the creatures over it at once, as the
/// Rust engine's `set_max_capacity` truncates them.
#[test]
fn test_lowering_the_cap_removes_creatures_over_it() {
    let Some((driver, mut params)) = uploaded(340) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    params.max_capacity = 50;
    driver.dispatch_sub_ticks(1, &params);

    let telemetry = driver.readback_telemetry();
    assert!(
        telemetry.population <= 50,
        "population {}",
        telemetry.population
    );
    let states = driver.readback_agent_states(340);
    let live_over_cap = (50..340)
        .filter(|&i| states[i].meta_flags & DEAD == 0)
        .count();
    assert_eq!(
        live_over_cap, 0,
        "live creatures in slots at or above the cap"
    );
    assert!(
        telemetry.freelist_top as usize <= 50,
        "only slots under the cap are free: {}",
        telemetry.freelist_top
    );
}

/// App frames cover only the slots that can hold a creature: the cap, or more if the world
/// was uploaded with more creatures, never more than the driver has.
#[test]
fn test_active_slots_cover_the_cap_and_the_uploaded_creatures() {
    let Some((driver, _)) = uploaded(340) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    assert_eq!(driver.active_slots(340), 340);
    assert_eq!(driver.active_slots(50), 72, "the 72 uploaded creatures");
    assert_eq!(driver.active_slots(100_000), driver.max_agents);
}

/// Lowering the cap removes creatures over it even when frames cover fewer slots than the
/// population once filled: the capacity pass covers every slot.
#[test]
fn test_lowering_the_cap_removes_creatures_born_above_the_new_frame_size() {
    let Some((driver, mut params)) = uploaded(340) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    params.max_agents = driver.active_slots(340);
    let mut highest = 0;
    for _ in 0..400 {
        driver.dispatch_sub_ticks(1, &params);
        params.tick += 1;
        let states = driver.readback_agent_states(340);
        highest = (0..340)
            .filter(|&i| states[i].meta_flags & DEAD == 0)
            .max()
            .unwrap_or(0);
        if highest >= 100 {
            break;
        }
    }
    assert!(highest >= 100, "no creature was born above slot 100");

    params.max_capacity = 50;
    params.max_agents = driver.active_slots(50);
    driver.dispatch_sub_ticks(1, &params);
    let states = driver.readback_agent_states(340);
    let live_over_cap: Vec<usize> = (50..340)
        .filter(|&i| states[i].meta_flags & DEAD == 0)
        .collect();
    assert!(
        live_over_cap.is_empty(),
        "live creatures at or above the cap: {live_over_cap:?}"
    );
}
