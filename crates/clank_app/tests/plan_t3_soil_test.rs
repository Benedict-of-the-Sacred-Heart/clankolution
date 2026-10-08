//! Plan Task 3: GPU soil chemistry (bloom renewal, food clamp [0, 2.5], taint ×0.994 − 0.0001,
//! scent ×0.954), boundary clamping, and the soil tools — on the real `soil_step.wgsl`,
//! compared with the reference `SoilGrid::step_soil` / `World` tools.

#[path = "support/gpu_harness.rs"]
mod gpu_harness;

use clank_app::gpu::soil_pipeline::GpuSoilPipeline;
use clank_app::sim::SimWorld;
use gpu_harness::*;

fn gpu_soil(gpu: &Gpu) -> Vec<(f32, f32, f32)> {
    gpu.driver
        .readback_soil()
        .iter()
        .map(GpuSoilPipeline::to_physics)
        .collect()
}

/// An empty seed-42 world with taint and scent sprinkled in, so all three decays run.
fn soil_world(growth: f64) -> SimWorld {
    let mut sim = world_with(Vec::new(), growth);
    let cells = sim.world.soil.food.len();
    for i in (0..cells).step_by(7) {
        sim.world.soil.taint[i] = 0.05 + (i % 9) as f32 * 0.2;
    }
    for i in (0..cells).step_by(5) {
        sim.world.soil.scent[i] = 0.01 + (i % 4) as f32 * 0.3;
    }
    sim
}

/// One soil tick on the GPU equals the reference cell by cell: regrowth, clamp and decays.
/// Cells where either engine rolled a random food spawn (different PRNGs) are excluded.
#[test]
fn test_one_soil_tick_matches_the_reference_cell_by_cell() {
    let mut sim = soil_world(100.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    sim.world.evolve();
    let got = gpu_soil(&gpu);
    let soil = &sim.world.soil;
    let (mut spawns, mut wrong) = (0, Vec::new());
    for (k, &(f, t, s)) in got.iter().enumerate() {
        let (rf, rt, rs) = (soil.food[k], soil.taint[k], soil.scent[k]);
        if (f - rf).abs() > 0.1 {
            spawns += 1;
            continue;
        }
        if (f - rf).abs() > 2e-6 || (t - rt).abs() > 2e-6 || (s - rs).abs() > 2e-6 {
            wrong.push(format!(
                "cell {k}: GPU ({f:.7}, {t:.7}, {s:.7}) vs reference ({rf:.7}, {rt:.7}, {rs:.7})"
            ));
        }
    }
    assert!(spawns <= 4, "{spawns} cells differ by a spawn-sized amount");
    assert!(
        wrong.is_empty(),
        "{} of {} cells differ:\n{}",
        wrong.len(),
        got.len(),
        wrong[..wrong.len().min(8)].join("\n")
    );
}

/// Food above the 2.5 ceiling is clamped to it, as the reference does.
#[test]
fn test_food_is_clamped_to_two_and_a_half() {
    let mut sim = soil_world(100.0);
    sim.world.soil.food[10] = 2.6;
    sim.world.soil.food[11] = 3.0;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    let got = gpu_soil(&gpu);
    assert_eq!(got[10].0, 2.5);
    assert_eq!(got[11].0, 2.5);
}

/// A creature at the far corner of the world grazes the last cell (74, 49): coordinates at
/// the seam are clamped into the grid instead of running past the buffer.
#[test]
fn test_grazing_at_the_world_corner_uses_the_corner_cell() {
    let corner = 49 * 75 + 74;
    let alone = world_with(Vec::new(), 0.0);
    let grazed = world_with(vec![creature(1, 899.999, 599.999, 0.0)], 0.0);
    let (Some(mut a), Some(mut b)) = (upload(&alone), upload(&grazed)) else {
        return;
    };
    a.tick();
    b.tick();
    let (fa, fb) = (gpu_soil(&a), gpu_soil(&b));
    // The creature moves before it eats (reference order): heading +x it crosses the seam
    // into cell (0, 49) on the same row.
    let across = 49 * 75;
    let eaten_corner = fa[corner].0 - fb[corner].0;
    let eaten_across = fa[across].0 - fb[across].0;
    assert!(
        eaten_corner > 0.0 || eaten_across > 0.0,
        "no food eaten at the corner cell ({eaten_corner}) or across the seam ({eaten_across})"
    );
    let elsewhere: Vec<usize> = (0..fa.len())
        .filter(|&k| k != corner && k != across && (fa[k].0 - fb[k].0).abs() > 1e-6)
        .collect();
    assert!(
        elsewhere.is_empty(),
        "grazing changed cells away from the creature: {elsewhere:?}"
    );
}

/// The Nourish tool on the GPU deposits food exactly like the reference `nourish_at`.
#[test]
fn test_nourish_tool_matches_the_reference() {
    tool_matches_reference(1, |w, x, y| w.nourish_at(x, y));
}

/// The Blight tool on the GPU deposits taint and removes food like the reference `blight_at`.
#[test]
fn test_blight_tool_matches_the_reference() {
    tool_matches_reference(2, |w, x, y| w.blight_at(x, y));
}

fn tool_matches_reference(tool: u32, reference: impl Fn(&mut clank_core::World, f64, f64)) {
    let mut sim = soil_world(0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let (x, y) = (452.0, 298.0);
    gpu.params.tool_type = tool;
    gpu.params.tool_pos = [x as f32, y as f32];
    gpu.params.tool_radius = 45.0;
    gpu.tick();
    reference(&mut sim.world, x, y);
    sim.world.evolve();
    let got = gpu_soil(&gpu);
    let soil = &sim.world.soil;
    let wrong: Vec<String> = got
        .iter()
        .enumerate()
        .filter(|(k, (f, t, _))| {
            (f - soil.food[*k]).abs() > 2e-6 || (t - soil.taint[*k]).abs() > 2e-6
        })
        .map(|(k, (f, t, _))| {
            format!(
                "cell ({}, {}): GPU food {f:.6} taint {t:.6} vs reference {:.6} {:.6}",
                k % 75,
                k / 75,
                soil.food[k],
                soil.taint[k]
            )
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{} cells differ:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(8)].join("\n")
    );
}

fn totals(cells: impl Iterator<Item = (f32, f32)>) -> (f64, f64) {
    cells.fold((0.0, 0.0), |(f, t), (cf, ct)| {
        (f + cf as f64, t + ct as f64)
    })
}

/// With no creatures, 600 ticks of renewal and random spawns keep the GPU's total food within
/// 3% of the reference's at every 50th tick from 100 to 600, and spawns push some cell above
/// the starting maximum. The reference steps its soil alone: `World::evolve` would seed
/// spores into the empty world at tick 45 and they would eat.
#[test]
fn test_600_soil_ticks_track_the_reference_food_total() {
    let mut sim = world_with(Vec::new(), 100.0);
    let initial_max = sim.world.soil.food.iter().copied().fold(0.0f32, f32::max);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let mut off = Vec::new();
    for tick in 1..=600 {
        gpu.tick();
        sim.world.soil.step_soil(&mut sim.world.prng, 1.0);
        if tick >= 100 && tick % 50 == 0 {
            let (g, _) = totals(gpu_soil(&gpu).into_iter().map(|(f, t, _)| (f, t)));
            let (r, _) = totals(sim.world.soil.food.iter().map(|&f| (f, 0.0)));
            if (g - r).abs() / r > 0.03 {
                off.push(format!(
                    "tick {tick}: GPU {g:.1} vs reference {r:.1} ({:+.1}%)",
                    (g - r) / r * 100.0
                ));
            }
        }
    }
    let gpu_max = gpu_soil(&gpu).iter().map(|c| c.0).fold(0.0f32, f32::max);
    assert!(
        off.is_empty(),
        "total food drifts more than 3%:\n{}",
        off.join("\n")
    );
    assert!(
        gpu_max > initial_max + 0.1,
        "no spawn pushed a cell above the starting maximum {initial_max:.3} (GPU max {gpu_max:.3})"
    );
}

/// An eclipse darkens the soil at the reference rate: on even ticks 160 random cells lose 27%
/// of their food and gain 0.14 taint (capped at 2). Over a 20-tick eclipse the GPU's food
/// loss and taint gain are within 20% of the reference's (different random cells). The app
/// counts the eclipse down once per tick in GPU mode; the test does the same with the params.
#[test]
fn test_an_eclipse_darkens_the_soil_at_the_reference_rate() {
    let mut sim = world_with(Vec::new(), 0.0);
    sim.world.eclipse = 20;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let (f0, t0) = totals(gpu_soil(&gpu).into_iter().map(|(f, t, _)| (f, t)));
    let (rf0, rt0) = totals(
        sim.world
            .soil
            .food
            .iter()
            .copied()
            .zip(sim.world.soil.taint.iter().copied()),
    );
    for _ in 0..20 {
        gpu.tick();
        gpu.params.eclipse = gpu.params.eclipse.saturating_sub(1);
        sim.world.evolve();
    }
    assert_eq!(
        sim.world.eclipse, 0,
        "reference eclipse did not end; scenario is wrong"
    );
    let (f1, t1) = totals(gpu_soil(&gpu).into_iter().map(|(f, t, _)| (f, t)));
    let (rf1, rt1) = totals(
        sim.world
            .soil
            .food
            .iter()
            .copied()
            .zip(sim.world.soil.taint.iter().copied()),
    );
    let (lost, gained, ref_lost, ref_gained) = (f0 - f1, t1 - t0, rf0 - rf1, rt1 - rt0);
    assert!(
        ref_lost > 0.0 && ref_gained > 0.0,
        "reference eclipse had no effect; scenario is wrong"
    );
    assert!(
        (lost - ref_lost).abs() / ref_lost < 0.2 && (gained - ref_gained).abs() / ref_gained < 0.2,
        "GPU eclipse: food lost {lost:.2}, taint gained {gained:.2}; reference: food lost {ref_lost:.2}, taint gained {ref_gained:.2}"
    );
}

/// In GPU mode the eclipse counts down once per tick, as the reference does: after GPU frames
/// and a switch back to the Rust engine, the countdown has dropped by the ticks that passed.
#[test]
fn test_gpu_mode_counts_the_eclipse_down() {
    let Some(mut app) = gpu_app(|sim| sim.world.eclipse = 60) else {
        return;
    };
    run(&mut app, 20);
    settle(&mut app);
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = clank_app::sim::ActiveEngine::Rust;
        sim.paused = false;
    }
    app.update();
    let sim = app.world().resource::<SimWorld>();
    let passed = sim.world.tick;
    assert_eq!(
        sim.world.eclipse,
        60u32.saturating_sub(passed),
        "after {passed} ticks (GPU then Rust) the eclipse countdown is {}",
        sim.world.eclipse
    );
}

/// An eclipse that ends partway through a frame stops darkening the soil there: with two
/// ticks of eclipse left and 8 sub-ticks in one frame, the soil takes one round of damage
/// (the one even tick), as the reference does over the same 8 ticks.
#[test]
fn test_an_eclipse_ends_mid_frame() {
    let mut sim = world_with(Vec::new(), 0.0);
    sim.world.eclipse = 2;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let t0: f64 = gpu_soil(&gpu).iter().map(|&(_, t, _)| t as f64).sum();
    let rt0: f64 = sim.world.soil.taint.iter().map(|&t| t as f64).sum();
    gpu.params.sub_ticks_per_frame = 8;
    gpu.driver.dispatch_sub_ticks(8, &gpu.params);
    for _ in 0..8 {
        sim.world.evolve();
    }
    let gained = gpu_soil(&gpu)
        .iter()
        .map(|&(_, t, _)| t as f64)
        .sum::<f64>()
        - t0;
    let ref_gained = sim.world.soil.taint.iter().map(|&t| t as f64).sum::<f64>() - rt0;
    assert!(ref_gained > 0.0, "reference eclipse had no effect");
    assert!(
        (gained - ref_gained).abs() / ref_gained < 0.2,
        "taint gained {gained:.2} vs reference {ref_gained:.2}"
    );
}

/// The eclipse strikes only on the reference's even ticks: with one tick of eclipse left
/// at tick 0, the reference's tick 1 is odd and the soil takes no damage.
#[test]
fn test_an_eclipse_skips_odd_ticks() {
    let mut sim = world_with(Vec::new(), 0.0);
    sim.world.eclipse = 1;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let t0: f64 = gpu_soil(&gpu).iter().map(|&(_, t, _)| t as f64).sum();
    gpu.tick();
    sim.world.evolve();
    let t1: f64 = gpu_soil(&gpu).iter().map(|&(_, t, _)| t as f64).sum();
    assert_eq!(sim.world.eclipse, 0, "scenario is wrong");
    assert!(
        t1 <= t0,
        "taint rose from {t0:.3} to {t1:.3} on an odd tick"
    );
}
