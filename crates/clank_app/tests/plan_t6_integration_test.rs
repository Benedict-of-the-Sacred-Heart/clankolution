//! Plan Task 6: GPU-mode stats, the telemetry census, audio and the long
//! run — on the real shaders. App-level tests run the real `ClankSimPlugin` systems headless
//! (no renderer, a GPU device of their own). None of them picks (tool 0).

#[path = "support/gpu_harness.rs"]
mod gpu_harness;

use bevy::prelude::*;
use clank_app::audio::AudioVoiceQueue;
use clank_app::gpu::soil_pipeline::GpuSoilPipeline;
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};
use clank_core::agent::AgentData;
use gpu_harness::*;

/// In GPU mode the GENERATION and LINEAGES values the UI shows are the live GPU creatures'
/// highest generation and number of distinct lineages, after births have happened.
#[test]
fn test_gpu_mode_stats_follow_the_gpu() {
    let Some(mut app) = gpu_app(|sim| sim.speed = 4) else {
        return;
    };
    run(&mut app, 100);
    settle(&mut app);
    let live = gpu_creatures(app_driver(&app));
    let max_gen = live.iter().map(|a| a.gen).max().unwrap_or(0);
    assert!(
        max_gen > 0,
        "no births in 400 ticks; the test proves nothing"
    );
    let sim = app.world().resource::<SimWorld>();
    assert_eq!(
        (sim.max_generation(), sim.living_lineages()),
        (max_gen, distinct_roots(&live)),
        "UI (GENERATION, LINEAGES) vs the live GPU creatures"
    );
    let gpu_soil = app_driver(&app).readback_soil();
    let expected_mean_food: f32 = gpu_soil
        .iter()
        .map(|c| GpuSoilPipeline::to_physics(c).0)
        .sum::<f32>()
        / gpu_soil.len() as f32;
    assert!(
        (sim.active_mean_food() - expected_mean_food).abs() < 2e-3,
        "UI THE RECORD mean food ({}) must track live GPU soil ({expected_mean_food})",
        sim.active_mean_food()
    );
}

/// The census counts the creatures alive after the tick, that tick's newborns included, and
/// `apex_record_milli` is the highest live energy (before or after the tick).
#[test]
fn test_telemetry_census_matches_the_live_creatures() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(60);
    let mut wrong = Vec::new();
    let mut births = 0;
    for tick in 60..100 {
        let before = gpu.live();
        let t = gpu.tick();
        let after = gpu.live();
        births += after.iter().filter(|(_, a)| a.age == 0).count();
        if t.population as usize != after.len() {
            wrong.push(format!(
                "tick {tick}: population {} but {} alive ({} newborn)",
                t.population,
                after.len(),
                after.iter().filter(|(_, a)| a.age == 0).count()
            ));
        }
        let max_e =
            |v: &[(usize, AgentData)]| v.iter().map(|(_, a)| a.energy).fold(0.0f64, f64::max);
        let apex = t.apex_record_milli as f64 / 1000.0;
        if (apex - max_e(&after)).abs() > 0.01 && (apex - max_e(&before)).abs() > 0.01 {
            wrong.push(format!(
                "tick {tick}: apex {apex:.3} but highest energy {:.3} before, {:.3} after",
                max_e(&before),
                max_e(&after)
            ));
        }
    }
    assert!(
        births > 0,
        "no births in ticks 60..100; the census of newborns is untested"
    );
    assert!(
        wrong.is_empty(),
        "{} census errors:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(10)].join("\n")
    );
}

/// Kill and birth voices in the camera's view reach `AudioVoiceQueue` through the app's frame
/// readback; voices outside the view do not. The camera shows 180×120px around (450, 300).
#[test]
fn test_audio_voices_reach_the_playback_queue() {
    let attacker = |id: u32, x: f64, y: f64| {
        let mut a = creature(id, x, y, 0.0);
        a.genes[260 + 3 * 11 + 10] = 127;
        a.tr = [0.8, 0.5, 0.5, 0.5, 0.5, 0.9];
        a.age = 10;
        a
    };
    let victim = |id: u32, x: f64, y: f64| {
        let mut v = creature(id, x, y, std::f64::consts::PI);
        v.energy = 0.4;
        v.age = 10;
        v
    };
    let parent = |id: u32, x: f64, y: f64| {
        let mut p = creature(id, x, y, 0.0);
        p.energy = 100.0;
        p.age = 70;
        p
    };
    let (seen_kill, seen_birth, hidden_kill, hidden_birth) = (
        (450.0, 300.0),
        (470.0, 330.0),
        (100.0, 100.0),
        (800.0, 500.0),
    );
    let agents = vec![
        attacker(1, seen_kill.0, seen_kill.1),
        victim(2, seen_kill.0 + 8.0, seen_kill.1),
        attacker(3, hidden_kill.0, hidden_kill.1),
        victim(4, hidden_kill.0 + 8.0, hidden_kill.1),
        parent(5, seen_birth.0, seen_birth.1),
        parent(6, hidden_birth.0, hidden_birth.1),
    ];
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    app.insert_resource(AudioVoiceQueue::new());
    app.world_mut().spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: 0.2,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(450.0, 600.0 - 300.0, 0.0),
    ));
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Gpu;
        sim.world.next_id = 7;
        sim.world.roots = 7;
        sim.world.agents = agents;
        sim.world.growth = 0.0;
        sim.world.sync_pos_cache();
    }
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }
    let mut voices = Vec::new();
    for _ in 0..4 {
        app.update();
        voices.extend(app.world().resource::<AudioVoiceQueue>().voices.clone());
    }
    settle(&mut app);
    voices.extend(app.world().resource::<AudioVoiceQueue>().voices.clone());
    let near = |v: &clank_app::gpu::types::AudioVoice, (x, y): (f64, f64), r: f32| {
        (v.pos[0] - x as f32).abs() < r && (v.pos[1] - y as f32).abs() < r
    };
    let mut wrong = Vec::new();
    if !voices
        .iter()
        .any(|v| v.event_type == 1 && near(v, seen_kill, 10.0))
    {
        wrong.push("no kill voice from the kill in view".to_string());
    }
    if !voices
        .iter()
        .any(|v| v.event_type == 2 && near(v, seen_birth, 15.0))
    {
        wrong.push("no birth voice from the birth in view".to_string());
    }
    let leaked: Vec<_> = voices
        .iter()
        .filter(|v| near(v, hidden_kill, 30.0) || near(v, hidden_birth, 30.0))
        .collect();
    if !leaked.is_empty() {
        wrong.push(format!("voices from outside the view: {leaked:?}"));
    }
    assert!(
        wrong.is_empty(),
        "{} (all voices: {voices:?})",
        wrong.join("; ")
    );
}

/// From the same seed-42 world, the app's GPU engine and the reference stay within ±20% on
/// population, cumulative births, kills and total soil food at ticks 200, 400, 600, 800 and
/// 1000 (statistical, not lockstep: different random draws). Small counts (kills) get the
/// count's own noise instead when it is larger: 3√n.
#[test]
fn test_long_run_matches_the_reference_statistically() {
    let mut reference = SimWorld::new(42);
    let Some(mut app) = gpu_app(|_| {}) else {
        return;
    };
    let mut wrong = Vec::new();
    let mut rows = Vec::new();
    for checkpoint in [200u32, 400, 600, 800, 1000] {
        while app.world().resource::<SimWorld>().world.tick < checkpoint {
            run(&mut app, 1);
        }
        settle(&mut app);
        while reference.world.tick < app.world().resource::<SimWorld>().world.tick {
            reference.world.evolve();
        }
        let food: f64 = app_driver(&app)
            .readback_soil()
            .iter()
            .map(|c| GpuSoilPipeline::to_physics(c).0 as f64)
            .sum();
        let sim = app.world().resource::<SimWorld>();
        let r = &reference.world;
        let ref_food: f64 = r.soil.food.iter().map(|&f| f as f64).sum();
        let pairs = [
            (
                "population",
                sim.active_population() as f64,
                r.agents.len() as f64,
            ),
            ("births", sim.world.births as f64, r.births as f64),
            ("kills", sim.world.kills as f64, r.kills as f64),
            ("soil food", food, ref_food),
        ];
        rows.push(format!(
            "tick {}: {}",
            sim.world.tick,
            pairs
                .iter()
                .map(|(n, g, r)| format!("{n} {g:.0}/{r:.0}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        for (name, g, r) in pairs {
            if (g - r).abs() > (0.2 * r).max(3.0 * r.sqrt()).max(1.0) {
                wrong.push(format!(
                    "tick {}: {name} {g:.0} vs reference {r:.0}",
                    sim.world.tick
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{}\n(GPU/reference per checkpoint:\n{})",
        wrong.join("\n"),
        rows.join("\n")
    );
}

/// clankolution.html brain shape: N=15 inputs, Q=10 hidden, O=6 outputs.
fn output_bias(j: usize) -> usize {
    10 * (15 + 10 + 1) + j * 11 + 10
}

/// An attacker on its 7th kill beside a victim with 0.4 energy: the kill is its 8th.
fn kill_milestone_world() -> SimWorld {
    let mut a = creature(1, 400.0, 300.0, 0.0);
    a.genes[output_bias(3)] = 127;
    a.tr = [0.8, 0.5, 0.5, 0.5, 0.5, 0.9];
    a.age = 10;
    a.kills = 7;
    a.root = 40;
    let mut victim = creature(2, 407.0, 300.0, 0.0);
    victim.energy = 0.4;
    victim.age = 10;
    world_with(vec![a, victim], 0.0)
}

/// Chronicle events of one GPU frame, through the per-frame readback the app uses.
fn frame_events(gpu: &mut Gpu) -> Vec<(u32, u32, u32, u32)> {
    gpu.tick();
    gpu.driver.request_frame_readback(None, false);
    // A synchronous read waits for the GPU to go idle, so the frame has landed.
    gpu.driver.readback_telemetry();
    gpu.driver
        .poll_frame_readbacks()
        .iter()
        .flat_map(|f| f.chronicle.iter())
        .map(|e| (e.tick, e.event_type, e.p1, e.p2))
        .collect()
}

/// The reference chronicles "Lineage R has taken N lives." on a creature's every 8th kill;
/// the GPU reports the same event, with the same tick, lineage and count.
#[test]
fn test_a_kill_milestone_is_chronicled_like_the_reference() {
    let mut sim = kill_milestone_world();
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let events = frame_events(&mut gpu);
    sim.world.evolve();
    let reference: Vec<(u32, u32, u32, u32)> = sim
        .world
        .events
        .iter()
        .map(|e| (e.tick, e.event_type, e.p1, e.p2))
        .collect();
    assert_eq!(reference, vec![(1, 3, 40, 8)], "scenario is wrong");
    assert_eq!(events, reference, "GPU chronicle (tick, type, p1, p2)");
}

/// The reference chronicles "Generation G opens in lineage R." for 8% of births whose child
/// generation is a positive multiple of 12. 450 parents of generation 11 each give birth
/// once: about 36 events (σ ≈ 5.8), each naming generation 12 and the parent's lineage.
#[test]
fn test_generation_milestones_are_chronicled_at_the_reference_rate() {
    let parents: Vec<AgentData> = (0..450u32)
        .map(|k| {
            let mut a = creature(
                1 + k,
                15.0 + (k % 30) as f64 * 30.0,
                15.0 + (k / 30) as f64 * 40.0,
                0.0,
            );
            a.energy = 100.0;
            a.age = 70;
            a.gen = 11;
            a
        })
        .collect();
    let mut sim = world_with(parents, 0.0);
    sim.world.max_cap = 4000; // room for every child
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let births_before = gpu.driver.readback_telemetry().total_births;
    let events = frame_events(&mut gpu);
    let births = gpu.driver.readback_telemetry().total_births - births_before;
    assert_eq!(
        births, 450,
        "every parent gives birth; the scenario is wrong"
    );
    let wrong: Vec<_> = events
        .iter()
        .filter(|&&(tick, t, gen, root)| {
            tick != 1 || t != 2 || gen != 12 || !(1..=450).contains(&root)
        })
        .collect();
    assert!(wrong.is_empty(), "malformed generation events: {wrong:?}");
    assert!(
        (16..=60).contains(&events.len()),
        "{} generation events from 450 generation-12 births; the reference expects about 36",
        events.len()
    );
}

/// In GPU mode the app's chronicle receives the GPU's events (the UI drains
/// `sim.world.events`), as it receives the reference's in Rust mode.
#[test]
fn test_gpu_events_reach_the_chronicle() {
    let Some(mut app) = gpu_app(|sim| sim.world = kill_milestone_world().world) else {
        return;
    };
    run(&mut app, 3);
    settle(&mut app);
    let sim = app.world().resource::<SimWorld>();
    assert!(
        sim.world
            .events
            .iter()
            .any(|e| e.event_type == 3 && e.p1 == 40 && e.p2 == 8),
        "the kill milestone never reached the chronicle: {:?}",
        sim.world.events
    );
}

/// In GPU mode, interactive tools (picking, Nourish, Blight, Extinguish) take effect at once
/// even while the simulation is paused, without advancing the simulation tick.
#[test]
fn test_gpu_tools_act_at_once_while_paused() {
    use clank_app::gpu::soil_pipeline::GpuSoilPipeline;
    use clank_app::sim::{GpuDriverResource, PendingTool};

    let Some(mut app) = gpu_app(|_| {}) else {
        return;
    };
    run(&mut app, 4);
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    settle(&mut app);

    let paused_tick = app.world().resource::<SimWorld>().world.tick;
    let live_before = gpu_creatures(app_driver(&app));
    let target = live_before
        .iter()
        .find(|a| {
            live_before
                .iter()
                .all(|b| b.id == a.id || torus_dist(a.x, a.y, b.x, b.y, 900.0, 600.0) > 50.0)
        })
        .expect("an isolated live creature")
        .clone();

    // 1. Picking while paused selects the creature and populates its specimen cache.
    app.world_mut().resource_mut::<SimWorld>().pending_tool = Some(PendingTool {
        tool_type: 0,
        tool_pos: [target.x as f32, target.y as f32],
        tool_radius: 25.0,
    });
    app.update();
    settle(&mut app);
    {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(
            sim.world.tick, paused_tick,
            "paused pick must not advance tick"
        );
        assert_eq!(
            sim.selected_agent_id,
            Some(target.id),
            "paused pick must select the creature under the cursor"
        );
        assert_eq!(
            sim.get_selected_agent().map(|a| a.id),
            Some(target.id),
            "paused pick must populate the selected specimen cache"
        );
    }

    // 2. Nourish while paused immediately increases GPU soil food at the target cell.
    let cell_idx = 10 * 75 + 10; // (126.0, 126.0)
    let food_before = {
        let gpu = app.world().resource::<GpuDriverResource>();
        let soil = gpu.driver.as_ref().unwrap().readback_soil();
        GpuSoilPipeline::to_physics(&soil[cell_idx]).0
    };
    app.world_mut().resource_mut::<SimWorld>().pending_tool = Some(PendingTool {
        tool_type: 1,
        tool_pos: [126.0, 126.0],
        tool_radius: 45.0,
    });
    app.update();
    let food_after = {
        let gpu = app.world().resource::<GpuDriverResource>();
        let soil = gpu.driver.as_ref().unwrap().readback_soil();
        GpuSoilPipeline::to_physics(&soil[cell_idx]).0
    };
    assert!(
        food_after > food_before + 0.1,
        "Nourish while paused must increase GPU soil food immediately ({food_before} -> {food_after})"
    );

    // 3. Blight while paused immediately increases GPU soil taint at the target cell.
    let blight_cell = 20 * 75 + 20; // (246.0, 246.0)
    let taint_before = {
        let gpu = app.world().resource::<GpuDriverResource>();
        let soil = gpu.driver.as_ref().unwrap().readback_soil();
        GpuSoilPipeline::to_physics(&soil[blight_cell]).1
    };
    app.world_mut().resource_mut::<SimWorld>().pending_tool = Some(PendingTool {
        tool_type: 2,
        tool_pos: [246.0, 246.0],
        tool_radius: 45.0,
    });
    app.update();
    let taint_after = {
        let gpu = app.world().resource::<GpuDriverResource>();
        let soil = gpu.driver.as_ref().unwrap().readback_soil();
        GpuSoilPipeline::to_physics(&soil[blight_cell]).1
    };
    assert!(
        taint_after > taint_before + 0.1,
        "Blight while paused must increase GPU soil taint immediately ({taint_before} -> {taint_after})"
    );

    // 4. Extinguish while paused immediately removes the target creature on the GPU.
    app.world_mut().resource_mut::<SimWorld>().pending_tool = Some(PendingTool {
        tool_type: 3,
        tool_pos: [target.x as f32, target.y as f32],
        tool_radius: 23.0,
    });
    app.update();
    let live_after = gpu_creatures(app_driver(&app));
    assert!(
        live_after.iter().all(|a| a.id != target.id),
        "Extinguish while paused must remove creature {} immediately",
        target.id
    );

    // 5. Seed Life while paused immediately spawns two founders in VRAM and updates population/lineages/births/sparks.
    let pop_before_seed = app.world().resource::<SimWorld>().active_population();
    let lineages_before_seed = app.world().resource::<SimWorld>().living_lineages();
    let births_before_seed = app.world().resource::<SimWorld>().world.births;
    app.world_mut()
        .resource_mut::<SimWorld>()
        .world
        .spark_events
        .clear();
    app.world_mut()
        .resource_scope(|world, mut sim: bevy::prelude::Mut<SimWorld>| {
            let gpu = world.resource::<GpuDriverResource>();
            clank_app::sim::seed_life_gpu_aware(&mut sim, gpu.driver.as_ref(), &[(450.0, 300.0)]);
        });
    assert_eq!(
        app.world().resource::<SimWorld>().active_population(),
        pop_before_seed + 2,
        "Seed Life while paused must immediately increment active_population by 2"
    );
    assert_eq!(
        app.world().resource::<SimWorld>().living_lineages(),
        lineages_before_seed + 2,
        "Seed Life while paused must immediately increment living_lineages by 2"
    );
    assert_eq!(
        app.world().resource::<SimWorld>().world.births,
        births_before_seed + 2,
        "Seed Life in GPU mode must increment world.births by 2 like clankolution.html"
    );
    assert!(
        !app.world()
            .resource::<SimWorld>()
            .world
            .spark_events
            .is_empty(),
        "Seed Life in GPU mode must emit a teal spark event like clankolution.html"
    );
    assert_eq!(
        app_driver(&app).readback_telemetry().population as usize,
        pop_before_seed + 2,
        "Seed Life while paused must immediately update GPU telemetry population"
    );

    // 6. Lowering Creature capacity while paused in GPU mode immediately trims excess creatures.
    app.world_mut()
        .resource_mut::<SimWorld>()
        .world
        .set_max_capacity(50);
    app.update();
    settle(&mut app);
    assert!(
        app.world().resource::<SimWorld>().active_population() <= 50,
        "Lowering max_cap to 50 while paused in GPU mode must trim population to <= 50"
    );
    assert_eq!(
        app.world().resource::<SimWorld>().world.tick,
        paused_tick,
        "paused tools and capacity changes must never advance the simulation tick"
    );
}

/// In GPU mode, birth sparks carry the parent's lineage color (`root % 8`, count 9) rather than
/// eclipse red (`color_idx == 9`), death sparks carry the dying creature's lineage color
/// (`root % 8`, count 5), and running Nourish/Blight emit gold (`10`) and blight-red (`11`) sparks.
#[test]
fn test_gpu_sparks_use_lineage_colors_for_births_and_deaths_and_tool_colors_for_nourish_blight() {
    let mut parent = creature(1, 450.0, 300.0, 0.0);
    parent.root = 5; // lineage color 5
    parent.energy = 100.0;
    parent.age = 70;
    parent.birth = 0;

    let mut starving = creature(2, 500.0, 300.0, 0.0);
    starving.root = 3; // lineage color 3
    starving.energy = 0.05; // starves on tick 1
    starving.age = 10;

    let Some(mut app) = gpu_app(|sim| {
        sim.world.agents = vec![parent, starving];
        sim.world.next_id = 3;
        sim.world.roots = 6;
        sim.world.growth = 0.0;
        sim.world.sync_pos_cache();
        for f in sim.world.soil.food.iter_mut() {
            *f = 0.0;
        }
    }) else {
        return;
    };

    app.world_mut()
        .resource_mut::<SimWorld>()
        .world
        .spark_events
        .clear();
    run(&mut app, 2);
    settle(&mut app);

    let sparks = app
        .world()
        .resource::<SimWorld>()
        .world
        .spark_events
        .clone();
    assert!(
        sparks.iter().any(|s| s.color_idx == 5 && s.count == 9),
        "Expected birth spark with parent lineage color_idx 5 and count 9, got {sparks:?}"
    );
    assert!(
        sparks.iter().any(|s| s.color_idx == 3 && s.count == 5),
        "Expected death spark with dying creature lineage color_idx 3 and count 5, got {sparks:?}"
    );

    // Running Nourish and Blight via pending_tool must emit tool sparks (10 and 11).
    app.world_mut()
        .resource_mut::<SimWorld>()
        .world
        .spark_events
        .clear();
    app.world_mut().resource_mut::<SimWorld>().pending_tool = Some(clank_app::sim::PendingTool {
        tool_type: 1,
        tool_pos: [200.0, 200.0],
        tool_radius: 45.0,
    });
    run(&mut app, 1);
    app.world_mut().resource_mut::<SimWorld>().pending_tool = Some(clank_app::sim::PendingTool {
        tool_type: 2,
        tool_pos: [250.0, 250.0],
        tool_radius: 45.0,
    });
    run(&mut app, 1);

    let tool_sparks = app
        .world()
        .resource::<SimWorld>()
        .world
        .spark_events
        .clone();
    assert!(
        tool_sparks.iter().any(|s| s.color_idx == 10 && s.count == 2),
        "Expected running GPU Nourish to emit gold spark (color_idx 10, count 2), got {tool_sparks:?}"
    );
    assert!(
        tool_sparks.iter().any(|s| s.color_idx == 11 && s.count == 2),
        "Expected running GPU Blight to emit blight spark (color_idx 11, count 2), got {tool_sparks:?}"
    );
}
