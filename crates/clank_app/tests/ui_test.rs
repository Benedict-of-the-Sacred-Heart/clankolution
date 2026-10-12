use clank_app::sim::SimWorld;
use clank_app::ui::{
    compute_cycle_subtitle, set_simulation_speed, toggle_pause, trigger_catastrophe, UiState,
};

#[test]
fn test_ui_state_defaults() {
    let state = UiState::default();
    assert_eq!(state.speed, 1.0);
    assert!(state.show_stats);
    assert!(state.show_controls);
}

#[test]
fn test_ui_toggle_pause() {
    let mut sim = SimWorld::new(42);
    assert!(!sim.paused);

    toggle_pause(&mut sim);
    assert!(sim.paused);

    toggle_pause(&mut sim);
    assert!(!sim.paused);
}

#[test]
fn test_ui_set_speed() {
    let mut sim = SimWorld::new(42);
    let mut state = UiState::default();

    set_simulation_speed(&mut sim, &mut state, 4.0);
    assert_eq!(state.speed, 4.0);
    assert_eq!(sim.speed, 4);

    // Speed clamped to 1..32 (matching HTML range)
    set_simulation_speed(&mut sim, &mut state, 25.0);
    assert_eq!(state.speed, 25.0);
    assert_eq!(sim.speed, 25);

    set_simulation_speed(&mut sim, &mut state, 40.0);
    assert_eq!(state.speed, 32.0);
    assert_eq!(sim.speed, 32);

    set_simulation_speed(&mut sim, &mut state, 0.2);
    assert_eq!(state.speed, 1.0);
    assert_eq!(sim.speed, 1);
}

#[test]
fn test_persistence_filenames_and_status() {
    use clank_app::ui::{
        compute_export_clank_filename, compute_export_clank_status, compute_export_json_filename,
        compute_export_json_status, compute_import_clank_status, compute_import_json_status,
        DEFAULT_SAVE_HINT, WHAT_SAVES_NOTE,
    };

    assert_eq!(
        compute_export_clank_filename(123),
        "clankolution-cycle-123.clank"
    );
    assert_eq!(
        compute_export_json_filename(456),
        "clankolution-cycle-456.json"
    );

    assert_eq!(
        compute_export_clank_status(123, 2048),
        "Cycle 123 saved to binary snapshot (.clank, 2.0 KB)."
    );
    assert_eq!(
        compute_export_json_status(456),
        "Cycle 456 exported. Keep the JSON file to restore this world."
    );

    assert_eq!(
        compute_import_clank_status(123, 4096),
        "Cycle 123 restored from .clank binary snapshot (4.0 KB)."
    );
    assert_eq!(
        compute_import_json_status(456),
        "Cycle 456 restored. The simulation continues here."
    );

    assert_eq!(
        WHAT_SAVES_NOTE,
        ".clank saves an instant binary snapshot in microseconds via rkyv. .json saves a portable human-readable format."
    );
    assert_eq!(
        DEFAULT_SAVE_HINT,
        "Export a snapshot to continue generations later. Import it here on this or another device."
    );
}

#[test]
fn test_ui_trigger_catastrophe() {
    let mut sim = SimWorld::new(42);
    assert_eq!(sim.world.eclipse, 0);

    trigger_catastrophe(&mut sim);
    assert!(sim.world.eclipse > 0);
}

#[test]
fn test_ui_active_tool_selection() {
    let mut state = UiState::default();
    assert_eq!(state.active_tool, clank_app::ui::ActiveTool::Observe);
    state.active_tool = clank_app::ui::ActiveTool::Nourish;
    assert_eq!(state.active_tool, clank_app::ui::ActiveTool::Nourish);
}

#[test]
fn test_history_buffer_push() {
    let mut state = UiState::default();
    state.record_history(100, 1.2, 5);
    assert_eq!(state.history.len(), 1);
    assert_eq!(state.history[0].population, 100);
    assert_eq!(state.history[0].kills, 5);
}

#[test]
fn test_living_roots_count() {
    let mut sim = SimWorld::new(42);
    // Genesis starts with 72 creatures, all root 1..72
    let living_roots = sim
        .world
        .agents
        .iter()
        .filter(|a| a.dead == 0)
        .map(|a| a.root)
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert_eq!(living_roots, 72);

    // If one agent dies, unique roots count should still reflect living roots
    let _killed_root = sim.world.agents[0].root;
    sim.world.agents[0].dead = 1;
    let living_roots_after = sim
        .world
        .agents
        .iter()
        .filter(|a| a.dead == 0)
        .map(|a| a.root)
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert_eq!(living_roots_after, 71);
    assert_ne!(sim.world.roots, living_roots_after as u32);
}

#[test]
fn test_chronicle_event_formatting() {
    let mut state = UiState::default();
    state.add_chronicle("00100  Generation 2 opens in lineage 5.".to_string());
    assert_eq!(state.chronicle.len(), 2); // 1 default + 1 added
    assert_eq!(
        state.chronicle[0],
        "00100  Generation 2 opens in lineage 5."
    );
}

#[test]
fn test_compute_cycle_subtitle() {
    // Tick 0 is genesis
    assert_eq!(compute_cycle_subtitle(0, 0, 72), "THE FIRST HUNGER");
    // Eclipse takes precedence
    assert_eq!(compute_cycle_subtitle(100, 50, 250), "THE ECLIPSE");
    // Swarm when population > 200
    assert_eq!(compute_cycle_subtitle(100, 0, 340), "THE SWARM");
    // Remnant when population < 25
    assert_eq!(compute_cycle_subtitle(100, 0, 18), "THE REMNANT");
    // Standard hunger
    assert_eq!(compute_cycle_subtitle(100, 0, 120), "THE HUNGER");
}

#[test]
fn test_compute_tool_hint() {
    use clank_app::ui::{compute_tool_hint, ActiveTool};
    assert_eq!(
        compute_tool_hint(ActiveTool::Observe),
        "Click a creature to inspect its lineage. Choose a tool, then paint on the world."
    );
    assert_eq!(
        compute_tool_hint(ActiveTool::Nourish),
        "Drag on the world to grow food."
    );
    assert_eq!(
        compute_tool_hint(ActiveTool::Blight),
        "Drag on the world to spread blight."
    );
    assert_eq!(
        compute_tool_hint(ActiveTool::SeedLife),
        "Drag on the world to seed life."
    );
    assert_eq!(
        compute_tool_hint(ActiveTool::Extinguish),
        "Drag on the world to extinguish creatures."
    );
    assert_eq!(
        compute_tool_hint(ActiveTool::Eclipse),
        "Click a creature to inspect its lineage. Choose a tool, then paint on the world."
    );
}

#[test]
fn test_specimen_card_stays_inside_its_width_for_long_ids() {
    use bevy_egui::egui;
    use clank_app::ui::render_specimen_box;

    let mut sim = SimWorld::new(42);
    let agent = &mut sim.world.agents[0];
    agent.id = 454_337;
    agent.root = 123_456;
    agent.gen = 98_765;
    sim.selected_agent_id = Some(454_337);

    let ctx = egui::Context::default();
    clank_app::theme::setup_clank_theme(&ctx);
    let mut card_width = 0.0;
    for _ in 0..2 {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            card_width = render_specimen_box(ui, &sim).width();
        });
    }
    // The card spans the 290 px sidebar column, as the stat grid above it does.
    assert!(card_width <= 290.0, "specimen card is {card_width} px wide");
}

#[test]
fn test_two_stat_boxes_fit_the_sidebar_column() {
    use bevy_egui::egui;
    use clank_app::ui::render_stat_box;

    let ctx = egui::Context::default();
    clank_app::theme::setup_clank_theme(&ctx);
    let mut box_width = 0.0;
    for _ in 0..2 {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            box_width =
                render_stat_box(ui, "POPULATION", "65536", " / 65536", Some("capacity")).width();
        });
    }
    // The sidebar column is 290 px; two boxes share it with the grid's 8 px gap between them
    // (as the reference's two-column stat grid shares its column).
    assert!(
        2.0 * box_width + 8.0 <= 290.0,
        "a stat box is {box_width} px wide"
    );
}
