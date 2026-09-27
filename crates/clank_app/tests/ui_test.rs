use clank_app::sim::SimWorld;
use clank_app::ui::{UiState, trigger_spore_catastrophe, set_simulation_speed, toggle_pause};

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

    // Speed clamped to 1..10
    set_simulation_speed(&mut sim, &mut state, 25.0);
    assert_eq!(state.speed, 10.0);
    assert_eq!(sim.speed, 10);

    set_simulation_speed(&mut sim, &mut state, 0.2);
    assert_eq!(state.speed, 1.0);
    assert_eq!(sim.speed, 1);
}

#[test]
fn test_ui_trigger_spore_catastrophe() {
    let mut sim = SimWorld::new(42);
    assert_eq!(sim.world.eclipse, 0);

    trigger_spore_catastrophe(&mut sim);
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
    let living_roots = sim.world.agents.iter()
        .filter(|a| a.dead == 0)
        .map(|a| a.root)
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert_eq!(living_roots, 72);

    // If one agent dies, unique roots count should still reflect living roots
    let _killed_root = sim.world.agents[0].root;
    sim.world.agents[0].dead = 1;
    let living_roots_after = sim.world.agents.iter()
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
    assert_eq!(state.chronicle[0], "00100  Generation 2 opens in lineage 5.");
}

