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
