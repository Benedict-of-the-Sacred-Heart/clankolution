use bevy::prelude::*;
use clank_app::sim::{ClankSimPlugin, SimWorld};

#[test]
fn test_sim_world_initialization() {
    let sim = SimWorld::new(123);
    assert_eq!(sim.world.tick, 0);
    assert_eq!(sim.speed, 1);
    assert!(!sim.paused);
    assert_eq!(sim.world_width, 900.0);
    assert_eq!(sim.world_height, 600.0);
    assert_eq!(sim.world.agents.len(), 72);
    assert_eq!(sim.world.max_cap, 340);
}

#[test]
fn test_sim_world_stepping_and_pausing() {
    let mut sim = SimWorld::new(42);
    assert_eq!(sim.world.tick, 0);

    sim.step(5);
    assert_eq!(sim.world.tick, 5);

    sim.paused = true;
    sim.step(10);
    assert_eq!(sim.world.tick, 5); // Must not advance while paused

    sim.paused = false;
    sim.step(10);
    assert_eq!(sim.world.tick, 15);
}

#[test]
fn test_sim_world_selection() {
    let mut sim = SimWorld::new(99);
    assert!(sim.selected_agent_id.is_none());
    assert!(sim.get_selected_agent().is_none());

    let first_id = sim.world.agents[0].id;
    sim.selected_agent_id = Some(first_id);

    let selected = sim.get_selected_agent();
    assert!(selected.is_some());
    assert_eq!(selected.unwrap().id, first_id);

    sim.selected_agent_id = Some(0xFFFFFFFE);
    assert!(sim.get_selected_agent().is_none());
}

#[test]
fn test_bevy_sim_plugin_step_system() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(ClankSimPlugin);

    // Initial update should register resource and step once (speed = 1)
    app.update();
    {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(sim.world.tick, 1);
    }

    // Change speed to 4
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.speed = 4;
    }
    app.update();
    {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(sim.world.tick, 5);
    }

    // Pause simulation
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = true;
    }
    app.update();
    {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(sim.world.tick, 5);
    }
}
