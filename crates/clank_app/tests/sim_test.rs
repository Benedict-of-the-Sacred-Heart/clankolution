use bevy::prelude::*;
use clank_app::sim::{ClankSimPlugin, SimWorld};

#[test]
fn test_sim_world_initialization() {
    let sim = SimWorld::new(123);
    assert_eq!(sim.world.tick, 0);
    assert_eq!(sim.speed, 1);
    assert!(!sim.paused);
    assert_eq!(sim.world_width, 950.0);
    assert_eq!(sim.world_height, 747.0);
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

#[test]
fn test_sim_world_extinguish_tool() {
    let mut sim = SimWorld::new(42);
    let target_agent = &sim.world.agents[0];
    let tx = target_agent.x;
    let ty = target_agent.y;
    let initial_sparks = sim.world.spark_events.len();

    sim.world.extinguish_at(tx, ty, 25.0);

    // Target agent must now be dead
    let dead_count = sim.world.agents.iter().filter(|a| a.dead != 0).count();
    assert!(dead_count >= 1);
    // Sparks must have been generated
    assert!(sim.world.spark_events.len() > initial_sparks);
}

#[test]
fn test_sim_world_nourish_tool() {
    let mut sim = SimWorld::new(42);
    let initial_sparks = sim.world.spark_events.len();
    sim.world.nourish_at(450.0, 300.0);

    // Food deposit should have occurred at (450, 300)
    let soil = &sim.world.soil;
    let col = ((450.0 / soil.w * soil.cols as f64) as usize).clamp(0, soil.cols - 1);
    let row = ((300.0 / soil.h * soil.rows as f64) as usize).clamp(0, soil.rows - 1);
    let idx = row * soil.cols + col;
    assert!(soil.food[idx] > 0.0);

    // Sparks should include color_idx 10 (gold)
    assert!(sim.world.spark_events.len() > initial_sparks);
    let last_spark = sim.world.spark_events.last().unwrap();
    assert_eq!(last_spark.color_idx, 10);
    assert_eq!(last_spark.count, 2);
}

#[test]
fn test_sim_world_blight_tool() {
    let mut sim = SimWorld::new(42);
    let initial_sparks = sim.world.spark_events.len();
    sim.world.blight_at(450.0, 300.0);

    let soil = &sim.world.soil;
    let col = ((450.0 / soil.w * soil.cols as f64) as usize).clamp(0, soil.cols - 1);
    let row = ((300.0 / soil.h * soil.rows as f64) as usize).clamp(0, soil.rows - 1);
    let idx = row * soil.cols + col;
    assert!(soil.taint[idx] > 0.0);

    // Sparks should include color_idx 8 (red)
    assert!(sim.world.spark_events.len() > initial_sparks);
    let last_spark = sim.world.spark_events.last().unwrap();
    assert_eq!(last_spark.color_idx, 8);
    assert_eq!(last_spark.count, 2);
}

#[test]
fn test_sim_world_seed_life_tool() {
    let mut sim = SimWorld::new(42);
    let initial_living = sim.world.agents.iter().filter(|a| a.dead == 0).count();
    let initial_sparks = sim.world.spark_events.len();
    sim.world.seed_life_at(450.0, 300.0);

    let new_living = sim.world.agents.iter().filter(|a| a.dead == 0).count();
    assert_eq!(new_living, initial_living + 2);

    // Sparks should include color_idx 9 (cyan)
    assert!(sim.world.spark_events.len() > initial_sparks);
    let last_spark = sim.world.spark_events.last().unwrap();
    assert_eq!(last_spark.color_idx, 9);
    assert_eq!(last_spark.count, 5);
}
