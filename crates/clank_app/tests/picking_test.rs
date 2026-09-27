use bevy::prelude::*;
use clank_app::sim::SimWorld;
use clank_app::rendering::find_agent_at_position;

#[test]
fn test_find_agent_at_position() {
    let mut sim = SimWorld::new(42);
    // Move agent 0 to a known position
    sim.world.agents[0].x = 100.0;
    sim.world.agents[0].y = 100.0;
    sim.world.agents[0].dead = 0;
    let target_id = sim.world.agents[0].id;

    // Direct hit
    let hit = find_agent_at_position(&sim, Vec2::new(100.0, 100.0), 5.0);
    assert_eq!(hit, Some(target_id));

    // Near hit within radius + margin
    let near_hit = find_agent_at_position(&sim, Vec2::new(103.0, 102.0), 5.0);
    assert_eq!(near_hit, Some(target_id));

    // Miss far away
    let miss = find_agent_at_position(&sim, Vec2::new(500.0, 500.0), 5.0);
    // Might hit another agent if one happens to be at 500,500, but certainly not target_id
    assert_ne!(miss, Some(target_id));

    // Dead agent should not be hit
    sim.world.agents[0].dead = 1;
    let dead_hit = find_agent_at_position(&sim, Vec2::new(100.0, 100.0), 5.0);
    assert_ne!(dead_hit, Some(target_id));
}
