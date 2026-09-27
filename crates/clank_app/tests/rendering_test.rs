use bevy::prelude::*;
use clank_app::rendering::{
    sim_to_bevy_coord, bevy_to_sim_coord, lineage_color, agent_body_color,
    agent_triangle_vertices, extract_agent_render_data, generate_soil_rgba,
};
use clank_app::sim::SimWorld;

#[test]
fn test_coordinate_mapping() {
    let world_height = 600.0;

    // Top-left
    let tl_sim = Vec2::new(0.0, 0.0);
    let tl_bevy = sim_to_bevy_coord(tl_sim, world_height);
    assert_eq!(tl_bevy, Vec2::new(0.0, 600.0));
    assert_eq!(bevy_to_sim_coord(tl_bevy, world_height), tl_sim);

    // Center
    let center_sim = Vec2::new(450.0, 300.0);
    let center_bevy = sim_to_bevy_coord(center_sim, world_height);
    assert_eq!(center_bevy, Vec2::new(450.0, 300.0));
    assert_eq!(bevy_to_sim_coord(center_bevy, world_height), center_sim);

    // Bottom-right
    let br_sim = Vec2::new(900.0, 600.0);
    let br_bevy = sim_to_bevy_coord(br_sim, world_height);
    assert_eq!(br_bevy, Vec2::new(900.0, 0.0));
    assert_eq!(bevy_to_sim_coord(br_bevy, world_height), br_sim);
}

#[test]
fn test_lineage_colors() {
    // 8 distinct lineage colors
    let c0 = lineage_color(0);
    let c1 = lineage_color(1);
    let c7 = lineage_color(7);
    let c8 = lineage_color(8); // should wrap to 0

    assert_eq!(c0, c8);
    assert_ne!(c0, c1);
    assert_ne!(c1, c7);

    // Alpha modulation by energy
    let full_energy = agent_body_color(0, 100.0);
    let low_energy = agent_body_color(0, 0.0);
    assert!(full_energy.to_srgba().alpha > low_energy.to_srgba().alpha);
    assert!(low_energy.to_srgba().alpha >= 0.55);
}

#[test]
fn test_agent_triangle_vertices() {
    let pos = Vec2::new(100.0, 100.0);
    let angle = 0.0; // facing +X
    let bulk = 0.5;

    let (v0, v1, v2) = agent_triangle_vertices(pos, angle, bulk);
    // Tip should point forward (+X)
    assert!(v0.x > pos.x);
    // Wings should be behind the tip
    assert!(v1.x < v0.x);
    assert!(v2.x < v0.x);
    // Wings should be symmetric about Y
    assert!((v1.y - pos.y).abs() - (pos.y - v2.y).abs() < 1e-4);
}

#[test]
fn test_extract_agent_render_data() {
    let sim = SimWorld::new(42);
    let render_items = extract_agent_render_data(&sim);

    // Sim starts with 72 creatures
    assert_eq!(render_items.len(), 72);

    for item in render_items {
        assert!(item.bevy_pos.x >= 0.0 && item.bevy_pos.x <= 900.0);
        assert!(item.bevy_pos.y >= 0.0 && item.bevy_pos.y <= 600.0);
        assert!(item.radius > 2.0 && item.radius < 10.0);
    }
}

#[test]
fn test_generate_soil_rgba() {
    let mut sim = SimWorld::new(42);
    // Seed some food in soil
    sim.world.soil.food[0] = 5.0;
    sim.world.soil.taint[1] = 0.8;

    let cols = sim.world.soil.cols;
    let rows = sim.world.soil.rows;
    let mut buf = vec![0u8; cols * rows * 4];

    generate_soil_rgba(&sim.world.soil, &mut buf);

    // Pixel 0 (food) should have high G/B channels
    let r0 = buf[0];
    let g0 = buf[1];
    let a0 = buf[3];
    assert_eq!(a0, 255);
    assert!(g0 > 20);

    // Pixel 1 (taint) should have elevated R channel
    let r1 = buf[4];
    let a1 = buf[7];
    assert_eq!(a1, 255);
    assert!(r1 > r0);
}

#[test]
fn test_dart_body_morphology() {
    use clank_app::rendering::compute_dart_polygon;
    let agent = clank_core::agent::AgentData::default();
    let vertices = compute_dart_polygon(&agent);
    assert_eq!(vertices.len(), 4);
    // Nose must point forward (+X)
    assert!(vertices[0].x > 0.0);
    // Wing tips must have opposite Y values
    assert!((vertices[1].y + vertices[3].y).abs() < 1e-4);
    // Rear notch must be indented
    assert!(vertices[2].x < 0.0);
}
