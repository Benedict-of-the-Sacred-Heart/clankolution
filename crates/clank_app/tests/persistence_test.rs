use clank_app::sim::SimWorld;
use clank_app::persistence::{save_clank_snapshot, load_clank_snapshot, save_json_world, load_json_world};

#[test]
fn test_clank_binary_persistence_roundtrip() {
    let mut sim = SimWorld::new(42);
    sim.step(150);

    let tick_saved = sim.world.tick;
    let kills_saved = sim.world.kills;
    let births_saved = sim.world.births;
    let agents_count = sim.world.agents.len();
    let first_agent = sim.world.agents[0];

    // Save binary snapshot bytes
    let bytes = save_clank_snapshot(&sim).expect("Failed to serialize clank snapshot");
    assert!(!bytes.is_empty());

    // Advance simulation 300 cycles so state diverges completely
    sim.step(300);
    assert!(sim.world.tick > tick_saved);

    // Restore from binary snapshot
    load_clank_snapshot(&mut sim, &bytes).expect("Failed to deserialize clank snapshot");

    assert_eq!(sim.world.tick, tick_saved);
    assert_eq!(sim.world.kills, kills_saved);
    assert_eq!(sim.world.births, births_saved);
    assert_eq!(sim.world.agents.len(), agents_count);
    assert_eq!(sim.world.agents[0].id, first_agent.id);
    assert!((sim.world.agents[0].x - first_agent.x).abs() < 1e-6);
    assert!((sim.world.agents[0].y - first_agent.y).abs() < 1e-6);
    assert_eq!(sim.world.agents[0].genes, first_agent.genes);

    // Ensure simulation continues running cleanly post-restore
    sim.step(50);
    assert_eq!(sim.world.tick, tick_saved + 50);
}

#[test]
fn test_json_persistence_roundtrip() {
    let mut sim = SimWorld::new(777);
    sim.step(80);

    let tick_saved = sim.world.tick;
    let agents_count = sim.world.agents.len();
    let first_id = sim.world.agents[0].id;

    // Export to JSON string
    let json_str = save_json_world(&sim).expect("Failed to serialize world to JSON");
    assert!(json_str.contains("\"version\":1"));

    // Advance state
    sim.step(120);
    assert!(sim.world.tick > tick_saved);

    // Import from JSON string
    load_json_world(&mut sim, &json_str).expect("Failed to deserialize world from JSON");

    assert_eq!(sim.world.tick, tick_saved);
    assert_eq!(sim.world.agents.len(), agents_count);
    assert_eq!(sim.world.agents[0].id, first_id);

    // Continue stepping
    sim.step(30);
    assert_eq!(sim.world.tick, tick_saved + 30);
}

#[test]
fn test_file_save_and_load_roundtrip() {
    use clank_app::persistence::{save_clank_file, load_clank_file, export_json_file, import_json_file};

    let mut sim = SimWorld::new(1234);
    sim.step(60);
    let original_tick = sim.world.tick;

    let clank_path = "/tmp/test_clank_roundtrip.clank";
    let json_path = "/tmp/test_json_roundtrip.json";

    // Test .clank file save and load
    let bytes_written = save_clank_file(&sim, clank_path).expect("Failed to write clank file");
    assert!(bytes_written > 1000);

    sim.step(100);
    assert_ne!(sim.world.tick, original_tick);

    load_clank_file(&mut sim, clank_path).expect("Failed to load clank file");
    assert_eq!(sim.world.tick, original_tick);

    // Test .json file export and import
    let json_written = export_json_file(&sim, json_path).expect("Failed to write json file");
    assert!(json_written > 1000);

    sim.step(100);
    assert_ne!(sim.world.tick, original_tick);

    import_json_file(&mut sim, json_path).expect("Failed to import json file");
    assert_eq!(sim.world.tick, original_tick);

    // Cleanup
    let _ = std::fs::remove_file(clank_path);
    let _ = std::fs::remove_file(json_path);
}

