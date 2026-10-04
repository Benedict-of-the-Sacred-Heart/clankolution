use clank_app::gpu::lbvh::{common_prefix_length, LbvhTree};
use clank_app::gpu::spatial_index::{
    compute_morton_32, get_cell_id, CellOffsetsTable, MORTON_COLS, MORTON_ROWS, TOTAL_CELLS,
};
use clank_app::gpu::types::GpuAgentState;

#[test]
fn test_cell_offsets_sentinel_initialization() {
    assert_eq!(MORTON_COLS, 9);
    assert_eq!(MORTON_ROWS, 6);
    assert_eq!(TOTAL_CELLS, 54);

    let mut table = CellOffsetsTable::new();
    table.clear();
    for i in 0..TOTAL_CELLS {
        let entry = table.get(i);
        assert_eq!(entry, [0xFFFFFFFF, 0xFFFFFFFF]);
    }

    // Coordinate mapping tests
    assert_eq!(get_cell_id([0.0, 0.0]), 0);
    assert_eq!(get_cell_id([95.0, 95.0]), 0);
    assert_eq!(get_cell_id([105.0, 0.0]), 1);
    assert_eq!(get_cell_id([900.0, 600.0]), 53); // Clamped to 8x5 = 53
}

#[test]
fn test_lbvh_degenerate_population_guard() {
    // When population is 0 or 1, builder returns early without allocating or underflowing
    let nodes_0 = LbvhTree::build(&[]);
    assert!(nodes_0.is_empty());

    let dummy_agent = GpuAgentState {
        pos_vel: [450.0, 300.0, 0.0, 0.0],
        angle_energy: [0.0, 100.0, 0.0, 0.0],
        traits: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        hidden: [0.0; 10],
        id: 42,
        meta_flags: 0,
        age_gen: 0,
        morton_code: compute_morton_32([450.0, 300.0]),
        packed_color: 0,
        visual_cache: 0,
    };

    let nodes_1 = LbvhTree::build(&[dummy_agent]);
    assert_eq!(nodes_1.len(), 1);
    assert_eq!(nodes_1[0].leaf_idx, 0);
}

#[test]
fn test_lbvh_identical_key_tie_breaking() {
    let key = compute_morton_32([450.0, 300.0]);
    let keys = vec![[key, 10], [key, 20]];

    // If morton keys are identical, LCP must break ties on agent ID using 32 + clz(id_a ^ id_b)
    let lcp = common_prefix_length(0, 1, &keys);
    assert!(lcp >= 32);
    assert_eq!(lcp, 32 + (10u32 ^ 20u32).leading_zeros() as i32);

    // Out of bounds queries return -1
    assert_eq!(common_prefix_length(0, -1, &keys), -1);
    assert_eq!(common_prefix_length(0, 2, &keys), -1);
}

#[test]
fn test_lbvh_uncapped_mouse_picking_with_dynamic_radius_shrinking() {
    // Create 3 agents:
    // Agent 0 at (100.0, 100.0) with trait 0.0 (visual_r = 2.0)
    // Agent 1 at (102.5, 100.0) with trait 0.0 (visual_r = 2.0) - halo candidate for cursor at (100.0, 100.0)
    // Agent 2 far away at (500.0, 500.0)
    let mut agents = Vec::new();
    for (i, &(x, y)) in [(100.0f32, 100.0f32), (102.5f32, 100.0f32), (500.0f32, 500.0f32)].iter().enumerate() {
        agents.push(GpuAgentState {
            pos_vel: [x, y, 0.0, 0.0],
            angle_energy: [0.0, 100.0, 0.0, 0.0],
            traits: [0.0; 8],
            hidden: [0.0; 10],
            id: (i + 1) as u32,
            meta_flags: 0,
            age_gen: 0,
            morton_code: compute_morton_32([x, y]),
            packed_color: 0,
            visual_cache: 0,
        });
    }

    let tree = LbvhTree::build(&agents);

    // Cursor directly at (100.0, 100.0)
    // Agent 0 is direct body hit (dist 0.0 <= visual_r 2.0 -> Priority 0)
    // Agent 1 is at dist 2.5 (halo -> Priority 1)
    let (best_idx, best_id) = tree.pick_agent([100.0, 100.0], &agents, 16.0);
    assert_eq!(best_idx, Some(0));
    assert_eq!(best_id, Some(1));

    // Cursor at (102.0, 100.0):
    // Dist to Agent 1 is 0.5 (Priority 0 body hit)
    // Dist to Agent 0 is 2.0 (Priority 0 body hit)
    // Tie-break: Agent 1 dist (0.5) < Agent 0 dist (2.0) -> selects Agent 1!
    let (best_idx2, best_id2) = tree.pick_agent([102.0, 100.0], &agents, 16.0);
    assert_eq!(best_idx2, Some(1));
    assert_eq!(best_id2, Some(2));
}

#[test]
fn test_lbvh_aoe_tool_bounding_box_query() {
    let mut agents = Vec::new();
    for &(x, y) in &[(100.0f32, 100.0f32), (120.0f32, 100.0f32), (150.0f32, 100.0f32), (400.0f32, 400.0f32)] {
        agents.push(GpuAgentState {
            pos_vel: [x, y, 0.0, 0.0],
            angle_energy: [0.0, 100.0, 0.0, 0.0],
            traits: [0.0; 8],
            hidden: [0.0; 10],
            id: agents.len() as u32 + 1,
            meta_flags: 0,
            age_gen: 0,
            morton_code: compute_morton_32([x, y]),
            packed_color: 0,
            visual_cache: 0,
        });
    }

    let tree = LbvhTree::build(&agents);
    // Query circle of radius 30.0 at (100.0, 100.0)
    // Should include Agent 0 (dist 0.0) and Agent 1 (dist 20.0), but not Agent 2 (dist 50.0) or Agent 3 (dist ~424)
    let hits = tree.query_aoe([100.0, 100.0], 30.0, &agents);
    assert_eq!(hits.len(), 2);
    assert!(hits.contains(&0));
    assert!(hits.contains(&1));
}
