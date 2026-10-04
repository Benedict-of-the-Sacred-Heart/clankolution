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

#[test]
fn test_spatial_query_single_agent_non_zero_slot() {
    let mut agents = vec![GpuAgentState {
        pos_vel: [0.0; 4],
        angle_energy: [0.0; 4],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 0,
        meta_flags: 1 << 13, // dead tombstone
        age_gen: 0,
        morton_code: 0,
        packed_color: 0,
        visual_cache: 0,
    }; 100];

    // Single living agent at slot 42
    agents[42] = GpuAgentState {
        pos_vel: [200.0, 200.0, 0.0, 0.0],
        angle_energy: [0.0, 100.0, 0.0, 0.0],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 999,
        meta_flags: 0, // alive
        age_gen: 0,
        morton_code: compute_morton_32([200.0, 200.0]),
        packed_color: 0,
        visual_cache: 0,
    };

    let keys = vec![[agents[42].morton_code, 42u32]];
    let tree = LbvhTree::build_from_keys(&keys, &agents);
    assert_eq!(tree.nodes.len(), 1);
    assert_eq!(tree.nodes[0].leaf_idx, 42);

    let (picked_slot, picked_id) = tree.pick_agent([200.0, 200.0], &agents, 16.0);
    assert_eq!(picked_slot, Some(42));
    assert_eq!(picked_id, Some(999));
}

#[test]
fn test_cell_offsets_population_from_sorted_keys() {
    let mut agents = Vec::new();
    let mut keys = Vec::new();

    // 2 agents in cell 0 ([50.0, 50.0])
    for i in 0..2 {
        agents.push(GpuAgentState {
            pos_vel: [50.0, 50.0, 0.0, 0.0],
            angle_energy: [0.0; 4],
            traits: [0.0; 8],
            hidden: [0.0; 10],
            id: i as u32,
            meta_flags: 0,
            age_gen: 0,
            morton_code: compute_morton_32([50.0, 50.0]),
            packed_color: 0,
            visual_cache: 0,
        });
        keys.push([agents[i].morton_code, i as u32]);
    }

    // 1 agent in cell 1 ([150.0, 50.0])
    agents.push(GpuAgentState {
        pos_vel: [150.0, 50.0, 0.0, 0.0],
        angle_energy: [0.0; 4],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 2,
        meta_flags: 0,
        age_gen: 0,
        morton_code: compute_morton_32([150.0, 50.0]),
        packed_color: 0,
        visual_cache: 0,
    });
    keys.push([agents[2].morton_code, 2u32]);

    let mut table = CellOffsetsTable::new();
    table.populate_from_keys(&keys, &agents);

    // Cell 0 has indices [0, 2]
    assert_eq!(table.get(0), [0, 2]);
    // Cell 1 has indices [2, 3]
    assert_eq!(table.get(1), [2, 3]);
    // Cell 2 has no agents -> sentinel
    assert_eq!(table.get(2), [0xFFFFFFFF, 0xFFFFFFFF]);
}

#[test]
fn test_lbvh_phase2_aabb_fitting_correctness() {
    let mut agents = Vec::new();
    let coords = [
        [50.0f32, 50.0f32],
        [60.0f32, 70.0f32],
        [200.0f32, 300.0f32],
        [800.0f32, 500.0f32],
    ];
    for (i, &pos) in coords.iter().enumerate() {
        agents.push(GpuAgentState {
            pos_vel: [pos[0], pos[1], 0.0, 0.0],
            angle_energy: [0.0; 4],
            traits: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            hidden: [0.0; 10],
            id: i as u32,
            meta_flags: (i as u32) % 4,
            age_gen: 0,
            morton_code: compute_morton_32(pos),
            packed_color: 0,
            visual_cache: 0,
        });
    }

    let tree = LbvhTree::build(&agents);
    assert_eq!(tree.nodes.len(), 7); // 3 internal + 4 leaves

    // Internal nodes must have valid AABB bounding boxes enclosing children
    for i in 0..3 {
        let node = &tree.nodes[i];
        assert!(node.aabb_min[0] <= node.aabb_max[0]);
        assert!(node.aabb_min[1] <= node.aabb_max[1]);
        assert!(node.count >= 2);

        let left = &tree.nodes[node.left_child as usize];
        let right = &tree.nodes[node.right_child as usize];

        assert!(node.aabb_min[0] <= left.aabb_min[0]);
        assert!(node.aabb_min[1] <= left.aabb_min[1]);
        assert!(node.aabb_max[0] >= left.aabb_max[0]);
        assert!(node.aabb_max[1] >= left.aabb_max[1]);

        assert!(node.aabb_min[0] <= right.aabb_min[0]);
        assert!(node.aabb_min[1] <= right.aabb_min[1]);
        assert!(node.aabb_max[0] >= right.aabb_max[0]);
        assert!(node.aabb_max[1] >= right.aabb_max[1]);
    }
}

#[test]
fn test_lbvh_minimap_lod_clusters() {
    let mut agents = Vec::new();
    for i in 0..16 {
        agents.push(GpuAgentState {
            pos_vel: [(i as f32) * 50.0, 100.0, 0.0, 0.0],
            angle_energy: [0.0; 4],
            traits: [0.0; 8],
            hidden: [0.0; 10],
            id: i as u32,
            meta_flags: i as u32 % 4,
            age_gen: 0,
            morton_code: compute_morton_32([(i as f32) * 50.0, 100.0]),
            packed_color: 0,
            visual_cache: 0,
        });
    }

    let tree = LbvhTree::build(&agents);
    let clusters = tree.extract_minimap_clusters(2);
    assert!(!clusters.is_empty());
    let total_counted: u32 = clusters.iter().map(|c| c.count).sum();
    assert_eq!(total_counted, 16);
}

#[test]
fn test_lbvh_camera_frustum_culling() {
    let mut agents = Vec::new();
    // Agent 0 inside viewport [100..300, 100..300]
    agents.push(GpuAgentState {
        pos_vel: [200.0, 200.0, 0.0, 0.0],
        angle_energy: [0.0; 4],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 10,
        meta_flags: 0,
        age_gen: 0,
        morton_code: compute_morton_32([200.0, 200.0]),
        packed_color: 0,
        visual_cache: 1,
    });
    // Agent 1 outside viewport [700..800, 500..600]
    agents.push(GpuAgentState {
        pos_vel: [750.0, 550.0, 0.0, 0.0],
        angle_energy: [0.0; 4],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 20,
        meta_flags: 0,
        age_gen: 0,
        morton_code: compute_morton_32([750.0, 550.0]),
        packed_color: 0,
        visual_cache: 1,
    });

    let tree = LbvhTree::build(&agents);
    let visible = tree.cull_frustum([100.0, 100.0], [300.0, 300.0], &agents);
    assert_eq!(visible, vec![0]);
}



