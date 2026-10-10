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
            root: 0,
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
        root: 0,
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
