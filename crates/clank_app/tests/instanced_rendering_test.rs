use clank_app::gpu::types::GpuAgentState;
use clank_app::rendering::{
    generate_dart_mesh_from_gpu_states, generate_outline_mesh_from_gpu_states, unpack_visual_cache,
};

#[test]
fn test_unpack_visual_cache_zero_cost_dead_check() {
    let packed_color = 0xFF112233;
    // Dead agent: visual_cache == 0 must return None immediately
    let result = unpack_visual_cache(0, packed_color);
    assert!(result.is_none());
}

#[test]
fn test_unpack_visual_cache_attributes() {
    let packed_color = 0xFF556677; // rgba
    let r_u8 = 128u32; // ~50% bulk -> r = 2.3 + 0.502 * 4.5 = ~4.56
    let glow_u8 = 200u32;
    let e_u8 = 255u32; // max energy -> alpha = 1.0
    let flags = (1u32 << 24) | (1u32 << 25); // attack + birth
    let visual_cache = r_u8 | (glow_u8 << 8) | (e_u8 << 16) | flags;

    let (radius, color, is_attacking, has_birth) = unpack_visual_cache(visual_cache, packed_color).expect("Must unpack living agent");
    assert!((radius - 4.56).abs() < 0.1);
    assert!(is_attacking);
    assert!(has_birth);
    assert!(color.to_srgba().alpha > 0.9);
}

#[test]
fn test_generate_dart_mesh_from_gpu_states_skips_dead() {
    let living_agent = GpuAgentState {
        pos_vel: [100.0, 200.0, 0.0, 0.0],
        angle_energy: [0.0, 80.0, 0.0, 0.0],
        traits: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        hidden: [0.0; 10],
        id: 1,
        meta_flags: 1,
        age_gen: 0,
        morton_code: 0,
        packed_color: 0xFF00FF00,
        visual_cache: 128 | (100 << 8) | (200 << 16),
    };

    let dead_agent = GpuAgentState {
        pos_vel: [300.0, 400.0, 0.0, 0.0],
        angle_energy: [0.0; 4],
        traits: [0.0; 8],
        hidden: [0.0; 10],
        id: 2,
        meta_flags: 1 << 13,
        age_gen: 0,
        morton_code: 0,
        packed_color: 0,
        visual_cache: 0, // Dead
    };

    let states = vec![living_agent, dead_agent];
    let mut positions = Vec::new();
    let mut colors = Vec::new();

    generate_dart_mesh_from_gpu_states(&states, 747.0, &mut positions, &mut colors);

    // Living agent generates 6 body vertices (2 triangles), dead agent is skipped (0 vertices)
    assert_eq!(positions.len(), 6);
    assert_eq!(colors.len(), 6);

    let mut outline_pos = Vec::new();
    let mut outline_col = Vec::new();
    generate_outline_mesh_from_gpu_states(&states, 747.0, &mut outline_pos, &mut outline_col);

    // Living agent generates 8 line-list vertices (4 edges), dead agent is skipped
    assert_eq!(outline_pos.len(), 8);
    assert_eq!(outline_col.len(), 8);
}
