use clank_app::gpu::soil_pipeline::{
    GpuSoilPipeline, SoilParams, CELL_COUNT, GRID_HEIGHT, GRID_WIDTH,
};

#[test]
fn test_soil_atomic_buffer_and_texture_bounds() {
    // 1. GpuSoilCell fixed-point conversions (micro-units, SOIL_FIXED_SCALE)
    let initial_cell = GpuSoilPipeline::from_physics(1.234, 0.567, 0.891);
    assert_eq!(initial_cell.food_micro, 1_234_000);
    assert_eq!(initial_cell.taint_micro, 567_000);
    assert_eq!(initial_cell.scent_micro, 891_000);
    assert_eq!(initial_cell.fertility_micro, 1_000_000);

    let (f, t, s) = GpuSoilPipeline::to_physics(&initial_cell);
    assert!((f - 1.234).abs() < 1e-4);
    assert!((t - 0.567).abs() < 1e-4);
    assert!((s - 0.891).abs() < 1e-4);

    // 2. Renewal formula: f += renewal * bloom * (1 - f / 1.7) clamped to [0.0, 2.5]
    // 3. Taint decay: t * 0.994 - 0.0001, scent decay: s * 0.954
    let mut cells = vec![GpuSoilPipeline::from_physics(1.0, 0.5, 0.5); CELL_COUNT];
    let bloom_table = vec![1.0f32; CELL_COUNT];
    let renewal = 0.05f32;

    GpuSoilPipeline::step_soil(&mut cells, &bloom_table, renewal);

    let (f_new, t_new, s_new) = GpuSoilPipeline::to_physics(&cells[0]);
    let expected_f = (1.0f32 + renewal * 1.0 * (1.0 - 1.0 / 1.7)).clamp(0.0, 2.5);
    let expected_t = (0.5f32 * 0.994 - 0.0001).max(0.0);
    let expected_s = 0.5f32 * 0.954;
    assert!((f_new - expected_f).abs() < 1e-3);
    assert!((t_new - expected_t).abs() < 1e-3);
    assert!((s_new - expected_s).abs() < 1e-3);

    // 4. Output texture: soil_display (colormap) is 75x50
    assert_eq!(GRID_WIDTH, 75);
    assert_eq!(GRID_HEIGHT, 50);
    assert_eq!(CELL_COUNT, 3750);
    assert_eq!(std::mem::size_of::<SoilParams>(), 16);

    let col_sample = GpuSoilPipeline::evaluate_soil_color(1.5, 0.2, 0.1);
    assert!(col_sample[0] >= 0.0 && col_sample[0] <= 1.0);
    assert!(col_sample[1] >= 0.0 && col_sample[1] <= 1.0);
    assert!(col_sample[2] >= 0.0 && col_sample[2] <= 1.0);
    assert_eq!(col_sample[3], 1.0);

    // 5. Boundary coordinate clamping: min(pos.x / 12.0, 74) and min(pos.y / 12.0, 49) prevents OOB write at seam (900.0, 600.0)
    let (cx_seam, cy_seam) = GpuSoilPipeline::pos_to_cell([900.0, 600.0]);
    assert_eq!(cx_seam, 74);
    assert_eq!(cy_seam, 49);
    let idx_seam = GpuSoilPipeline::cell_index(cx_seam, cy_seam);
    assert_eq!(idx_seam, 3749);
    assert!(idx_seam < CELL_COUNT);

    let (cx_zero, cy_zero) = GpuSoilPipeline::pos_to_cell([0.0, 0.0]);
    assert_eq!(cx_zero, 0);
    assert_eq!(cy_zero, 0);
    assert_eq!(GpuSoilPipeline::cell_index(cx_zero, cy_zero), 0);
}
