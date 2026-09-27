use clank_app::api::*;

#[test]
fn test_metrics_computation() {
    let mut metrics = LiveMetrics::default();
    metrics.update(144.0, 0.00694, 500, 120, 10.5);
    assert_eq!(metrics.fps, 144.0);
    assert!((metrics.frame_time_ms - 6.94).abs() < 0.1);
}

#[test]
fn test_metrics_seconds_and_ms_normalization() {
    let mut metrics = LiveMetrics::default();
    metrics.update(60.0, 16.66, 100, 50, 5.0);
    assert_eq!(metrics.fps, 60.0);
    assert!((metrics.frame_time_ms - 16.66).abs() < 0.1);
}
