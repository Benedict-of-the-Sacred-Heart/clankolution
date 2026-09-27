use bevy::prelude::*;
use clank_app::camera::{ClankCameraPlugin, MainCamera, clamp_zoom, compute_camera_pan, track_target_position};

#[test]
fn test_zoom_clamping() {
    // Standard zooms
    assert_eq!(clamp_zoom(1.0, 1.2), 1.2);
    assert_eq!(clamp_zoom(1.0, 0.8), 0.8);

    // Minimum zoom clamp (0.1)
    assert_eq!(clamp_zoom(0.15, 0.1), 0.1);
    assert_eq!(clamp_zoom(0.1, 0.5), 0.1);

    // Maximum zoom clamp (10.0)
    assert_eq!(clamp_zoom(9.0, 1.5), 10.0);
    assert_eq!(clamp_zoom(10.0, 2.0), 10.0);
}

#[test]
fn test_camera_pan_calculation() {
    let current_pos = Vec3::new(450.0, 300.0, 0.0);
    let mouse_delta = Vec2::new(10.0, -5.0);
    let zoom_scale = 2.0;

    let new_pos = compute_camera_pan(current_pos, mouse_delta, zoom_scale);
    // Dragging right moves camera left (or world right relative to camera)
    assert_eq!(new_pos.x, 450.0 - (10.0 * 2.0));
    assert_eq!(new_pos.y, 300.0 - (-5.0 * 2.0));
    assert_eq!(new_pos.z, 0.0);
}

#[test]
fn test_track_target_position() {
    let current_cam = Vec3::new(0.0, 0.0, 0.0);
    let target = Vec2::new(100.0, 50.0);
    let lerp_factor = 0.1;

    let interpolated = track_target_position(current_cam, target, lerp_factor);
    assert_eq!(interpolated.x, 10.0);
    assert_eq!(interpolated.y, 5.0);
    assert_eq!(interpolated.z, 0.0);
}

#[test]
fn test_camera_plugin_spawns_main_camera() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(ClankCameraPlugin);

    app.update();

    let mut query = app.world_mut().query_filtered::<&Transform, With<MainCamera>>();
    let count = query.iter(app.world()).count();
    assert_eq!(count, 1);
}

#[test]
fn test_world_pointer_conversion() {
    use clank_app::camera::screen_to_world;
    let win_pos = Vec2::new(100.0, 150.0);
    let world_pos = screen_to_world(win_pos, Vec2::new(900.0, 600.0), 1.0, Vec2::ZERO);
    assert!(world_pos.x >= 0.0 && world_pos.x <= 900.0);
    assert!(world_pos.y >= 0.0 && world_pos.y <= 600.0);
}

#[test]
fn test_compute_arena_viewport() {
    use clank_app::camera::compute_arena_viewport;
    let (vp, arena_size) = compute_arena_viewport(Vec2::new(1280.0, 800.0), 1.0);
    assert_eq!(arena_size, Vec2::new(950.0, 747.0));
    assert_eq!(vp.physical_position, UVec2::new(0, 53));
    assert_eq!(vp.physical_size, UVec2::new(950, 747));

    let (vp2, arena_size2) = compute_arena_viewport(Vec2::new(1280.0, 800.0), 2.0);
    assert_eq!(arena_size2, Vec2::new(950.0, 747.0));
    assert_eq!(vp2.physical_position, UVec2::new(0, 106));
    assert_eq!(vp2.physical_size, UVec2::new(1900, 1494));
}
