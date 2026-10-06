use bevy::prelude::*;
use crate::sim::SimWorld;

#[derive(Component)]
pub struct MainCamera;

#[inline]
pub fn clamp_zoom(current: f32, factor: f32) -> f32 {
    (current * factor).clamp(0.1, 10.0)
}

#[inline]
pub fn compute_camera_pan(current: Vec3, mouse_delta: Vec2, zoom_scale: f32) -> Vec3 {
    Vec3::new(
        current.x - mouse_delta.x * zoom_scale,
        current.y - mouse_delta.y * zoom_scale,
        current.z,
    )
}

#[inline]
pub fn screen_to_world(win_pos: Vec2, world_size: Vec2, _zoom_scale: f32, _cam_pos: Vec2) -> Vec2 {
    Vec2::new(
        win_pos.x.clamp(0.0, world_size.x),
        win_pos.y.clamp(0.0, world_size.y),
    )
}

#[inline]
pub fn track_target_position(current: Vec3, target: Vec2, lerp_factor: f32) -> Vec3 {
    Vec3::new(
        current.x + (target.x - current.x) * lerp_factor,
        current.y + (target.y - current.y) * lerp_factor,
        current.z,
    )
}

use bevy::camera::Viewport;

#[inline]
pub fn compute_arena_viewport(window_size: Vec2, scale_factor: f32) -> (Viewport, Vec2) {
    let top_bar_h = 53.0;
    let sidebar_w = 330.0;
    let arena_w = (window_size.x - sidebar_w).max(320.0);
    let arena_h = (window_size.y - top_bar_h).max(250.0);
    let arena_size = Vec2::new(arena_w, arena_h);
    let vp = Viewport {
        physical_position: UVec2::new(0, (top_bar_h * scale_factor).round() as u32),
        physical_size: UVec2::new(
            (arena_size.x * scale_factor).round() as u32,
            (arena_size.y * scale_factor).round() as u32,
        ),
        depth: 0.0..1.0,
    };
    (vp, arena_size)
}

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Transform::from_xyz(450.0, 300.0, 0.0),
        MainCamera,
    ));
}

pub fn camera_viewport_sync_system(
    window_query: Query<&Window>,
    mut camera_query: Query<(&mut Camera, &mut Transform, &mut Projection), With<MainCamera>>,
    mut sim: Option<ResMut<SimWorld>>,
) {
    let Ok(window) = window_query.single() else { return };
    let (vp, arena_size) = compute_arena_viewport(
        Vec2::new(window.width(), window.height()),
        window.scale_factor(),
    );

    if let Ok((mut camera, mut transform, mut projection)) = camera_query.single_mut() {
        camera.viewport = Some(vp);
        transform.translation.x = 450.0;
        transform.translation.y = 300.0;

        if let Projection::Orthographic(ref mut ortho) = *projection {
            let scale_x = 900.0 / arena_size.x;
            let scale_y = 600.0 / arena_size.y;
            ortho.scale = scale_x.max(scale_y);
        }

        if let Some(ref mut sim) = sim {
            sim.world_width = 900.0;
            sim.world_height = 600.0;
        }
    }
}

pub fn camera_control_system(
    mut camera_query: Query<&mut Projection, With<MainCamera>>,
) {
    let Ok(mut projection) = camera_query.single_mut() else {
        return;
    };

    if let Projection::Orthographic(ref mut ortho) = *projection {
        ortho.scale = 1.0;
    }
}

pub struct ClankCameraPlugin;

impl Plugin for ClankCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_camera)
            .add_systems(Update, (camera_viewport_sync_system, camera_control_system));
    }
}
