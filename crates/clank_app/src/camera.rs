use bevy::prelude::*;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
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
        Transform::from_xyz(640.0, 400.0, 0.0),
        MainCamera,
    ));
}

pub fn camera_viewport_sync_system(
    window_query: Query<&Window>,
    mut camera_query: Query<&mut Transform, With<MainCamera>>,
    mut sim: Option<ResMut<SimWorld>>,
) {
    let Ok(window) = window_query.single() else { return };
    let arena_w = (window.width() - 330.0).max(320.0);
    let arena_h = (window.height() - 53.0).max(250.0);

    if let Ok(mut transform) = camera_query.single_mut() {
        transform.translation.x = window.width() * 0.5;
        transform.translation.y = window.height() * 0.5;

        if let Some(ref mut sim) = sim {
            if (sim.world_width - arena_w as f64).abs() > 0.5
                || (sim.world_height - arena_h as f64).abs() > 0.5
            {
                let w = arena_w as f64;
                let h = arena_h as f64;
                sim.world_width = w;
                sim.world_height = h;
                let cols = ((arena_w / 12.0).ceil() as usize).clamp(20, 200);
                let rows = ((arena_h / 12.0).ceil() as usize).clamp(20, 200);
                sim.world.resize(w, h, cols, rows);
            }
        }
    }
}

pub fn camera_control_system(
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    mouse_motion: Option<Res<AccumulatedMouseMotion>>,
    mouse_scroll: Option<Res<AccumulatedMouseScroll>>,
    sim: Option<Res<SimWorld>>,
    mut camera_query: Query<(&mut Transform, &mut Projection), With<MainCamera>>,
) {
    let Ok((mut transform, mut projection)) = camera_query.single_mut() else {
        return;
    };

    let mut current_scale = match *projection {
        Projection::Orthographic(ref ortho) => ortho.scale,
        _ => 1.0,
    };

    // Zooming
    if let Some(scroll) = mouse_scroll {
        if scroll.delta.y != 0.0 {
            let zoom_factor = if scroll.delta.y > 0.0 { 0.9 } else { 1.1 };
            current_scale = clamp_zoom(current_scale, zoom_factor);
            if let Projection::Orthographic(ref mut ortho) = *projection {
                ortho.scale = current_scale;
            }
        }
    }

    // Panning (Right click or Middle click drag)
    if let (Some(buttons), Some(motion)) = (mouse_buttons, mouse_motion) {
        if buttons.pressed(MouseButton::Right) || buttons.pressed(MouseButton::Middle) {
            if motion.delta != Vec2::ZERO {
                let world_delta = Vec2::new(motion.delta.x, -motion.delta.y);
                transform.translation = compute_camera_pan(transform.translation, world_delta, current_scale);
            }
        }
    }

    // Creature tracking
    if let Some(sim) = sim {
        if let Some(agent) = sim.get_selected_agent() {
            transform.translation = track_target_position(
                transform.translation,
                Vec2::new(agent.x as f32, (sim.world_height - agent.y) as f32),
                0.1,
            );
        }
    }
}

pub struct ClankCameraPlugin;

impl Plugin for ClankCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_camera)
            .add_systems(Update, (camera_viewport_sync_system, camera_control_system));
    }
}
