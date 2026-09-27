use bevy::prelude::*;
use bevy::window::PresentMode;
use clank_app::sim::ClankSimPlugin;
use clank_app::camera::ClankCameraPlugin;
use clank_app::rendering::ClankRenderPlugin;
use clank_app::ui::ClankUiPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Clankolution 2.0 — Native Desktop (Bevy 0.19)".into(),
                    resolution: (1280u32, 800u32).into(),
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
        )
        .insert_resource(ClearColor(Color::srgb(0.04, 0.09, 0.10)))
        .add_plugins((
            ClankSimPlugin,
            ClankCameraPlugin,
            ClankRenderPlugin,
            ClankUiPlugin,
        ))
        .run();
}
