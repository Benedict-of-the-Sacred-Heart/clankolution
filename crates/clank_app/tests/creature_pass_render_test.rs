//! End-to-end check of the Bevy creature pass: the real sim and render plugins draw into an
//! offscreen image (no window), and the frame is read back with Bevy's screenshot API.

use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use clank_app::camera::MainCamera;
use clank_app::rendering::ClankRenderPlugin;
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};
use std::sync::{Arc, Mutex};

const W: u32 = 900;
const H: u32 = 600;

#[derive(Resource)]
struct Target(Handle<Image>);

fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..default()
            })
            .disable::<WinitPlugin>()
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins((ClankSimPlugin, ClankRenderPlugin));
    app.finish();
    app.cleanup();

    let mut image = Image::new_target_texture(W, H, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let handle = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    // The arena camera, framing the whole 900×600 world as the app does at startup.
    app.world_mut().spawn((
        Camera2d,
        RenderTarget::Image(handle.clone().into()),
        Transform::from_xyz(W as f32 * 0.5, H as f32 * 0.5, 0.0),
        MainCamera,
    ));
    app.insert_resource(Target(handle));
    app
}

fn capture(app: &mut App) -> Vec<u8> {
    let slot: Arc<Mutex<Option<Image>>> = Arc::default();
    let sink = slot.clone();
    let handle = app.world().resource::<Target>().0.clone();
    app.world_mut().spawn(Screenshot::image(handle)).observe(
        move |shot: On<ScreenshotCaptured>| {
            *sink.lock().unwrap() = Some(shot.image.clone());
        },
    );
    for _ in 0..20 {
        app.update();
        if let Some(image) = slot.lock().unwrap().take() {
            let data = image.data.expect("screenshot pixels");
            assert_eq!(data.len(), (W * H * 4) as usize);
            return data;
        }
    }
    panic!("screenshot never arrived");
}

fn differs(a: &[u8], b: &[u8], i: usize) -> bool {
    (0..3).any(|c| (a[i * 4 + c] as i32 - b[i * 4 + c] as i32).abs() > 8)
}

/// Paused Rust-engine world with seed 42, run long enough for full nine-point trails.
fn paused_rust_world() -> Option<App> {
    let mut app = headless_app();
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return None;
    }
    for _ in 0..30 {
        app.update();
    }
    let sim = app.world().resource::<SimWorld>();
    assert!(sim.world.agents.iter().any(|a| a.trail_count == 9));
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    for _ in 0..3 {
        app.update();
    }
    Some(app)
}

/// Creatures drawn by the GPU creature pass land where the simulation puts them: every pixel
/// they change is within reach of a creature (glow, trail and halo included).
#[test]
fn test_rust_creatures_are_drawn_at_their_positions() {
    let Some(mut app) = paused_rust_world() else {
        return;
    };
    let with = capture(&mut app);
    let positions: Vec<(f32, f32)> = {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        let p = sim
            .world
            .agents
            .iter()
            .filter(|a| a.dead == 0)
            .map(|a| (a.x as f32, a.y as f32))
            .collect();
        sim.world.agents.clear();
        p
    };
    let without = capture(&mut app);

    let near_creature = |px: f32, py: f32| {
        positions.iter().any(|&(x, y)| {
            let dx = (px - x).abs().min(W as f32 - (px - x).abs());
            let dy = (py - y).abs().min(H as f32 - (py - y).abs());
            dx * dx + dy * dy <= 40.0 * 40.0
        })
    };
    let (mut changed, mut stray) = (0, 0);
    for i in 0..(W * H) as usize {
        if differs(&with, &without, i) {
            changed += 1;
            // Image rows run top-down, as simulation y does.
            let (px, py) = ((i as u32 % W) as f32 + 0.5, (i as u32 / W) as f32 + 0.5);
            if !near_creature(px, py) {
                stray += 1;
            }
        }
    }
    assert!(
        changed > positions.len() * 20,
        "creatures drawn: {changed} pixels changed for {} creatures",
        positions.len()
    );
    assert!(
        stray * 100 <= changed,
        "{stray} of {changed} changed pixels are nowhere near a creature"
    );
}

/// The same paused world, handed to the GPU engine, looks the same: creatures, trails and soil
/// are presented from the GPU buffers exactly as from the Rust snapshot.
#[test]
fn test_gpu_engine_presents_the_same_world() {
    let Some(mut app) = paused_rust_world() else {
        return;
    };
    let rust_frame = capture(&mut app);
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Gpu;
        sim.gpu_resync_pending = true; // uploads now, while paused
    }
    app.update();
    assert!(app
        .world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap()
        .is_initialized());
    let gpu_frame = capture(&mut app);

    let n = (W * H) as usize;
    let diff = (0..n)
        .filter(|&i| differs(&rust_frame, &gpu_frame, i))
        .count();
    // Identical on the reference machine; dropping the trails on the switch changes ~300.
    assert!(
        diff * 10_000 <= n,
        "GPU-engine frame differs from the Rust-engine frame on {diff} of {n} pixels"
    );
}
