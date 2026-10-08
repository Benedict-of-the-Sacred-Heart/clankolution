//! Bevy integration of the GPU creature renderer: both engines' creatures are culled and drawn
//! straight from the simulation driver's buffers, and the GPU engine's soil colormap is copied
//! texture-to-texture into the soil sprite. No per-frame readback.

use crate::camera::MainCamera;
use crate::gpu::creature_render::{
    CreatureDrawKey, CreatureLayerKind, CreatureRenderParams, CreatureRenderer, CreatureSource,
};
use crate::rendering::SoilTextureHandle;
use crate::sim::{GpuDriverResource, SimWorld, GPU_MAX_AGENTS};
use bevy::core_pipeline::core_2d::{Transparent2d, CORE_2D_DEPTH_FORMAT};
use bevy::ecs::system::lifetimeless::SRes;
use bevy::ecs::system::SystemParamItem;
use bevy::math::FloatOrd;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::CachedRenderPipelineId;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::sync_world::MainEntity;
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, Msaa};
use bevy::render::{Render, RenderApp, RenderSystems};

/// The four creature layers' main-world entities (distinct keys for their phase items).
#[derive(Resource, Clone, Copy)]
pub struct CreatureLayerEntities(pub [Entity; 4]);

/// What the renderer draws this frame, published by the main world after the simulation step.
#[derive(Resource, Clone, ExtractResource)]
pub struct CreatureFrame {
    pub source: CreatureSource,
    pub slot_count: u32,
    pub world_size: [f32; 2],
    /// The GPU engine's soil colormap and the sprite image it is copied into.
    pub soil: Option<(wgpu::Texture, AssetId<Image>)>,
    pub layers: [Entity; 4],
}

fn spawn_creature_layers(mut commands: Commands) {
    let layers = CreatureLayerKind::ALL
        .map(|kind| commands.spawn(Name::new(format!("{kind:?} layer"))).id());
    commands.insert_resource(CreatureLayerEntities(layers));
}

/// Publishes the buffers to draw. While the GPU engine holds the world (`is_initialized`) they
/// are its live buffers; otherwise the Rust engine's creatures are written into them first.
pub fn publish_creature_frame(
    mut commands: Commands,
    sim: Res<SimWorld>,
    gpu: Res<GpuDriverResource>,
    layers: Option<Res<CreatureLayerEntities>>,
    soil_handle: Option<Res<SoilTextureHandle>>,
) {
    let (Some(driver), Some(layers)) = (gpu.driver.as_ref(), layers) else {
        commands.remove_resource::<CreatureFrame>();
        return;
    };
    let world_size = [sim.world_width as f32, sim.world_height as f32];
    let (slot_count, soil) = if driver.is_initialized() {
        let soil = soil_handle.map(|h| (driver.soil_display_texture().clone(), h.0.id()));
        (driver.max_agents, soil)
    } else {
        let (states, trails) = crate::gpu::bridge::render_snapshot(&sim);
        (driver.upload_render_snapshot(&states, &trails), None)
    };
    commands.insert_resource(CreatureFrame {
        source: driver.creature_source(),
        slot_count,
        world_size,
        soil,
        layers: layers.0,
    });
}

#[derive(Resource)]
pub struct CreatureRenderState {
    renderer: CreatureRenderer,
}

/// The arena camera's draw target, which the pipelines are specialized for.
fn draw_key(view: &ExtractedView, msaa: &Msaa) -> CreatureDrawKey {
    CreatureDrawKey {
        color_format: view.target_format,
        samples: msaa.samples(),
        depth: Some(CORE_2D_DEPTH_FORMAT),
    }
}

/// Cull parameters for the arena view, matching what the main world culled with: the camera
/// center in simulation coordinates and the orthographic view size in world units.
fn render_params(view: &ExtractedView, frame: &CreatureFrame) -> CreatureRenderParams {
    let clip_from_world = view
        .clip_from_world
        .unwrap_or(view.clip_from_view * view.world_from_view.to_matrix().inverse());
    let center = view.world_from_view.translation().truncate();
    let size = Vec2::new(
        2.0 / view.clip_from_view.x_axis.x,
        2.0 / view.clip_from_view.y_axis.y,
    )
    .abs();
    CreatureRenderParams {
        clip_from_world: clip_from_world.to_cols_array_2d(),
        camera_pos: crate::rendering::bevy_to_sim_coord(center, frame.world_size[1]).to_array(),
        camera_size: size.to_array(),
        world_size: frame.world_size,
        slot_count: frame.slot_count,
        _pad: 0,
    }
}

/// Culls into the indirect args and copies the GPU soil, submitted ahead of the frame's passes.
fn prepare_creature_frame(
    frame: Option<Res<CreatureFrame>>,
    state: Option<ResMut<CreatureRenderState>>,
    views: Query<&ExtractedView, (With<MainCamera>, With<ExtractedCamera>)>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let (Some(frame), Some(mut state), Ok(view)) = (frame, state, views.single()) else {
        return;
    };
    let device = device.wgpu_device();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("creature_frame_encoder"),
    });
    let rp = render_params(view, &frame);
    state
        .renderer
        .encode_cull(device, &queue, &mut encoder, &frame.source, &rp);
    if let Some((soil, image)) = &frame.soil {
        if let Some(gpu_image) = gpu_images.get(*image) {
            if gpu_image.texture.size() == soil.size() {
                encoder.copy_texture_to_texture(
                    soil.as_image_copy(),
                    gpu_image.texture.as_image_copy(),
                    soil.size(),
                );
            }
        }
    }
    queue.submit([encoder.finish()]);
}

fn queue_creature_layers(
    frame: Option<Res<CreatureFrame>>,
    state: Option<ResMut<CreatureRenderState>>,
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa), (With<MainCamera>, With<ExtractedCamera>)>,
    device: Res<RenderDevice>,
) {
    let (Some(frame), Some(mut state)) = (frame, state) else {
        return;
    };
    let draw_function = draw_functions.read().id::<DrawCreatureLayer>();
    for (view, msaa) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        state
            .renderer
            .ensure_draw_pipelines(device.wgpu_device(), draw_key(view, msaa));
        for (kind, entity) in CreatureLayerKind::ALL.iter().zip(frame.layers) {
            phase.add_transient(Transparent2d {
                sort_key: FloatOrd(kind.sort_z()),
                entity: (Entity::PLACEHOLDER, MainEntity::from(entity)),
                pipeline: CachedRenderPipelineId::INVALID,
                draw_function,
                batch_range: 0..1,
                extracted_index: usize::MAX,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
            });
        }
    }
}

pub struct DrawCreatureLayer;

impl RenderCommand<Transparent2d> for DrawCreatureLayer {
    type Param = (SRes<CreatureRenderState>, SRes<CreatureFrame>);
    type ViewQuery = (&'static ExtractedView, &'static Msaa);
    type ItemQuery = ();

    fn render<'w>(
        item: &Transparent2d,
        (view, msaa): (&'w ExtractedView, &'w Msaa),
        _entity: Option<()>,
        (state, frame): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let main = item.main_entity().id();
        let Some(k) = frame.layers.iter().position(|e| *e == main) else {
            return RenderCommandResult::Skip;
        };
        // `wgpu_pass` resets Bevy's tracked state, so the next item rebinds what it needs.
        state.into_inner().renderer.draw_layer(
            pass.wgpu_pass(),
            draw_key(view, msaa),
            CreatureLayerKind::ALL[k],
        );
        RenderCommandResult::Success
    }
}

pub struct CreaturePassPlugin;

impl Plugin for CreaturePassPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<CreatureFrame>::default(),
            ExtractComponentPlugin::<MainCamera>::default(),
        ))
        .add_systems(Startup, spawn_creature_layers)
        .add_systems(
            Update,
            publish_creature_frame.after(crate::sim::sim_step_system),
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_render_command::<Transparent2d, DrawCreatureLayer>()
            .add_systems(
                Render,
                (
                    queue_creature_layers.in_set(RenderSystems::Queue),
                    prepare_creature_frame.in_set(RenderSystems::Prepare),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let Some(device) = render_app.world().get_resource::<RenderDevice>() else {
            return;
        };
        let renderer = CreatureRenderer::new(device.wgpu_device(), GPU_MAX_AGENTS);
        render_app.insert_resource(CreatureRenderState { renderer });
    }
}
