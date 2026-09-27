use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::mesh::PrimitiveTopology;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::image::ImageSampler;
use clank_core::agent::AgentData;
use clank_core::soil::SoilGrid;
use crate::sim::SimWorld;
use crate::theme::PALETTE;

#[derive(Resource)]
pub struct SoilTextureHandle(pub Handle<Image>);

#[derive(Resource)]
pub struct VignetteTextureHandle(pub Handle<Image>);

#[derive(Resource)]
pub struct AgentMeshResource {
    pub mesh_handle: Handle<Mesh>,
}

#[derive(Component)]
pub struct SoilSprite;

#[derive(Component)]
pub struct VignetteSprite;

#[inline]
pub fn sim_to_bevy_coord(sim_pos: Vec2, world_height: f32) -> Vec2 {
    Vec2::new(sim_pos.x, world_height - sim_pos.y)
}

#[inline]
pub fn bevy_to_sim_coord(bevy_pos: Vec2, world_height: f32) -> Vec2 {
    Vec2::new(bevy_pos.x, world_height - bevy_pos.y)
}

#[inline]
pub fn lineage_color(root: u32) -> Color {
    let c = PALETTE[(root as usize) % PALETTE.len()];
    Color::srgb_u8(c.r(), c.g(), c.b())
}

#[inline]
pub fn agent_body_color(root: u32, energy: f64) -> Color {
    let base = lineage_color(root).to_srgba();
    let alpha = (0.55 + (energy / 160.0)).clamp(0.55, 1.0) as f32;
    Color::srgba(base.red, base.green, base.blue, alpha)
}

#[inline]
pub fn compute_dart_polygon(a: &AgentData) -> [Vec2; 4] {
    let r = (2.3 + a.tr[0] * 4.5) as f32;
    let nose = Vec2::new(r * 1.5, 0.0);
    let right_wing = Vec2::new(-r * 0.75, r * (0.5 + a.tr[3] as f32 * 0.45));
    let rear_notch = Vec2::new(-r * (0.45 + a.tr[5] as f32), 0.0);
    let left_wing = Vec2::new(-r * 0.75, -r * (0.5 + a.tr[3] as f32 * 0.45));
    [nose, right_wing, rear_notch, left_wing]
}

#[inline]
pub fn agent_triangle_vertices(pos: Vec2, angle: f32, bulk_trait: f32) -> (Vec2, Vec2, Vec2) {
    let r = 2.3 + bulk_trait * 4.5;
    let rot = Mat2::from_angle(-angle);
    let tip = pos + rot * Vec2::new(r * 1.5, 0.0);
    let wing1 = pos + rot * Vec2::new(-r * 0.75, r * 0.75);
    let wing2 = pos + rot * Vec2::new(-r * 0.75, -r * 0.75);
    (tip, wing1, wing2)
}

pub struct AgentRenderItem {
    pub id: u32,
    pub sim_pos: Vec2,
    pub bevy_pos: Vec2,
    pub angle: f32,
    pub radius: f32,
    pub color: Color,
    pub is_selected: bool,
    pub attack: f32,
    pub birth: u32,
    pub signal: f32,
    pub sight: f32,
    pub bulk: f32,
    pub armor: f32,
    pub carnivory: f32,
    pub trail: Vec<Vec2>,
}

pub fn extract_agent_render_data(sim: &SimWorld) -> Vec<AgentRenderItem> {
    let h = sim.world_height as f32;
    let selected_id = sim.selected_agent_id;

    sim.world
        .agents
        .iter()
        .filter(|a| a.dead == 0)
        .map(|a| {
            let sim_pos = Vec2::new(a.x as f32, a.y as f32);
            let bevy_pos = sim_to_bevy_coord(sim_pos, h);
            let radius = (2.3 + a.tr[0] * 4.5) as f32;
            let color = agent_body_color(a.root, a.energy);
            let is_selected = selected_id == Some(a.id);

            let mut trail = Vec::with_capacity(a.trail_count as usize);
            for i in 0..(a.trail_count as usize).min(9) {
                let tp = Vec2::new(a.trail_x[i] as f32, a.trail_y[i] as f32);
                trail.push(sim_to_bevy_coord(tp, h));
            }

            AgentRenderItem {
                id: a.id,
                sim_pos,
                bevy_pos,
                angle: a.angle as f32,
                radius,
                color,
                is_selected,
                attack: a.attack as f32,
                birth: a.birth,
                signal: a.signal as f32,
                sight: a.tr[2] as f32,
                bulk: a.tr[0] as f32,
                armor: a.tr[3] as f32,
                carnivory: a.tr[5] as f32,
                trail,
            }
        })
        .collect()
}

pub fn generate_soil_rgba(soil: &SoilGrid, out_buf: &mut [u8]) {
    let inv18 = 1.0 / 1.8;
    for i in 0..soil.grid_size {
        let f_val = ((soil.food[i] * inv18) as f64).min(1.0);
        let mut r = 9.5 + f_val * 42.0;
        let mut g = 22.5 + f_val * 61.0;
        let mut b = 25.5 + f_val * 44.0;

        let t = soil.taint[i] as f64;
        if t > 0.0 {
            let tc = t.min(1.0);
            r += tc * 98.0;
            g -= tc * 13.0;
            b += tc * 23.0;
        }

        let s = soil.scent[i] as f64;
        if s > 0.0 {
            let sc = s.min(1.0);
            r += sc * 27.0;
            g += sc * 20.0;
            b += sc * 33.0;
        }

        let offset = i * 4;
        if offset + 3 < out_buf.len() {
            out_buf[offset] = r.clamp(0.0, 255.0) as u8;
            out_buf[offset + 1] = g.clamp(0.0, 255.0) as u8;
            out_buf[offset + 2] = b.clamp(0.0, 255.0) as u8;
            out_buf[offset + 3] = 255;
        }
    }
}

pub fn generate_vignette_rgba(w: usize, h: usize, out_buf: &mut [u8]) {
    let inv_w = 1.0 / (w as f32);
    let inv_h = 1.0 / (h as f32);
    for y in 0..h {
        let ny = (y as f32) * inv_h - 0.45;
        for x in 0..w {
            let nx = (x as f32) * inv_w - 0.5;
            let dist = (nx * nx * 1.2 + ny * ny).sqrt();
            let norm_dist = ((dist - 0.25) / 0.45).clamp(0.0, 1.0);
            let alpha = (norm_dist * norm_dist * 220.0).clamp(0.0, 255.0) as u8;
            let idx = (y * w + x) * 4;
            if idx + 3 < out_buf.len() {
                out_buf[idx] = 7;
                out_buf[idx + 1] = 16;
                out_buf[idx + 2] = 18;
                out_buf[idx + 3] = alpha;
            }
        }
    }
}

pub fn setup_soil_rendering(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // 1. Dynamic Soil Image with Linear Sampler for Smooth Cellular Texture
    let mut soil_img = Image::new_fill(
        Extent3d { width: 75, height: 50, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[10, 23, 26, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    soil_img.sampler = ImageSampler::linear();
    let soil_handle = images.add(soil_img);

    commands.spawn((
        Sprite {
            image: soil_handle.clone(),
            custom_size: Some(Vec2::new(950.0, 747.0)),
            ..default()
        },
        Transform::from_xyz(475.0, 373.5, -20.0),
        SoilSprite,
    ));
    commands.insert_resource(SoilTextureHandle(soil_handle));

    // 2. Atmospheric Radial Vignette Overlay Quad
    let vig_size = 128;
    let mut vig_buf = vec![0u8; vig_size * vig_size * 4];
    generate_vignette_rgba(vig_size, vig_size, &mut vig_buf);
    let mut vig_img = Image::new_fill(
        Extent3d { width: vig_size as u32, height: vig_size as u32, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &vig_buf,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    vig_img.sampler = ImageSampler::linear();
    let vig_handle = images.add(vig_img);

    commands.spawn((
        Sprite {
            image: vig_handle.clone(),
            custom_size: Some(Vec2::new(950.0, 747.0)),
            ..default()
        },
        Transform::from_xyz(475.0, 373.5, -5.0),
        VignetteSprite,
    ));
    commands.insert_resource(VignetteTextureHandle(vig_handle));
}

pub fn update_soil_texture_system(
    sim: Option<Res<SimWorld>>,
    soil_handle: Option<Res<SoilTextureHandle>>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<(&mut Sprite, &mut Transform), (With<SoilSprite>, Without<VignetteSprite>)>,
    mut vig_query: Query<(&mut Sprite, &mut Transform), (With<VignetteSprite>, Without<SoilSprite>)>,
) {
    let (Some(sim), Some(soil_handle)) = (sim, soil_handle) else { return };
    let Some(mut image) = images.get_mut(&soil_handle.0) else { return };

    let cols = sim.world.soil.cols;
    let rows = sim.world.soil.rows;
    let expected_len = cols * rows * 4;

    let needs_resize = image.texture_descriptor.size.width != cols as u32
        || image.texture_descriptor.size.height != rows as u32
        || match &image.data {
            Some(d) => d.len() != expected_len,
            None => true,
        };

    if needs_resize {
        image.resize(Extent3d {
            width: cols as u32,
            height: rows as u32,
            depth_or_array_layers: 1,
        });
        if let Some(ref mut d) = image.data {
            d.resize(expected_len, 0);
        } else {
            image.data = Some(vec![0u8; expected_len]);
        }
    }

    if let Some(ref mut data) = image.data {
        generate_soil_rgba(&sim.world.soil, data);
    }

    let w = sim.world_width as f32;
    let h = sim.world_height as f32;

    if let Ok((mut sprite, mut transform)) = query.single_mut() {
        sprite.custom_size = Some(Vec2::new(w, h));
        transform.translation.x = w * 0.5;
        transform.translation.y = h * 0.5;
    }
    if let Ok((mut sprite, mut transform)) = vig_query.single_mut() {
        sprite.custom_size = Some(Vec2::new(w, h));
        transform.translation.x = w * 0.5;
        transform.translation.y = h * 0.5;
    }
}

pub fn generate_dart_mesh_data(
    sim: &SimWorld,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();

    let items = extract_agent_render_data(sim);
    positions.reserve(items.len() * 12);
    colors.reserve(items.len() * 12);

    for item in &items {
        let rot = Mat2::from_angle(-item.angle);
        let r = item.radius;
        let c = item.color.to_srgba();

        // 1. Glow Halo: 2 Triangles scaled by (1.5 + item.signal * 0.4), with low alpha (0.15 + item.signal * 0.15)
        let halo_scale = 1.5 + item.signal * 0.4;
        let halo_r = r * halo_scale;
        let halo_nose = item.bevy_pos + rot * Vec2::new(halo_r * 1.5, 0.0);
        let halo_right = item.bevy_pos + rot * Vec2::new(-halo_r * 0.75, halo_r * (0.5 + item.armor * 0.45));
        let halo_rear = item.bevy_pos + rot * Vec2::new(-halo_r * (0.45 + item.carnivory), 0.0);
        let halo_left = item.bevy_pos + rot * Vec2::new(-halo_r * 0.75, -halo_r * (0.5 + item.armor * 0.45));

        let halo_alpha = (0.15 + item.signal * 0.18).min(0.4);
        let halo_rgba = [c.red, c.green, c.blue, halo_alpha];

        // Halo Triangle 1: [nose, right, rear]
        positions.push([halo_nose.x, halo_nose.y, -3.0]);
        positions.push([halo_right.x, halo_right.y, -3.0]);
        positions.push([halo_rear.x, halo_rear.y, -3.0]);
        colors.push(halo_rgba);
        colors.push(halo_rgba);
        colors.push(halo_rgba);

        // Halo Triangle 2: [nose, rear, left]
        positions.push([halo_nose.x, halo_nose.y, -3.0]);
        positions.push([halo_rear.x, halo_rear.y, -3.0]);
        positions.push([halo_left.x, halo_left.y, -3.0]);
        colors.push(halo_rgba);
        colors.push(halo_rgba);
        colors.push(halo_rgba);

        // 2. Solid Body: 2 Triangles
        let nose = item.bevy_pos + rot * Vec2::new(r * 1.5, 0.0);
        let right = item.bevy_pos + rot * Vec2::new(-r * 0.75, r * (0.5 + item.armor * 0.45));
        let rear = item.bevy_pos + rot * Vec2::new(-r * (0.45 + item.carnivory), 0.0);
        let left = item.bevy_pos + rot * Vec2::new(-r * 0.75, -r * (0.5 + item.armor * 0.45));

        let body_rgba = [c.red, c.green, c.blue, c.alpha];

        // Body Triangle 1: [nose, right, rear]
        positions.push([nose.x, nose.y, -2.0]);
        positions.push([right.x, right.y, -2.0]);
        positions.push([rear.x, rear.y, -2.0]);
        colors.push(body_rgba);
        colors.push(body_rgba);
        colors.push(body_rgba);

        // Body Triangle 2: [nose, rear, left]
        positions.push([nose.x, nose.y, -2.0]);
        positions.push([rear.x, rear.y, -2.0]);
        positions.push([left.x, left.y, -2.0]);
        colors.push(body_rgba);
        colors.push(body_rgba);
        colors.push(body_rgba);
    }
}

pub fn setup_agent_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    let mesh_handle = meshes.add(mesh);
    let mat_handle = materials.add(ColorMaterial::default());

    commands.spawn((
        Mesh2d(mesh_handle.clone()),
        MeshMaterial2d(mat_handle),
        Transform::default(),
    ));
    commands.insert_resource(AgentMeshResource { mesh_handle });
}

pub fn update_agent_mesh_system(
    sim: Option<Res<SimWorld>>,
    res: Option<Res<AgentMeshResource>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (Some(sim), Some(res)) = (sim, res) else { return };
    let Some(mut mesh) = meshes.get_mut(&res.mesh_handle) else { return };

    let mut positions = Vec::new();
    let mut colors = Vec::new();
    generate_dart_mesh_data(&sim, &mut positions, &mut colors);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
}

pub fn render_sim_gizmos_system(sim: Option<Res<SimWorld>>, mut gizmos: Gizmos) {
    let Some(sim) = sim else { return };

    let w = sim.world_width as f32;
    let h = sim.world_height as f32;

    let items = extract_agent_render_data(&sim);
    for item in items {
        // 1. Trails (Faded multi-segment history lines in lineage palette)
        if item.trail.len() > 1 {
            let trail_color = item.color.with_alpha(0.25);
            for window in item.trail.windows(2) {
                if (window[0].x - window[1].x).abs() < w * 0.5
                    && (window[0].y - window[1].y).abs() < h * 0.5
                {
                    gizmos.line_2d(window[0], window[1], trail_color);
                }
            }
        }

        // 2. Exact Dart Polygon Body Outline
        let rot = Mat2::from_angle(-item.angle);
        let r = item.radius;
        let nose = item.bevy_pos + rot * Vec2::new(r * 1.5, 0.0);
        let right_wing = item.bevy_pos + rot * Vec2::new(-r * 0.75, r * (0.5 + item.armor * 0.45));
        let rear_notch = item.bevy_pos + rot * Vec2::new(-r * (0.45 + item.carnivory), 0.0);
        let left_wing = item.bevy_pos + rot * Vec2::new(-r * 0.75, -r * (0.5 + item.armor * 0.45));

        // Body outline (Red when attacking, else #153034)
        let border_color = if item.attack > 0.5 {
            Color::srgb(1.0, 0.33, 0.31)
        } else {
            Color::srgb(0.082, 0.188, 0.204)
        };

        gizmos.line_2d(nose, right_wing, border_color);
        gizmos.line_2d(right_wing, rear_notch, border_color);
        gizmos.line_2d(rear_notch, left_wing, border_color);
        gizmos.line_2d(left_wing, nose, border_color);

        // 3. Sensory Antennae (Whiskers if sight trait > 0.56)
        if item.sight > 0.56 {
            let antenna_color = item.color.with_alpha(0.6);
            let ant1_start = item.bevy_pos + rot * Vec2::new(-r * 0.3, r * 0.6);
            let ant1_end = item.bevy_pos + rot * Vec2::new(-r * (1.5 + item.sight), r * (1.1 + item.signal));
            let ant2_start = item.bevy_pos + rot * Vec2::new(-r * 0.3, -r * 0.6);
            let ant2_end = item.bevy_pos + rot * Vec2::new(-r * (1.5 + item.sight), -r * (1.1 + item.signal));
            gizmos.line_2d(ant1_start, ant1_end, antenna_color);
            gizmos.line_2d(ant2_start, ant2_end, antenna_color);
        }

        // 4. Birth Halo Ring
        if item.birth > 0 {
            let halo_color = item.color.with_alpha((item.birth as f32) / 120.0);
            let halo_radius = r + 3.0 + ((95.0 - item.birth as f32).max(0.0)) * 0.12;
            gizmos.circle_2d(item.bevy_pos, halo_radius, halo_color);
        }

        // 5. Selection Reticle (Dashed circle #fff1d6)
        if item.is_selected {
            gizmos.circle_2d(item.bevy_pos, r + 8.0, Color::srgb(1.0, 0.945, 0.839));
        }
    }

    // 6. Combat / Bite Sparks
    for spark in &sim.world.spark_events {
        let spark_pos = sim_to_bevy_coord(Vec2::new(spark.x, spark.y), h);
        let spark_color = if spark.color_idx == 8 {
            Color::srgb(1.0, 0.46, 0.40)
        } else if spark.color_idx == 9 {
            Color::srgb(1.0, 0.40, 0.35)
        } else {
            lineage_color(spark.color_idx)
        };
        gizmos.rect_2d(spark_pos, Vec2::splat(3.0), spark_color);
    }

    // 7. Eclipse Visual Overlay
    if sim.world.eclipse > 0 {
        let alpha = ((sim.world.eclipse as f32 / 210.0) * 0.42).clamp(0.0, 0.5);
        let eclipse_color = Color::srgba(0.95, 0.4, 0.32, alpha);
        gizmos.rect_2d(Vec2::new(w * 0.5, h * 0.5), Vec2::new(w, h), eclipse_color);
    }
}

pub fn find_agent_at_position(sim: &SimWorld, sim_pos: Vec2, margin: f32) -> Option<u32> {
    let mut closest: Option<(u32, f32)> = None;

    for a in &sim.world.agents {
        if a.dead != 0 {
            continue;
        }
        let dx = (a.x as f32) - sim_pos.x;
        let dy = (a.y as f32) - sim_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let hit_radius = (2.3 + a.tr[0] * 4.5) as f32 + margin;

        if dist <= hit_radius {
            if let Some((_, best_dist)) = closest {
                if dist < best_dist {
                    closest = Some((a.id, dist));
                }
            } else {
                closest = Some((a.id, dist));
            }
        }
    }

    closest.map(|(id, _)| id)
}

pub fn agent_picking_system(
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    window_query: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<crate::camera::MainCamera>>,
    sim: Option<ResMut<SimWorld>>,
    ui_state: Option<Res<crate::ui::UiState>>,
    mut egui_contexts: Option<bevy_egui::EguiContexts>,
) {
    let (Some(mouse_buttons), Some(mut sim)) = (mouse_buttons, sim) else { return };
    let is_down = mouse_buttons.pressed(MouseButton::Left);
    if !is_down {
        return;
    }

    if let Some(ref mut contexts) = egui_contexts {
        if let Ok(ctx) = contexts.ctx_mut() {
            if ctx.egui_wants_pointer_input() || ctx.is_pointer_over_egui() {
                return;
            }
        }
    }

    let Ok(window) = window_query.single() else { return };
    let Ok((camera, camera_transform)) = camera_query.single() else { return };

    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_pos) else { return };

    let sim_pos = bevy_to_sim_coord(world_pos, sim.world_height as f32);
    let active_tool = ui_state.map(|s| s.active_tool).unwrap_or(crate::ui::ActiveTool::Observe);

    match active_tool {
        crate::ui::ActiveTool::Observe => {
            if mouse_buttons.just_pressed(MouseButton::Left) {
                sim.selected_agent_id = find_agent_at_position(&sim, sim_pos, 8.0);
            }
        }
        crate::ui::ActiveTool::Nourish => {
            let soil = &mut sim.world.soil;
            let col = ((sim_pos.x as f64 / soil.w * soil.cols as f64) as usize).clamp(0, soil.cols - 1);
            let row = ((sim_pos.y as f64 / soil.h * soil.rows as f64) as usize).clamp(0, soil.rows - 1);
            let idx = row * soil.cols + col;
            soil.food[idx] = (soil.food[idx] + 0.6).min(10.0);
        }
        crate::ui::ActiveTool::Blight => {
            let soil = &mut sim.world.soil;
            let col = ((sim_pos.x as f64 / soil.w * soil.cols as f64) as usize).clamp(0, soil.cols - 1);
            let row = ((sim_pos.y as f64 / soil.h * soil.rows as f64) as usize).clamp(0, soil.rows - 1);
            let idx = row * soil.cols + col;
            soil.taint[idx] = (soil.taint[idx] + 0.8).min(5.0);
            soil.food[idx] = 0.0;
        }
        crate::ui::ActiveTool::SeedLife => {
            if mouse_buttons.just_pressed(MouseButton::Left) {
                sim.world.create_agent(sim_pos.x as f64, sim_pos.y as f64, None, None);
            }
        }
        crate::ui::ActiveTool::Extinguish => {
            if let Some(target_id) = find_agent_at_position(&sim, sim_pos, 25.0) {
                if let Some(a) = sim.world.agents.iter_mut().find(|a| a.id == target_id) {
                    a.dead = 1;
                    a.energy = 0.0;
                }
            }
        }
        crate::ui::ActiveTool::Eclipse => {}
    }
}

pub struct ClankRenderPlugin;

impl Plugin for ClankRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_soil_rendering, setup_agent_rendering))
            .add_systems(
                Update,
                (
                    update_soil_texture_system,
                    update_agent_mesh_system,
                    render_sim_gizmos_system,
                    agent_picking_system,
                ),
            );
    }
}
