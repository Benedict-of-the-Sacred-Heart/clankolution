use crate::sim::SimWorld;
use crate::theme::PALETTE;
use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use clank_core::agent::AgentData;
use clank_core::soil::SoilGrid;

#[derive(Resource)]
pub struct SoilTextureHandle(pub Handle<Image>);

#[derive(Resource)]
pub struct VignetteTextureHandle(pub Handle<Image>);

#[derive(Component)]
pub struct SoilSprite;

#[derive(Component)]
pub struct VignetteSprite;

#[derive(Component)]
pub struct EclipseOverlaySprite;

#[derive(Clone, Copy, Debug)]
pub struct SparkParticle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub life: f32,
    pub max_life: f32,
    pub color: Color,
}

#[derive(Resource)]
pub struct ParticleMeshResource {
    pub mesh_handle: Handle<Mesh>,
}

#[derive(Resource)]
pub struct ParticleSystemResource {
    pub particles: Vec<SparkParticle>,
    pub prng_state: u64,
}

impl Default for ParticleSystemResource {
    fn default() -> Self {
        Self {
            particles: Vec::with_capacity(1024),
            prng_state: 0x9e3779b97f4a7c15,
        }
    }
}

impl ParticleSystemResource {
    #[inline]
    pub fn next_f32(&mut self, min: f32, max: f32) -> f32 {
        self.prng_state ^= self.prng_state << 13;
        self.prng_state ^= self.prng_state >> 7;
        self.prng_state ^= self.prng_state << 17;
        let frac = (self.prng_state & 0x00ffffff) as f32 / 16777216.0;
        min + frac * (max - min)
    }
}

#[inline]
pub fn step_particles(particles: &mut Vec<SparkParticle>) {
    particles.retain_mut(|p| {
        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.97;
        p.vy *= 0.97;
        p.life -= 1.0;
        p.life > 0.0
    });
}

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

pub fn setup_soil_rendering(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // 1. Dynamic Soil Image with Linear Sampler for Smooth Cellular Texture
    let mut soil_img = Image::new_fill(
        Extent3d {
            width: 75,
            height: 50,
            depth_or_array_layers: 1,
        },
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
            custom_size: Some(Vec2::new(900.0, 600.0)),
            ..default()
        },
        Transform::from_xyz(450.0, 300.0, -20.0),
        SoilSprite,
    ));
    commands.insert_resource(SoilTextureHandle(soil_handle));

    // 2. Atmospheric Radial Vignette Overlay Quad
    let vig_size = 128;
    let mut vig_buf = vec![0u8; vig_size * vig_size * 4];
    generate_vignette_rgba(vig_size, vig_size, &mut vig_buf);
    let mut vig_img = Image::new_fill(
        Extent3d {
            width: vig_size as u32,
            height: vig_size as u32,
            depth_or_array_layers: 1,
        },
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
            custom_size: Some(Vec2::new(900.0, 600.0)),
            ..default()
        },
        Transform::from_xyz(450.0, 300.0, -5.0),
        VignetteSprite,
    ));
    commands.insert_resource(VignetteTextureHandle(vig_handle));

    // 3. Eclipse Darkening Overlay Quad (`clankolution.html`: rgba(20, 5, 7, (eclipse / 210) * 0.42))
    commands.spawn((
        Sprite {
            color: Color::srgba(20.0 / 255.0, 5.0 / 255.0, 7.0 / 255.0, 0.0),
            custom_size: Some(Vec2::new(900.0, 600.0)),
            ..default()
        },
        Transform::from_xyz(450.0, 300.0, 6.0),
        EclipseOverlaySprite,
    ));
}

pub fn update_soil_texture_system(
    sim: Option<Res<SimWorld>>,
    gpu_driver: Option<Res<crate::sim::GpuDriverResource>>,
    soil_handle: Option<Res<SoilTextureHandle>>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<
        (&mut Sprite, &mut Transform),
        (
            With<SoilSprite>,
            Without<VignetteSprite>,
            Without<EclipseOverlaySprite>,
        ),
    >,
    mut vig_query: Query<
        (&mut Sprite, &mut Transform),
        (
            With<VignetteSprite>,
            Without<SoilSprite>,
            Without<EclipseOverlaySprite>,
        ),
    >,
    mut eclipse_query: Query<
        (&mut Sprite, &mut Transform),
        (
            With<EclipseOverlaySprite>,
            Without<SoilSprite>,
            Without<VignetteSprite>,
        ),
    >,
) {
    let (Some(sim), Some(soil_handle)) = (sim, soil_handle) else {
        return;
    };
    let cols = sim.world.soil.cols;
    let rows = sim.world.soil.rows;
    let expected_len = cols * rows * 4;
    let needs_resize = images.get(&soil_handle.0).is_some_and(|image| {
        image.texture_descriptor.size.width != cols as u32
            || image.texture_descriptor.size.height != rows as u32
            || image.data.as_ref().is_none_or(|d| d.len() != expected_len)
    });
    // While the GPU engine holds the world, the creature pass copies its soil colormap into
    // this image on the GPU. Touching the asset here would re-upload stale CPU pixels.
    let gpu_owns_soil = gpu_driver
        .as_ref()
        .and_then(|g| g.driver.as_ref())
        .is_some_and(|d| d.is_initialized());

    if needs_resize || !gpu_owns_soil {
        let Some(mut image) = images.get_mut(&soil_handle.0) else {
            return;
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
    if let Ok((mut sprite, mut transform)) = eclipse_query.single_mut() {
        let alpha = if sim.world.eclipse > 0 {
            ((sim.world.eclipse as f32 / 210.0) * 0.42).clamp(0.0, 0.42)
        } else {
            0.0
        };
        sprite.color = Color::srgba(20.0 / 255.0, 5.0 / 255.0, 7.0 / 255.0, alpha);
        sprite.custom_size = Some(Vec2::new(w, h));
        transform.translation.x = w * 0.5;
        transform.translation.y = h * 0.5;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VisualCacheUnpacked {
    pub radius: f32,
    pub color: Color,
    pub is_attacking: bool,
    pub birth_val: u8,
    pub signal: f32,
}

#[inline]
pub fn unpack_visual_cache_detailed(
    visual_cache: u32,
    packed_color: u32,
) -> Option<VisualCacheUnpacked> {
    if visual_cache == 0 {
        return None;
    }
    let r_u8 = (visual_cache & 0xFF) as f32;
    let radius = 2.3 + (r_u8 / 255.0) * 4.5;
    let signal = ((visual_cache >> 8) & 0xFF) as f32 / 255.0;
    let e_u8 = ((visual_cache >> 16) & 0xFF) as f32;
    let alpha = (0.55 + (e_u8 / 255.0 * 100.0 / 160.0)).clamp(0.55, 1.0);

    let r = (packed_color & 0xFF) as u8;
    let g = ((packed_color >> 8) & 0xFF) as u8;
    let b = ((packed_color >> 16) & 0xFF) as u8;
    let base_color = Color::srgba_u8(r, g, b, (alpha * 255.0) as u8);

    let is_attacking = (visual_cache & (1 << 24)) != 0;
    let birth_val = ((visual_cache >> 25) & 0x7F) as u8;

    Some(VisualCacheUnpacked {
        radius,
        color: base_color,
        is_attacking,
        birth_val,
        signal,
    })
}

#[inline]
pub fn unpack_visual_cache(
    visual_cache: u32,
    packed_color: u32,
) -> Option<(f32, Color, bool, bool)> {
    let d = unpack_visual_cache_detailed(visual_cache, packed_color)?;
    Some((d.radius, d.color, d.is_attacking, d.birth_val > 0))
}

pub fn generate_particle_mesh_data(
    particles: &ParticleSystemResource,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    let count = particles.particles.len();
    positions.reserve(count * 6);
    colors.reserve(count * 6);

    for p in &particles.particles {
        let alpha = (p.life / p.max_life).clamp(0.0, 1.0);
        let c = p.color.to_srgba();
        let rgba = [c.red, c.green, c.blue, alpha];

        let x0 = p.x - 1.0;
        let x1 = p.x + 1.0;
        let y0 = p.y - 1.0;
        let y1 = p.y + 1.0;
        let z = 5.0;

        // Triangle 1: (x0, y0), (x1, y0), (x1, y1)
        positions.push([x0, y0, z]);
        positions.push([x1, y0, z]);
        positions.push([x1, y1, z]);
        colors.push(rgba);
        colors.push(rgba);
        colors.push(rgba);

        // Triangle 2: (x0, y0), (x1, y1), (x0, y1)
        positions.push([x0, y0, z]);
        positions.push([x1, y1, z]);
        positions.push([x0, y1, z]);
        colors.push(rgba);
        colors.push(rgba);
        colors.push(rgba);
    }

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn setup_particle_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, -100.0]; 3]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0, 0.0, 0.0, 0.0]; 3]);
    let mesh_handle = meshes.add(mesh);
    let mat_handle = materials.add(ColorMaterial::default());

    commands.spawn((
        Mesh2d(mesh_handle.clone()),
        MeshMaterial2d(mat_handle),
        Transform::default(),
    ));
    commands.insert_resource(ParticleMeshResource { mesh_handle });
}

pub fn update_particle_mesh_system(
    particles: Option<Res<ParticleSystemResource>>,
    res: Option<Res<ParticleMeshResource>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (Some(particles), Some(res)) = (particles, res) else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&res.mesh_handle) else {
        return;
    };

    let mut positions = Vec::new();
    let mut colors = Vec::new();
    generate_particle_mesh_data(&particles, &mut positions, &mut colors);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
}

pub fn update_particles_system(
    sim: Option<ResMut<SimWorld>>,
    res: Option<ResMut<ParticleSystemResource>>,
) {
    let (Some(mut sim), Some(mut res)) = (sim, res) else {
        return;
    };
    let h = sim.world_height as f32;

    // Drain spark events so each event is consumed exactly once
    let events: Vec<_> = sim.world.spark_events.drain(..).collect();
    for spark in events {
        let spark_pos = sim_to_bevy_coord(Vec2::new(spark.x as f32, spark.y as f32), h);
        let color = if spark.color_idx == 8 {
            Color::srgb(1.0, 0.46, 0.40) // #ff7667 death
        } else if spark.color_idx == 9 {
            Color::srgb(1.0, 0.40, 0.35) // #ff6659 eclipse
        } else if spark.color_idx == 10 {
            Color::srgb(0.87, 0.74, 0.47) // #dfbd79 nourish
        } else if spark.color_idx == 11 {
            Color::srgb(0.88, 0.47, 0.48) // #e0777a blight
        } else if spark.color_idx == 12 {
            Color::srgb(0.61, 0.91, 0.84) // #9ce7d5 seed life
        } else {
            lineage_color(spark.color_idx)
        };

        let count = spark.count.clamp(1, 30);
        for _ in 0..count {
            let vx = res.next_f32(-2.4, 2.4);
            let vy = res.next_f32(-2.4, 2.4);
            let life = res.next_f32(15.0, 36.0);

            res.particles.push(SparkParticle {
                x: spark_pos.x,
                y: spark_pos.y,
                vx,
                vy,
                life,
                max_life: 36.0,
                color,
            });
        }
    }

    if !sim.paused {
        let steps = sim.speed.clamp(1, 10);
        for _ in 0..steps {
            step_particles(&mut res.particles);
        }
    }

    if res.particles.len() > 850 {
        let excess = res.particles.len() - 850;
        res.particles.drain(0..excess);
    }
}

pub fn render_sim_gizmos_system(sim: Option<Res<SimWorld>>, mut gizmos: Gizmos) {
    let Some(sim) = sim else { return };

    let w = sim.world_width as f32;
    let h = sim.world_height as f32;

    // Render high-detail focus gizmos (reticle & trail) ONLY for the selected specimen.
    // Background dart outlines and birth halos are rendered in the unified GPU mesh.
    if let Some(a) = sim.get_selected_agent() {
        if a.dead == 0 {
            let sim_pos = Vec2::new(a.x as f32, a.y as f32);
            let bevy_pos = sim_to_bevy_coord(sim_pos, h);
            let r = (2.3 + a.tr[0] * 4.5) as f32;
            let c = agent_body_color(a.root, a.energy);

            // Selection Reticle (#fff1d6)
            gizmos.circle_2d(bevy_pos, r + 8.0, Color::srgb(1.0, 0.945, 0.839));

            // Selected specimen history trail (the GPU engine fills the cached agent's trail)
            let trail_color = c.with_alpha(0.45);
            if a.trail_count > 1 {
                let trail_len = (a.trail_count as usize).min(9);
                for i in 0..trail_len.saturating_sub(1) {
                    let p1 =
                        sim_to_bevy_coord(Vec2::new(a.trail_x[i] as f32, a.trail_y[i] as f32), h);
                    let p2 = sim_to_bevy_coord(
                        Vec2::new(a.trail_x[i + 1] as f32, a.trail_y[i + 1] as f32),
                        h,
                    );
                    if (p1.x - p2.x).abs() < w * 0.5 && (p1.y - p2.y).abs() < h * 0.5 {
                        gizmos.line_2d(p1, p2, trail_color);
                    }
                }
            }
        }
    }

    // Eclipse Visual Border (`clankolution.html`: rgba(244, 102, 83, 0.4) at strokeRect(1, 1, W-2, H-2))
    if sim.world.eclipse > 0 {
        let eclipse_border = Color::srgba(244.0 / 255.0, 102.0 / 255.0, 83.0 / 255.0, 0.4);
        gizmos.rect_2d(
            Vec2::new(w * 0.5, h * 0.5),
            Vec2::new((w - 2.0).max(1.0), (h - 2.0).max(1.0)),
            eclipse_border,
        );
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
    gpu_driver: Option<ResMut<crate::sim::GpuDriverResource>>,
    ui_state: Option<Res<crate::ui::UiState>>,
    mut egui_contexts: Option<bevy_egui::EguiContexts>,
) {
    let (Some(mouse_buttons), Some(mut sim)) = (mouse_buttons, sim) else {
        return;
    };
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

    let Ok(window) = window_query.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };

    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_pos) else {
        return;
    };

    let sim_pos = bevy_to_sim_coord(world_pos, sim.world_height as f32);
    let active_tool = ui_state
        .map(|s| s.active_tool)
        .unwrap_or(crate::ui::ActiveTool::Observe);

    match active_tool {
        crate::ui::ActiveTool::Observe => {
            if mouse_buttons.just_pressed(MouseButton::Left) {
                if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                    sim.pending_tool = Some(crate::sim::PendingTool {
                        tool_type: 0,
                        tool_pos: [sim_pos.x, sim_pos.y],
                        tool_radius: 25.0,
                    });
                } else {
                    sim.selected_agent_id = find_agent_at_position(&sim, sim_pos, 25.0);
                }
            }
        }
        crate::ui::ActiveTool::Nourish => {
            sim.world.nourish_at(sim_pos.x as f64, sim_pos.y as f64);
            if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                sim.pending_tool = Some(crate::sim::PendingTool {
                    tool_type: 1,
                    tool_pos: [sim_pos.x, sim_pos.y],
                    tool_radius: 45.0,
                });
            }
        }
        crate::ui::ActiveTool::Blight => {
            sim.world.blight_at(sim_pos.x as f64, sim_pos.y as f64);
            if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                sim.pending_tool = Some(crate::sim::PendingTool {
                    tool_type: 2,
                    tool_pos: [sim_pos.x, sim_pos.y],
                    tool_radius: 45.0,
                });
            }
        }
        crate::ui::ActiveTool::SeedLife => {
            if mouse_buttons.just_pressed(MouseButton::Left) || (sim.world.tick % 8 == 0) {
                let drv = gpu_driver.as_ref().and_then(|g| g.driver.as_ref());
                crate::sim::seed_life_gpu_aware(&mut sim, drv, &[(sim_pos.x, sim_pos.y)]);
            }
        }
        crate::ui::ActiveTool::Extinguish => {
            sim.world
                .extinguish_at(sim_pos.x as f64, sim_pos.y as f64, 23.0);
            sim.validate_cpu_selection();
            if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                sim.pending_tool = Some(crate::sim::PendingTool {
                    tool_type: 3,
                    tool_pos: [sim_pos.x, sim_pos.y],
                    tool_radius: 23.0,
                });
            }
        }
        crate::ui::ActiveTool::Eclipse => {
            if mouse_buttons.just_pressed(MouseButton::Left) {
                sim.world.trigger_eclipse();
            }
        }
    }
}

pub struct ClankRenderPlugin;

impl Plugin for ClankRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(crate::creature_pass::CreaturePassPlugin)
            .init_resource::<ParticleSystemResource>()
            .add_systems(Startup, (setup_soil_rendering, setup_particle_rendering))
            .add_systems(
                Update,
                (
                    update_soil_texture_system,
                    update_particles_system,
                    update_particle_mesh_system,
                    render_sim_gizmos_system,
                    agent_picking_system,
                ),
            );
    }
}
