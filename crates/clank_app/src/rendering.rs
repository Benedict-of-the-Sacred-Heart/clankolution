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

pub fn extract_gpu_agent_render_data(
    agents: &[crate::gpu::types::GpuAgentState],
    world_height: f32,
    selected_id: Option<u32>,
    camera_bounds: Option<[f32; 4]>,
) -> Vec<AgentRenderItem> {
    agents
        .iter()
        .filter(|a| (a.meta_flags & (1 << 13)) == 0) // living only
        .filter(|a| {
            if let Some([min_x, min_y, max_x, max_y]) = camera_bounds {
                let x = a.pos_vel[0];
                let y = a.pos_vel[1];
                let r = 2.3 + a.traits[0] * 4.5;
                x + r >= min_x && x - r <= max_x && y + r >= min_y && y - r <= max_y
            } else {
                true
            }
        })
        .map(|a| {
            let sim_pos = Vec2::new(a.pos_vel[0], a.pos_vel[1]);
            let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
            let radius = 2.3 + a.traits[0] * 4.5;
            let root = a.meta_flags & 0x0F;
            let energy = a.angle_energy[1] as f64;
            let color = agent_body_color(root, energy);
            let is_selected = selected_id == Some(a.id);
            let a_birth = (a.meta_flags >> 6) & 0x7F;

            AgentRenderItem {
                id: a.id,
                sim_pos,
                bevy_pos,
                angle: a.angle_energy[0],
                radius,
                color,
                is_selected,
                attack: a.angle_energy[3],
                birth: a_birth,
                signal: a.traits[6],
                sight: a.traits[2],
                bulk: a.traits[0],
                armor: a.traits[3],
                carnivory: a.traits[5],
                trail: Vec::new(),
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
            custom_size: Some(Vec2::new(900.0, 600.0)),
            ..default()
        },
        Transform::from_xyz(450.0, 300.0, -5.0),
        VignetteSprite,
    ));
    commands.insert_resource(VignetteTextureHandle(vig_handle));
}

pub fn update_soil_texture_system(
    sim: Option<Res<SimWorld>>,
    gpu_driver: Option<Res<crate::sim::GpuDriverResource>>,
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
        if sim.active_engine == crate::sim::ActiveEngine::Gpu {
            let mut copied = false;
            if let Some(gpu) = gpu_driver.as_ref() {
                if let Some(ref driver) = gpu.driver {
                    if driver.is_initialized() {
                        if sim.world.tick % 2 == 0 {
                            driver.copy_soil_display_rgba(data);
                        }
                        copied = true;
                    }
                }
            }
            if !copied {
                generate_soil_rgba(&sim.world.soil, data);
            }
        } else {
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
pub fn unpack_visual_cache_detailed(visual_cache: u32, packed_color: u32) -> Option<VisualCacheUnpacked> {
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
pub fn unpack_visual_cache(visual_cache: u32, packed_color: u32) -> Option<(f32, Color, bool, bool)> {
    let d = unpack_visual_cache_detailed(visual_cache, packed_color)?;
    Some((d.radius, d.color, d.is_attacking, d.birth_val > 0))
}

pub fn generate_dart_mesh_from_gpu_states(
    agents: &[crate::gpu::types::GpuAgentState],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(agents.len() * 6);
    colors.reserve(agents.len() * 6);

    for a in agents {
        let Some((radius, color, _is_attacking, _has_birth)) = unpack_visual_cache(a.visual_cache, a.packed_color) else {
            continue;
        };

        let sim_pos = Vec2::new(a.pos_vel[0], a.pos_vel[1]);
        let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
        let rot = Mat2::from_angle(-a.angle_energy[0]);
        let c = color.to_srgba();
        let body_rgba = [c.red, c.green, c.blue, c.alpha];

        let r = radius;
        let armor = a.traits[3];
        let carnivory = a.traits[5];

        let nose = bevy_pos + rot * Vec2::new(r * 1.5, 0.0);
        let right = bevy_pos + rot * Vec2::new(-r * 0.75, r * (0.5 + armor * 0.45));
        let rear = bevy_pos + rot * Vec2::new(-r * (0.45 + carnivory), 0.0);
        let left = bevy_pos + rot * Vec2::new(-r * 0.75, -r * (0.5 + armor * 0.45));

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

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn generate_outline_mesh_from_gpu_states(
    agents: &[crate::gpu::types::GpuAgentState],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(agents.len() * 12);
    colors.reserve(agents.len() * 12);

    for a in agents {
        let Some(vis) = unpack_visual_cache_detailed(a.visual_cache, a.packed_color) else {
            continue;
        };

        let sim_pos = Vec2::new(a.pos_vel[0], a.pos_vel[1]);
        let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
        let rot = Mat2::from_angle(-a.angle_energy[0]);

        let r = vis.radius;
        let armor = a.traits[3];
        let carnivory = a.traits[5];

        let nose = bevy_pos + rot * Vec2::new(r * 1.5, 0.0);
        let right = bevy_pos + rot * Vec2::new(-r * 0.75, r * (0.5 + armor * 0.45));
        let rear = bevy_pos + rot * Vec2::new(-r * (0.45 + carnivory), 0.0);
        let left = bevy_pos + rot * Vec2::new(-r * 0.75, -r * (0.5 + armor * 0.45));

        let border_rgba = if vis.is_attacking {
            [1.0, 0.33, 0.31, 1.0]
        } else {
            [0.082, 0.188, 0.204, 1.0]
        };

        // 4 lines = 8 vertices for LineList
        positions.push([nose.x, nose.y, -1.9]);
        positions.push([right.x, right.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        positions.push([right.x, right.y, -1.9]);
        positions.push([rear.x, rear.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        positions.push([rear.x, rear.y, -1.9]);
        positions.push([left.x, left.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        positions.push([left.x, left.y, -1.9]);
        positions.push([nose.x, nose.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        // Sensory antennae whiskers if sight > 0.56
        if a.traits[2] > 0.56 {
            let c = vis.color.to_srgba();
            let ant_color = [c.red, c.green, c.blue, 0.55];
            let sight = a.traits[2];
            let signal = a.traits[6];
            let ant1_start = bevy_pos + rot * Vec2::new(-r * 0.3, r * 0.6);
            let ant1_end = bevy_pos + rot * Vec2::new(-r * (1.5 + sight), r * (1.1 + signal));
            let ant2_start = bevy_pos + rot * Vec2::new(-r * 0.3, -r * 0.6);
            let ant2_end = bevy_pos + rot * Vec2::new(-r * (1.5 + sight), -r * (1.1 + signal));

            positions.push([ant1_start.x, ant1_start.y, -1.9]);
            positions.push([ant1_end.x, ant1_end.y, -1.9]);
            colors.push(ant_color);
            colors.push(ant_color);

            positions.push([ant2_start.x, ant2_start.y, -1.9]);
            positions.push([ant2_end.x, ant2_end.y, -1.9]);
            colors.push(ant_color);
            colors.push(ant_color);
        }

        // Birth halo ring if newborn (dynamic expanding ring in creature lineage color)
        let birth_val = if vis.birth_val > 0 {
            vis.birth_val
        } else {
            ((a.meta_flags >> 6) & 0x7F) as u8
        };
        if birth_val > 0 {
            let birth_f = birth_val as f32;
            let halo_r = r + 3.0 + ((95.0 - birth_f).max(0.0)) * 0.12;
            let c = vis.color.to_srgba();
            let halo_color = [c.red, c.green, c.blue, (birth_f / 120.0).clamp(0.0, 1.0)];
            for seg in 0..16 {
                let theta1 = (seg as f32) * std::f32::consts::TAU / 16.0;
                let theta2 = ((seg + 1) as f32) * std::f32::consts::TAU / 16.0;
                let p1 = bevy_pos + Vec2::new(theta1.cos() * halo_r, theta1.sin() * halo_r);
                let p2 = bevy_pos + Vec2::new(theta2.cos() * halo_r, theta2.sin() * halo_r);
                positions.push([p1.x, p1.y, -1.8]);
                positions.push([p2.x, p2.y, -1.8]);
                colors.push(halo_color);
                colors.push(halo_color);
            }
        }
    }

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn generate_dart_mesh_from_instances(
    instances: &[crate::gpu::types::GpuDartInstance],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(instances.len() * 6);
    colors.reserve(instances.len() * 6);

    for inst in instances {
        let Some((radius, color, _is_attacking, _has_birth)) = unpack_visual_cache(inst.vis_data[1], inst.vis_data[0]) else {
            continue;
        };

        let sim_pos = Vec2::new(inst.pos_angle[0], inst.pos_angle[1]);
        let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
        let rot = Mat2::from_angle(-inst.pos_angle[2]);
        let c = color.to_srgba();
        let body_rgba = [c.red, c.green, c.blue, c.alpha];

        let r = radius;
        let armor = inst.pad0;
        let carnivory = ((inst.pad1[0] >> 24) as f32) / 255.0;

        let nose = bevy_pos + rot * Vec2::new(r * 1.5, 0.0);
        let right = bevy_pos + rot * Vec2::new(-r * 0.75, r * (0.5 + armor * 0.45));
        let rear = bevy_pos + rot * Vec2::new(-r * (0.45 + carnivory), 0.0);
        let left = bevy_pos + rot * Vec2::new(-r * 0.75, -r * (0.5 + armor * 0.45));

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

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn generate_outline_mesh_from_instances(
    instances: &[crate::gpu::types::GpuDartInstance],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(instances.len() * 12);
    colors.reserve(instances.len() * 12);

    for inst in instances {
        let Some(vis) = unpack_visual_cache_detailed(inst.vis_data[1], inst.vis_data[0]) else {
            continue;
        };

        let sim_pos = Vec2::new(inst.pos_angle[0], inst.pos_angle[1]);
        let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
        let rot = Mat2::from_angle(-inst.pos_angle[2]);

        let r = vis.radius;
        let armor = inst.pad0;
        let carnivory = ((inst.pad1[0] >> 24) as f32) / 255.0;
        let sight = f32::from_bits(inst.pad1[1]);

        let nose = bevy_pos + rot * Vec2::new(r * 1.5, 0.0);
        let right = bevy_pos + rot * Vec2::new(-r * 0.75, r * (0.5 + armor * 0.45));
        let rear = bevy_pos + rot * Vec2::new(-r * (0.45 + carnivory), 0.0);
        let left = bevy_pos + rot * Vec2::new(-r * 0.75, -r * (0.5 + armor * 0.45));

        let border_rgba = if vis.is_attacking {
            [1.0, 0.33, 0.31, 1.0]
        } else {
            [0.082, 0.188, 0.204, 1.0]
        };

        // 4 lines = 8 vertices for LineList
        positions.push([nose.x, nose.y, -1.9]);
        positions.push([right.x, right.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        positions.push([right.x, right.y, -1.9]);
        positions.push([rear.x, rear.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        positions.push([rear.x, rear.y, -1.9]);
        positions.push([left.x, left.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        positions.push([left.x, left.y, -1.9]);
        positions.push([nose.x, nose.y, -1.9]);
        colors.push(border_rgba);
        colors.push(border_rgba);

        // Sensory antennae whiskers if sight > 0.56
        if sight > 0.56 {
            let c = vis.color.to_srgba();
            let ant_color = [c.red, c.green, c.blue, 0.55];
            let signal = vis.signal;
            let ant1_start = bevy_pos + rot * Vec2::new(-r * 0.3, r * 0.6);
            let ant1_end = bevy_pos + rot * Vec2::new(-r * (1.5 + sight), r * (1.1 + signal));
            let ant2_start = bevy_pos + rot * Vec2::new(-r * 0.3, -r * 0.6);
            let ant2_end = bevy_pos + rot * Vec2::new(-r * (1.5 + sight), -r * (1.1 + signal));

            positions.push([ant1_start.x, ant1_start.y, -1.9]);
            positions.push([ant1_end.x, ant1_end.y, -1.9]);
            colors.push(ant_color);
            colors.push(ant_color);

            positions.push([ant2_start.x, ant2_start.y, -1.9]);
            positions.push([ant2_end.x, ant2_end.y, -1.9]);
            colors.push(ant_color);
            colors.push(ant_color);
        }

        // Birth halo ring if newborn (dynamic expanding ring in creature lineage color)
        if vis.birth_val > 0 {
            let birth_f = vis.birth_val as f32;
            let halo_r = r + 3.0 + (95.0 - birth_f).max(0.0) * 0.12;
            let c = vis.color.to_srgba();
            let halo_color = [c.red, c.green, c.blue, (birth_f / 120.0).clamp(0.0, 1.0)];
            for seg in 0..16 {
                let theta1 = (seg as f32) * std::f32::consts::TAU / 16.0;
                let theta2 = ((seg + 1) as f32) * std::f32::consts::TAU / 16.0;
                let p1 = bevy_pos + Vec2::new(theta1.cos() * halo_r, theta1.sin() * halo_r);
                let p2 = bevy_pos + Vec2::new(theta2.cos() * halo_r, theta2.sin() * halo_r);
                positions.push([p1.x, p1.y, -1.8]);
                positions.push([p2.x, p2.y, -1.8]);
                colors.push(halo_color);
                colors.push(halo_color);
            }
        }
    }

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn generate_dart_mesh_data(
    sim: &SimWorld,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    let (gpu_states, _, _, _, _) = crate::gpu::bridge::sync_rust_to_gpu(sim);
    generate_dart_mesh_from_gpu_states(&gpu_states, sim.world_height as f32, positions, colors);
}

#[derive(Resource)]
pub struct AgentOutlineMeshResource {
    pub mesh_handle: Handle<Mesh>,
}

#[derive(Resource)]
pub struct AgentGlowMeshResource {
    pub mesh_handle: Handle<Mesh>,
}

#[derive(Resource)]
pub struct AgentTrailMeshResource {
    pub mesh_handle: Handle<Mesh>,
}

#[derive(Resource, Default)]
pub struct AgentTrailsTracker {
    pub trails: std::collections::HashMap<u32, std::collections::VecDeque<Vec2>>,
    pub last_clean_tick: u32,
}

pub fn generate_glow_mesh_from_instances(
    instances: &[crate::gpu::types::GpuDartInstance],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(instances.len() * 60);
    colors.reserve(instances.len() * 60);

    for inst in instances {
        let Some(vis) = unpack_visual_cache_detailed(inst.vis_data[1], inst.vis_data[0]) else {
            continue;
        };

        let sim_pos = Vec2::new(inst.pos_angle[0], inst.pos_angle[1]);
        let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
        let r = vis.radius;
        let c = vis.color.to_srgba();
        let signal = vis.signal;

        let glow_r = r + 5.0 + 8.0 * signal;
        let center_alpha = (0.20 + 0.18 * signal).clamp(0.12, 0.40);
        let center_color = [c.red, c.green, c.blue, center_alpha];
        let outer_color = [c.red, c.green, c.blue, 0.0];

        for seg in 0..20 {
            let theta1 = (seg as f32) * std::f32::consts::TAU / 20.0;
            let theta2 = ((seg + 1) as f32) * std::f32::consts::TAU / 20.0;
            let p1 = bevy_pos + Vec2::new(theta1.cos() * glow_r, theta1.sin() * glow_r);
            let p2 = bevy_pos + Vec2::new(theta2.cos() * glow_r, theta2.sin() * glow_r);

            positions.push([bevy_pos.x, bevy_pos.y, -2.2]);
            positions.push([p1.x, p1.y, -2.2]);
            positions.push([p2.x, p2.y, -2.2]);
            colors.push(center_color);
            colors.push(outer_color);
            colors.push(outer_color);
        }
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

pub fn generate_glow_mesh_from_gpu_states(
    agents: &[crate::gpu::types::GpuAgentState],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(agents.len() * 60);
    colors.reserve(agents.len() * 60);

    for a in agents {
        let Some(vis) = unpack_visual_cache_detailed(a.visual_cache, a.packed_color) else {
            continue;
        };

        let sim_pos = Vec2::new(a.pos_vel[0], a.pos_vel[1]);
        let bevy_pos = sim_to_bevy_coord(sim_pos, world_height);
        let r = vis.radius;
        let c = vis.color.to_srgba();
        let signal = a.traits[6];

        let glow_r = r + 5.0 + 8.0 * signal;
        let center_alpha = (0.20 + 0.18 * signal).clamp(0.12, 0.40);
        let center_color = [c.red, c.green, c.blue, center_alpha];
        let outer_color = [c.red, c.green, c.blue, 0.0];

        for seg in 0..20 {
            let theta1 = (seg as f32) * std::f32::consts::TAU / 20.0;
            let theta2 = ((seg + 1) as f32) * std::f32::consts::TAU / 20.0;
            let p1 = bevy_pos + Vec2::new(theta1.cos() * glow_r, theta1.sin() * glow_r);
            let p2 = bevy_pos + Vec2::new(theta2.cos() * glow_r, theta2.sin() * glow_r);

            positions.push([bevy_pos.x, bevy_pos.y, -2.2]);
            positions.push([p1.x, p1.y, -2.2]);
            positions.push([p2.x, p2.y, -2.2]);
            colors.push(center_color);
            colors.push(outer_color);
            colors.push(outer_color);
        }
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

pub fn generate_swarm_trails_from_sim(
    sim: &SimWorld,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    let w = sim.world_width as f32;
    let h = sim.world_height as f32;

    for a in sim.world.agents.iter().filter(|ag| ag.dead == 0) {
        let tc = (a.trail_count as usize).min(9);
        if tc < 2 { continue; }
        let c = lineage_color(a.root).to_srgba();

        for i in 0..(tc - 1) {
            let p1 = sim_to_bevy_coord(Vec2::new(a.trail_x[i] as f32, a.trail_y[i] as f32), h);
            let p2 = sim_to_bevy_coord(Vec2::new(a.trail_x[i + 1] as f32, a.trail_y[i + 1] as f32), h);

            if (p1.x - p2.x).abs() < w * 0.5 && (p1.y - p2.y).abs() < h * 0.5 {
                let alpha = (0.18 + 0.32 * ((i + 1) as f32 / (tc - 1) as f32)).clamp(0.15, 0.50);
                let seg_color = [c.red, c.green, c.blue, alpha];
                positions.push([p1.x, p1.y, -2.05]);
                positions.push([p2.x, p2.y, -2.05]);
                colors.push(seg_color);
                colors.push(seg_color);
            }
        }
    }

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn generate_swarm_trails_from_tracker(
    tracker: &AgentTrailsTracker,
    instances: &[crate::gpu::types::GpuDartInstance],
    world_width: f32,
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(instances.len() * 18);
    colors.reserve(instances.len() * 18);

    for inst in instances {
        let agent_id = inst.pad1[0] & 0x00FFFFFF;
        let Some(vis) = unpack_visual_cache_detailed(inst.vis_data[1], inst.vis_data[0]) else {
            continue;
        };
        let c = vis.color.to_srgba();

        if let Some(trail) = tracker.trails.get(&agent_id) {
            let tc = trail.len();
            if tc < 2 { continue; }
            for i in 0..(tc - 1) {
                let p1 = trail[i];
                let p2 = trail[i + 1];
                if (p1.x - p2.x).abs() < world_width * 0.5 && (p1.y - p2.y).abs() < world_height * 0.5 {
                    let alpha = (0.18 + 0.32 * ((i + 1) as f32 / (tc - 1) as f32)).clamp(0.15, 0.50);
                    let seg_color = [c.red, c.green, c.blue, alpha];
                    positions.push([p1.x, p1.y, -2.05]);
                    positions.push([p2.x, p2.y, -2.05]);
                    colors.push(seg_color);
                    colors.push(seg_color);
                }
            }
        }
    }

    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

pub fn setup_agent_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, -100.0]; 3]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0, 0.0, 0.0, 0.0]; 3]);
    let mesh_handle = meshes.add(mesh);
    let mat_handle = materials.add(ColorMaterial::default());

    commands.spawn((
        Mesh2d(mesh_handle.clone()),
        MeshMaterial2d(mat_handle.clone()),
        Transform::default(),
    ));
    commands.insert_resource(AgentMeshResource { mesh_handle });

    let mut outline_mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    outline_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, -100.0]; 2]);
    outline_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0, 0.0, 0.0, 0.0]; 2]);
    let outline_handle = meshes.add(outline_mesh);

    commands.spawn((
        Mesh2d(outline_handle.clone()),
        MeshMaterial2d(mat_handle.clone()),
        Transform::default(),
    ));
    commands.insert_resource(AgentOutlineMeshResource { mesh_handle: outline_handle });

    // Bioluminescent glow aura mesh (interpolated radial gradient)
    let mut glow_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    glow_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, -100.0]; 3]);
    glow_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0, 0.0, 0.0, 0.0]; 3]);
    let glow_handle = meshes.add(glow_mesh);

    commands.spawn((
        Mesh2d(glow_handle.clone()),
        MeshMaterial2d(mat_handle.clone()),
        Transform::default(),
    ));
    commands.insert_resource(AgentGlowMeshResource { mesh_handle: glow_handle });

    // Fading swarm motion trail mesh
    let mut trail_mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    trail_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, -100.0]; 2]);
    trail_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0, 0.0, 0.0, 0.0]; 2]);
    let trail_handle = meshes.add(trail_mesh);

    commands.spawn((
        Mesh2d(trail_handle.clone()),
        MeshMaterial2d(mat_handle),
        Transform::default(),
    ));
    commands.insert_resource(AgentTrailMeshResource { mesh_handle: trail_handle });
}

pub fn update_agent_mesh_system(
    sim: Option<Res<SimWorld>>,
    gpu_driver: Option<Res<crate::sim::GpuDriverResource>>,
    camera_query: Query<(&Camera, &GlobalTransform, &Projection), With<crate::camera::MainCamera>>,
    res: Option<Res<AgentMeshResource>>,
    outline_res: Option<Res<AgentOutlineMeshResource>>,
    glow_res: Option<Res<AgentGlowMeshResource>>,
    trail_res: Option<Res<AgentTrailMeshResource>>,
    mut tracker: Option<ResMut<AgentTrailsTracker>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (Some(sim), Some(res)) = (sim, res) else { return };

    if sim.active_engine == crate::sim::ActiveEngine::Gpu {
        if let Some(ref gpu) = gpu_driver {
            if let Some(ref driver) = gpu.driver {
                if driver.is_initialized() {
                    let (cam_pos, cam_size) = if let Ok((camera, transform, proj)) = camera_query.single() {
                        let pos_bevy = transform.translation().truncate();
                        let pos_sim = bevy_to_sim_coord(pos_bevy, sim.world_height as f32);
                        let mut size = camera.logical_viewport_size().unwrap_or(Vec2::new(sim.world_width as f32, sim.world_height as f32));
                        if let Projection::Orthographic(ref ortho) = *proj {
                            size *= ortho.scale;
                        }
                        ([pos_sim.x, pos_sim.y], [size.x, size.y])
                    } else {
                        ([(sim.world_width * 0.5) as f32, (sim.world_height * 0.5) as f32], [sim.world_width as f32, sim.world_height as f32])
                    };

                    let cull_params = crate::gpu::types::GpuSimParams {
                        world_size: [sim.world_width as f32, sim.world_height as f32],
                        camera_pos: cam_pos,
                        camera_size: cam_size,
                        agent_count: sim.gpu_population,
                        max_capacity: sim.world.max_cap as u32,
                        max_agents: driver.max_agents,
                        ..Default::default()
                    };

                    let visible_count = driver.dispatch_culling(&cull_params) as usize;
                    let instances = if visible_count > 0 {
                        driver.readback_dart_instances(visible_count)
                    } else {
                        Vec::new()
                    };

                    if let Some(mut mesh) = meshes.get_mut(&res.mesh_handle) {
                        let mut positions = Vec::new();
                        let mut colors = Vec::new();
                        generate_dart_mesh_from_instances(&instances, sim.world_height as f32, &mut positions, &mut colors);
                        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                    }

                    if let Some(ref outline_res) = outline_res {
                        if let Some(mut mesh) = meshes.get_mut(&outline_res.mesh_handle) {
                            let mut positions = Vec::new();
                            let mut colors = Vec::new();
                            generate_outline_mesh_from_instances(&instances, sim.world_height as f32, &mut positions, &mut colors);
                            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                        }
                    }

                    if let Some(ref glow_res) = glow_res {
                        if let Some(mut mesh) = meshes.get_mut(&glow_res.mesh_handle) {
                            let mut positions = Vec::new();
                            let mut colors = Vec::new();
                            generate_glow_mesh_from_instances(&instances, sim.world_height as f32, &mut positions, &mut colors);
                            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                        }
                    }

                    if let Some(ref mut tracker) = tracker {
                        let mut current_ids = std::collections::HashSet::new();
                        for inst in &instances {
                            let agent_id = inst.pad1[0] & 0x00FFFFFF;
                            if agent_id > 0 {
                                current_ids.insert(agent_id);
                                let sim_pos = Vec2::new(inst.pos_angle[0], inst.pos_angle[1]);
                                let bevy_pos = sim_to_bevy_coord(sim_pos, sim.world_height as f32);
                                let queue = tracker.trails.entry(agent_id).or_default();
                                if queue.back().map_or(true, |last: &Vec2| last.distance_squared(bevy_pos) > 0.25) {
                                    queue.push_back(bevy_pos);
                                    if queue.len() > 9 {
                                        queue.pop_front();
                                    }
                                }
                            }
                        }
                        if sim.world.tick.saturating_sub(tracker.last_clean_tick) >= 30 {
                            tracker.trails.retain(|id, _| current_ids.contains(id));
                            tracker.last_clean_tick = sim.world.tick;
                        }
                    }

                    if let (Some(ref trail_res), Some(ref tracker)) = (trail_res.as_ref(), tracker.as_ref()) {
                        if let Some(mut mesh) = meshes.get_mut(&trail_res.mesh_handle) {
                            let mut positions = Vec::new();
                            let mut colors = Vec::new();
                            generate_swarm_trails_from_tracker(tracker, &instances, sim.world_width as f32, sim.world_height as f32, &mut positions, &mut colors);
                            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
                        }
                    }
                    return;
                }
            }
        }
        return;
    }

    let (gpu_states, _, _, _, _) = crate::gpu::bridge::sync_rust_to_gpu(&sim);

    if let Some(mut mesh) = meshes.get_mut(&res.mesh_handle) {
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        generate_dart_mesh_from_gpu_states(&gpu_states, sim.world_height as f32, &mut positions, &mut colors);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    }

    if let Some(ref outline_res) = outline_res {
        if let Some(mut mesh) = meshes.get_mut(&outline_res.mesh_handle) {
            let mut positions = Vec::new();
            let mut colors = Vec::new();
            generate_outline_mesh_from_gpu_states(&gpu_states, sim.world_height as f32, &mut positions, &mut colors);
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }
    }

    if let Some(ref glow_res) = glow_res {
        if let Some(mut mesh) = meshes.get_mut(&glow_res.mesh_handle) {
            let mut positions = Vec::new();
            let mut colors = Vec::new();
            generate_glow_mesh_from_gpu_states(&gpu_states, sim.world_height as f32, &mut positions, &mut colors);
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }
    }

    if let Some(ref trail_res) = trail_res {
        if let Some(mut mesh) = meshes.get_mut(&trail_res.mesh_handle) {
            let mut positions = Vec::new();
            let mut colors = Vec::new();
            generate_swarm_trails_from_sim(&sim, &mut positions, &mut colors);
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        }
    }
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
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
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
    let (Some(particles), Some(res)) = (particles, res) else { return };
    let Some(mut mesh) = meshes.get_mut(&res.mesh_handle) else { return };

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
    let (Some(mut sim), Some(mut res)) = (sim, res) else { return };
    let h = sim.world_height as f32;

    // Drain spark events so each event is consumed exactly once
    let events: Vec<_> = sim.world.spark_events.drain(..).collect();
    for spark in events {
        let spark_pos = sim_to_bevy_coord(Vec2::new(spark.x as f32, spark.y as f32), h);
        let color = if spark.color_idx == 8 {
            Color::srgb(1.0, 0.46, 0.40)
        } else if spark.color_idx == 9 {
            Color::srgb(0.61, 0.91, 0.84)
        } else if spark.color_idx == 10 {
            Color::srgb(0.87, 0.74, 0.47)
        } else if spark.color_idx == 11 {
            Color::srgb(0.88, 0.47, 0.48)
        } else {
            lineage_color(spark.color_idx)
        };

        let count = spark.count.clamp(1, 15);
        for _ in 0..count {
            let vx = res.next_f32(-2.4, 2.4);
            let vy = res.next_f32(-2.4, 2.4);
            let life = res.next_f32(36.0, 86.0);

            res.particles.push(SparkParticle {
                x: spark_pos.x,
                y: spark_pos.y,
                vx,
                vy,
                life,
                max_life: 86.0,
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

pub fn render_sim_gizmos_system(
    sim: Option<Res<SimWorld>>,
    tracker: Option<Res<AgentTrailsTracker>>,
    mut gizmos: Gizmos,
) {
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

            // Selected specimen history trail (from tracker in GPU mode, or world agent in Rust mode)
            let trail_color = c.with_alpha(0.45);
            if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                if let Some(ref tr) = tracker {
                    if let Some(queue) = tr.trails.get(&a.id) {
                        let q_vec: Vec<Vec2> = queue.iter().copied().collect();
                        for i in 0..q_vec.len().saturating_sub(1) {
                            let p1 = q_vec[i];
                            let p2 = q_vec[i + 1];
                            if (p1.x - p2.x).abs() < w * 0.5 && (p1.y - p2.y).abs() < h * 0.5 {
                                gizmos.line_2d(p1, p2, trail_color);
                            }
                        }
                    }
                }
            } else if a.trail_count > 1 {
                let trail_len = (a.trail_count as usize).min(9);
                for i in 0..trail_len.saturating_sub(1) {
                    let p1 = sim_to_bevy_coord(Vec2::new(a.trail_x[i] as f32, a.trail_y[i] as f32), h);
                    let p2 = sim_to_bevy_coord(Vec2::new(a.trail_x[i + 1] as f32, a.trail_y[i + 1] as f32), h);
                    if (p1.x - p2.x).abs() < w * 0.5 && (p1.y - p2.y).abs() < h * 0.5 {
                        gizmos.line_2d(p1, p2, trail_color);
                    }
                }
            }
        }
    }

    // Eclipse Visual Overlay
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
    mut gpu_driver: Option<ResMut<crate::sim::GpuDriverResource>>,
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
                if let Some(ref gpu) = gpu_driver {
                    if let Some(ref driver) = gpu.driver {
                        if driver.is_initialized() {
                            let cx = (sim_pos.x / 12.0).floor() as i32;
                            let cy = (sim_pos.y / 12.0).floor() as i32;
                            driver.sync_soil_cells_gpu(&sim.world.soil, cx, cy, 3);
                        }
                    }
                }
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
                if let Some(ref gpu) = gpu_driver {
                    if let Some(ref driver) = gpu.driver {
                        if driver.is_initialized() {
                            let cx = (sim_pos.x / 12.0).floor() as i32;
                            let cy = (sim_pos.y / 12.0).floor() as i32;
                            driver.sync_soil_cells_gpu(&sim.world.soil, cx, cy, 3);
                        }
                    }
                }
            }
        }
        crate::ui::ActiveTool::SeedLife => {
            if mouse_buttons.just_pressed(MouseButton::Left) || (sim.world.tick % 8 == 0) {
                sim.world.seed_life_at(sim_pos.x as f64, sim_pos.y as f64);
                if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                    if let Some(ref mut gpu) = gpu_driver {
                        if let Some(ref driver) = gpu.driver {
                            if driver.is_initialized() {
                                driver.seed_agents_gpu(&[(sim_pos.x, sim_pos.y)]);
                            }
                        }
                    }
                }
            }
        }
        crate::ui::ActiveTool::Extinguish => {
            sim.world.extinguish_at(sim_pos.x as f64, sim_pos.y as f64, 23.0);
            if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                sim.pending_tool = Some(crate::sim::PendingTool {
                    tool_type: 3,
                    tool_pos: [sim_pos.x, sim_pos.y],
                    tool_radius: 35.0,
                });
            }
        }
        crate::ui::ActiveTool::Eclipse => {
            if mouse_buttons.just_pressed(MouseButton::Left) {
                sim.world.eclipse = 120;
            }
        }
    }
}

pub struct ClankRenderPlugin;

impl Plugin for ClankRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleSystemResource>()
            .init_resource::<AgentTrailsTracker>()
            .add_systems(
                Startup,
                (
                    setup_soil_rendering,
                    setup_agent_rendering,
                    setup_particle_rendering,
                ),
            )
            .add_systems(
                Update,
                (
                    update_soil_texture_system,
                    update_agent_mesh_system,
                    update_particles_system,
                    update_particle_mesh_system,
                    render_sim_gizmos_system,
                    agent_picking_system,
                ),
            );
    }
}
