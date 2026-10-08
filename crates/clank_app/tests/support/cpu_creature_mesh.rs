//! Frozen copy of the CPU creature meshes (rendering.rs, 2026-10-06) that the GPU creature
//! renderer replaces. Tests draw these as the pixel reference; do not edit to make tests pass.
#![allow(dead_code)]

use bevy::prelude::{Color, Mat2, Vec2};

#[inline]
pub fn sim_to_bevy_coord(sim_pos: Vec2, world_height: f32) -> Vec2 {
    Vec2::new(sim_pos.x, world_height - sim_pos.y)
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

pub fn generate_dart_mesh_from_instances(
    instances: &[clank_app::gpu::types::GpuDartInstance],
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(instances.len() * 6);
    colors.reserve(instances.len() * 6);

    for inst in instances {
        let Some((radius, color, _is_attacking, _has_birth)) =
            unpack_visual_cache(inst.vis_data[1], inst.vis_data[0])
        else {
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
    instances: &[clank_app::gpu::types::GpuDartInstance],
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

pub fn generate_glow_mesh_from_instances(
    instances: &[clank_app::gpu::types::GpuDartInstance],
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

/// Appends one creature's trail (Bevy coordinates, oldest first) as line segments, skipping
/// segments that wrap around the torus. Shared by both engines so their trails look identical.
fn push_trail_segments(
    pts: &[Vec2],
    c: bevy::color::Srgba,
    world_width: f32,
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    let tc = pts.len();
    if tc < 2 {
        return;
    }
    for i in 0..(tc - 1) {
        let (p1, p2) = (pts[i], pts[i + 1]);
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

fn finish_trail_mesh(positions: &mut Vec<[f32; 3]>, colors: &mut Vec<[f32; 4]>) {
    if positions.is_empty() {
        positions.push([0.0, 0.0, -100.0]);
        positions.push([0.0, 0.0, -100.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
        colors.push([0.0, 0.0, 0.0, 0.0]);
    }
}

/// Trail mesh for the GPU engine: `trails[i]` is the per-tick trail of `instances[i]`
/// (both read back after the same `dispatch_culling`).
pub fn generate_swarm_trails_from_gpu(
    instances: &[clank_app::gpu::types::GpuDartInstance],
    trails: &[clank_app::gpu::types::GpuAgentTrail],
    world_width: f32,
    world_height: f32,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    positions.clear();
    colors.clear();
    positions.reserve(instances.len() * 16);
    colors.reserve(instances.len() * 16);

    let mut pts = [Vec2::ZERO; clank_app::gpu::types::TRAIL_LEN];
    for (inst, trail) in instances.iter().zip(trails) {
        let Some(vis) = unpack_visual_cache_detailed(inst.vis_data[1], inst.vis_data[0]) else {
            continue;
        };
        let tc = (trail.count as usize).min(clank_app::gpu::types::TRAIL_LEN);
        for i in 0..tc {
            pts[i] = sim_to_bevy_coord(Vec2::new(trail.pts[i][0], trail.pts[i][1]), world_height);
        }
        push_trail_segments(
            &pts[..tc],
            vis.color.to_srgba(),
            world_width,
            world_height,
            positions,
            colors,
        );
    }
    finish_trail_mesh(positions, colors);
}
