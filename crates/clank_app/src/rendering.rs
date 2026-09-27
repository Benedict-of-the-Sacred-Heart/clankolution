use bevy::prelude::*;
use clank_core::soil::SoilGrid;
use crate::sim::SimWorld;

pub const PALETTE: [Color; 8] = [
    Color::srgb(0.369, 0.918, 0.831), // #5eead4
    Color::srgb(0.376, 0.647, 0.980), // #60a5fa
    Color::srgb(0.753, 0.518, 0.988), // #c084fc
    Color::srgb(0.957, 0.447, 0.714), // #f472b6
    Color::srgb(0.984, 0.573, 0.235), // #fb923c
    Color::srgb(0.980, 0.800, 0.082), // #facc15
    Color::srgb(0.290, 0.871, 0.502), // #4ade80
    Color::srgb(0.176, 0.831, 0.749), // #2dd4bf
];

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
    PALETTE[(root as usize) % PALETTE.len()]
}

#[inline]
pub fn agent_body_color(root: u32, energy: f64) -> Color {
    let base = lineage_color(root).to_srgba();
    let alpha = (0.55 + (energy / 160.0)).clamp(0.55, 1.0) as f32;
    Color::srgba(base.red, base.green, base.blue, alpha)
}

#[inline]
pub fn agent_triangle_vertices(pos: Vec2, angle: f32, bulk_trait: f32) -> (Vec2, Vec2, Vec2) {
    let r = 2.3 + bulk_trait * 4.5;
    // Rotation by -angle because Bevy's +Y is upwards whereas web canvas +Y is downwards
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
                trail,
            }
        })
        .collect()
}

pub fn generate_soil_rgba(soil: &SoilGrid, out_buf: &mut [u8]) {
    let inv18 = 1.0 / 18.0;
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

pub fn render_sim_gizmos_system(sim: Option<Res<SimWorld>>, mut gizmos: Gizmos) {
    let Some(sim) = sim else { return };

    let w = sim.world_width as f32;
    let h = sim.world_height as f32;

    // Field boundary
    gizmos.rect_2d(
        Vec2::new(w * 0.5, h * 0.5),
        Vec2::new(w, h),
        Color::srgb(0.12, 0.22, 0.25),
    );

    let items = extract_agent_render_data(&sim);
    for item in items {
        // Trails
        if item.trail.len() > 1 {
            let trail_color = item.color.with_alpha(0.3);
            for window in item.trail.windows(2) {
                // Ignore wrapped coordinates across toroidal boundary
                if (window[0].x - window[1].x).abs() < w * 0.5
                    && (window[0].y - window[1].y).abs() < h * 0.5
                {
                    gizmos.line_2d(window[0], window[1], trail_color);
                }
            }
        }

        // Body triangle
        let (v0, v1, v2) = agent_triangle_vertices(item.bevy_pos, item.angle, (item.radius - 2.3) / 4.5);
        let border_color = if item.attack > 0.5 {
            Color::srgb(1.0, 0.33, 0.31)
        } else {
            item.color
        };
        gizmos.line_2d(v0, v1, border_color);
        gizmos.line_2d(v1, v2, border_color);
        gizmos.line_2d(v2, v0, border_color);

        // Core dot
        gizmos.circle_2d(item.bevy_pos, item.radius * 0.3, item.color);

        // Birth halo
        if item.birth > 0 {
            let halo_color = item.color.with_alpha((item.birth as f32) / 120.0);
            gizmos.circle_2d(item.bevy_pos, item.radius + 3.0, halo_color);
        }

        // Selection reticle
        if item.is_selected {
            gizmos.circle_2d(item.bevy_pos, item.radius + 7.0, Color::srgb(1.0, 0.95, 0.84));
        }
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
    mut egui_contexts: Option<bevy_egui::EguiContexts>,
) {
    let (Some(mouse_buttons), Some(mut sim)) = (mouse_buttons, sim) else { return };
    if !mouse_buttons.just_pressed(MouseButton::Left) {
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
    sim.selected_agent_id = find_agent_at_position(&sim, sim_pos, 6.0);
}

pub struct ClankRenderPlugin;

impl Plugin for ClankRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (render_sim_gizmos_system, agent_picking_system));
    }
}

