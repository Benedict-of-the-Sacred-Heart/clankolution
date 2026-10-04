//! GPU Soil Simulation Pipeline & Direct Texture Generation
//!
//! Manages the persistent 60 KB atomic soil storage buffer (3,750 cells)
//! and direct rasterization into dual `rgba16float` textures:
//! 1. `soil_data`: Raw physics scalars [f, t, s, 1.0] for zero-ALU bilinear feeler sensing.
//! 2. `soil_display`: Colormapped RGBA display for direct Bevy SoilSprite presentation.

use super::types::GpuSoilCell;

pub const GRID_WIDTH: u32 = 75;
pub const GRID_HEIGHT: u32 = 50;
pub const CELL_COUNT: usize = (GRID_WIDTH * GRID_HEIGHT) as usize; // 3,750 cells
pub const CELL_SIZE: f32 = 12.0;
pub const ARENA_WIDTH: f32 = 900.0;
pub const ARENA_HEIGHT: f32 = 600.0;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilParams {
    pub renewal: f32,
    pub width: u32,
    pub height: u32,
    pub pad: u32,
}

pub struct GpuSoilPipeline;

impl GpuSoilPipeline {
    /// Converts raw continuous physics values to fixed-point integer millijoules/milligrams.
    #[inline]
    pub fn from_physics(food: f32, taint: f32, scent: f32) -> GpuSoilCell {
        GpuSoilCell {
            food_milli: (food * 1000.0).round() as i32,
            taint_milli: (taint * 1000.0).round() as i32,
            scent_milli: (scent * 1000.0).round() as i32,
            pad: 0,
        }
    }

    /// Converts fixed-point atomic cell values to continuous physics floats.
    #[inline]
    pub fn to_physics(cell: &GpuSoilCell) -> (f32, f32, f32) {
        (
            cell.food_milli as f32 * 0.001,
            cell.taint_milli as f32 * 0.001,
            cell.scent_milli as f32 * 0.001,
        )
    }

    /// Maps continuous arena coordinates [x, y] to clamped grid coordinates [cx, cy].
    /// Clamping to [0..GRID_WIDTH-1, 0..GRID_HEIGHT-1] ensures boundary coordinates
    /// (e.g. at the 900.0, 600.0 seam) never access out of bounds.
    #[inline]
    pub fn pos_to_cell(pos: [f32; 2]) -> (u32, u32) {
        let cx = ((pos[0] / CELL_SIZE).max(0.0).floor() as u32).min(GRID_WIDTH - 1);
        let cy = ((pos[1] / CELL_SIZE).max(0.0).floor() as u32).min(GRID_HEIGHT - 1);
        (cx, cy)
    }

    /// Converts grid coordinates [cx, cy] to 1D buffer index.
    #[inline]
    pub fn cell_index(cx: u32, cy: u32) -> usize {
        (cy * GRID_WIDTH + cx) as usize
    }

    /// Reference simulation step matching `soil_step.wgsl` kernel math.
    pub fn step_soil(cells: &mut [GpuSoilCell], bloom_table: &[f32], renewal: f32) {
        for (i, cell) in cells.iter_mut().enumerate() {
            let (mut f, mut t, mut s) = Self::to_physics(cell);
            let bloom = bloom_table.get(i).copied().unwrap_or(1.0);

            // Environmental renewal & decay
            f += renewal * bloom * (1.0 - f / 1.7);
            f = f.clamp(0.0, 2.5);

            if t > 0.0 {
                t = (t * 0.994 - 0.0001).max(0.0);
            }
            if s > 0.0 {
                s *= 0.954;
            }

            *cell = Self::from_physics(f, t, s);
        }
    }

    /// Evaluates raw physics output for `soil_data` texture (`rgba16float`).
    #[inline]
    pub fn evaluate_soil_data(food: f32, taint: f32, scent: f32) -> [f32; 4] {
        [food, taint, scent, 1.0]
    }

    /// Evaluates display colormap for `soil_display` texture (`rgba16float`).
    #[inline]
    pub fn evaluate_soil_color(f: f32, t: f32, s: f32) -> [f32; 4] {
        let inv18 = 1.0 / 1.8;
        let f_val = (f * inv18).min(1.0);
        let mut r = 9.5 + f_val * 42.0;
        let mut g = 22.5 + f_val * 61.0;
        let mut b = 25.5 + f_val * 44.0;

        if t > 0.0 {
            let tc = t.min(1.0);
            r += tc * 98.0;
            g -= tc * 13.0;
            b += tc * 23.0;
        }

        if s > 0.0 {
            let sc = s.min(1.0);
            r += sc * 27.0;
            g += sc * 20.0;
            b += sc * 33.0;
        }

        [
            (r / 255.0).clamp(0.0, 1.0),
            (g / 255.0).clamp(0.0, 1.0),
            (b / 255.0).clamp(0.0, 1.0),
            1.0,
        ]
    }
}
