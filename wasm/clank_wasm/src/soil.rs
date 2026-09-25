use crate::math::{self, Prng};

pub const W: f64 = 900.0;
pub const H: f64 = 600.0;
pub const COLS: usize = 75;
pub const ROWS: usize = 50;
pub const GRID_SIZE: usize = COLS * ROWS; // 3750

pub const CELL_W: f64 = W / (COLS as f64); // 12.0
pub const CELL_H: f64 = H / (ROWS as f64); // 12.0
pub const INV_CELL_W: f64 = 1.0 / CELL_W;
pub const INV_CELL_H: f64 = 1.0 / CELL_H;

pub struct SoilGrid {
    pub food: [f32; GRID_SIZE],
    pub taint: [f32; GRID_SIZE],
    pub scent: [f32; GRID_SIZE],
    pub bloom: [f64; GRID_SIZE],
}

impl SoilGrid {
    pub fn new() -> Self {
        let mut grid = Self {
            food: [0.0; GRID_SIZE],
            taint: [0.0; GRID_SIZE],
            scent: [0.0; GRID_SIZE],
            bloom: [0.0; GRID_SIZE],
        };
        grid.init_bloom();
        grid
    }

    pub fn init_bloom(&mut self) {
        for y in 0..ROWS {
            let y_f = y as f64;
            for x in 0..COLS {
                let x_f = x as f64;
                let i = y * COLS + x;
                self.bloom[i] = 0.0008 + 0.0022 * (0.5 + 0.5 * math::sin(x_f * 0.13 + math::sin(y_f * 0.19)) * math::cos(y_f * 0.11));
            }
        }
    }

    #[inline(always)]
    pub fn idx(&self, x: f64, y: f64) -> usize {
        let mut cx = (math::wrap(x, W) * INV_CELL_W) as usize;
        if cx >= COLS {
            cx = COLS - 1;
        }
        let mut cy = (math::wrap(y, H) * INV_CELL_H) as usize;
        if cy >= ROWS {
            cy = ROWS - 1;
        }
        cy * COLS + cx
    }

    #[inline(always)]
    pub fn sample(arr: &[f32; GRID_SIZE], idx: usize) -> f64 {
        arr[idx] as f64
    }

    #[inline(always)]
    pub fn step_soil(&mut self, prng: &mut Prng, growth: f64) {
        let renewal = growth;
        let spawn_threshold = 0.00013 * renewal;
        for k in 0..GRID_SIZE {
            let mut f = self.food[k] as f64;
            let t = self.taint[k];
            let s = self.scent[k];

            // Spatial bloom lookup
            f += renewal * self.bloom[k] * (1.0 - f / 1.7);
            if prng.next_f64() < spawn_threshold {
                f += prng.rand(0.15, 0.6);
            }
            self.food[k] = if f < 0.0 { 0.0 } else if f > 2.5 { 2.5 } else { f as f32 };

            if t > 0.0 {
                let dec = (t as f64) * 0.994 - 0.0001;
                self.taint[k] = if dec > 0.0 { dec as f32 } else { 0.0 };
            }
            if s > 0.0 {
                self.scent[k] = ((s as f64) * 0.954) as f32;
            }
        }
    }

    pub fn deposit(arr: &mut [f32; GRID_SIZE], x: f64, y: f64, value: f64, radius: i32) {
        let cx = (math::wrap(x, W) * INV_CELL_W).floor() as i32;
        let cy = (math::wrap(y, H) * INV_CELL_H).floor() as i32;
        let rows_i = ROWS as i32;
        let cols_i = COLS as i32;

        if radius == 1 {
            let v18 = value / 1.8;
            let c0 = (cy * cols_i + cx) as usize;
            arr[c0] = math::clamp((arr[c0] as f64) + value, 0.0, 3.0) as f32;
            let c_up = (((cy - 1).rem_euclid(rows_i)) * cols_i + cx) as usize;
            arr[c_up] = math::clamp((arr[c_up] as f64) + v18, 0.0, 3.0) as f32;
            let c_down = (((cy + 1).rem_euclid(rows_i)) * cols_i + cx) as usize;
            arr[c_down] = math::clamp((arr[c_down] as f64) + v18, 0.0, 3.0) as f32;
            let c_left = (cy * cols_i + ((cx - 1).rem_euclid(cols_i))) as usize;
            arr[c_left] = math::clamp((arr[c_left] as f64) + v18, 0.0, 3.0) as f32;
            let c_right = (cy * cols_i + ((cx + 1).rem_euclid(cols_i))) as usize;
            arr[c_right] = math::clamp((arr[c_right] as f64) + v18, 0.0, 3.0) as f32;
            return;
        }

        let r_max = (radius * radius) as f64 + 0.5;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let rr = (dx * dx + dy * dy) as f64;
                if rr > r_max {
                    continue;
                }
                let cell_y = (cy + dy).rem_euclid(rows_i) as usize;
                let cell_x = (cx + dx).rem_euclid(cols_i) as usize;
                let idx = cell_y * COLS + cell_x;
                let sum = (arr[idx] as f64) + value / (1.0 + rr * 0.8);
                arr[idx] = math::clamp(sum, 0.0, 3.0) as f32;
            }
        }
    }
}
