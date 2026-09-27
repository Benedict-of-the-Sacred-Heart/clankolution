use crate::math::{self, Prng};

pub struct SoilGrid {
    pub w: f64,
    pub h: f64,
    pub cols: usize,
    pub rows: usize,
    pub grid_size: usize,
    pub cell_w: f64,
    pub cell_h: f64,
    pub inv_cell_w: f64,
    pub inv_cell_h: f64,
    pub food: Vec<f32>,
    pub taint: Vec<f32>,
    pub scent: Vec<f32>,
    pub bloom: Vec<f64>,
}

impl SoilGrid {
    pub fn new() -> Self {
        let cols = 75;
        let rows = 50;
        let w = 900.0;
        let h = 600.0;
        let grid_size = cols * rows;
        let cell_w = w / (cols as f64);
        let cell_h = h / (rows as f64);
        let inv_cell_w = 1.0 / cell_w;
        let inv_cell_h = 1.0 / cell_h;
        let mut grid = Self {
            w,
            h,
            cols,
            rows,
            grid_size,
            cell_w,
            cell_h,
            inv_cell_w,
            inv_cell_h,
            food: vec![0.0; grid_size],
            taint: vec![0.0; grid_size],
            scent: vec![0.0; grid_size],
            bloom: vec![0.0; grid_size],
        };
        grid.init_bloom();
        grid
    }

    pub fn resize(&mut self, w: f64, h: f64, cols: usize, rows: usize) {
        self.w = w;
        self.h = h;
        self.cols = cols;
        self.rows = rows;
        self.grid_size = cols * rows;
        self.cell_w = w / (cols as f64);
        self.cell_h = h / (rows as f64);
        self.inv_cell_w = 1.0 / self.cell_w;
        self.inv_cell_h = 1.0 / self.cell_h;
        self.food.resize(self.grid_size, 0.0);
        self.taint.resize(self.grid_size, 0.0);
        self.scent.resize(self.grid_size, 0.0);
        self.bloom.resize(self.grid_size, 0.0);
        self.init_bloom();
    }

    pub fn init_bloom(&mut self) {
        for y in 0..self.rows {
            let y_f = y as f64;
            for x in 0..self.cols {
                let x_f = x as f64;
                let i = y * self.cols + x;
                self.bloom[i] = 0.0008 + 0.0022 * (0.5 + 0.5 * math::sin(x_f * 0.13 + math::sin(y_f * 0.19)) * math::cos(y_f * 0.11));
            }
        }
    }

    #[inline(always)]
    pub fn idx(&self, x: f64, y: f64) -> usize {
        let mut cx = (math::wrap(x, self.w) * self.inv_cell_w) as usize;
        if cx >= self.cols {
            cx = self.cols - 1;
        }
        let mut cy = (math::wrap(y, self.h) * self.inv_cell_h) as usize;
        if cy >= self.rows {
            cy = self.rows - 1;
        }
        cy * self.cols + cx
    }

    #[inline(always)]
    pub fn idx_fast(&self, x: f64, y: f64) -> usize {
        let mut cx = (x * self.inv_cell_w) as usize;
        if cx >= self.cols {
            cx = self.cols - 1;
        }
        let mut cy = (y * self.inv_cell_h) as usize;
        if cy >= self.rows {
            cy = self.rows - 1;
        }
        cy * self.cols + cx
    }

    #[inline(always)]
    pub fn sample(arr: &[f32], idx: usize) -> f64 {
        arr[idx] as f64
    }

    #[inline(always)]
    pub fn step_soil(&mut self, prng: &mut Prng, growth: f64) {
        let renewal = growth;
        let spawn_threshold = 0.00013 * renewal;
        let food_slice = self.food.as_mut_slice();
        let taint_slice = self.taint.as_mut_slice();
        let scent_slice = self.scent.as_mut_slice();
        let bloom_slice = self.bloom.as_slice();

        for k in 0..self.grid_size {
            let f_ptr = unsafe { food_slice.get_unchecked_mut(k) };
            let t_ptr = unsafe { taint_slice.get_unchecked_mut(k) };
            let s_ptr = unsafe { scent_slice.get_unchecked_mut(k) };
            let b_val = unsafe { *bloom_slice.get_unchecked(k) };

            let mut f = (*f_ptr) as f64;
            let t = *t_ptr;
            let s = *s_ptr;

            // Spatial bloom lookup
            f += renewal * b_val * (1.0 - f / 1.7);
            if prng.next_f64() < spawn_threshold {
                f += prng.rand(0.15, 0.6);
            }
            *f_ptr = if f < 0.0 { 0.0 } else if f > 2.5 { 2.5 } else { f as f32 };

            if t > 0.0 {
                let dec = (t as f64) * 0.994 - 0.0001;
                *t_ptr = if dec > 0.0 { dec as f32 } else { 0.0 };
            }
            if s > 0.0 {
                *s_ptr = ((s as f64) * 0.954) as f32;
            }
        }
    }

    pub fn deposit(&mut self, target: usize, x: f64, y: f64, value: f64, radius: i32) {
        let arr = match target {
            0 => &mut self.food,
            1 => &mut self.taint,
            _ => &mut self.scent,
        };
        let cx = (math::wrap(x, self.w) * self.inv_cell_w).floor() as i32;
        let cy = (math::wrap(y, self.h) * self.inv_cell_h).floor() as i32;
        let rows_i = self.rows as i32;
        let cols_i = self.cols as i32;
        let cx = cx.rem_euclid(cols_i);
        let cy = cy.rem_euclid(rows_i);

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
                let idx = cell_y * self.cols + cell_x;
                let sum = (arr[idx] as f64) + value / (1.0 + rr * 0.8);
                arr[idx] = math::clamp(sum, 0.0, 3.0) as f32;
            }
        }
    }
}
