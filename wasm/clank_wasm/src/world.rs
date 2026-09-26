use crate::math::{self, Prng};
use crate::agent::{AgentData, GENES};
use crate::soil::SoilGrid;

pub const TAU: f64 = std::f64::consts::TAU;
pub const SCALE_H: f64 = 0.61 / 127.0;
pub const SCALE_O: f64 = 0.66 / 127.0;

pub struct NearResult {
    pub best_idx: Option<usize>,
    pub best_dx: f64,
    pub best_dy: f64,
    pub best_d: f64,
    pub density: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SimEvent {
    pub tick: u32,
    pub event_type: u32, // 1 = SPORES, 2 = GENERATION, 3 = KILLS
    pub p1: u32,         // gen (type 2) or root (type 3)
    pub p2: u32,         // root (type 2) or kills (type 3)
}

pub struct World {
    pub prng: Prng,
    pub soil: SoilGrid,
    pub agents: Vec<AgentData>,
    pub pos_x: Vec<f64>,
    pub pos_y: Vec<f64>,
    pub events: Vec<SimEvent>,
    pub tick: u32,
    pub kills: u32,
    pub births: u32,
    pub roots: u32,
    pub next_id: u32,
    pub eclipse: u32,
    pub mutation: f64,
    pub growth: f64,
    pub hostility: f64,
    pub max_cap: usize,
    pub w: f64,
    pub h: f64,
    pub grid_nx: usize,
    pub grid_ny: usize,
    pub cell_w: f64,
    pub cell_h: f64,
    pub grid_head: Vec<i32>,
    pub grid_prev: Vec<i32>,
    pub grid_next: Vec<i32>,
    pub agent_cell: Vec<usize>,
    pub neighbor_cells: Vec<usize>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        let mut w = Self {
            prng: Prng::new(seed),
            soil: SoilGrid::new(),
            agents: Vec::with_capacity(340),
            pos_x: Vec::with_capacity(340),
            pos_y: Vec::with_capacity(340),
            events: Vec::with_capacity(128),
            tick: 0,
            kills: 0,
            births: 0,
            roots: 1,
            next_id: 1,
            eclipse: 0,
            mutation: 16.0,
            growth: 100.0,
            hostility: 100.0,
            max_cap: 340,
            w: 900.0,
            h: 600.0,
            grid_nx: 9,
            grid_ny: 6,
            cell_w: 100.0,
            cell_h: 100.0,
            grid_head: Vec::new(),
            grid_prev: Vec::new(),
            grid_next: Vec::new(),
            agent_cell: Vec::new(),
            neighbor_cells: Vec::new(),
        };
        w.rebuild_grid_layout(900.0, 600.0);
        // Initial startup draws numbers during initial resize() -> setupGrid(false)
        for _ in 0..w.soil.grid_size {
            w.prng.rand(0.15, 0.5);
        }
        w.reset();
        w
    }

    pub fn set_max_capacity(&mut self, cap: u32) {
        self.max_cap = (cap as usize).clamp(15, 10_000);
        self.agents.reserve(self.max_cap);
        self.pos_x.reserve(self.max_cap);
        self.pos_y.reserve(self.max_cap);
        self.ensure_grid_capacity(self.max_cap);
    }

    pub fn resize(&mut self, w: f64, h: f64, cols: usize, rows: usize) {
        self.w = w;
        self.h = h;
        self.soil.resize(w, h, cols, rows);
        self.rebuild_grid_layout(w, h);
        if !self.agents.is_empty() {
            self.build_grid();
        }
    }

    pub fn rebuild_grid_layout(&mut self, w: f64, h: f64) {
        let nx = ((w / 100.0).floor() as usize).clamp(3, 100);
        let ny = ((h / 100.0).floor() as usize).clamp(3, 100);
        self.grid_nx = nx;
        self.grid_ny = ny;
        self.cell_w = w / (nx as f64);
        self.cell_h = h / (ny as f64);
        let num_cells = nx * ny;
        self.grid_head = vec![-1; num_cells];
        self.neighbor_cells = Vec::with_capacity(num_cells * 9);
        let nx_i = nx as i32;
        let ny_i = ny as i32;
        for cy in 0..ny_i {
            for cx in 0..nx_i {
                self.neighbor_cells.push((cy as usize) * nx + (cx as usize));
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let ncy = (cy + dy).rem_euclid(ny_i) as usize;
                        let ncx = (cx + dx).rem_euclid(nx_i) as usize;
                        self.neighbor_cells.push(ncy * nx + ncx);
                    }
                }
            }
        }
    }

    pub fn ensure_grid_capacity(&mut self, cap: usize) {
        if self.grid_prev.len() < cap {
            self.grid_prev.resize(cap, -1);
            self.grid_next.resize(cap, -1);
            self.agent_cell.resize(cap, 0);
        }
    }

    #[inline(always)]
    pub fn get_agent_cell(&self, x: f64, y: f64) -> usize {
        let mut cx = (math::wrap(x, self.w) / self.cell_w) as usize;
        if cx >= self.grid_nx { cx = self.grid_nx - 1; }
        let mut cy = (math::wrap(y, self.h) / self.cell_h) as usize;
        if cy >= self.grid_ny { cy = self.grid_ny - 1; }
        cy * self.grid_nx + cx
    }

    pub fn sync_pos_cache(&mut self) {
        let count = self.agents.len();
        self.pos_x.resize(count, 0.0);
        self.pos_y.resize(count, 0.0);
        for i in 0..count {
            self.pos_x[i] = self.agents[i].x;
            self.pos_y[i] = self.agents[i].y;
        }
    }

    pub fn build_grid(&mut self) {
        self.grid_head.fill(-1);
        let count = self.agents.len();
        self.ensure_grid_capacity(count);
        if self.pos_x.len() != count || self.pos_y.len() != count {
            self.sync_pos_cache();
        }
        for i in 0..count {
            let c = self.get_agent_cell(self.pos_x[i], self.pos_y[i]);
            self.agent_cell[i] = c;
            let old_head = self.grid_head[c];
            self.grid_next[i] = old_head;
            self.grid_prev[i] = -1;
            if old_head != -1 {
                self.grid_prev[old_head as usize] = i as i32;
            }
            self.grid_head[c] = i as i32;
        }
    }

    #[inline(always)]
    pub fn update_agent_cell(&mut self, i: usize, new_x: f64, new_y: f64) {
        let new_c = self.get_agent_cell(new_x, new_y);
        let cur_c = self.agent_cell[i];
        if new_c == cur_c { return; }
        let p = self.grid_prev[i];
        let n = self.grid_next[i];
        if p != -1 {
            self.grid_next[p as usize] = n;
        } else {
            self.grid_head[cur_c] = n;
        }
        if n != -1 {
            self.grid_prev[n as usize] = p;
        }
        self.agent_cell[i] = new_c;
        let h = self.grid_head[new_c];
        self.grid_next[i] = h;
        self.grid_prev[i] = -1;
        if h != -1 {
            self.grid_prev[h as usize] = i as i32;
        }
        self.grid_head[new_c] = i as i32;
    }

    #[inline(always)]
    pub fn insert_agent_cell(&mut self, i: usize, x: f64, y: f64) {
        self.ensure_grid_capacity(i + 1);
        let c = self.get_agent_cell(x, y);
        self.agent_cell[i] = c;
        let h = self.grid_head[c];
        self.grid_next[i] = h;
        self.grid_prev[i] = -1;
        if h != -1 {
            self.grid_prev[h as usize] = i as i32;
        }
        self.grid_head[c] = i as i32;
    }

    pub fn reset(&mut self) {
        self.agents.clear();
        self.pos_x.clear();
        self.pos_y.clear();
        self.events.clear();
        self.tick = 0;
        self.kills = 0;
        self.births = 0;
        self.roots = 1;
        self.next_id = 1;
        self.eclipse = 0;

        self.soil.food.fill(0.0);
        self.soil.taint.fill(0.0);
        self.soil.scent.fill(0.0);
        self.soil.init_bloom();

        // Initial background food noise: food[i] = rand(0.15, 0.5)
        for i in 0..self.soil.grid_size {
            self.soil.food[i] = self.prng.rand(0.15, 0.5) as f32;
        }

        // 12 initial food patches
        for _ in 0..12 {
            let cx = self.prng.rand(0.0, self.w);
            let cy = self.prng.rand(0.0, self.h);
            for _ in 0..16 {
                let px = cx + self.prng.rand(-80.0, 80.0);
                let py = cy + self.prng.rand(-80.0, 80.0);
                let val = self.prng.rand(0.2, 0.5);
                self.soil.deposit(0, px, py, val, 2);
            }
        }

        // 72 initial creatures
        for _ in 0..72 {
            let x = self.prng.rand(0.0, self.w);
            let y = self.prng.rand(0.0, self.h);
            self.create_agent(x, y, None, None);
        }
    }

    #[inline(always)]
    pub fn spark_prng(&mut self, n: u32) {
        for _ in 0..n {
            self.prng.rand(-2.4, 2.4);
            self.prng.rand(-2.4, 2.4);
            self.prng.rand(15.0, 36.0);
        }
    }

    pub fn create_agent(
        &mut self,
        x: f64,
        y: f64,
        parent: Option<&AgentData>,
        other: Option<&AgentData>,
    ) -> usize {
        let mut a = AgentData::default();
        a.id = self.next_id;
        self.next_id += 1;
        a.x = math::wrap(x, self.w);
        a.y = math::wrap(y, self.h);
        a.angle = self.prng.rand(0.0, TAU);
        a.vx = 0.0;
        a.vy = 0.0;
        a.energy = if parent.is_some() { 25.0 } else { self.prng.rand(34.0, 52.0) };
        a.age = 0;

        if let Some(p) = parent {
            let other_gen = other.map(|o| o.gen).unwrap_or(0);
            a.gen = p.gen.max(other_gen) + 1;
            a.root = p.root;
        } else {
            a.gen = 0;
            a.root = self.roots;
            self.roots += 1;
        }

        let mut_rate = self.mutation / 100.0;

        // Genome crossover and mutation
        if let Some(p) = parent {
            for i in 0..GENES {
                let mut base = if let Some(o) = other {
                    if self.prng.next_f64() < 0.48 {
                        o.genes[i]
                    } else {
                        p.genes[i]
                    }
                } else {
                    p.genes[i]
                };

                if self.prng.next_f64() < 0.11 {
                    let r1 = self.prng.next_f64();
                    let r2 = self.prng.next_f64();
                    let delta = ((r1 + r2 - 1.0) * 100.0 * mut_rate).round() as i32;
                    let val = (base as i32) + delta;
                    base = if val < -127 { -127 } else if val > 127 { 127 } else { val as i8 };
                }
                a.genes[i] = base;
            }
        } else {
            for i in 0..GENES {
                a.genes[i] = self.prng.rand(-58.0, 58.0).round() as i8;
            }
        }

        // Traits crossover and mutation
        if let Some(p) = parent {
            for i in 0..6 {
                let base = if let Some(o) = other {
                    if self.prng.next_f64() < 0.45 {
                        o.tr[i]
                    } else {
                        p.tr[i]
                    }
                } else {
                    p.tr[i]
                };
                let r1 = self.prng.next_f64();
                let r2 = self.prng.next_f64();
                let delta = (r1 + r2 - 1.0) * mut_rate * 0.6;
                let val = base + delta;
                a.tr[i] = if val < 0.03 { 0.03 } else if val > 0.98 { 0.98 } else { val };
            }
        } else {
            a.tr[0] = self.prng.rand(0.25, 0.8);
            a.tr[1] = self.prng.rand(0.3, 0.8);
            a.tr[2] = self.prng.rand(0.3, 0.8);
            a.tr[3] = self.prng.rand(0.25, 0.75);
            a.tr[4] = self.prng.rand(0.2, 0.8);
            a.tr[5] = self.prng.rand(0.2, 0.8);
        }

        let idx = self.agents.len();
        self.pos_x.push(a.x);
        self.pos_y.push(a.y);
        self.agents.push(a);
        self.births += 1;
        idx
    }

    #[inline(always)]
    pub fn near(&self, a_idx: usize) -> NearResult {
        let a = &self.agents[a_idx];
        let ax = a.x;
        let ay = a.y;
        let half_w = self.w * 0.5;
        let half_h = self.h * 0.5;
        let mut bd = 1e9f64;
        let mut cutoff = 1e9f64;
        let mut density = 0.0f64;
        let mut best_idx = None;
        let mut best_dx = 0.0f64;
        let mut best_dy = 0.0f64;

        let cell = self.agent_cell[a_idx];
        let neighbor_base = cell * 9;

        for k in 0..9 {
            let neighbor_cell = unsafe { *self.neighbor_cells.get_unchecked(neighbor_base + k) };
            let mut b_idx = unsafe { *self.grid_head.get_unchecked(neighbor_cell) };

            while b_idx != -1 {
                let b_u = b_idx as usize;
                if b_u != a_idx {
                    let bx = unsafe { *self.pos_x.get_unchecked(b_u) };
                    let mut dx = bx - ax;
                    if dx > half_w {
                        dx -= self.w;
                    } else if dx < -half_w {
                        dx += self.w;
                    }
                    let dx2 = dx * dx;
                    if dx2 < cutoff {
                        let by = unsafe { *self.pos_y.get_unchecked(b_u) };
                        let mut dy = by - ay;
                        if dy > half_h {
                            dy -= self.h;
                        } else if dy < -half_h {
                            dy += self.h;
                        }
                        let d = dx2 + dy * dy;
                        if d < 10000.0 {
                            density += 1.0;
                        }
                        if d < bd || (d == bd && best_idx.map_or(true, |prev| b_u < prev)) {
                            bd = d;
                            cutoff = if bd > 10000.0 { bd } else { 10000.0 };
                            best_idx = Some(b_u);
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }
                b_idx = unsafe { *self.grid_next.get_unchecked(b_u) };
            }
        }

        // Global fallback if no creature found in Moore neighborhood (sparse population)
        if bd >= 10000.0 {
            let len = self.agents.len();
            for i in 0..len {
                if i == a_idx { continue; }
                let bx = unsafe { *self.pos_x.get_unchecked(i) };
                let mut dx = bx - ax;
                if dx > half_w {
                    dx -= self.w;
                } else if dx < -half_w {
                    dx += self.w;
                }
                let dx2 = dx * dx;
                if dx2 >= cutoff { continue; }
                let by = unsafe { *self.pos_y.get_unchecked(i) };
                let mut dy = by - ay;
                if dy > half_h {
                    dy -= self.h;
                } else if dy < -half_h {
                    dy += self.h;
                }
                let d = dx2 + dy * dy;
                if d < bd || (d == bd && best_idx.map_or(true, |prev| i < prev)) {
                    bd = d;
                    cutoff = bd;
                    best_idx = Some(i);
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        NearResult {
            best_idx,
            best_dx,
            best_dy,
            best_d: bd.sqrt(),
            density,
        }
    }

    #[inline(always)]
    pub fn brain(a: &mut AgentData, ins: &[f64; 15]) -> [f32; 6] {
        let mut p = 0;
        let mut new_h = [0.0f32; 10];
        let mut out = [0.0f32; 6];

        // Recurrent Hidden Layer (Q = 10, N = 15)
        for j in 0..10 {
            let mut s = 0.0f64;
            let mut k = 0;
            while k + 4 <= 15 {
                let w0 = a.genes[p] as f64;
                let w1 = a.genes[p + 1] as f64;
                let w2 = a.genes[p + 2] as f64;
                let w3 = a.genes[p + 3] as f64;
                s += w0 * ins[k] + w1 * ins[k + 1] + w2 * ins[k + 2] + w3 * ins[k + 3];
                p += 4;
                k += 4;
            }
            while k < 15 {
                s += (a.genes[p] as f64) * ins[k];
                p += 1;
                k += 1;
            }

            k = 0;
            while k + 2 <= 10 {
                let w0 = a.genes[p] as f64;
                let w1 = a.genes[p + 1] as f64;
                s += w0 * (a.h[k] as f64) + w1 * (a.h[k + 1] as f64);
                p += 2;
                k += 2;
            }

            s += a.genes[p] as f64;
            p += 1;

            new_h[j] = math::tanh(s * SCALE_H) as f32;
        }

        // Output Actuator Layer (O = 6, Q = 10)
        for j in 0..6 {
            let mut s = 0.0f64;
            let mut k = 0;
            while k + 2 <= 10 {
                let w0 = a.genes[p] as f64;
                let w1 = a.genes[p + 1] as f64;
                s += w0 * (new_h[k] as f64) + w1 * (new_h[k + 1] as f64);
                p += 2;
                k += 2;
            }

            s += a.genes[p] as f64;
            p += 1;

            out[j] = math::tanh(s * SCALE_O) as f32;
        }

        a.h = new_h;
        out
    }

    pub fn evolve(&mut self) {
        self.tick += 1;
        let growth_val = self.growth / 100.0;
        let hostility_val = self.hostility / 100.0;

        // Environmental soil step
        self.soil.step_soil(&mut self.prng, growth_val);

        // Eclipse catastrophe dynamics
        if self.eclipse > 0 {
            self.eclipse -= 1;
            if self.tick % 2 == 0 {
                for _ in 0..160 {
                    let k = (self.prng.next_f64() * (self.soil.grid_size as f64)).floor() as usize;
                    let k = k.min(self.soil.grid_size - 1);
                    self.soil.food[k] = ((self.soil.food[k] as f64) * 0.73) as f32;
                    let new_t = (self.soil.taint[k] as f64) + 0.14;
                    self.soil.taint[k] = if new_t < 0.0 { 0.0 } else if new_t > 2.0 { 2.0 } else { new_t as f32 };
                }
            }
        }

        let tick008 = (self.tick as f64) * 0.08;
        self.build_grid();
        let initial_len = self.agents.len();
        let mut sensory_inputs = [0.0f64; 15];

        for i in 0..initial_len {
            if self.agents[i].dead != 0 {
                continue;
            }
            if self.agents[i].energy <= 0.0 {
                self.agents[i].energy = 0.0;
                let ax = self.agents[i].x;
                let ay = self.agents[i].y;
                let ae = self.agents[i].energy;
                let deposit_food = math::clamp(ae * 0.016 + 0.6, 0.3, 2.0);
                self.soil.deposit(0, ax, ay, deposit_food, 2);
                self.soil.deposit(1, ax, ay, 0.1, 1);
                self.spark_prng(5);
                self.agents[i].dead = 1;
                continue;
            }

            self.agents[i].age += 1;
            if self.agents[i].cooldown > 0 {
                self.agents[i].cooldown -= 1;
            }
            if self.agents[i].birth > 0 {
                self.agents[i].birth -= 1;
            }

            let tr0 = self.agents[i].tr[0];
            let tr1 = self.agents[i].tr[1];
            let tr2 = self.agents[i].tr[2];
            let tr3 = self.agents[i].tr[3];
            let tr4 = self.agents[i].tr[4];
            let tr5 = self.agents[i].tr[5];

            let fwd_x = math::cos(self.agents[i].angle);
            let fwd_y = math::sin(self.agents[i].angle);
            let reach = 19.0 + 38.0 * tr2;

            let fwd_reach_x_term = fwd_x * reach;
            let fwd_reach_y_term = fwd_y * reach;

            let ax = self.agents[i].x;
            let ay = self.agents[i].y;
            let idx_here = self.soil.idx_fast(ax, ay);
            let here = self.soil.food[idx_here] as f64;

            let fwd_reach_x = ax + fwd_reach_x_term;
            let fwd_reach_y = ay + fwd_reach_y_term;
            let idx_fwd = self.soil.idx(fwd_reach_x, fwd_reach_y);
            let forward = self.soil.food[idx_fwd] as f64;

            let mid_x = ax + fwd_reach_x_term * 0.7;
            let mid_y = ay + fwd_reach_y_term * 0.7;
            let off_x = -fwd_reach_y_term * 0.6;
            let off_y = fwd_reach_x_term * 0.6;
            let left = self.soil.food[self.soil.idx(mid_x + off_x, mid_y + off_y)] as f64;
            let right = self.soil.food[self.soil.idx(mid_x - off_x, mid_y - off_y)] as f64;

            let near_res = self.near(i);
            let d = near_res.best_d;
            if near_res.best_idx.is_some() {
                let bearing = math::atan2(near_res.best_dy, near_res.best_dx) - self.agents[i].angle;
                sensory_inputs[7] = math::sin(bearing);
                sensory_inputs[8] = math::cos(bearing);
            } else {
                sensory_inputs[7] = 0.0;
                sensory_inputs[8] = 1.0;
            }

            let mut v = self.agents[i].energy / 75.0 - 1.0;
            sensory_inputs[0] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = here - 1.0;
            sensory_inputs[1] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = forward - left;
            sensory_inputs[2] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = forward - right;
            sensory_inputs[3] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = forward - here;
            sensory_inputs[4] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = self.soil.taint[idx_here] as f64;
            sensory_inputs[5] = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
            v = (self.soil.scent[idx_fwd] as f64) - (self.soil.scent[idx_here] as f64);
            sensory_inputs[6] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = 1.0 - d / (100.0 + 70.0 * tr2);
            sensory_inputs[9] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            v = near_res.density / 9.0;
            sensory_inputs[10] = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
            v = (self.agents[i].age as f64) / 600.0;
            sensory_inputs[11] = if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v };
            sensory_inputs[12] = math::sin(tick008 + (self.agents[i].id as f64));
            v = self.agents[i].vx * fwd_x + self.agents[i].vy * fwd_y;
            sensory_inputs[13] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };

            if let Some(b_idx) = near_res.best_idx {
                v = self.agents[b_idx].energy / 70.0 - 1.0;
                sensory_inputs[14] = if v < -1.0 { -1.0 } else if v > 1.0 { 1.0 } else { v };
            } else {
                sensory_inputs[14] = 0.0;
            }

            let o = Self::brain(&mut self.agents[i], &sensory_inputs);

            let o0 = o[0] as f64;
            let o1 = o[1] as f64;
            let o2 = o[2] as f64;
            let o3 = o[3] as f64;
            let o4 = o[4] as f64;
            let o5 = o[5] as f64;

            // Actuator 0: Steering
            self.agents[i].angle += o0 * (0.11 + 0.09 * tr1);

            // Actuator 1: Thrust & Physics
            let thrust = (o1 + 1.0) * 0.5;
            let mot = 0.45 + 1.1 * tr1;
            self.agents[i].vx = (self.agents[i].vx + fwd_x * thrust * mot * 0.22) * 0.89;
            self.agents[i].vy = (self.agents[i].vy + fwd_y * thrust * mot * 0.22) * 0.89;
            self.agents[i].x = math::wrap(self.agents[i].x + self.agents[i].vx, self.w);
            self.agents[i].y = math::wrap(self.agents[i].y + self.agents[i].vy, self.h);
            self.pos_x[i] = self.agents[i].x;
            self.pos_y[i] = self.agents[i].y;
            self.update_agent_cell(i, self.agents[i].x, self.agents[i].y);

            // Trail management (in-place ring buffer)
            let cur_x = self.agents[i].x;
            let cur_y = self.agents[i].y;
            let tc = self.agents[i].trail_count as usize;
            if tc >= 9 {
                for t in 0..8 {
                    self.agents[i].trail_x[t] = self.agents[i].trail_x[t + 1];
                    self.agents[i].trail_y[t] = self.agents[i].trail_y[t + 1];
                }
                self.agents[i].trail_x[8] = cur_x;
                self.agents[i].trail_y[8] = cur_y;
            } else {
                self.agents[i].trail_x[tc] = cur_x;
                self.agents[i].trail_y[tc] = cur_y;
                self.agents[i].trail_count += 1;
            }

            // Actuators 2, 3, 4: Feeding, Attack, Signal
            self.agents[i].feeding = if o2 > 0.0 { o2 } else { 0.0 };
            self.agents[i].attack = if o3 > 0.0 { o3 } else { 0.0 };
            self.agents[i].signal = if o4 > 0.0 { o4 } else { 0.0 };

            if self.agents[i].signal > 0.4 {
                let s_val = (self.agents[i].signal - 0.4) * 0.035;
                self.soil.deposit(2, self.agents[i].x, self.agents[i].y, s_val, 1);
            }

            let fi = self.soil.idx_fast(self.agents[i].x, self.agents[i].y);
            let intake_cap = (0.016 + 0.064 * self.agents[i].feeding) * (0.7 + tr4);
            let cur_food = self.soil.food[fi] as f64;
            let eaten = if cur_food < intake_cap { cur_food } else { intake_cap };
            self.soil.food[fi] = ((self.soil.food[fi] as f64) - eaten) as f32;
            self.agents[i].energy += eaten * (9.0 + 9.0 * tr4);

            let taint_val = self.soil.taint[fi] as f64;
            self.agents[i].energy -= 0.10 +
                0.12 * tr0 +
                0.07 * tr1 +
                0.035 * tr2 +
                0.055 * tr3 +
                0.035 * self.agents[i].attack +
                0.014 * self.agents[i].signal +
                thrust * 0.06 +
                taint_val * (0.10 + 0.18 * (1.0 - tr3));

            // Predation / Combat
            if let Some(b_idx) = near_res.best_idx {
                if d < 14.0 + 10.0 * tr0
                    && self.agents[i].attack > 0.25
                    && self.agents[i].cooldown == 0
                    && self.agents[b_idx].energy > 0.0
                {
                    let b_tr3 = self.agents[b_idx].tr[3];
                    let damage = (0.5 + self.agents[i].attack * 2.2) * hostility_val * (0.8 + tr0) * (1.0 - 0.65 * b_tr3);
                    self.agents[b_idx].energy -= damage;
                    self.agents[i].energy += damage * (0.1 + 0.55 * tr5);
                    self.agents[i].cooldown = 3;
                    self.agents[i].last_victim = self.agents[b_idx].id;

                    if self.prng.next_f64() < 0.13 {
                        self.spark_prng(2);
                    }

                    if self.agents[b_idx].energy <= 0.0 {
                        self.kills += 1;
                        self.agents[i].kills += 1;
                        let kill_energy = 8.0 * tr5;
                        self.agents[i].energy += if kill_energy < 9.0 { kill_energy } else { 9.0 };
                        if self.agents[i].kills % 8 == 0 && self.events.len() < 1024 {
                            self.events.push(SimEvent {
                                tick: self.tick,
                                event_type: 3,
                                p1: self.agents[i].root,
                                p2: self.agents[i].kills,
                            });
                        }
                    }
                }
            }

            // Reproduction
            if self.agents[i].energy > 58.0 + 12.0 * tr0
                && self.agents[i].age > 65
                && self.agents[i].birth == 0
                && o5 > -0.15
                && self.agents.len() < self.max_cap
            {
                let mut mate_idx = None;
                if let Some(b_idx) = near_res.best_idx {
                    if d < 18.0
                        && self.agents[b_idx].energy > 42.0
                        && self.agents[b_idx].root != self.agents[i].root
                        && self.prng.next_f64() < 0.15
                    {
                        mate_idx = Some(b_idx);
                    }
                }

                let parent_clone = self.agents[i];
                let other_clone = mate_idx.map(|idx| self.agents[idx]);

                let cx = self.agents[i].x + self.prng.rand(-9.0, 9.0);
                let cy = self.agents[i].y + self.prng.rand(-9.0, 9.0);

                let c_idx = self.create_agent(cx, cy, Some(&parent_clone), other_clone.as_ref());
                self.insert_agent_cell(c_idx, self.agents[c_idx].x, self.agents[c_idx].y);
                self.agents[c_idx].energy = 24.0;
                self.agents[i].energy -= 24.0;
                if let Some(b_idx) = mate_idx {
                    self.agents[b_idx].energy -= 6.0;
                }
                self.agents[i].birth = 95;
                self.spark_prng(9);

                if self.agents[c_idx].gen > 0 && self.agents[c_idx].gen % 12 == 0 && self.prng.next_f64() < 0.08 {
                    if self.events.len() < 1024 {
                        self.events.push(SimEvent {
                            tick: self.tick,
                            event_type: 2,
                            p1: self.agents[c_idx].gen,
                            p2: self.agents[c_idx].root,
                        });
                    }
                }
            }

            self.agents[i].energy = math::clamp(self.agents[i].energy, -10.0, 110.0);
            if self.agents[i].energy <= 0.0 || self.agents[i].age > 2100 {
                self.agents[i].energy = if self.agents[i].energy > 0.0 { self.agents[i].energy } else { 0.0 };
                let ax = self.agents[i].x;
                let ay = self.agents[i].y;
                let ae = self.agents[i].energy;
                let deposit_food = math::clamp(ae * 0.016 + 0.6, 0.3, 2.0);
                self.soil.deposit(0, ax, ay, deposit_food, 2);
                self.soil.deposit(1, ax, ay, 0.1, 1);
                self.spark_prng(5);
                self.agents[i].dead = 1;
            }
        }

        // Filter dead agents in-place (stable compaction)
        let mut alive_count = 0;
        for i in 0..self.agents.len() {
            if self.agents[i].dead == 0 {
                if alive_count != i {
                    self.agents[alive_count] = self.agents[i];
                    self.pos_x[alive_count] = self.pos_x[i];
                    self.pos_y[alive_count] = self.pos_y[i];
                }
                alive_count += 1;
            }
        }
        self.agents.truncate(alive_count);
        self.pos_x.truncate(alive_count);
        self.pos_y.truncate(alive_count);

        // Spore replenishment if population collapses
        if self.agents.len() < 15 && self.tick % 45 == 0 {
            let n = 15 - self.agents.len();
            for _ in 0..n {
                let x = self.prng.rand(0.0, self.w);
                let y = self.prng.rand(0.0, self.h);
                self.create_agent(x, y, None, None);
            }
            if self.events.len() < 1024 {
                self.events.push(SimEvent {
                    tick: self.tick,
                    event_type: 1,
                    p1: 0,
                    p2: 0,
                });
            }
        }
    }
}
