use crate::math::{self, Prng};
use crate::agent::{AgentData, MAX_CAP, GENES};
use crate::soil::{SoilGrid, W, H, COLS, ROWS, GRID_SIZE, INV_CELL_W, INV_CELL_H};

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

pub struct World {
    pub prng: Prng,
    pub soil: SoilGrid,
    pub agents: Vec<AgentData>,
    pub tick: u32,
    pub kills: u32,
    pub births: u32,
    pub roots: u32,
    pub next_id: u32,
    pub eclipse: u32,
    pub mutation: f64,
    pub growth: f64,
    pub hostility: f64,
}

impl World {
    pub fn new(seed: u32) -> Self {
        let mut w = Self {
            prng: Prng::new(seed),
            soil: SoilGrid::new(),
            agents: Vec::with_capacity(MAX_CAP),
            tick: 0,
            kills: 0,
            births: 0,
            roots: 1,
            next_id: 1,
            eclipse: 0,
            mutation: 16.0,
            growth: 100.0,
            hostility: 100.0,
        };
        // Initial startup draws 3,750 numbers during initial resize() -> setupGrid(false)
        for _ in 0..GRID_SIZE {
            w.prng.rand(0.15, 0.5);
        }
        w.reset();
        w
    }

    pub fn reset(&mut self) {
        self.agents.clear();
        self.tick = 0;
        self.kills = 0;
        self.births = 0;
        self.roots = 1;
        self.next_id = 1;
        self.eclipse = 0;

        self.soil.food = [0.0; GRID_SIZE];
        self.soil.taint = [0.0; GRID_SIZE];
        self.soil.scent = [0.0; GRID_SIZE];
        self.soil.init_bloom();

        // Initial background food noise: food[i] = rand(0.15, 0.5)
        for i in 0..GRID_SIZE {
            self.soil.food[i] = self.prng.rand(0.15, 0.5) as f32;
        }

        // 12 initial food patches
        for _ in 0..12 {
            let cx = self.prng.rand(0.0, W);
            let cy = self.prng.rand(0.0, H);
            for _ in 0..16 {
                let px = cx + self.prng.rand(-80.0, 80.0);
                let py = cy + self.prng.rand(-80.0, 80.0);
                let val = self.prng.rand(0.2, 0.5);
                SoilGrid::deposit(&mut self.soil.food, px, py, val, 2);
            }
        }

        // 72 initial creatures
        for _ in 0..72 {
            let x = self.prng.rand(0.0, W);
            let y = self.prng.rand(0.0, H);
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
        a.x = math::wrap(x, W);
        a.y = math::wrap(y, H);
        a.angle = self.prng.rand(0.0, TAU);
        a.vx = 0.0;
        a.vy = 0.0;
        a.energy = if parent.is_some() { 25.0 } else { self.prng.rand(34.0, 52.0) };
        a.age = 0;
        a.gen = if let Some(p) = parent {
            let og = other.map_or(0, |o| o.gen);
            std::cmp::max(p.gen, og) + 1
        } else {
            0
        };
        a.root = if let Some(p) = parent {
            p.root
        } else {
            let r = self.roots;
            self.roots += 1;
            r
        };

        // Genome
        let m_rate = self.mutation / 100.0;
        if let Some(p) = parent {
            for i in 0..GENES {
                let base = if other.is_some() && self.prng.next_f64() < 0.48 {
                    other.unwrap().genes[i]
                } else {
                    p.genes[i]
                };
                let mutation = if self.prng.next_f64() < 0.11 {
                    let r = (self.prng.next_f64() + self.prng.next_f64() - 1.0) * 100.0 * m_rate;
                    (r + 0.5).floor() as i32
                } else {
                    0
                };
                let v = base as i32 + mutation;
                a.genes[i] = if v < -127 { -127 } else if v > 127 { 127 } else { v as i8 };
            }
        } else {
            for i in 0..GENES {
                let r = self.prng.rand(-58.0, 58.0);
                a.genes[i] = (r + 0.5).floor() as i8;
            }
        }

        // Traits
        if let Some(p) = parent {
            for i in 0..6 {
                let base = if other.is_some() && self.prng.next_f64() < 0.45 {
                    other.unwrap().tr[i]
                } else {
                    p.tr[i]
                };
                let mutation = (self.prng.next_f64() + self.prng.next_f64() - 1.0) * m_rate * 0.6;
                a.tr[i] = math::clamp(base + mutation, 0.03, 0.98);
            }
        } else {
            a.tr[0] = self.prng.rand(0.25, 0.8);
            a.tr[1] = self.prng.rand(0.30, 0.8);
            a.tr[2] = self.prng.rand(0.30, 0.8);
            a.tr[3] = self.prng.rand(0.25, 0.75);
            a.tr[4] = self.prng.rand(0.20, 0.8);
            a.tr[5] = self.prng.rand(0.20, 0.8);
        }

        self.agents.push(a);
        self.births += 1;
        self.agents.len() - 1
    }

    #[inline(always)]
    pub fn near(&self, a_idx: usize) -> NearResult {
        let a = &self.agents[a_idx];
        let ax = a.x;
        let ay = a.y;
        let half_w = W * 0.5;
        let half_h = H * 0.5;
        let mut bd = 1e9f64;
        let mut cutoff = 1e9f64;
        let mut density = 0.0f64;
        let mut best_idx = None;
        let mut best_dx = 0.0f64;
        let mut best_dy = 0.0f64;

        let len = self.agents.len();
        for i in 0..len {
            if i == a_idx {
                continue;
            }
            let b = &self.agents[i];
            let mut dx = b.x - ax;
            if dx > half_w {
                dx -= W;
            } else if dx < -half_w {
                dx += W;
            }

            let dx2 = dx * dx;
            if dx2 >= cutoff {
                continue;
            }

            let mut dy = b.y - ay;
            if dy > half_h {
                dy -= H;
            } else if dy < -half_h {
                dy += H;
            }

            let d = dx2 + dy * dy;
            if d < 10000.0 {
                density += 1.0;
            }
            if d < bd {
                bd = d;
                cutoff = if bd > 10000.0 { bd } else { 10000.0 };
                best_idx = Some(i);
                best_dx = dx;
                best_dy = dy;
            }
        }

        NearResult {
            best_idx,
            best_dx,
            best_dy,
            best_d: if best_idx.is_some() { math::sqrt(bd) } else { 999.0 },
            density,
        }
    }

    pub fn brain(a: &mut AgentData, ins: &[f64; 15]) -> [f32; 6] {
        let mut p = 0;
        let w = &a.genes;
        let h_prev = a.h;
        let mut new_h = [0.0f32; 10];
        let mut brain_out = [0.0f32; 6];

        let in0 = ins[0]; let in1 = ins[1]; let in2 = ins[2]; let in3 = ins[3]; let in4 = ins[4];
        let in5 = ins[5]; let in6 = ins[6]; let in7 = ins[7]; let in8 = ins[8]; let in9 = ins[9];
        let in10 = ins[10]; let in11 = ins[11]; let in12 = ins[12]; let in13 = ins[13]; let in14 = ins[14];

        let hp0 = h_prev[0] as f64; let hp1 = h_prev[1] as f64; let hp2 = h_prev[2] as f64;
        let hp3 = h_prev[3] as f64; let hp4 = h_prev[4] as f64; let hp5 = h_prev[5] as f64;
        let hp6 = h_prev[6] as f64; let hp7 = h_prev[7] as f64; let hp8 = h_prev[8] as f64;
        let hp9 = h_prev[9] as f64;

        for j in 0..10 {
            let mut s = 0.0f64;
            s += (w[p] as f64) * in0; p += 1;
            s += (w[p] as f64) * in1; p += 1;
            s += (w[p] as f64) * in2; p += 1;
            s += (w[p] as f64) * in3; p += 1;
            s += (w[p] as f64) * in4; p += 1;
            s += (w[p] as f64) * in5; p += 1;
            s += (w[p] as f64) * in6; p += 1;
            s += (w[p] as f64) * in7; p += 1;
            s += (w[p] as f64) * in8; p += 1;
            s += (w[p] as f64) * in9; p += 1;
            s += (w[p] as f64) * in10; p += 1;
            s += (w[p] as f64) * in11; p += 1;
            s += (w[p] as f64) * in12; p += 1;
            s += (w[p] as f64) * in13; p += 1;
            s += (w[p] as f64) * in14; p += 1;

            s += (w[p] as f64) * hp0; p += 1;
            s += (w[p] as f64) * hp1; p += 1;
            s += (w[p] as f64) * hp2; p += 1;
            s += (w[p] as f64) * hp3; p += 1;
            s += (w[p] as f64) * hp4; p += 1;
            s += (w[p] as f64) * hp5; p += 1;
            s += (w[p] as f64) * hp6; p += 1;
            s += (w[p] as f64) * hp7; p += 1;
            s += (w[p] as f64) * hp8; p += 1;
            s += (w[p] as f64) * hp9; p += 1;

            s += w[p] as f64; p += 1; // bias
            new_h[j] = math::tanh(s * SCALE_H) as f32;
        }

        let nh0 = new_h[0] as f64; let nh1 = new_h[1] as f64; let nh2 = new_h[2] as f64;
        let nh3 = new_h[3] as f64; let nh4 = new_h[4] as f64; let nh5 = new_h[5] as f64;
        let nh6 = new_h[6] as f64; let nh7 = new_h[7] as f64; let nh8 = new_h[8] as f64;
        let nh9 = new_h[9] as f64;

        for j in 0..6 {
            let mut s = 0.0f64;
            s += (w[p] as f64) * nh0; p += 1;
            s += (w[p] as f64) * nh1; p += 1;
            s += (w[p] as f64) * nh2; p += 1;
            s += (w[p] as f64) * nh3; p += 1;
            s += (w[p] as f64) * nh4; p += 1;
            s += (w[p] as f64) * nh5; p += 1;
            s += (w[p] as f64) * nh6; p += 1;
            s += (w[p] as f64) * nh7; p += 1;
            s += (w[p] as f64) * nh8; p += 1;
            s += (w[p] as f64) * nh9; p += 1;

            s += w[p] as f64; p += 1; // bias
            brain_out[j] = math::tanh(s * SCALE_O) as f32;
        }

        a.h = new_h;
        brain_out
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
                    let k = (self.prng.next_f64() * (GRID_SIZE as f64)).floor() as usize;
                    let k = k.min(GRID_SIZE - 1);
                    self.soil.food[k] = ((self.soil.food[k] as f64) * 0.73) as f32;
                    let new_t = (self.soil.taint[k] as f64) + 0.14;
                    self.soil.taint[k] = if new_t < 0.0 { 0.0 } else if new_t > 2.0 { 2.0 } else { new_t as f32 };
                }
            }
        }

        let tick008 = (self.tick as f64) * 0.08;
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
                SoilGrid::deposit(&mut self.soil.food, ax, ay, deposit_food, 2);
                SoilGrid::deposit(&mut self.soil.taint, ax, ay, 0.1, 1);
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
            let idx_here = self.soil.idx(ax, ay);
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
            self.agents[i].x = math::wrap(self.agents[i].x + self.agents[i].vx, W);
            self.agents[i].y = math::wrap(self.agents[i].y + self.agents[i].vy, H);

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
                SoilGrid::deposit(&mut self.soil.scent, self.agents[i].x, self.agents[i].y, s_val, 1);
            }

            let fi = self.soil.idx(self.agents[i].x, self.agents[i].y);
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
                    }
                }
            }

            // Reproduction
            if self.agents[i].energy > 58.0 + 12.0 * tr0
                && self.agents[i].age > 65
                && self.agents[i].birth == 0
                && o5 > -0.15
                && self.agents.len() < MAX_CAP
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
                self.agents[c_idx].energy = 24.0;
                self.agents[i].energy -= 24.0;
                if let Some(b_idx) = mate_idx {
                    self.agents[b_idx].energy -= 6.0;
                }
                self.agents[i].birth = 95;
                self.spark_prng(9);

                if self.agents[c_idx].gen > 0 && self.agents[c_idx].gen % 12 == 0 && self.prng.next_f64() < 0.08 {
                    // lineage notification event
                }
            }

            self.agents[i].energy = math::clamp(self.agents[i].energy, -10.0, 110.0);
            if self.agents[i].energy <= 0.0 || self.agents[i].age > 2100 {
                self.agents[i].energy = if self.agents[i].energy > 0.0 { self.agents[i].energy } else { 0.0 };
                let ax = self.agents[i].x;
                let ay = self.agents[i].y;
                let ae = self.agents[i].energy;
                let deposit_food = math::clamp(ae * 0.016 + 0.6, 0.3, 2.0);
                SoilGrid::deposit(&mut self.soil.food, ax, ay, deposit_food, 2);
                SoilGrid::deposit(&mut self.soil.taint, ax, ay, 0.1, 1);
                self.spark_prng(5);
                self.agents[i].dead = 1;
            }
        }

        // Filter dead agents in-place (stable compaction)
        let mut alive_count = 0;
        for i in 0..self.agents.len() {
            if self.agents[i].dead == 0 {
                self.agents[alive_count] = self.agents[i];
                alive_count += 1;
            }
        }
        self.agents.truncate(alive_count);

        // Spore replenishment if population collapses
        if self.agents.len() < 15 && self.tick % 45 == 0 {
            let n = 15 - self.agents.len();
            for _ in 0..n {
                let x = self.prng.rand(0.0, W);
                let y = self.prng.rand(0.0, H);
                self.create_agent(x, y, None, None);
            }
        }
    }
}
