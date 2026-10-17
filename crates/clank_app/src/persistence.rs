use crate::sim::SimWorld;
use clank_core::archive::WorldSnapshot;
use rkyv::Deserialize;
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

#[derive(Debug)]
pub enum PersistenceError {
    Serialization(String),
    Deserialization(String),
    InvalidArchive(String),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(s) => write!(f, "Serialization error: {}", s),
            Self::Deserialization(s) => write!(f, "Deserialization error: {}", s),
            Self::InvalidArchive(s) => write!(f, "Invalid archive: {}", s),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
        }
    }
}

impl std::error::Error for PersistenceError {}

pub fn save_clank_snapshot(sim: &SimWorld) -> Result<Vec<u8>, PersistenceError> {
    let w = &sim.world;
    let snap = WorldSnapshot::new(
        w.tick,
        w.kills,
        w.births,
        w.roots,
        w.next_id,
        w.eclipse,
        w.w,
        w.h,
        w.soil.cols as u32,
        w.soil.rows as u32,
        w.mutation,
        w.growth,
        w.hostility,
        w.prng.s,
        &w.soil.food,
        &w.soil.taint,
        &w.soil.scent,
        &w.agents,
    );
    Ok(snap.to_bytes())
}

pub fn load_clank_snapshot(sim: &mut SimWorld, bytes: &[u8]) -> Result<(), PersistenceError> {
    let archived = rkyv::check_archived_root::<WorldSnapshot>(bytes)
        .map_err(|e| PersistenceError::InvalidArchive(format!("{:?}", e)))?;

    if archived.version != WorldSnapshot::CURRENT_VERSION {
        return Err(PersistenceError::UnsupportedVersion(
            archived.version.into(),
        ));
    }

    let w = &mut sim.world;

    let target_w: f64 = archived.width.into();
    let target_h: f64 = archived.height.into();
    let target_cols: u32 = archived.cols.into();
    let target_rows: u32 = archived.rows.into();
    if w.w != target_w
        || w.h != target_h
        || w.soil.cols != target_cols as usize
        || w.soil.rows != target_rows as usize
    {
        w.resize(
            target_w,
            target_h,
            target_cols as usize,
            target_rows as usize,
        );
    }

    w.tick = archived.tick.into();
    w.kills = archived.kills.into();
    w.births = archived.births.into();
    w.roots = archived.roots.into();
    w.next_id = archived.next_id.into();
    w.eclipse = archived.eclipse.into();
    w.mutation = archived.mutation.into();
    w.growth = archived.growth.into();
    w.hostility = archived.hostility.into();
    w.prng.s = archived.prng_state.into();

    let food_len = archived.food.len().min(w.soil.food.len());
    for i in 0..food_len {
        w.soil.food[i] = archived.food[i].into();
    }
    let taint_len = archived.taint.len().min(w.soil.taint.len());
    for i in 0..taint_len {
        w.soil.taint[i] = archived.taint[i].into();
    }
    let scent_len = archived.scent.len().min(w.soil.scent.len());
    for i in 0..scent_len {
        w.soil.scent[i] = archived.scent[i].into();
    }

    let snap_agents: Vec<clank_core::agent::AgentData> = archived
        .agents
        .deserialize(&mut rkyv::Infallible)
        .map_err(|e| PersistenceError::Deserialization(format!("{:?}", e)))?;

    if snap_agents.len() > w.max_cap {
        w.set_max_capacity(snap_agents.len() as u32);
    }

    w.agents.clear();
    w.agents.extend(snap_agents);
    for a in &w.agents {
        w.next_id = w.next_id.max(a.id.saturating_add(1));
        w.roots = w.roots.max(a.root.saturating_add(1));
    }
    w.build_grid();

    sim.world_width = target_w;
    sim.world_height = target_h;
    sim.selected_agent_id = None;
    sim.selected_agent_slot = None;
    sim.selected_agent_cache = None;
    sim.pending_tool = None;
    sim.gpu_resync_pending = true;
    sim.sync_gpu_cached_stats_from_cpu();

    Ok(())
}

#[derive(Clone, Debug, SerdeSerialize, SerdeDeserialize)]
pub struct JsonHistoryPoint {
    #[serde(alias = "population")]
    pub p: usize,
    #[serde(alias = "food")]
    pub f: f32,
    #[serde(alias = "kills")]
    pub k: usize,
}

#[derive(SerdeSerialize, SerdeDeserialize)]
pub struct JsonAgent {
    pub id: u32,
    pub root: u32,
    pub gen: u32,
    pub age: u32,
    #[serde(default)]
    pub cooldown: u32,
    #[serde(default)]
    pub birth: u32,
    #[serde(default)]
    pub kills: u32,
    #[serde(default)]
    pub dead: bool,
    #[serde(default, alias = "lastVictim")]
    pub last_victim: Option<u32>,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub angle: f64,
    pub energy: f64,
    #[serde(default)]
    pub feeding: f64,
    #[serde(default)]
    pub attack: f64,
    #[serde(default)]
    pub signal: f64,
    pub tr: Vec<f64>,
    pub w: Vec<i8>,
    pub h: Vec<f32>,
    #[serde(default)]
    pub trail: Vec<[f64; 2]>,
}

#[derive(SerdeSerialize, SerdeDeserialize)]
pub struct JsonWorld {
    pub version: u32,
    pub width: f64,
    pub height: f64,
    #[serde(default)]
    pub cap: usize,
    #[serde(default)]
    pub tick: u32,
    #[serde(default)]
    pub kills: u32,
    #[serde(default)]
    pub births: u32,
    #[serde(default, alias = "nextId")]
    pub next_id: u32,
    #[serde(default)]
    pub roots: u32,
    #[serde(default)]
    pub eclipse: u32,
    pub cols: u32,
    pub rows: u32,
    pub food: Vec<f32>,
    #[serde(default)]
    pub taint: Vec<f32>,
    #[serde(default)]
    pub scent: Vec<f32>,
    #[serde(default)]
    pub history: Vec<JsonHistoryPoint>,
    #[serde(default = "default_mutation")]
    pub mutation: f64,
    #[serde(default = "default_hundred")]
    pub growth: f64,
    #[serde(default = "default_hundred")]
    pub hostility: f64,
    pub agents: Vec<JsonAgent>,
}

fn default_mutation() -> f64 {
    16.0
}

fn default_hundred() -> f64 {
    100.0
}

pub fn save_json_world(sim: &SimWorld) -> Result<String, PersistenceError> {
    let w = &sim.world;
    let json_agents = w
        .agents
        .iter()
        .map(|a| {
            let trail_len = (a.trail_count as usize).min(9);
            let trail = (0..trail_len)
                .map(|i| [a.trail_x[i], a.trail_y[i]])
                .collect();
            JsonAgent {
                id: a.id,
                root: a.root,
                gen: a.gen,
                age: a.age,
                cooldown: a.cooldown,
                birth: a.birth,
                kills: a.kills,
                dead: a.dead != 0,
                last_victim: if a.last_victim != 0 {
                    Some(a.last_victim)
                } else {
                    None
                },
                x: a.x,
                y: a.y,
                vx: a.vx,
                vy: a.vy,
                angle: a.angle,
                energy: a.energy,
                feeding: a.feeding,
                attack: a.attack,
                signal: a.signal,
                tr: a.tr.to_vec(),
                w: a.genes.to_vec(),
                h: a.h.to_vec(),
                trail,
            }
        })
        .collect();

    let json_world = JsonWorld {
        version: 1,
        width: w.w,
        height: w.h,
        cap: w.max_cap,
        tick: w.tick,
        kills: w.kills,
        births: w.births,
        next_id: w.next_id,
        roots: w.roots,
        eclipse: w.eclipse,
        cols: w.soil.cols as u32,
        rows: w.soil.rows as u32,
        food: w.soil.food.clone(),
        taint: w.soil.taint.clone(),
        scent: w.soil.scent.clone(),
        history: Vec::new(),
        mutation: w.mutation,
        growth: w.growth,
        hostility: w.hostility,
        agents: json_agents,
    };

    serde_json::to_string(&json_world).map_err(|e| PersistenceError::Serialization(e.to_string()))
}

pub fn load_json_world(sim: &mut SimWorld, json_str: &str) -> Result<(), PersistenceError> {
    let data: JsonWorld = serde_json::from_str(json_str)
        .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

    if data.version != 1 {
        return Err(PersistenceError::UnsupportedVersion(data.version));
    }
    if data.cols < 1
        || data.rows < 1
        || data.width < 1.0
        || data.height < 1.0
        || (data.cols as usize).saturating_mul(data.rows as usize) != data.food.len()
    {
        return Err(PersistenceError::InvalidArchive(
            "Invalid world grid dimensions".to_string(),
        ));
    }

    let w = &mut sim.world;
    if w.w != data.width
        || w.h != data.height
        || w.soil.cols != data.cols as usize
        || w.soil.rows != data.rows as usize
    {
        w.resize(
            data.width,
            data.height,
            data.cols as usize,
            data.rows as usize,
        );
    }

    w.tick = data.tick;
    w.kills = data.kills;
    w.births = data.births;
    w.eclipse = data.eclipse.min(210);
    w.mutation = data.mutation.clamp(0.0, 50.0);
    w.growth = data.growth.clamp(0.0, 200.0);
    w.hostility = data.hostility.clamp(0.0, 200.0);

    w.soil.food.fill(0.0);
    w.soil.taint.fill(0.0);
    w.soil.scent.fill(0.0);

    let food_len = data.food.len().min(w.soil.food.len());
    for i in 0..food_len {
        w.soil.food[i] = data.food[i].clamp(0.0, 2.5);
    }

    let taint_len = data.taint.len().min(w.soil.taint.len());
    for i in 0..taint_len {
        w.soil.taint[i] = data.taint[i].clamp(0.0, 3.0);
    }

    let scent_len = data.scent.len().min(w.soil.scent.len());
    for i in 0..scent_len {
        w.soil.scent[i] = data.scent[i].clamp(0.0, 3.0);
    }

    let req_cap = data.cap.max(data.agents.len());
    if req_cap > 0 {
        w.set_max_capacity(req_cap.clamp(50, 65_536) as u32);
    }

    let mut max_id = data.next_id.max(1);
    let mut max_root = data.roots.max(1);

    w.agents.clear();
    for ja in data.agents {
        if ja.dead {
            continue;
        }
        let mut a = clank_core::agent::AgentData::default();
        a.id = ja.id;
        a.root = ja.root;
        a.gen = ja.gen;
        a.age = ja.age;
        a.cooldown = ja.cooldown;
        a.birth = ja.birth;
        a.kills = ja.kills;
        a.dead = 0;
        a.last_victim = ja.last_victim.unwrap_or(0);
        a.x = clank_core::math::wrap(ja.x, w.w);
        a.y = clank_core::math::wrap(ja.y, w.h);
        a.vx = ja.vx;
        a.vy = ja.vy;
        a.angle = ja.angle;
        a.energy = ja.energy;
        a.feeding = ja.feeding;
        a.attack = ja.attack;
        a.signal = ja.signal;
        for (i, v) in ja.tr.iter().enumerate().take(6) {
            a.tr[i] = v.clamp(0.0, 1.0);
        }
        for (i, v) in ja.w.iter().enumerate().take(clank_core::agent::GENES) {
            a.genes[i] = *v;
        }
        for (i, v) in ja.h.iter().enumerate().take(10) {
            a.h[i] = *v;
            a.h_next[i] = *v;
        }
        let trail_len = ja.trail.len().min(9);
        for (i, pt) in ja.trail.iter().take(trail_len).enumerate() {
            a.trail_x[i] = pt[0];
            a.trail_y[i] = pt[1];
        }
        a.trail_count = trail_len as u32;
        max_id = max_id.max(a.id.saturating_add(1));
        max_root = max_root.max(a.root.saturating_add(1));
        w.agents.push(a);
    }
    w.next_id = max_id;
    w.roots = max_root;

    w.build_grid();
    sim.world_width = data.width;
    sim.world_height = data.height;
    sim.selected_agent_id = None;
    sim.selected_agent_slot = None;
    sim.selected_agent_cache = None;
    sim.pending_tool = None;
    sim.gpu_resync_pending = true;
    sim.sync_gpu_cached_stats_from_cpu();

    Ok(())
}

pub fn save_clank_file(sim: &SimWorld, path: &str) -> Result<usize, PersistenceError> {
    let bytes = save_clank_snapshot(sim)?;
    std::fs::write(path, &bytes).map_err(|e| PersistenceError::Serialization(e.to_string()))?;
    Ok(bytes.len())
}

pub fn load_clank_file(sim: &mut SimWorld, path: &str) -> Result<(), PersistenceError> {
    let bytes =
        std::fs::read(path).map_err(|e| PersistenceError::Deserialization(e.to_string()))?;
    load_clank_snapshot(sim, &bytes)
}

pub fn export_json_file(sim: &SimWorld, path: &str) -> Result<usize, PersistenceError> {
    let json_str = save_json_world(sim)?;
    std::fs::write(path, json_str.as_bytes())
        .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
    Ok(json_str.len())
}

pub fn import_json_file(sim: &mut SimWorld, path: &str) -> Result<(), PersistenceError> {
    let json_str = std::fs::read_to_string(path)
        .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;
    load_json_world(sim, &json_str)
}
