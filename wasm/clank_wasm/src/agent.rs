use rkyv::{Archive, Deserialize, Serialize};

pub const MAX_CAP: usize = 340;
pub const GENES: usize = 326;

#[repr(C)]
#[derive(Clone, Copy, Debug, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct AgentData {
    // 8-byte aligned float fields (0..72)
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub angle: f64,
    pub energy: f64,
    pub feeding: f64,
    pub attack: f64,
    pub signal: f64,

    // Morphological traits [bulk, speed, sight, armor, foraging, carnivory] (72..120)
    pub tr: [f64; 6],

    // Trail history coordinates (120..264)
    pub trail_x: [f64; 9],
    pub trail_y: [f64; 9],

    // 4-byte aligned neural hidden states (264..344)
    pub h: [f32; 10],
    pub h_next: [f32; 10],

    // 4-byte aligned metadata and integer counters (344..384)
    pub id: u32,
    pub root: u32,
    pub gen: u32,
    pub age: u32,
    pub cooldown: u32,
    pub birth: u32,
    pub kills: u32,
    pub dead: u32,
    pub last_victim: u32,
    pub trail_count: u32,

    // 1-byte aligned quantized genome weights (384..710)
    pub genes: [i8; GENES],

    // Explicit cache-alignment padding (710..768)
    pub _pad: [u8; 58],
}

impl Default for AgentData {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            angle: 0.0,
            energy: 50.0,
            feeding: 0.0,
            attack: 0.0,
            signal: 0.0,
            tr: [0.5; 6],
            trail_x: [0.0; 9],
            trail_y: [0.0; 9],
            h: [0.0; 10],
            h_next: [0.0; 10],
            id: 0,
            root: 0,
            gen: 0,
            age: 0,
            cooldown: 0,
            birth: 0,
            kills: 0,
            dead: 0,
            last_victim: 0,
            trail_count: 0,
            genes: [0; GENES],
            _pad: [0; 58],
        }
    }
}

// Compile-time assertion that AgentData has exact 768-byte size and 8-byte alignment (12 cache lines)
const _: () = assert!(std::mem::size_of::<AgentData>() == 768);
const _: () = assert!(std::mem::align_of::<AgentData>() == 8);
