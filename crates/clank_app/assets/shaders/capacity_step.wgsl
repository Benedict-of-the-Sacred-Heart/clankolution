// crates/clank_app/assets/shaders/capacity_step.wgsl
// Rebuilds the tombstone freelist for a new carrying capacity (the creature-capacity slider).
// Dead slots under the cap become free; live creatures at or above it are removed, as the Rust
// engine's `set_max_capacity` truncates them. Runs alone between frames; `freelist_top` is
// zeroed before it.

struct GpuSimParams {
    tick: u32,
    agent_count: u32,
    max_agents: u32,
    max_capacity: u32,

    hostility: f32,
    mut_rate: f32,
    speed: f32,
    renewal: f32,

    sub_tick: u32,
    sub_ticks_per_frame: u32,
    tool_type: u32,
    tool_radius: f32,

    tool_pos: vec2f,
    camera_pos: vec2f,

    camera_size: vec2f,
    world_size: vec2f,

    soil_grid: vec2u,
    eclipse: u32,
    epoch: u32,
}

struct GpuAgentState {
    pos_vel: vec4f,
    angle_energy: vec4f,
    traits: array<vec4f, 2>,
    hidden: array<vec4f, 2>,
    hidden_tail: vec2f,
    id: u32,
    meta_flags: u32,
    age_gen: u32,
    morton_code: u32,
    root: u32,
    visual_cache: u32,
}

struct GpuAgentAtomic {
    energy_milli: i32,
    mate_claim: u32,
    mate_energy_milli: u32,
    dead_claimed: u32,
}

const DEAD_BIT: u32 = 1u << 13u;
// Word index of `telemetry.freelist_top` at the start of the consolidated queue buffer.
const FREELIST_TOP_WORD: u32 = 7u;

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(2) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(3) var<storage, read_write> queue_words: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: GpuSimParams;

@compute @workgroup_size(64)
fn rebuild_freelist_main(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    if (slot >= params.max_agents) { return; }
    let cap = min(params.max_capacity, params.max_agents);
    // A claimed slot is free even before its creature's next pass sets the dead bit.
    let dead = (agent_states[slot].meta_flags & DEAD_BIT) != 0u
        || agent_atomics[slot].dead_claimed != 0u;
    if (slot >= cap) {
        if (!dead) {
            agent_states[slot].meta_flags |= DEAD_BIT;
            agent_states[slot].visual_cache = 0u;
            agent_atomics[slot].energy_milli = 0;
            agent_atomics[slot].dead_claimed = 1u;
        }
        return;
    }
    if (dead) {
        let top = atomicAdd(&queue_words[FREELIST_TOP_WORD], 1u);
        freelist[top] = slot;
    }
}
