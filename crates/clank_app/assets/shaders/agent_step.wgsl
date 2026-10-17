// crates/clank_app/assets/shaders/agent_step.wgsl
// GPU Agent Simulation Step Kernel
//
// 128B GpuAgentState + 352B GpuAgentGenome Struct-of-Arrays
// Branchless 88-Word Vectorized RNN with unpack4x8snorm
// Strict <= 8 Storage Buffer Limit for 100% WebGPU Portability

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

struct GpuAgentGenome {
    packed_genes: array<u32, 88>,
}

struct GpuAgentAtomic {
    energy_milli: atomic<i32>,
    mate_claim: atomic<u32>,
    mate_energy_milli: atomic<u32>,
    dead_claimed: atomic<u32>,
}

// Fixed-point scale of SoilCell fields (micro-units). Must match
// SOIL_FIXED_SCALE in src/gpu/types.rs and SOIL_SCALE in soil_step.wgsl.
const SOIL_SCALE: f32 = 1000000.0;
// The reference's deposit() clamps every touched cell to [0, 3]. Every deposit is positive, so
// capping with atomicMin after the add leaves the same value as the reference's sequential clamp.
const SOIL_CELL_MAX_MICRO: i32 = 3000000;

struct SoilCell {
    food_micro: atomic<i32>,
    taint_micro: atomic<i32>,
    scent_micro: atomic<i32>,
    fertility_micro: atomic<i32>,
}

struct GpuTelemetry {
    population: atomic<u32>,
    kills: atomic<u32>,
    starvations: atomic<u32>,
    apex_record_milli: atomic<u32>,
    food_grazed_milli: atomic<u32>,
    sub_ticks_elapsed: atomic<u32>,
    apex_agent_id: atomic<u32>,
    freelist_top: atomic<u32>,
    birth_count: atomic<u32>,
    audio_voice_count: atomic<u32>,
    selected_agent_idx: u32,
    selected_agent_id: u32,
    total_births: atomic<u32>,
    total_deaths: atomic<u32>,
    max_generation: atomic<u32>,
    living_lineages: atomic<u32>,
    live_count: atomic<u32>,
    pending_free_count: atomic<u32>,
    available_slots: atomic<u32>,
    chronicle_count: atomic<u32>,
    reserved1: array<u32, 12>,
}

struct BirthEvent {
    parent_a: u32,
    parent_b: u32,
    child_slot: u32,
    birth_tick: u32,
}

struct AudioVoice {
    pos: vec2f,
    event_type: u32,
    volume: f32,
}

// GpuChronicleEvent: 2 = generation milestone (gen, root), 3 = kill milestone (root, kills).
struct ChronicleEvent {
    tick: u32,
    event_type: u32,
    p1: u32,
    p2: u32,
}

struct ConsolidatedQueue {
    telemetry: GpuTelemetry,
    births: array<BirthEvent, 65536>,
    audio: array<AudioVoice, 256>,
    chronicle: array<ChronicleEvent, 64>,
}

// Exactly 8 storage buffers in Group 0
@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read> agent_genomes: array<GpuAgentGenome>;
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(3) var<storage, read_write> soil_buffer: array<SoilCell>;
@group(0) @binding(4) var<storage, read> spatial_keys: array<vec2u>;
@group(0) @binding(6) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(7) var<storage, read_write> queue_buffer: ConsolidatedQueue;

@group(1) @binding(0) var<uniform> params: GpuSimParams;

fn wrap_coords(p: vec2f) -> vec2f {
    return vec2f(
        p.x - params.world_size.x * floor(p.x / params.world_size.x),
        p.y - params.world_size.y * floor(p.y / params.world_size.y)
    );
}

fn sample_soil_probe(p: vec2f) -> vec4f {
    let wrapped = wrap_coords(p);
    let cell_w = params.world_size.x / f32(params.soil_grid.x);
    let cell_h = params.world_size.y / f32(params.soil_grid.y);
    let cx = min(u32(max(0.0, wrapped.x) / cell_w), params.soil_grid.x - 1u);
    let cy = min(u32(max(0.0, wrapped.y) / cell_h), params.soil_grid.y - 1u);
    let k = cy * params.soil_grid.x + cx;
    let f = f32(atomicLoad(&soil_buffer[k].food_micro)) / SOIL_SCALE;
    let t = f32(atomicLoad(&soil_buffer[k].taint_micro)) / SOIL_SCALE;
    let s = f32(atomicLoad(&soil_buffer[k].scent_micro)) / SOIL_SCALE;
    return vec4f(f, t, s, 1.0);
}

fn deposit_food_radius2(p: vec2f, val: f32) {
    let wrapped = wrap_coords(p);
    let cell_w = params.world_size.x / f32(params.soil_grid.x);
    let cell_h = params.world_size.y / f32(params.soil_grid.y);
    let cx = i32(min(u32(max(0.0, wrapped.x) / cell_w), params.soil_grid.x - 1u));
    let cy = i32(min(u32(max(0.0, wrapped.y) / cell_h), params.soil_grid.y - 1u));
    let cols = i32(params.soil_grid.x);
    let rows = i32(params.soil_grid.y);
    for (var dy = -2; dy <= 2; dy++) {
        let ny = u32((cy + dy + rows) % rows);
        for (var dx = -2; dx <= 2; dx++) {
            let rr = f32(dx * dx + dy * dy);
            if (rr > 4.5) { continue; }
            let nx = u32((cx + dx + cols) % cols);
            let idx = ny * params.soil_grid.x + nx;
            let w = 1.0 / (1.0 + rr * 0.8);
            let deposit_micro = i32(round(val * w * SOIL_SCALE));
            atomicAdd(&soil_buffer[idx].food_micro, deposit_micro);
            atomicMin(&soil_buffer[idx].food_micro, SOIL_CELL_MAX_MICRO);
        }
    }
}

fn deposit_taint_radius1(p: vec2f, val: f32) {
    let wrapped = wrap_coords(p);
    let cell_w = params.world_size.x / f32(params.soil_grid.x);
    let cell_h = params.world_size.y / f32(params.soil_grid.y);
    let cx = i32(min(u32(max(0.0, wrapped.x) / cell_w), params.soil_grid.x - 1u));
    let cy = i32(min(u32(max(0.0, wrapped.y) / cell_h), params.soil_grid.y - 1u));
    let cols = i32(params.soil_grid.x);
    let rows = i32(params.soil_grid.y);

    let c0 = u32(cy) * params.soil_grid.x + u32(cx);
    atomicAdd(&soil_buffer[c0].taint_micro, i32(round(val * SOIL_SCALE)));
    atomicMin(&soil_buffer[c0].taint_micro, SOIL_CELL_MAX_MICRO);

    let v18_micro = i32(round((val / 1.8) * SOIL_SCALE));
    let c_up = u32((cy - 1 + rows) % rows) * params.soil_grid.x + u32(cx);
    let c_down = u32((cy + 1 + rows) % rows) * params.soil_grid.x + u32(cx);
    let c_left = u32(cy) * params.soil_grid.x + u32((cx - 1 + cols) % cols);
    let c_right = u32(cy) * params.soil_grid.x + u32((cx + 1 + cols) % cols);

    atomicAdd(&soil_buffer[c_up].taint_micro, v18_micro);
    atomicMin(&soil_buffer[c_up].taint_micro, SOIL_CELL_MAX_MICRO);
    atomicAdd(&soil_buffer[c_down].taint_micro, v18_micro);
    atomicMin(&soil_buffer[c_down].taint_micro, SOIL_CELL_MAX_MICRO);
    atomicAdd(&soil_buffer[c_left].taint_micro, v18_micro);
    atomicMin(&soil_buffer[c_left].taint_micro, SOIL_CELL_MAX_MICRO);
    atomicAdd(&soil_buffer[c_right].taint_micro, v18_micro);
    atomicMin(&soil_buffer[c_right].taint_micro, SOIL_CELL_MAX_MICRO);
}

fn deposit_scent_radius1(p: vec2f, val: f32) {
    let wrapped = wrap_coords(p);
    let cell_w = params.world_size.x / f32(params.soil_grid.x);
    let cell_h = params.world_size.y / f32(params.soil_grid.y);
    let cx = i32(min(u32(max(0.0, wrapped.x) / cell_w), params.soil_grid.x - 1u));
    let cy = i32(min(u32(max(0.0, wrapped.y) / cell_h), params.soil_grid.y - 1u));
    let cols = i32(params.soil_grid.x);
    let rows = i32(params.soil_grid.y);

    let c0 = u32(cy) * params.soil_grid.x + u32(cx);
    atomicAdd(&soil_buffer[c0].scent_micro, i32(round(val * SOIL_SCALE)));
    atomicMin(&soil_buffer[c0].scent_micro, SOIL_CELL_MAX_MICRO);

    let v18_micro = i32(round((val / 1.8) * SOIL_SCALE));
    let c_up = u32((cy - 1 + rows) % rows) * params.soil_grid.x + u32(cx);
    let c_down = u32((cy + 1 + rows) % rows) * params.soil_grid.x + u32(cx);
    let c_left = u32(cy) * params.soil_grid.x + u32((cx - 1 + cols) % cols);
    let c_right = u32(cy) * params.soil_grid.x + u32((cx + 1 + cols) % cols);

    atomicAdd(&soil_buffer[c_up].scent_micro, v18_micro);
    atomicMin(&soil_buffer[c_up].scent_micro, SOIL_CELL_MAX_MICRO);
    atomicAdd(&soil_buffer[c_down].scent_micro, v18_micro);
    atomicMin(&soil_buffer[c_down].scent_micro, SOIL_CELL_MAX_MICRO);
    atomicAdd(&soil_buffer[c_left].scent_micro, v18_micro);
    atomicMin(&soil_buffer[c_left].scent_micro, SOIL_CELL_MAX_MICRO);
    atomicAdd(&soil_buffer[c_right].scent_micro, v18_micro);
    atomicMin(&soil_buffer[c_right].scent_micro, SOIL_CELL_MAX_MICRO);
}

fn get_gene(agent_idx: u32, p: u32) -> f32 {
    let word = agent_genomes[agent_idx].packed_genes[p >> 2u];
    let shift = (p & 3u) << 3u;
    let b = (i32(word >> shift) << 24) >> 24;
    return f32(b);
}

// Random streams: every use draws from its own, so no two draws are correlated.
const MATE_ROLL_STREAM: u32 = 0x20000000u;
const AUDIO_STREAM: u32 = 0x21000000u;

// pcg3d (Jarzynski & Olano 2020): a stateless hash of (id, stream, tick).
fn pcg3d(id: u32, stream: u32, tick: u32) -> u32 {
    var v = vec3u(id, stream, tick) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v.x;
}

// Uniform in [0, 1) from the top 24 bits (exact in f32), as the reference's Math.random().
fn pcg3d_float(id: u32, stream: u32, tick: u32) -> f32 {
    return f32(pcg3d(id, stream, tick) >> 8u) / 16777216.0;
}

fn toroidal_aabb_dist_1d(p: f32, b_min: f32, b_max: f32, w: f32) -> f32 {
    if (p >= b_min && p <= b_max) { return 0.0; }
    let direct = select(p - b_max, b_min - p, p < b_min);
    let wrapped = select(w - p + b_min, w - b_max + p, p < b_min);
    return min(direct, wrapped);
}

fn distance_to_aabb(pos: vec2f, aabb_min: vec2f, aabb_max: vec2f) -> f32 {
    let dx = toroidal_aabb_dist_1d(pos.x, aabb_min.x, aabb_max.x, params.world_size.x);
    let dy = toroidal_aabb_dist_1d(pos.y, aabb_min.y, aabb_max.y, params.world_size.y);
    return sqrt(dx * dx + dy * dy);
}

fn toroidal_dist(p1: vec2f, p2: vec2f) -> f32 {
    let dx = abs(p1.x - p2.x);
    let x_dist = min(dx, params.world_size.x - dx);
    let dy = abs(p1.y - p2.y);
    let y_dist = min(dy, params.world_size.y - dy);
    return sqrt(x_dist * x_dist + y_dist * y_dist);
}

// Squared toroidal distance (the reference wraps each axis to the nearer image).
fn toroidal_dist2(p1: vec2f, p2: vec2f) -> f32 {
    let dx = abs(p1.x - p2.x);
    let x_dist = min(dx, params.world_size.x - dx);
    let dy = abs(p1.y - p2.y);
    let y_dist = min(dy, params.world_size.y - dy);
    return x_dist * x_dist + y_dist * y_dist;
}

// Queues a chronicle entry for this frame's readback; past 64 in one frame, entries are dropped.
fn push_chronicle(tick: u32, event_type: u32, p1: u32, p2: u32) {
    let k = atomicAdd(&queue_buffer.telemetry.chronicle_count, 1u);
    if (k < 64u) {
        queue_buffer.chronicle[k] = ChronicleEvent(tick, event_type, p1, p2);
    }
}

// Called only by the thread that won a death claim. The slot waits on the pending-free list
// (freelist[max_agents..]) until the merge after the birth pass, so a newborn never lands in
// the slot of a creature that died this sub-tick.
fn push_pending_free(slot: u32) {
    let pending = atomicAdd(&queue_buffer.telemetry.pending_free_count, 1u);
    if (pending < params.max_agents) {
        freelist[params.max_agents + pending] = slot;
    }
}

@compute @workgroup_size(64)
fn agent_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.max_agents) { return; }
    // Threads take the slots in Morton order: the sort's first max_agents keys hold every slot
    // once, live creatures first by position, so neighbouring threads read neighbouring
    // creatures, cell lists and soil cells.
    let agent_idx = spatial_keys[id.x].y;
    // The number of the tick being computed: the reference's evolve() does tick++ before the
    // creatures run, and the driver already writes each sub-tick's own tick into params.tick.
    let step_tick = params.tick + 1u;

    // Top-of-Shader Tombstone Dead-Check Guard with dead_claimed CAS synchronization:
    let m_flags = agent_states[agent_idx].meta_flags;
    let is_dead = (m_flags & (1u << 13u)) != 0u;
    let dead_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed) != 0u;
    if (is_dead || dead_claimed) {
        if (!is_dead) {
            agent_states[agent_idx].meta_flags |= (1u << 13u);
            agent_states[agent_idx].angle_energy[1] = 0.0;
            agent_states[agent_idx].visual_cache = 0u;
        }
        return;
    }

    let pos = agent_states[agent_idx].pos_vel.xy;
    let vel = agent_states[agent_idx].pos_vel.zw;

    var angle = agent_states[agent_idx].angle_energy[0];
    let atomic_energy_milli = atomicLoad(&agent_atomics[agent_idx].energy_milli);
    if (atomic_energy_milli <= 0) {
        let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
        if (claim_death.exchanged) {
            atomicAdd(&queue_buffer.telemetry.starvations, 1u);
            atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
            push_pending_free(agent_idx);
            deposit_food_radius2(pos, 0.6);
            deposit_taint_radius1(pos, 0.1);
        }
        agent_states[agent_idx].meta_flags |= (1u << 13u);
        agent_states[agent_idx].angle_energy[1] = 0.0;
        agent_states[agent_idx].visual_cache = 0u;
        return;
    }

    var a_energy = max(0.0, f32(atomic_energy_milli) * 0.001);
    var a_feed = agent_states[agent_idx].angle_energy[2];
    var a_attack = agent_states[agent_idx].angle_energy[3];

    let tr0 = agent_states[agent_idx].traits[0][0]; // bulk
    let tr1 = agent_states[agent_idx].traits[0][1];
    let tr2 = agent_states[agent_idx].traits[0][2]; // sight
    let tr3 = agent_states[agent_idx].traits[0][3]; // armor
    let tr4 = agent_states[agent_idx].traits[1][0];
    let tr5 = agent_states[agent_idx].traits[1][1]; // carnivory
    let a_root = agent_states[agent_idx].root;
    var a_cooldown = (m_flags >> 4u) & 0x03u;
    var a_birth = (m_flags >> 6u) & 0x7Fu;
    var a_kills = (m_flags >> 14u) & 0x3FFFFu;

    var a_age = agent_states[agent_idx].age_gen & 0xFFFFu;
    let a_gen = agent_states[agent_idx].age_gen >> 16u;

    a_age = min(a_age + 1u, 65535u);
    if (a_cooldown > 0u) { a_cooldown -= 1u; }
    if (a_birth > 0u) { a_birth -= 1u; }

    // Antennae feelers
    // HTML: reach = 19 + 38 * tr2 (sight trait)
    let reach = 19.0 + 38.0 * tr2;
    let fwd_term = vec2f(cos(angle), sin(angle)) * reach;

    let probe_here    = sample_soil_probe(pos);
    let probe_forward = sample_soil_probe(pos + fwd_term);
    let mid_antennae  = pos + fwd_term * 0.7;
    let off_antennae  = vec2f(-fwd_term.y, fwd_term.x) * 0.6;
    let probe_left    = sample_soil_probe(mid_antennae + off_antennae);
    let probe_right   = sample_soil_probe(mid_antennae - off_antennae);

    let here_food    = probe_here.r;
    let forward_food = probe_forward.r;
    let left_food    = probe_left.r;
    let right_food   = probe_right.r;
    let here_taint   = probe_here.g;
    let scent_diff   = probe_forward.b - probe_here.b;

    // Nearest neighbour and density, as the reference near(): the 3×3 cells first (density
    // counts d² < 10000), then every live creature when nothing is within 100 px. A creature
    // outside the 3×3 cells is at least 100 px away, so the result is the true nearest.
    // spatial_keys: [0, N) sorted keys, [N, N + 54) cell table, [N + 54, ...) cell_slots.
    var best_d2 = 1e30;
    var best_neighbor = 0xFFFFFFFFu;
    var density = 0.0;
    let n_keys = params.max_agents;
    {
        let gx = i32(min(u32(max(0.0, pos.x) / (params.world_size.x / 9.0)), 8u));
        let gy = i32(min(u32(max(0.0, pos.y) / (params.world_size.y / 6.0)), 5u));
        for (var dy = -1; dy <= 1; dy++) {
            let n_gy = (gy + dy + 6) % 6;
            for (var dx = -1; dx <= 1; dx++) {
                let n_gx = (gx + dx + 9) % 9;
                let range = spatial_keys[n_keys + u32(n_gy * 9 + n_gx)];
                if (range.x == 0xFFFFFFFFu) { continue; }
                for (var k = 0u; k < range.y; k += 1u) {
                    let other = spatial_keys[n_keys + 54u + range.x + k].x;
                    if (other == agent_idx || (agent_states[other].meta_flags & (1u << 13u)) != 0u) { continue; }
                    let d2 = toroidal_dist2(pos, agent_states[other].pos_vel.xy);
                    if (d2 < 10000.0) { density += 1.0; }
                    if (d2 < best_d2 || (d2 == best_d2 && other < best_neighbor)) {
                        best_d2 = d2;
                        best_neighbor = other;
                    }
                }
            }
        }
        if (best_d2 >= 10000.0) {
            let live_count = min(atomicLoad(&queue_buffer.telemetry.live_count), n_keys);
            for (var i = 0u; i < live_count; i += 1u) {
                let other = spatial_keys[i].y;
                if (other == agent_idx || (agent_states[other].meta_flags & (1u << 13u)) != 0u) { continue; }
                let d2 = toroidal_dist2(pos, agent_states[other].pos_vel.xy);
                if (d2 < best_d2 || (d2 == best_d2 && other < best_neighbor)) {
                    best_d2 = d2;
                    best_neighbor = other;
                }
            }
        }
    }
    let best_dist = sqrt(best_d2);

    // 15 Canonical Sensory Inputs (matching clankolution.html lines 1334-1362)
    var sensory_inputs: array<f32, 15>;
    sensory_inputs[0] = clamp(a_energy / 75.0 - 1.0, -1.0, 1.0);
    sensory_inputs[1] = clamp(here_food - 1.0, -1.0, 1.0);
    sensory_inputs[2] = clamp(forward_food - left_food, -1.0, 1.0);
    sensory_inputs[3] = clamp(forward_food - right_food, -1.0, 1.0);
    sensory_inputs[4] = clamp(forward_food - here_food, -1.0, 1.0);
    sensory_inputs[5] = clamp(here_taint, 0.0, 1.0);
    sensory_inputs[6] = clamp(scent_diff, -1.0, 1.0);

    if (best_neighbor != 0xFFFFFFFFu) {
        let neighbor_pos = agent_states[best_neighbor].pos_vel.xy;
        var dx = neighbor_pos.x - pos.x;
        if (dx > params.world_size.x * 0.5) { dx -= params.world_size.x; }
        else if (dx < -params.world_size.x * 0.5) { dx += params.world_size.x; }

        var dy = neighbor_pos.y - pos.y;
        if (dy > params.world_size.y * 0.5) { dy -= params.world_size.y; }
        else if (dy < -params.world_size.y * 0.5) { dy += params.world_size.y; }

        let angle_to_neighbor = atan2(dy, dx);
        let bearing = angle_to_neighbor - angle;

        sensory_inputs[7] = sin(bearing);
        sensory_inputs[8] = cos(bearing);
        sensory_inputs[9] = clamp(1.0 - best_dist / (100.0 + 70.0 * tr2), -1.0, 1.0);
        sensory_inputs[14] = clamp(agent_states[best_neighbor].angle_energy[1] / 70.0 - 1.0, -1.0, 1.0);
    } else {
        sensory_inputs[7] = 0.0;
        sensory_inputs[8] = 1.0;
        sensory_inputs[9] = -1.0;
        sensory_inputs[14] = 0.0;
    }

    sensory_inputs[10] = clamp(density / 9.0, 0.0, 1.0);
    sensory_inputs[11] = clamp(f32(a_age) / 600.0, 0.0, 1.0);
    sensory_inputs[12] = sin(f32(step_tick) * 0.08 + f32(agent_states[agent_idx].id));
    sensory_inputs[13] = clamp(vel.x * cos(angle) + vel.y * sin(angle), -1.0, 1.0);

    // Canonical 326-Weight Elman RNN Forward Pass
    let h_prev = array<f32, 10>(
        agent_states[agent_idx].hidden[0].x, agent_states[agent_idx].hidden[0].y,
        agent_states[agent_idx].hidden[0].z, agent_states[agent_idx].hidden[0].w,
        agent_states[agent_idx].hidden[1].x, agent_states[agent_idx].hidden[1].y,
        agent_states[agent_idx].hidden[1].z, agent_states[agent_idx].hidden[1].w,
        agent_states[agent_idx].hidden_tail.x, agent_states[agent_idx].hidden_tail.y
    );

    let scale_h = 0.61 / 127.0;
    let scale_o = 0.66 / 127.0;

    var p = 0u;
    var new_h: array<f32, 10>;
    for (var j = 0u; j < 10u; j += 1u) {
        var s = 0.0;
        for (var k = 0u; k < 15u; k += 1u) {
            s += get_gene(agent_idx, p) * sensory_inputs[k];
            p += 1u;
        }
        for (var k = 0u; k < 10u; k += 1u) {
            s += get_gene(agent_idx, p) * h_prev[k];
            p += 1u;
        }
        s += get_gene(agent_idx, p); // bias
        p += 1u;
        new_h[j] = tanh(s * scale_h);
    }

    var out: array<f32, 6>;
    for (var j = 0u; j < 6u; j += 1u) {
        var s = 0.0;
        for (var k = 0u; k < 10u; k += 1u) {
            s += get_gene(agent_idx, p) * new_h[k];
            p += 1u;
        }
        s += get_gene(agent_idx, p); // bias
        p += 1u;
        out[j] = tanh(s * scale_o);
    }

    // Heading direction before steering (matching HTML line 1299 and clank_core line 712)
    let fwd_dir = vec2f(cos(angle), sin(angle));

    // Actuator 0: Steering (HTML: a.angle += o[0] * (0.11 + 0.09 * tr1))
    angle += out[0] * (0.11 + 0.09 * tr1);

    // Actuator 1: Thrust & Kinematics (HTML: thrust = (o[1] + 1) * 0.5; mot = 0.45 + 1.1 * tr1)
    let thrust = (out[1] + 1.0) * 0.5;
    let mot = 0.45 + 1.1 * tr1;
    let thrust_force = fwd_dir * (thrust * mot * 0.22);
    let new_vel = (vel + thrust_force) * 0.89;
    var new_pos = wrap_coords(pos + new_vel);

    a_feed = max(0.0, out[2]);
    a_attack = max(0.0, out[3]);
    let a_signal = max(0.0, out[4]);
    var a_last_victim = u32(agent_states[agent_idx].traits[1][3]);

    // 1. Soil Grazing
    let cell_w = params.world_size.x / f32(params.soil_grid.x);
    let cell_h = params.world_size.y / f32(params.soil_grid.y);
    let cx = min(u32(max(0.0, new_pos.x) / cell_w), params.soil_grid.x - 1u);
    let cy = min(u32(max(0.0, new_pos.y) / cell_h), params.soil_grid.y - 1u);
    let cell_idx = cy * params.soil_grid.x + cx;
    let intake_cap = (0.016 + 0.064 * a_feed) * (0.7 + tr4);
    let intake_micro = i32(round(intake_cap * SOIL_SCALE));
    var eaten_micro = 0;
    if (intake_micro > 0) {
        var cur_food_micro = atomicLoad(&soil_buffer[cell_idx].food_micro);
        while (cur_food_micro > 0) {
            let take = min(intake_micro, cur_food_micro);
            let cas = atomicCompareExchangeWeak(
                &soil_buffer[cell_idx].food_micro,
                cur_food_micro,
                cur_food_micro - take,
            );
            if (cas.exchanged) {
                eaten_micro = take;
                break;
            }
            cur_food_micro = cas.old_value;
        }
    }
    let eaten_float = f32(eaten_micro) / SOIL_SCALE;
    if (eaten_micro > 0) {
        atomicAdd(&queue_buffer.telemetry.food_grazed_milli, u32(round(eaten_float * 1000.0)));
    }

    // 2. Scent emission (HTML line 1394, radius 1)
    if (a_signal > 0.4) {
        deposit_scent_radius1(new_pos, (a_signal - 0.4) * 0.035);
    }

    // 3. Basal, Thrust, and Taint Metabolic Costs
    let energy_gain = eaten_float * (9.0 + 9.0 * tr4);
    let basal_cost = 0.10 + 0.12 * tr0 + 0.07 * tr1 + 0.035 * tr2 + 0.055 * tr3 + 0.035 * a_attack + 0.014 * a_signal;
    let thrust_cost = thrust * 0.06;
    let post_taint = f32(atomicLoad(&soil_buffer[cell_idx].taint_micro)) / SOIL_SCALE;
    let taint_cost = post_taint * (0.10 + 0.18 * (1.0 - tr3));
    let internal_delta_milli = i32((energy_gain - basal_cost - thrust_cost - taint_cost) * 1000.0);
    atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);

    // 4. Combat Resolution & Decisive Killer Attribution
    let contact_dist = 14.0 + 10.0 * tr0;
    if (best_neighbor != 0xFFFFFFFFu && best_dist < contact_dist && a_attack > 0.25 && a_cooldown == 0u) {
        let victim_idx = best_neighbor;
        let victim_tr3 = agent_states[victim_idx].traits[0][3];
        let victim_energy_milli = atomicLoad(&agent_atomics[victim_idx].energy_milli);
        if (victim_energy_milli > 0) {
            let damage = (0.5 + a_attack * 2.2) * params.hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3);
            let damage_milli = i32(damage * 1000.0);
            let old_energy_milli = atomicSub(&agent_atomics[victim_idx].energy_milli, damage_milli);

            a_cooldown = 3u;
            a_last_victim = agent_states[victim_idx].id;
            let combat_gain = damage * (0.1 + 0.55 * tr5);
            atomicAdd(&agent_atomics[agent_idx].energy_milli, i32(combat_gain * 1000.0));

            // Decisive killer attribution:
            if (old_energy_milli > 0 && old_energy_milli <= damage_milli) {
                a_kills += 1u;
                atomicAdd(&queue_buffer.telemetry.kills, 1u);
                // Reference: every 8th kill, "Lineage R has taken N lives."
                if (a_kills % 8u == 0u) {
                    push_chronicle(step_tick, 3u, a_root, a_kills);
                }
                let kill_bonus = min(9.0, 8.0 * tr5);
                atomicAdd(&agent_atomics[agent_idx].energy_milli, i32(kill_bonus * 1000.0));

                // Frustum-culled stochastic audio emission:
                let dx = abs(pos.x - params.camera_pos.x);
                let dist_x = min(dx, params.world_size.x - dx);
                let dy = abs(pos.y - params.camera_pos.y);
                let dist_y = min(dy, params.world_size.y - dy);
                let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
                if (in_view) {
                    let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (params.world_size.x - 150.0), 0.0, 1.0);
                    let kill_volume = mix(0.08, 1.0, zoom_factor);
                    let live_pop = max(params.agent_count, atomicLoad(&queue_buffer.telemetry.live_count));
                    let density_filter = select(1u, 4u, live_pop > 10000u);
                    if (pcg3d(agent_idx, AUDIO_STREAM ^ victim_idx, step_tick) % density_filter == 0u) {
                        let voice_slot = atomicAdd(&queue_buffer.telemetry.audio_voice_count, 1u);
                        if (voice_slot < 256u) {
                            queue_buffer.audio[voice_slot] = AudioVoice(pos, 1u /* EVENT_KILL */, kill_volume);
                        }
                    }
                }

                // Atomic CAS death ownership: only the winner counts the death and frees the slot
                let claim_death = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
                if (claim_death.exchanged) {
                    atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
                    push_pending_free(victim_idx);

                    // Multi-cell corpse deposition from victim (HTML line 1210: food radius 2, taint radius 1)
                    let vpos = agent_states[victim_idx].pos_vel.xy;
                    deposit_food_radius2(vpos, 0.6);
                    deposit_taint_radius1(vpos, 0.1);
                }
            }
        }
    }

    // 5. Reproduction (evaluated AFTER grazing, metabolism & combat in strict parity with HTML lines 1400-1444)
    let current_pre_birth_e = f32(atomicLoad(&agent_atomics[agent_idx].energy_milli)) * 0.001;
    let can_reproduce = (out[5] > -0.15)
        && (current_pre_birth_e > (58.0 + 12.0 * tr0))
        && (a_age > 65u)
        && (a_birth == 0u);

    if (can_reproduce) {
        // Reference mating: no exclusive claim. Several suitors may take the same mate in one
        // tick, each costing it 6 energy.
        var mate_partner = 0xFFFFFFFFu;
        if (best_neighbor != 0xFFFFFFFFu && best_dist < 18.0) {
            let partner_e_milli = atomicLoad(&agent_atomics[best_neighbor].energy_milli);
            if (partner_e_milli > 42000 && agent_states[best_neighbor].root != a_root
                && pcg3d_float(agent_idx, MATE_ROLL_STREAM, step_tick) < 0.15) {
                mate_partner = best_neighbor;
            }
        }

        // The reference's `agents.length < CAP`: reserve one of this sub-tick's free slots.
        // Only a birth that gets one proceeds and charges energy, so the cap holds across
        // sub-ticks and nobody pays for a birth that cannot happen.
        var reserved = false;
        var avail = atomicLoad(&queue_buffer.telemetry.available_slots);
        while (avail > 0u) {
            let r = atomicCompareExchangeWeak(&queue_buffer.telemetry.available_slots, avail, avail - 1u);
            if (r.exchanged) { reserved = true; break; }
            avail = r.old_value;
        }

        // Not select(): WGSL evaluates both of its arguments, so the atomicAdd would queue
        // births that got no slot.
        var queue_idx = 0xFFFFFFFFu;
        if (reserved) { queue_idx = atomicAdd(&queue_buffer.telemetry.birth_count, 1u); }
        if (reserved && queue_idx < 65536u) {
            atomicSub(&agent_atomics[agent_idx].energy_milli, 24000);
            if (mate_partner != 0xFFFFFFFFu) {
                atomicSub(&agent_atomics[mate_partner].energy_milli, 6000);
            }
            a_birth = 95u;
            queue_buffer.births[queue_idx] = BirthEvent(agent_idx, mate_partner, 0xFFFFFFFFu, 0u);

            // Frustum-culled birth audio voice
            let dx = abs(pos.x - params.camera_pos.x);
            let dist_x = min(dx, params.world_size.x - dx);
            let dy = abs(pos.y - params.camera_pos.y);
            let dist_y = min(dy, params.world_size.y - dy);
            let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
            if (in_view) {
                let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (params.world_size.x - 150.0), 0.0, 1.0);
                let birth_volume = mix(0.08, 0.7, zoom_factor);
                let voice_slot = atomicAdd(&queue_buffer.telemetry.audio_voice_count, 1u);
                if (voice_slot < 256u) {
                    queue_buffer.audio[voice_slot] = AudioVoice(pos, 2u /* EVENT_BIRTH */, birth_volume);
                }
            }
        }
    }

    // 6. Strict Single-Writer Death Check (Energy depletion or Senescence at age > 2100)
    let already_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed);
    let current_energy_milli = atomicLoad(&agent_atomics[agent_idx].energy_milli);

    if (already_claimed != 0u || current_energy_milli <= 0 || a_age > 2100u) {
        if (already_claimed == 0u) {
            let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
            if (claim_death.exchanged) {
                atomicAdd(&queue_buffer.telemetry.starvations, 1u);
                atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
                push_pending_free(agent_idx);

                // Corpse deposition into soil (HTML line 1210: food radius 2, taint radius 1)
                let corpse_food = clamp(f32(max(0, current_energy_milli)) * 0.000016 + 0.6, 0.3, 2.0);
                deposit_food_radius2(pos, corpse_food);
                deposit_taint_radius1(pos, 0.1);
            }
        }
        agent_states[agent_idx].meta_flags |= (1u << 13u);
        agent_states[agent_idx].angle_energy[1] = 0.0;
        agent_states[agent_idx].visual_cache = 0u;
        return;
    }

    // Clamp to the reference's 110 ceiling atomically: a plain store of what this thread read
    // would erase damage another creature's thread subtracts in between.
    atomicMin(&agent_atomics[agent_idx].energy_milli, 110000);
    a_energy = clamp(f32(current_energy_milli) * 0.001, 0.0, 110.0);
    atomicMax(&queue_buffer.telemetry.apex_record_milli, u32(a_energy * 1000.0));

    // Commit state
    agent_states[agent_idx].pos_vel = vec4f(new_pos.x, new_pos.y, new_vel.x, new_vel.y);
    agent_states[agent_idx].angle_energy = vec4f(angle, a_energy, a_feed, a_attack);
    agent_states[agent_idx].traits[1] = vec4f(tr4, tr5, a_signal, f32(a_last_victim));
    agent_states[agent_idx].hidden[0] = vec4f(new_h[0], new_h[1], new_h[2], new_h[3]);
    agent_states[agent_idx].hidden[1] = vec4f(new_h[4], new_h[5], new_h[6], new_h[7]);
    agent_states[agent_idx].hidden_tail = vec2f(new_h[8], new_h[9]);
    agent_states[agent_idx].meta_flags = (a_cooldown << 4u) | (a_birth << 6u) | (a_kills << 14u);
    agent_states[agent_idx].age_gen = (a_age & 0xFFFFu) | (a_gen << 16u);

    // Pack visual_cache: radius_u8 (0..7), glow_u8/signal (8..15), energy_u8 (16..23), is_attacking (24: a_attack > 0.5), birth_val (25..31)
    let r_u8 = u32(clamp(tr0, 0.0, 1.0) * 255.0);
    let glow_u8 = u32(clamp(a_signal, 0.0, 1.0) * 255.0);
    let e_u8 = u32(clamp(a_energy / 100.0, 0.0, 1.0) * 255.0);
    let vis_flags = select(0u, 1u << 24u, a_attack > 0.5) | ((a_birth & 0x7Fu) << 25u);
    agent_states[agent_idx].visual_cache = r_u8 | (glow_u8 << 8u) | (e_u8 << 16u) | vis_flags;
}
