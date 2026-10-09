// crates/clank_app/assets/shaders/birth_step.wgsl
// Decoupled Birth & Genome Mutation Pass
//
// 1 Workgroup per newborn child (32 parallel threads)
// Pass-Decoupled Freelist allocation (Zero ABA / Push-Pop Race)
// Workgroup-Uniform Bounds Guard

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

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> agent_genomes: array<GpuAgentGenome>;
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(3) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(4) var<storage, read_write> freelist: array<u32>;
@group(1) @binding(0) var<uniform> params: GpuSimParams;

var<workgroup> shared_child_slot: u32;
var<workgroup> shared_child_id: u32;

// Random streams, each offset by the gene or trait index: every draw has its own, so which
// parent a gene comes from is independent of whether it mutates.
const CROSSOVER_STREAM: u32 = 0x10000000u;
const MUTATE_ROLL_STREAM: u32 = 0x11000000u;
const MUTATE_R1_STREAM: u32 = 0x12000000u;
const MUTATE_R2_STREAM: u32 = 0x13000000u;
const TRAIT_CROSSOVER_STREAM: u32 = 0x14000000u;
const TRAIT_R1_STREAM: u32 = 0x15000000u;
const TRAIT_R2_STREAM: u32 = 0x16000000u;
const CHILD_X_STREAM: u32 = 0x17000000u;
const CHILD_Y_STREAM: u32 = 0x17000001u;
const CHILD_ANGLE_STREAM: u32 = 0x17000002u;
const GENERATION_EVENT_STREAM: u32 = 0x18000000u;

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

// r1 + r2 - 1 from two independent streams: triangular on (-1, 1), as the reference.
fn pcg3d_triangular(id: u32, stream1: u32, stream2: u32, tick: u32) -> f32 {
    return pcg3d_float(id, stream1, tick) + pcg3d_float(id, stream2, tick) - 1.0;
}

// JavaScript Math.round: ties toward +infinity (WGSL round() ties to even).
fn js_round(x: f32) -> f32 {
    return floor(x + 0.5);
}

fn wrap_coords(p: vec2f) -> vec2f {
    let w = params.world_size;
    return vec2f(
        p.x - w.x * floor(p.x / w.x),
        p.y - w.y * floor(p.y / w.y)
    );
}

// Trait c of a creature: traits[0] holds 0..3, traits[1].xy holds 4 and 5.
fn trait_of(slot: u32, c: u32) -> f32 {
    if (c < 4u) { return agent_states[slot].traits[0][c]; }
    return agent_states[slot].traits[1][c - 4u];
}

@compute @workgroup_size(32)
fn birth_main(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let total_births = min(atomicLoad(&queue_buffer.telemetry.birth_count), 65536u);
    if (wg_id.x >= total_births) {
        return; // Workgroup-Uniform Bounds Guard
    }

    let event = queue_buffer.births[wg_id.x];
    let parent_a = event.parent_a;
    let parent_b = event.parent_b;
    let sexual = parent_b != 0xFFFFFFFFu && parent_b < params.max_agents;
    // The number of the tick being computed (see agent_step.wgsl).
    let step_tick = params.tick + 1u;

    if (local_id.x == 0u) {
        var cur_top = atomicLoad(&queue_buffer.telemetry.freelist_top);
        var slot = 0xFFFFFFFFu;
        while (cur_top > 0u) {
            let cas = atomicCompareExchangeWeak(&queue_buffer.telemetry.freelist_top, cur_top, cur_top - 1u);
            if (cas.exchanged) {
                slot = freelist[cur_top - 1u];
                break;
            }
            cur_top = cas.old_value;
        }
        shared_child_slot = slot;
        shared_child_id = select(0u, atomicAdd(&queue_buffer.telemetry.apex_agent_id, 1u), slot != 0xFFFFFFFFu);
    }
    let child_idx = workgroupUniformLoad(&shared_child_slot);
    let child_id = workgroupUniformLoad(&shared_child_id);
    // Cannot happen when the agent pass reserved capacity (available_slots); kept as a guard.
    if (child_idx == 0xFFFFFFFFu) { return; }

    // Genome crossover & mutation, 4 genes per word (326 genes in words 0..81; 82..87 spare).
    // Per gene, as the reference: 48% from the mate, then an 11% chance of a mutation of
    // round((r1 + r2 - 1) * 100 * mutRate), clamped to [-127, 127].
    for (var w = local_id.x; w < 88u; w += 32u) {
        if (w >= 82u) {
            agent_genomes[child_idx].packed_genes[w] = 0u;
            continue;
        }
        let word_a = agent_genomes[parent_a].packed_genes[w];
        var word_b = word_a;
        if (sexual) {
            word_b = agent_genomes[parent_b].packed_genes[w];
        }
        var word = 0u;
        for (var b = 0u; b < 4u; b += 1u) {
            let g = w * 4u + b;
            if (g >= 326u) { break; }
            var gene = extractBits(i32(word_a), b * 8u, 8u);
            if (sexual && pcg3d_float(child_id, CROSSOVER_STREAM + g, step_tick) < 0.48) {
                gene = extractBits(i32(word_b), b * 8u, 8u);
            }
            if (pcg3d_float(child_id, MUTATE_ROLL_STREAM + g, step_tick) < 0.11) {
                let tri = pcg3d_triangular(child_id, MUTATE_R1_STREAM + g, MUTATE_R2_STREAM + g, step_tick);
                gene = clamp(gene + i32(js_round(tri * 100.0 * params.mut_rate)), -127, 127);
            }
            word |= (u32(gene) & 0xFFu) << (b * 8u);
        }
        agent_genomes[child_idx].packed_genes[w] = word;
    }

    if (local_id.x == 0u) {
        atomicAdd(&queue_buffer.telemetry.total_births, 1u);
        // Within ±9px of the parent (rand(-9, 9) per axis), wrapped.
        let parent_pos = agent_states[parent_a].pos_vel.xy;
        let offset = vec2f(
            pcg3d_float(child_id, CHILD_X_STREAM, step_tick) * 18.0 - 9.0,
            pcg3d_float(child_id, CHILD_Y_STREAM, step_tick) * 18.0 - 9.0
        );
        let child_pos = wrap_coords(parent_pos + offset);

        agent_states[child_idx].pos_vel = vec4f(child_pos.x, child_pos.y, 0.0, 0.0);
        agent_states[child_idx].angle_energy = vec4f(pcg3d_float(child_id, CHILD_ANGLE_STREAM, step_tick) * 6.2831853, 24.0, 0.0, 0.0);
        atomicStore(&agent_atomics[child_idx].energy_milli, 24000);
        atomicStore(&agent_atomics[child_idx].dead_claimed, 0u);
        atomicStore(&agent_atomics[child_idx].mate_claim, 0u);
        atomicStore(&agent_atomics[child_idx].mate_energy_milli, 0u);

        // Traits: 45% from the mate, then clamp(base + (r1 + r2 - 1) * mutRate * 0.6, .03, .98).
        var tr: array<f32, 6>;
        for (var c = 0u; c < 6u; c += 1u) {
            var base = trait_of(parent_a, c);
            if (sexual && pcg3d_float(child_id, TRAIT_CROSSOVER_STREAM + c, step_tick) < 0.45) {
                base = trait_of(parent_b, c);
            }
            let delta = pcg3d_triangular(child_id, TRAIT_R1_STREAM + c, TRAIT_R2_STREAM + c, step_tick) * params.mut_rate * 0.6;
            tr[c] = clamp(base + delta, 0.03, 0.98);
        }
        agent_states[child_idx].traits[0] = vec4f(tr[0], tr[1], tr[2], tr[3]);
        agent_states[child_idx].traits[1] = vec4f(tr[4], tr[5], 0.0, 0.0); // signal 0, last victim 0

        // Newborn recurrent hidden states must start at 0
        agent_states[child_idx].hidden[0] = vec4f(0.0);
        agent_states[child_idx].hidden[1] = vec4f(0.0);
        agent_states[child_idx].hidden_tail = vec2f(0.0);

        agent_states[child_idx].root = agent_states[parent_a].root;
        agent_states[child_idx].meta_flags = 0u; // birth = 0, dead = 0

        let parent_gen = agent_states[parent_a].age_gen >> 16u;
        var max_parent_gen = parent_gen;
        if (sexual) {
            max_parent_gen = max(parent_gen, agent_states[parent_b].age_gen >> 16u);
        }
        let child_gen = max_parent_gen + 1u;
        agent_states[child_idx].age_gen = (child_gen & 0xFFFFu) << 16u;
        // Reference: 8% of births at a positive multiple of 12 generations are chronicled.
        if (child_gen > 0u && child_gen % 12u == 0u
            && pcg3d_float(child_id, GENERATION_EVENT_STREAM, step_tick) < 0.08) {
            let k = atomicAdd(&queue_buffer.telemetry.chronicle_count, 1u);
            if (k < 64u) {
                queue_buffer.chronicle[k] =
                    ChronicleEvent(step_tick, 2u, child_gen, agent_states[parent_a].root);
            }
        }
        agent_states[child_idx].id = child_id;
        agent_states[child_idx].morton_code = 0u;

        let child_r_u8 = u32(clamp(tr[0], 0.0, 1.0) * 255.0);
        let child_e_u8 = u32(clamp(24.0 / 100.0, 0.0, 1.0) * 255.0);
        agent_states[child_idx].visual_cache = child_r_u8 | (child_e_u8 << 16u);
    }
}

var<workgroup> merge_base: u32;
var<workgroup> merge_count: u32;

// After the birth pass: moves this sub-tick's deaths from the pending-free list
// (freelist[max_agents..]) onto the stack, so a slot freed this sub-tick is reused only on
// the next one. One workgroup: thread 0 reserves the stack range, all threads copy.
@compute @workgroup_size(64)
fn merge_pending_free_main(@builtin(local_invocation_id) local_id: vec3u) {
    if (local_id.x == 0u) {
        let base = atomicLoad(&queue_buffer.telemetry.freelist_top);
        let count = min(atomicLoad(&queue_buffer.telemetry.pending_free_count), params.max_agents - min(base, params.max_agents));
        merge_base = base;
        merge_count = count;
        atomicStore(&queue_buffer.telemetry.freelist_top, base + count);
        atomicStore(&queue_buffer.telemetry.pending_free_count, 0u);
    }
    let count = workgroupUniformLoad(&merge_count);
    let base = workgroupUniformLoad(&merge_base);
    for (var i = local_id.x; i < count; i += 64u) {
        freelist[base + i] = freelist[params.max_agents + i];
    }
}
