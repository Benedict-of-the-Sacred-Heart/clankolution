// crates/clank_app/assets/shaders/birth_step.wgsl
// Decoupled Birth & Genome Mutation Pass
//
// 1 Workgroup per newborn child (32 parallel threads)
// Pass-Decoupled Freelist allocation (Zero ABA / Push-Pop Race)
// Workgroup-Uniform Bounds Guard

override ENABLE_EXPANDED_CORTEX: bool = false;

struct GpuSimParams {
    tick: u32,
    agent_count: u32,
    max_agents: u32,
    hostility: f32,
    mut_rate: f32,
    speed: f32,
    renewal: f32,
    sub_tick: u32,
    sub_ticks_per_frame: u32,
    tool_type: u32,
    tool_pos: vec2f,
    camera_pos: vec2f,
    camera_size: vec2f,
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
    packed_color: u32,
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
    _reserved0: array<u32, 4>,
    lineage_counts: array<atomic<u32>, 16>,
}

struct BirthEvent {
    parent_a: u32,
    parent_b: u32,
    child_slot: u32,
    pad: u32,
}

struct AudioVoice {
    pos: vec2f,
    event_type: u32,
    volume: f32,
}

struct ConsolidatedQueue {
    telemetry: GpuTelemetry,
    births: array<BirthEvent, 65536>,
    audio: array<AudioVoice, 256>,
}

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> agent_genomes: array<GpuAgentGenome>;
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(3) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(4) var<storage, read_write> freelist: array<u32>;
@group(1) @binding(0) var<uniform> params: GpuSimParams;

var<workgroup> shared_child_slot: u32;

fn pcg_hash(id: u32, stream: u32, tick: u32) -> u32 {
    let state = id * 747796405u + stream * 2891336453u + tick * 1013904223u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_float(id: u32, stream: u32, tick: u32) -> f32 {
    return f32(pcg_hash(id, stream, tick)) / 4294967295.0;
}

fn pcg_triangular(id: u32, stream: u32, tick: u32) -> f32 {
    let r1 = pcg_float(id, stream, tick);
    let r2 = pcg_float(id, stream + 100u, tick);
    return r1 + r2 - 1.0;
}

fn wrap_coords(p: vec2f) -> vec2f {
    return vec2f(
        p.x - 900.0 * floor(p.x / 900.0),
        p.y - 600.0 * floor(p.y / 600.0)
    );
}

fn mutate_gene_byte(val: i32, id: u32, stream: u32, tick: u32, mut_rate: f32) -> u32 {
    let tri = pcg_triangular(id, stream, tick);
    let delta = i32(round(tri * 100.0 * mut_rate));
    let new_val = clamp(val + delta, -127, 127);
    return u32(new_val & 0xFF);
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
    }
    workgroupBarrier();

    let child_idx = shared_child_slot;
    if (child_idx == 0xFFFFFFFFu) { return; } // Carrying capacity reached

    // Parallel genome crossover & mutation across 88 words
    for (var w = local_id.x; w < 88u; w += 32u) {
        var word_a = agent_genomes[parent_a].packed_genes[w];
        var word_b = select(word_a, agent_genomes[parent_b].packed_genes[w], parent_b != 0xFFFFFFFFu);
        let p_cross = pcg_float(child_idx, w, params.tick);
        var chosen = select(word_b, word_a, p_cross < 0.48);

        // Unpack 4 bytes, mutate each, and repack
        let b0 = (i32(chosen << 24u) >> 24);
        let b1 = (i32(chosen << 16u) >> 24);
        let b2 = (i32(chosen << 8u) >> 24);
        let b3 = (i32(chosen) >> 24);

        let m0 = mutate_gene_byte(b0, child_idx, w * 4u + 0u, params.tick, params.mut_rate);
        let m1 = mutate_gene_byte(b1, child_idx, w * 4u + 1u, params.tick, params.mut_rate);
        let m2 = mutate_gene_byte(b2, child_idx, w * 4u + 2u, params.tick, params.mut_rate);
        let m3 = mutate_gene_byte(b3, child_idx, w * 4u + 3u, params.tick, params.mut_rate);

        var mutated_word = m0 | (m1 << 8u) | (m2 << 16u) | (m3 << 24u);

        // Mutate dummy clamped bytes in baseline
        if (!ENABLE_EXPANDED_CORTEX) {
            if (w < 70u && (w % 7u) == 6u) {
                mutated_word = mutated_word & 0x0000FFFFu;
            }
            if (w >= 70u && ((w - 70u) % 3u) == 2u) {
                mutated_word = mutated_word & 0x00FFFFFFu;
            }
        }
        agent_genomes[child_idx].packed_genes[w] = mutated_word;
    }


    if (local_id.x == 0u) {
        let parent_pos = agent_states[parent_a].pos_vel.xy;
        let offset = vec2f(
            pcg_triangular(child_idx, 1u, params.tick) * 9.0,
            pcg_triangular(child_idx, 2u, params.tick) * 9.0
        );
        let child_pos = wrap_coords(parent_pos + offset);

        agent_states[child_idx].pos_vel = vec4f(child_pos.x, child_pos.y, 0.0, 0.0);
        agent_states[child_idx].angle_energy = vec4f(pcg_float(child_idx, 3u, params.tick) * 6.2831853, 24.0, 0.0, 0.0);
        atomicStore(&agent_atomics[child_idx].energy_milli, 24000);
        atomicStore(&agent_atomics[child_idx].dead_claimed, 0u);
        atomicStore(&agent_atomics[child_idx].mate_claim, 0u);
        atomicStore(&agent_atomics[child_idx].mate_energy_milli, 0u);

        // Inherit traits with mutation
        let mut_rate = params.mut_rate;
        for (var t = 0u; t < 2u; t += 1u) {
            var tr_vec = agent_states[parent_a].traits[t];
            for (var c = 0u; c < 4u; c += 1u) {
                let delta = pcg_triangular(child_idx, 10u + t * 4u + c, params.tick) * mut_rate * 0.6;
                tr_vec[c] = clamp(tr_vec[c] + delta, 0.03, 0.98);
            }
            agent_states[child_idx].traits[t] = tr_vec;
        }

        let parent_meta = agent_states[parent_a].meta_flags;
        let child_root = parent_meta & 0x0Fu;
        agent_states[child_idx].meta_flags = (child_root & 0x0Fu) | (95u << 6u); // root, birth = 95, dead = 0

        let parent_gen = agent_states[parent_a].age_gen >> 16u;
        agent_states[child_idx].age_gen = ((parent_gen + 1u) & 0xFFFFu) << 16u;
        agent_states[child_idx].id = atomicAdd(&queue_buffer.telemetry.apex_agent_id, 1u);
        agent_states[child_idx].morton_code = 0u;
        agent_states[child_idx].packed_color = agent_states[parent_a].packed_color;

        let child_r_u8 = u32(clamp(agent_states[child_idx].traits[0][0], 0.0, 1.0) * 255.0);
        let child_e_u8 = u32(clamp(24.0 / 100.0, 0.0, 1.0) * 255.0);
        let child_glow_u8 = 255u; // Newborn birth flash!
        agent_states[child_idx].visual_cache = child_r_u8 | (child_glow_u8 << 8u) | (child_e_u8 << 16u);
    }
}
