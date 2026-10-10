// crates/clank_app/assets/shaders/spatial_query.wgsl
// Unified Interactive Spatial Query Pass (Picking & AoE Tools)

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

struct GpuTelemetry {
    population: u32,
    kills: u32,
    starvations: u32,
    apex_record_milli: u32,
    food_grazed_milli: u32,
    sub_ticks_elapsed: u32,
    apex_agent_id: u32,
    freelist_top: u32,
    birth_count: u32,
    audio_voice_count: u32,
    selected_agent_idx: u32,
    selected_agent_id: u32,
    total_births: u32,
    total_deaths: u32,
    max_generation: u32,
    living_lineages: u32,
    live_count: u32,
    pending_free_count: u32,
    available_slots: u32,
    chronicle_count: u32,
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

struct CullOutput {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

struct DartInstance {
    pos_angle: vec3f,
    pad0: f32,
    vis_data: vec2u,
    pad1: vec2u,
}

struct GpuAgentAtomic {
    energy_milli: atomic<i32>,
    mate_claim: atomic<u32>,
    mate_energy_milli: atomic<u32>,
    dead_claimed: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
// [0, N) sorted keys, [N, N + 54) grid cell table, [N + 54, ...) cell_slots (N = max_agents).
@group(0) @binding(1) var<storage, read> spatial_keys: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(3) var<uniform> params: GpuSimParams;
@group(0) @binding(4) var<storage, read_write> visible_instances: array<u32>;
@group(0) @binding(5) var<storage, read_write> cull_output: CullOutput;
@group(0) @binding(6) var<storage, read_write> dart_instances: array<DartInstance>;
@group(0) @binding(7) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;

@compute @workgroup_size(64)
fn spatial_query_main(@builtin(global_invocation_id) id: vec3u) {
    let w = params.world_size;
    let tool_pos = vec2f(
        params.tool_pos[0] - w.x * floor(params.tool_pos[0] / w.x),
        params.tool_pos[1] - w.y * floor(params.tool_pos[1] / w.y)
    );

    if (params.tool_type == 0u /* inspect/pick */) {
        if (id.x != 0u) { return; }

        // As the Rust engine's find_agent_at_position: the nearest live creature whose body
        // radius plus tool_radius reaches the cursor, by plain distance, the lower id on a tie.
        // That reach (at most 6.8 + 25 px) is under one grid cell (world / 9 by world / 6), so
        // the cursor's 3×3 cells hold every candidate.
        let n_keys = params.max_agents;
        let gx = i32(min(u32(tool_pos.x / (w.x / 9.0)), 8u));
        let gy = i32(min(u32(tool_pos.y / (w.y / 6.0)), 5u));
        var best_idx = 0xFFFFFFFFu;
        var best_id = 0u;
        var best_d = 3.0e38;
        for (var dy = -1; dy <= 1; dy++) {
            let n_gy = (gy + dy + 6) % 6;
            for (var dx = -1; dx <= 1; dx++) {
                let n_gx = (gx + dx + 9) % 9;
                let range = spatial_keys[n_keys + u32(n_gy * 9 + n_gx)];
                if (range.x == 0xFFFFFFFFu) { continue; }
                for (var k = 0u; k < min(range.y, n_keys); k += 1u) {
                    let slot = spatial_keys[n_keys + 54u + range.x + k].x;
                    if (slot >= n_keys || (agent_states[slot].meta_flags & (1u << 13u)) != 0u) { continue; }
                    let a_id = agent_states[slot].id;
                    let d = distance(agent_states[slot].pos_vel.xy, params.tool_pos);
                    let reach = 2.3 + agent_states[slot].traits[0][0] * 4.5 + params.tool_radius;
                    if (d <= reach && (d < best_d || (d == best_d && a_id < best_id))) {
                        best_d = d;
                        best_idx = slot;
                        best_id = a_id;
                    }
                }
            }
        }
        queue_buffer.telemetry.selected_agent_idx = best_idx;
        queue_buffer.telemetry.selected_agent_id = best_id;
    } else {
        // AoE Tool Brushes: tool_type >= 1 (1 = nourish, 2 = blight, 3 = extinguish, 4 = seed)
        let agent_idx = id.x;
        if (agent_idx >= params.max_agents) { return; }
        let m_flags = agent_states[agent_idx].meta_flags;
        if ((m_flags & (1u << 13u)) != 0u) { return; }

        let pos = agent_states[agent_idx].pos_vel.xy;
        // Extinguish (World::extinguish_at): plain Math.hypot distance to the cursor as given,
        // not toroidal and not wrapped, strictly inside the radius.
        let d = distance(params.tool_pos, pos);
        let aoe_radius = select(params.tool_radius, 23.0, params.tool_radius <= 0.0);
        if (d < aoe_radius) {
            if (params.tool_type == 3u) {
                // Extinguish: kill agent by setting energy to 0.
                // Standard atomic death resolution runs in agent_step (depositing corpse & pushing freelist).
                atomicStore(&agent_atomics[agent_idx].energy_milli, 0);
                agent_states[agent_idx].angle_energy[1] = 0.0;
            }
        }
    }
}

@compute @workgroup_size(64)
fn frustum_cull_clear(@builtin(global_invocation_id) id: vec3u) {
    if (id.x == 0u) {
        cull_output.vertex_count = 12u;
        atomicStore(&cull_output.instance_count, 0u);
        cull_output.first_vertex = 0u;
        cull_output.first_instance = 0u;
    }
}

@compute @workgroup_size(64)
fn frustum_cull_main(@builtin(global_invocation_id) id: vec3u) {
    let agent_idx = id.x;
    if (agent_idx >= params.max_agents) {
        return;
    }

    let state = agent_states[agent_idx];

    // Culling invariant: dead agents (dead flag in meta_flags, visual_cache == 0, or dead_claimed) are culled
    if ((state.meta_flags & (1u << 13u)) != 0u || state.visual_cache == 0u || atomicLoad(&agent_atomics[agent_idx].dead_claimed) != 0u) {
        return;
    }

    // Toroidal frustum test
    let pos = state.pos_vel.xy;
    let cam_pos = params.camera_pos;
    let cam_size = params.camera_size;
    let half_w = cam_size.x * 0.5 + 24.0;
    let half_h = cam_size.y * 0.5 + 24.0;

    let dx = abs(pos.x - cam_pos.x);
    let dist_x = min(dx, params.world_size.x - dx);
    let dy = abs(pos.y - cam_pos.y);
    let dist_y = min(dy, params.world_size.y - dy);

    if (dist_x <= half_w && dist_y <= half_h) {
        let slot = atomicAdd(&cull_output.instance_count, 1u);
        if (slot < params.max_agents) {
            visible_instances[slot] = agent_idx;
            let angle = state.angle_energy[0];
            dart_instances[slot].pos_angle = vec3f(pos.x, pos.y, angle);
            dart_instances[slot].pad0 = state.traits[0].w;
            dart_instances[slot].vis_data = vec2u(state.root, state.visual_cache);
            let carn_u8 = u32(clamp(state.traits[1].y, 0.0, 1.0) * 255.0);
            let packed_carn_id = (carn_u8 << 24u) | (state.id & 0x00FFFFFFu);
            dart_instances[slot].pad1 = vec2u(packed_carn_id, bitcast<u32>(state.traits[0].z));
        }
    }
}

