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

struct GpuLbvhNode {
    aabb_min: vec2f,
    aabb_max: vec2f,
    center_of_mass: vec2f,
    count: u32,
    dominant_lineage: u32,
    left_child: u32,
    right_child: u32,
    parent: u32,
    leaf_idx: u32,
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
    extinctions: u32,
    lineage_counts: array<u32, 16>,
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

struct ConsolidatedQueue {
    telemetry: GpuTelemetry,
    births: array<BirthEvent, 65536>,
    audio: array<AudioVoice, 256>,
}

struct CullOutput {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct DartInstance {
    pos_angle: vec3f,
    pad0: f32,
    vis_data: vec2u,
    pad1: vec2u,
}

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read> lbvh_nodes: array<GpuLbvhNode>;
@group(0) @binding(2) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(3) var<uniform> params: GpuSimParams;
@group(0) @binding(4) var<storage, read_write> visible_instances: array<u32>;
@group(0) @binding(5) var<storage, read_write> cull_output: CullOutput;
@group(0) @binding(6) var<storage, read_write> dart_instances: array<DartInstance>;

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

@compute @workgroup_size(64)
fn spatial_query_main(@builtin(global_invocation_id) id: vec3u) {
    let w = params.world_size;
    let tool_pos = vec2f(
        params.tool_pos[0] - w.x * floor(params.tool_pos[0] / w.x),
        params.tool_pos[1] - w.y * floor(params.tool_pos[1] / w.y)
    );

    if (params.tool_type == 0u /* inspect/pick */) {
        if (id.x != 0u) { return; }

        if (params.agent_count == 0u) {
            queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
            queue_buffer.telemetry.selected_agent_id = 0u;
            return;
        }

        var search_r = clamp(16.0 * (params.camera_size[0] / params.world_size.x), 4.0, 24.0);

        if (params.agent_count == 1u) {
            let slot = lbvh_nodes[0].leaf_idx;
            if (slot < params.max_agents) {
                let agent_pos = agent_states[slot].pos_vel.xy;
                let d = toroidal_dist(tool_pos, agent_pos);
                if (d <= search_r) {
                    queue_buffer.telemetry.selected_agent_idx = slot;
                    queue_buffer.telemetry.selected_agent_id = agent_states[slot].id;
                } else {
                    queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
                    queue_buffer.telemetry.selected_agent_id = 0u;
                }
            } else {
                queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
                queue_buffer.telemetry.selected_agent_id = 0u;
            }
            return;
        }

        var best_idx = 0xFFFFFFFFu;
        var best_id = 0u;
        var best_priority = 2u; // 0 = body hit, 1 = halo, 2 = none
        var best_dist = search_r;

        var stack: array<u32, 64>;
        var stack_ptr = 0u;
        stack[stack_ptr] = 0u; // Root
        stack_ptr += 1u;

        while (stack_ptr > 0u) {
            stack_ptr -= 1u;
            let node_idx = stack[stack_ptr];
            let node = lbvh_nodes[node_idx];

            let box_dist = distance_to_aabb(tool_pos, node.aabb_min, node.aabb_max);
            if (box_dist > search_r) { continue; }

            if (node.leaf_idx != 0xFFFFFFFFu) {
                let agent_idx = node.leaf_idx;
                let agent_pos = agent_states[agent_idx].pos_vel.xy;
                let d = toroidal_dist(tool_pos, agent_pos);
                let visual_r = 2.0 + 3.0 * agent_states[agent_idx].traits[0][0];

                let priority = select(1u, 0u, d <= visual_r);
                if (d <= search_r) {
                    if (priority < best_priority || (priority == best_priority && d < best_dist)) {
                        best_priority = priority;
                        best_dist = d;
                        best_idx = agent_idx;
                        best_id = agent_states[agent_idx].id;

                        // Dynamic Radius Shrinking
                        if (priority == 0u) {
                            search_r = min(search_r, d);
                        }
                    }
                }
            } else {
                if (stack_ptr < 62u) {
                    if (node.right_child != 0xFFFFFFFFu) {
                        stack[stack_ptr] = node.right_child;
                        stack_ptr += 1u;
                    }
                    if (node.left_child != 0xFFFFFFFFu) {
                        stack[stack_ptr] = node.left_child;
                        stack_ptr += 1u;
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
        let d = toroidal_dist(tool_pos, pos);
        let aoe_radius = select(params.tool_radius, 60.0, params.tool_radius <= 0.0);
        if (d <= aoe_radius) {
            if (params.tool_type == 1u) {
                // Nourish: grant energy
                agent_states[agent_idx].angle_energy[1] = min(agent_states[agent_idx].angle_energy[1] + 25.0, 150.0);
            } else if (params.tool_type == 2u) {
                // Blight: deduct energy
                agent_states[agent_idx].angle_energy[1] = max(agent_states[agent_idx].angle_energy[1] - 30.0, 0.0);
            } else if (params.tool_type == 3u) {
                // Extinguish: kill agent
                agent_states[agent_idx].meta_flags |= (1u << 13u);
                agent_states[agent_idx].angle_energy[1] = 0.0;
                agent_states[agent_idx].visual_cache = 0u;
            }
        }
    }
}

@compute @workgroup_size(64)
fn frustum_cull_clear(@builtin(global_invocation_id) id: vec3u) {
    if (id.x == 0u) {
        atomicStore(&cull_output.count, 0u);
    }
}

@compute @workgroup_size(64)
fn frustum_cull_main(@builtin(global_invocation_id) id: vec3u) {
    let agent_idx = id.x;
    if (agent_idx >= params.max_capacity || agent_idx >= params.max_agents) {
        return;
    }

    let state = agent_states[agent_idx];

    // Culling invariant: dead agents (dead flag in meta_flags or visual_cache == 0) are culled
    if ((state.meta_flags & (1u << 13u)) != 0u || state.visual_cache == 0u) {
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
        let slot = atomicAdd(&cull_output.count, 1u);
        if (slot < params.max_capacity && slot < params.max_agents) {
            visible_instances[slot] = agent_idx;
            let angle = state.angle_energy[0];
            dart_instances[slot].pos_angle = vec3f(pos.x, pos.y, angle);
            dart_instances[slot].pad0 = 0.0;
            dart_instances[slot].vis_data = vec2u(state.packed_color, state.visual_cache);
            dart_instances[slot].pad1 = vec2u(0u, 0u);
        }
    }
}

