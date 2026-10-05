// crates/clank_app/assets/shaders/lbvh_build.wgsl
// Two-Phase Linear Bounding Volume Hierarchy (LBVH) Construction (Karras 2012)

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

@group(0) @binding(0) var<storage, read> spatial_keys: array<vec2u>; // (morton_key, agent_id)
@group(0) @binding(1) var<storage, read_write> lbvh_nodes: array<GpuLbvhNode>;
@group(0) @binding(2) var<storage, read_write> node_flags: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(4) var<uniform> params: GpuSimParams;

fn common_prefix_length(i: i32, j: i32, n: u32) -> i32 {
    if (j < 0 || j >= i32(n)) { return -1; }
    let key_i = agent_states[i].morton_code;
    let key_j = agent_states[j].morton_code;
    if (key_i != key_j) {
        return i32(countLeadingZeros(key_i ^ key_j));
    }
    // Tie-break with unique agent slot id:
    let id_i = agent_states[i].id;
    let id_j = agent_states[j].id;
    return 32 + i32(countLeadingZeros(id_i ^ id_j));
}

@compute @workgroup_size(64)
fn build_lbvh_hierarchy(@builtin(global_invocation_id) id: vec3u) {
    let n = params.agent_count;
    if (n < 2u || id.x >= n - 1u) { return; } // Guard N <= 1

    if (id.x == 0u) {
        lbvh_nodes[0].parent = 0xFFFFFFFFu;
    }

    let i = i32(id.x);
    atomicStore(&node_flags[id.x], 0u);

    let delta_next = common_prefix_length(i, i + 1, n);
    let delta_prev = common_prefix_length(i, i - 1, n);
    let d = select(-1, 1, delta_next > delta_prev);
    let delta_min = common_prefix_length(i, i - d, n);

    var l_max = 2;
    var bound_steps = 0u;
    while (common_prefix_length(i, i + l_max * d, n) > delta_min && bound_steps < 32u) {
        l_max *= 2;
        bound_steps += 1u;
    }

    var l = 0;
    var step = l_max / 2;
    while (step > 0) {
        if (common_prefix_length(i, i + (l + step) * d, n) > delta_min) {
            l += step;
        }
        step /= 2;
    }

    let j = i + l * d;
    let first = min(i, j);
    let last = max(i, j);
    let delta_node = common_prefix_length(first, last, n);

    var split = first;
    step = last - first;
    loop {
        step = (step + 1) / 2;
        let new_split = split + step;
        if (new_split < last) {
            if (common_prefix_length(first, new_split, n) > delta_node) {
                split = new_split;
            }
        }
        if (step <= 1) {
            break;
        }
    }
    let gamma = split;

    let num_internal = n - 1u;
    let left = select(u32(gamma), num_internal + u32(gamma), gamma == first);
    let right = select(u32(gamma + 1), num_internal + u32(gamma + 1), gamma + 1 == last);

    lbvh_nodes[id.x].left_child = left;
    lbvh_nodes[id.x].right_child = right;
    lbvh_nodes[id.x].leaf_idx = 0xFFFFFFFFu;
    lbvh_nodes[left].parent = id.x;
    lbvh_nodes[right].parent = id.x;
}
