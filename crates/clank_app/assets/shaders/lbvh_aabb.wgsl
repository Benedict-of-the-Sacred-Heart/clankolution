// crates/clank_app/assets/shaders/lbvh_aabb.wgsl
// Phase 2: Bottom-Up Bounding Box & Metric Hierarchy Fitting (Karras 2012)

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
    world_size: vec2f,
    soil_grid: vec2u,
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

@compute @workgroup_size(64)
fn fit_lbvh_aabbs(@builtin(global_invocation_id) id: vec3u) {
    let n = params.agent_count;
    if (id.x >= n) { return; }

    let agent_idx = spatial_keys[id.x].y;

    // Degenerate N=1 guard
    if (n == 1u) {
        let a = agent_states[agent_idx];
        let r = 2.0 + 3.0 * a.traits[0][0];
        lbvh_nodes[0].aabb_min = a.pos_vel.xy - vec2f(r, r);
        lbvh_nodes[0].aabb_max = a.pos_vel.xy + vec2f(r, r);
        lbvh_nodes[0].center_of_mass = a.pos_vel.xy;
        lbvh_nodes[0].count = 1u;
        lbvh_nodes[0].dominant_lineage = a.meta_flags & 0x0Fu;
        lbvh_nodes[0].left_child = 0xFFFFFFFFu;
        lbvh_nodes[0].right_child = 0xFFFFFFFFu;
        lbvh_nodes[0].parent = 0xFFFFFFFFu;
        lbvh_nodes[0].leaf_idx = agent_idx;
        return;
    }

    let num_internal = n - 1u;
    let leaf_node_idx = num_internal + id.x;

    let a = agent_states[agent_idx];
    let r = 2.0 + 3.0 * a.traits[0][0];
    lbvh_nodes[leaf_node_idx].aabb_min = a.pos_vel.xy - vec2f(r, r);
    lbvh_nodes[leaf_node_idx].aabb_max = a.pos_vel.xy + vec2f(r, r);
    lbvh_nodes[leaf_node_idx].center_of_mass = a.pos_vel.xy;
    lbvh_nodes[leaf_node_idx].count = 1u;
    lbvh_nodes[leaf_node_idx].dominant_lineage = a.meta_flags & 0x0Fu;
    lbvh_nodes[leaf_node_idx].left_child = 0xFFFFFFFFu;
    lbvh_nodes[leaf_node_idx].right_child = 0xFFFFFFFFu;
    lbvh_nodes[leaf_node_idx].leaf_idx = agent_idx;

    // Climb tree towards root
    var curr = lbvh_nodes[leaf_node_idx].parent;
    while (curr != 0xFFFFFFFFu) {
        let flag = atomicAdd(&node_flags[curr], 1u);
        if (flag < 1u) {
            // First child arrived; terminate thread to let second child process parent
            break;
        }

        // Second child arrived: compute bounding box and metrics from left and right children
        let left = lbvh_nodes[curr].left_child;
        let right = lbvh_nodes[curr].right_child;

        let min_x = min(lbvh_nodes[left].aabb_min.x, lbvh_nodes[right].aabb_min.x);
        let min_y = min(lbvh_nodes[left].aabb_min.y, lbvh_nodes[right].aabb_min.y);
        let max_x = max(lbvh_nodes[left].aabb_max.x, lbvh_nodes[right].aabb_max.x);
        let max_y = max(lbvh_nodes[left].aabb_max.y, lbvh_nodes[right].aabb_max.y);

        lbvh_nodes[curr].aabb_min = vec2f(min_x, min_y);
        lbvh_nodes[curr].aabb_max = vec2f(max_x, max_y);

        let c_left = lbvh_nodes[left].count;
        let c_right = lbvh_nodes[right].count;
        let total_count = c_left + c_right;
        lbvh_nodes[curr].count = total_count;

        let com = (lbvh_nodes[left].center_of_mass * f32(c_left) + lbvh_nodes[right].center_of_mass * f32(c_right)) / max(1.0, f32(total_count));
        lbvh_nodes[curr].center_of_mass = com;

        if (c_left >= c_right) {
            lbvh_nodes[curr].dominant_lineage = lbvh_nodes[left].dominant_lineage;
        } else {
            lbvh_nodes[curr].dominant_lineage = lbvh_nodes[right].dominant_lineage;
        }

        curr = lbvh_nodes[curr].parent;
    }
}
