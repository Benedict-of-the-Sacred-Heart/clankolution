// crates/clank_app/assets/shaders/morton_grid.wgsl
// Hybrid Morton Grid Spatial Indexing & Cell Offset Table Generation

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
    _pad0: u32,

    tool_pos: vec2f,
    camera_pos: vec2f,

    camera_size: vec2f,
    world_size: vec2f,

    soil_grid: vec2u,
    _pad1: vec2u,
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

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> spatial_keys: array<vec2u>; // (morton_key, agent_id)
@group(0) @binding(2) var<storage, read_write> cell_offsets: array<vec2u, 54>;
@group(0) @binding(3) var<uniform> params: GpuSimParams;

fn expand_bits(v_in: u32) -> u32 {
    var v = v_in & 0x0000FFFFu;
    v = (v | (v << 8u)) & 0x00FF00FFu;
    v = (v | (v << 4u)) & 0x0F0F0F0Fu;
    v = (v | (v << 2u)) & 0x33333333u;
    v = (v | (v << 1u)) & 0x55555555u;
    return v;
}

fn compute_morton_32(pos: vec2f) -> u32 {
    let x_norm = u32(clamp(pos.x / params.world_size.x, 0.0, 1.0) * 65535.0);
    let y_norm = u32(clamp(pos.y / params.world_size.y, 0.0, 1.0) * 65535.0);
    return expand_bits(x_norm) | (expand_bits(y_norm) << 1u);
}

@compute @workgroup_size(64)
fn clear_cell_offsets(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < 54u) {
        cell_offsets[id.x] = vec2u(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
}

@compute @workgroup_size(64)
fn morton_encode(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.max_agents) { return; }
    let is_dead = (agent_states[id.x].meta_flags & (1u << 13u)) != 0u;
    if (is_dead) {
        // Dead Agent Partitioning: Assign sentinel so tombstones sort to array tail
        spatial_keys[id.x] = vec2u(0xFFFFFFFFu, id.x);
    } else {
        let code = compute_morton_32(agent_states[id.x].pos_vel.xy);
        agent_states[id.x].morton_code = code;
        spatial_keys[id.x] = vec2u(code, id.x);
    }
}

fn get_cell_id(pos: vec2f) -> u32 {
    let cell_w = params.world_size.x / 9.0;
    let cell_h = params.world_size.y / 6.0;
    let gx = min(u32(max(0.0, pos.x) / cell_w), 8u);
    let gy = min(u32(max(0.0, pos.y) / cell_h), 5u);
    return gy * 9u + gx;
}

@compute @workgroup_size(64)
fn populate_cell_offsets(@builtin(global_invocation_id) id: vec3u) {
    let n = params.agent_count;
    if (id.x >= n) { return; }

    let slot = spatial_keys[id.x].y;
    let m_flags = agent_states[slot].meta_flags;
    if ((m_flags & (1u << 13u)) != 0u) { return; }

    let pos = agent_states[slot].pos_vel.xy;
    let cell_id = get_cell_id(pos);

    if (id.x == 0u) {
        cell_offsets[cell_id].x = id.x;
    } else {
        let prev_slot = spatial_keys[id.x - 1u].y;
        let prev_pos = agent_states[prev_slot].pos_vel.xy;
        let prev_cell = get_cell_id(prev_pos);
        if (prev_cell != cell_id) {
            cell_offsets[cell_id].x = id.x;
            cell_offsets[prev_cell].y = id.x;
        }
    }

    if (id.x == n - 1u) {
        cell_offsets[cell_id].y = n;
    }
}

