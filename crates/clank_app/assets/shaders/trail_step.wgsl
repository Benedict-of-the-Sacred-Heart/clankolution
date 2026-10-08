// crates/clank_app/assets/shaders/trail_step.wgsl
// Per-tick motion trails (clankolution.html: `a.trail.push([a.x,a.y]); if(a.trail.length>9)a.trail.shift()`)
// and the gather of visible trails for rendering.

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

const TRAIL_LEN: u32 = 9u;

struct GpuAgentTrail {
    pts: array<vec2f, 9>,
    count: u32,
    owner_id: u32,
    pad: vec4u,
}

struct CullOutput {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(0) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> agent_trails: array<GpuAgentTrail>;
@group(0) @binding(2) var<uniform> params: GpuSimParams;
@group(0) @binding(3) var<storage, read> visible_instances: array<u32>;
@group(0) @binding(4) var<storage, read> cull_output: CullOutput;
@group(0) @binding(5) var<storage, read_write> dart_trails: array<GpuAgentTrail>;

// Runs once per sub-tick after the agent step: append each living creature's new position.
@compute @workgroup_size(64)
fn trail_record_main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= params.max_agents) {
        return;
    }
    let state = agent_states[idx];
    if ((state.meta_flags & (1u << 13u)) != 0u || state.id == 0u) {
        return;
    }

    var trail = agent_trails[idx];
    if (trail.owner_id != state.id) {
        // Newborn, spore, or a fresh upload in this slot: no history yet.
        trail.count = 0u;
        trail.owner_id = state.id;
    }
    let p = state.pos_vel.xy;
    if (trail.count < TRAIL_LEN) {
        trail.pts[trail.count] = p;
        trail.count += 1u;
    } else {
        for (var k = 0u; k < TRAIL_LEN - 1u; k++) {
            trail.pts[k] = trail.pts[k + 1u];
        }
        trail.pts[TRAIL_LEN - 1u] = p;
    }
    agent_trails[idx] = trail;
}

// Runs after frustum culling: dart_trails[i] is the trail of dart_instances[i].
@compute @workgroup_size(64)
fn trail_gather_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= min(cull_output.instance_count, params.max_agents)) {
        return;
    }
    dart_trails[i] = agent_trails[visible_instances[i]];
}
