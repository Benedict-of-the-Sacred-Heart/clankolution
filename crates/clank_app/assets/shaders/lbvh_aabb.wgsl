// crates/clank_app/assets/shaders/lbvh_aabb.wgsl
// Phase 2: Bottom-Up Bounding Box & Metric Hierarchy Fitting (Karras 2012)

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
    root: u32,
    visual_cache: u32,
}

@group(0) @binding(0) var<storage, read> spatial_keys: array<vec2u>; // sorted (morton, slot), live first
// GpuLbvhNode as 12 words, so every access can be atomic (field word offsets below).
@group(0) @binding(1) var<storage, read_write> lbvh_words: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> node_flags: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(4) var<uniform> params: GpuSimParams;
@group(0) @binding(5) var<storage, read_write> queue_words: array<atomic<u32>>;

const LIVE_COUNT_WORD: u32 = 16u; // GpuTelemetry.live_count, written by the spatial sort

const NODE_WORDS: u32 = 12u;
const MIN_X: u32 = 0u;
const MIN_Y: u32 = 1u;
const MAX_X: u32 = 2u;
const MAX_Y: u32 = 3u;
const COM_X: u32 = 4u;
const COM_Y: u32 = 5u;
const COUNT: u32 = 6u;
const DOMINANT: u32 = 7u;
const LEFT: u32 = 8u;
const RIGHT: u32 = 9u;
const PARENT: u32 = 10u;
const LEAF: u32 = 11u;
const NONE: u32 = 0xFFFFFFFFu;

fn word(node: u32, field: u32) -> u32 { return node * NODE_WORDS + field; }

// Phase 2: each live creature writes its leaf, then climbs, growing each ancestor's box with
// atomicMin/atomicMax. The result does not depend on which thread arrives first (a child's box
// written by another workgroup is never read, which WGSL could not order). A thread stops once
// it changes nothing at a node: whoever set that node's extreme value climbs on with it.
// Box coordinates are clamped to be non-negative, so u32 order is float order; a box only has
// to contain its creatures' centres for picking to prune correctly.
@compute @workgroup_size(64)
fn fit_lbvh_aabbs(@builtin(global_invocation_id) id: vec3u) {
    let n = atomicLoad(&queue_words[LIVE_COUNT_WORD]);
    if (id.x >= n) { return; }

    // Leaf id.x is the id.x-th live creature in Morton order.
    let agent_idx = spatial_keys[id.x].y;
    let a = agent_states[agent_idx];
    let r = 2.0 + 3.0 * a.traits[0][0];
    let bmin = max(vec2f(0.0, 0.0), a.pos_vel.xy - vec2f(r, r));
    let bmax = a.pos_vel.xy + vec2f(r, r);
    let lo = vec2u(bitcast<u32>(bmin.x), bitcast<u32>(bmin.y));
    let hi = vec2u(bitcast<u32>(bmax.x), bitcast<u32>(bmax.y));

    // Degenerate N = 1: the root is the leaf.
    let leaf = select(n - 1u + id.x, 0u, n == 1u);
    atomicStore(&lbvh_words[word(leaf, MIN_X)], lo.x);
    atomicStore(&lbvh_words[word(leaf, MIN_Y)], lo.y);
    atomicStore(&lbvh_words[word(leaf, MAX_X)], hi.x);
    atomicStore(&lbvh_words[word(leaf, MAX_Y)], hi.y);
    atomicStore(&lbvh_words[word(leaf, COM_X)], bitcast<u32>(a.pos_vel.x));
    atomicStore(&lbvh_words[word(leaf, COM_Y)], bitcast<u32>(a.pos_vel.y));
    atomicStore(&lbvh_words[word(leaf, COUNT)], 1u);
    atomicStore(&lbvh_words[word(leaf, DOMINANT)], a.root);
    atomicStore(&lbvh_words[word(leaf, LEFT)], NONE);
    atomicStore(&lbvh_words[word(leaf, RIGHT)], NONE);
    atomicStore(&lbvh_words[word(leaf, LEAF)], agent_idx);
    if (n == 1u) {
        atomicStore(&lbvh_words[word(leaf, PARENT)], NONE);
        return;
    }

    // Parents were written by phase 1, in the previous dispatch. A Karras tree over 32-bit keys with
    // a 32-bit tie-break is at most 65 levels deep.
    var curr = atomicLoad(&lbvh_words[word(leaf, PARENT)]);
    var steps = 0u;
    while (curr < n - 1u && steps < 66u) {
        steps += 1u;
        var changed = atomicMin(&lbvh_words[word(curr, MIN_X)], lo.x) > lo.x;
        changed = (atomicMin(&lbvh_words[word(curr, MIN_Y)], lo.y) > lo.y) || changed;
        changed = (atomicMax(&lbvh_words[word(curr, MAX_X)], hi.x) < hi.x) || changed;
        changed = (atomicMax(&lbvh_words[word(curr, MAX_Y)], hi.y) < hi.y) || changed;
        if (!changed || curr == 0u) { break; }
        curr = atomicLoad(&lbvh_words[word(curr, PARENT)]);
    }
}
