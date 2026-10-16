// crates/clank_app/assets/shaders/creature_render.wgsl
// GPU-resident creature presentation: cull the simulation's agent slots into visible instances
// (with their trails), write indirect draw args, and draw the four creature layers of
// clankolution.html line 58 straight from those buffers. The vertex math mirrors the CPU meshes
// in rendering.rs (`generate_*_from_instances`) vertex for vertex.

struct RenderParams {
    clip_from_world: mat4x4f,
    camera_pos: vec2f,
    camera_size: vec2f,
    world_size: vec2f,
    slot_count: u32,
    pad: u32,
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

// Read-only view of GpuAgentAtomic.
struct AgentAtomicView {
    energy_milli: i32,
    mate_claim: u32,
    mate_energy_milli: u32,
    dead_claimed: u32,
}

struct GpuAgentTrail {
    pts: array<vec2f, 9>,
    count: u32,
    owner_id: u32,
    pad: vec4u,
}

struct DartInstance {
    pos_angle: vec3f,
    pad0: f32,      // armor trait
    vis_data: vec2u, // root, visual_cache
    pad1: vec2u,    // (carnivory_u8 << 24) | id, sight bits
}

struct CullCount {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

// ---- Cull (compute) ----

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(2) var<storage, read> agent_atomics: array<AgentAtomicView>;
@group(0) @binding(3) var<storage, read> agent_trails: array<GpuAgentTrail>;
@group(0) @binding(4) var<storage, read_write> cull_count: CullCount;
@group(0) @binding(5) var<storage, read_write> instances: array<DartInstance>;
@group(0) @binding(6) var<storage, read_write> instance_trails: array<GpuAgentTrail>;
@group(0) @binding(7) var<storage, read_write> draw_args: array<DrawArgs, 4>;

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= params.slot_count) {
        return;
    }
    let s = agent_states[idx];
    if ((s.meta_flags & (1u << 13u)) != 0u || s.visual_cache == 0u || agent_atomics[idx].dead_claimed != 0u) {
        return;
    }
    // Toroidal frustum test with a 24px margin.
    let dx = abs(s.pos_vel.x - params.camera_pos.x);
    let dy = abs(s.pos_vel.y - params.camera_pos.y);
    let half = params.camera_size * 0.5 + vec2f(24.0);
    if (min(dx, params.world_size.x - dx) > half.x || min(dy, params.world_size.y - dy) > half.y) {
        return;
    }
    let slot = atomicAdd(&cull_count.count, 1u);
    if (slot >= arrayLength(&instances)) {
        return;
    }
    let carn_u8 = u32(clamp(s.traits[1].y, 0.0, 1.0) * 255.0);
    instances[slot] = DartInstance(
        vec3f(s.pos_vel.xy, s.angle_energy.x),
        s.traits[0].w,
        vec2u(s.root, s.visual_cache),
        vec2u((carn_u8 << 24u) | (s.id & 0x00FFFFFFu), bitcast<u32>(s.traits[0].z)),
    );
    // A slot reused by another creature carries no trail yet.
    var t = agent_trails[idx];
    if (t.owner_id != s.id) {
        t.count = 0u;
    }
    instance_trails[slot] = t;
}

// Vertices per instance: Glow 20 fan triangles, Trail 8 segments, Body 2 triangles,
// Outline 4 edges + 2 antennae + 16 halo segments.
@compute @workgroup_size(1)
fn finalize_args_main() {
    let n = min(atomicLoad(&cull_count.count), arrayLength(&instances));
    draw_args[0] = DrawArgs(60u, n, 0u, 0u);
    draw_args[1] = DrawArgs(16u, n, 0u, 0u);
    draw_args[2] = DrawArgs(6u, n, 0u, 0u);
    draw_args[3] = DrawArgs(44u, n, 0u, 0u);
}

// ---- Draw (vertex + fragment) ----

@group(0) @binding(8) var<uniform> view_params: RenderParams;
@group(0) @binding(9) var<storage, read> draw_instances: array<DartInstance>;
@group(0) @binding(10) var<storage, read> draw_trails: array<GpuAgentTrail>;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
}

// Outside the clip volume: a primitive made only of these is discarded.
const OFF = vec4f(-2.0, -2.0, 0.5, 1.0);
const TAU: f32 = 6.28318530718;

fn to_bevy(p: vec2f) -> vec2f {
    return vec2f(p.x, view_params.world_size.y - p.y);
}

fn clip_at(p: vec2f, z: f32) -> vec4f {
    return view_params.clip_from_world * vec4f(p, z, 1.0);
}

// glam `Mat2::from_angle(-angle) * v`, as the CPU meshes rotate.
fn rot(angle: f32, v: vec2f) -> vec2f {
    let c = cos(-angle);
    let s = sin(-angle);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

// theme::PALETTE, as 0xBBGGRR: a lineage's colour is PALETTE[root % 8].
const PALETTE = array<u32, 8>(0xD2E591u, 0x78B8E6u, 0x6B78EEu, 0xD797BEu, 0xDFB87Fu, 0x8BDFD4u, 0xBBA3E4u, 0xA1C589u);

fn lineage_rgb(root: u32) -> vec3f {
    let pc = PALETTE[root % 8u];
    return vec3f(f32(pc & 0xFFu), f32((pc >> 8u) & 0xFFu), f32((pc >> 16u) & 0xFFu)) / 255.0;
}

fn radius(vc: u32) -> f32 {
    return 2.3 + f32(vc & 0xFFu) / 255.0 * 4.5;
}

fn signal(vc: u32) -> f32 {
    return f32((vc >> 8u) & 0xFFu) / 255.0;
}

// `Color::srgba_u8(.., (alpha * 255.0) as u8)`: the body alpha is quantized to 1/255.
fn body_alpha(vc: u32) -> f32 {
    let a = clamp(0.55 + f32((vc >> 16u) & 0xFFu) / 255.0 * 100.0 / 160.0, 0.55, 1.0);
    return floor(a * 255.0) / 255.0;
}

// 0 nose, 1 right wing, 2 rear notch, 3 left wing.
fn dart_point(inst: DartInstance, k: u32) -> vec2f {
    let r = radius(inst.vis_data.y);
    let armor = inst.pad0;
    let carn = f32(inst.pad1.x >> 24u) / 255.0;
    var local = vec2f(r * 1.5, 0.0);
    if (k == 1u) {
        local = vec2f(-r * 0.75, r * (0.5 + armor * 0.45));
    }
    if (k == 2u) {
        local = vec2f(-r * (0.45 + carn), 0.0);
    }
    if (k == 3u) {
        local = vec2f(-r * 0.75, -r * (0.5 + armor * 0.45));
    }
    return to_bevy(inst.pos_angle.xy) + rot(inst.pos_angle.z, local);
}

@vertex
fn vs_glow(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
    let inst = draw_instances[i];
    let vc = inst.vis_data.y;
    let sig = signal(vc);
    let glow_r = radius(vc) + 5.0 + 8.0 * sig;
    let rgb = lineage_rgb(inst.vis_data.x);
    let center = to_bevy(inst.pos_angle.xy);
    let seg = v / 3u;
    let corner = v % 3u;
    if (corner == 0u) {
        return VOut(clip_at(center, -2.2), vec4f(rgb, clamp(0.20 + 0.18 * sig, 0.12, 0.40)));
    }
    let theta = f32(seg + corner - 1u) * TAU / 20.0;
    return VOut(clip_at(center + vec2f(cos(theta) * glow_r, sin(theta) * glow_r), -2.2), vec4f(rgb, 0.0));
}

@vertex
fn vs_trail(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
    let inst = draw_instances[i];
    let t = draw_trails[i];
    let seg = v / 2u;
    let count = min(t.count, 9u);
    if (count < 2u || seg + 1u >= count) {
        return VOut(OFF, vec4f(0.0));
    }
    let p1 = to_bevy(t.pts[seg]);
    let p2 = to_bevy(t.pts[seg + 1u]);
    // Segments that wrap around the torus are not drawn.
    if (!(abs(p1.x - p2.x) < view_params.world_size.x * 0.5 && abs(p1.y - p2.y) < view_params.world_size.y * 0.5)) {
        return VOut(OFF, vec4f(0.0));
    }
    let alpha = clamp(0.18 + 0.32 * (f32(seg + 1u) / f32(count - 1u)), 0.15, 0.50);
    return VOut(clip_at(select(p1, p2, (v & 1u) == 1u), -2.05), vec4f(lineage_rgb(inst.vis_data.x), alpha));
}

@vertex
fn vs_body(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
    let inst = draw_instances[i];
    // [nose, right, rear], [nose, rear, left]
    var order = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let vc = inst.vis_data.y;
    return VOut(clip_at(dart_point(inst, order[v]), -2.0), vec4f(lineage_rgb(inst.vis_data.x), body_alpha(vc)));
}

@vertex
fn vs_outline(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
    let inst = draw_instances[i];
    let vc = inst.vis_data.y;
    let r = radius(vc);
    let rgb = lineage_rgb(inst.vis_data.x);
    let end = v & 1u;
    if (v < 8u) {
        // Perimeter: nose-right, right-rear, rear-left, left-nose.
        let k = (v / 2u + end) % 4u;
        var border = vec4f(0.082, 0.188, 0.204, 1.0);
        if ((vc & (1u << 24u)) != 0u) {
            border = vec4f(1.0, 0.33, 0.31, 1.0);
        }
        return VOut(clip_at(dart_point(inst, k), -1.9), border);
    }
    if (v < 12u) {
        // Sensory antennae when sight > 0.56.
        let sight = bitcast<f32>(inst.pad1.y);
        if (!(sight > 0.56)) {
            return VOut(OFF, vec4f(0.0));
        }
        let side = select(1.0, -1.0, v >= 10u);
        var local = vec2f(-r * 0.3, side * r * 0.6);
        if (end == 1u) {
            local = vec2f(-r * (1.5 + sight), side * r * (1.1 + signal(vc)));
        }
        return VOut(clip_at(to_bevy(inst.pos_angle.xy) + rot(inst.pos_angle.z, local), -1.9), vec4f(rgb, 0.55));
    }
    // Birth halo: 16 segments, expanding as the countdown runs out.
    let birth = f32((vc >> 25u) & 0x7Fu);
    if (birth <= 0.0) {
        return VOut(OFF, vec4f(0.0));
    }
    let halo_r = r + 3.0 + max(95.0 - birth, 0.0) * 0.12;
    let theta = f32((v - 12u) / 2u + end) * TAU / 16.0;
    let p = to_bevy(inst.pos_angle.xy) + vec2f(cos(theta) * halo_r, sin(theta) * halo_r);
    return VOut(clip_at(p, -1.8), vec4f(rgb, clamp(birth / 120.0, 0.0, 1.0)));
}

@fragment
fn fs_color(in: VOut) -> @location(0) vec4f {
    return in.color;
}
