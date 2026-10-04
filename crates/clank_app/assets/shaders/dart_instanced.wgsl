// crates/clank_app/assets/shaders/dart_instanced.wgsl
// High-Throughput Instanced Dart & Outline Vertex/Fragment Shader
//
// Streams contiguous vec2u(packed_color, visual_cache) from Cache Line 1.
// Bypasses Cache Line 0 (traits, kinematics) saving ~72 MB/frame vertex fetch bandwidth.
// Zero-cost dead check: visual_cache == 0u collapses scale to 0.0 px (instant hardware cull).

struct VertexInput {
    @location(0) position: vec2f,  // Base dart vertex position [-1.0, 1.5]
    @location(1) edge_flag: f32,   // 0.0 = interior fill, 1.0 = perimeter border
}

struct InstanceInput {
    @location(2) pos_angle: vec3f, // x, y, angle
    @location(3) vis_data: vec2u,  // (packed_color, visual_cache)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) color: vec4f,
    @location(1) edge: f32,
}

struct ViewUniform {
    view_proj: mat4x4f,
    world_size: vec2f,
}

@group(0) @binding(0) var<uniform> view: ViewUniform;

@vertex
fn vs_main(in_vertex: VertexInput, in_inst: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let visual_cache = in_inst.vis_data.y;
    let packed_color = in_inst.vis_data.x;

    // Zero-Cost Dead State Invariant: Collapse to degenerate clip coordinates
    if (visual_cache == 0u) {
        out.clip_position = vec4f(0.0, 0.0, -100.0, 1.0);
        out.color = vec4f(0.0);
        out.edge = 0.0;
        return out;
    }

    // Unpack visual_cache [radius_u8, glow_u8, energy_u8, flags_u8]
    let r_u8 = f32(visual_cache & 0xFFu);
    let radius = 2.3 + (r_u8 / 255.0) * 4.5;
    let glow = f32((visual_cache >> 8u) & 0xFFu) / 255.0;
    let energy_u8 = f32((visual_cache >> 16u) & 0xFFu);
    let is_attacking = (visual_cache & (1u << 24u)) != 0u;

    // Unpack packed_color (rgba8unorm)
    let cr = f32(packed_color & 0xFFu) / 255.0;
    let cg = f32((packed_color >> 8u) & 0xFFu) / 255.0;
    let cb = f32((packed_color >> 16u) & 0xFFu) / 255.0;
    let alpha = clamp(0.55 + (energy_u8 / 255.0 * 100.0 / 160.0), 0.55, 1.0);

    // Rotation & Scaling
    let angle = -in_inst.pos_angle.z;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rot = mat2x2f(cos_a, -sin_a, sin_a, cos_a);

    let local_pos = rot * (in_vertex.position * radius);
    let world_pos = vec2f(in_inst.pos_angle.x, view.world_size.y - in_inst.pos_angle.y) + local_pos;

    out.clip_position = view.view_proj * vec4f(world_pos, -2.0, 1.0);

    if (in_vertex.edge_flag > 0.5) {
        // Outline border: combat red when attacking, else dark teal #153034
        if (is_attacking) {
            out.color = vec4f(1.0, 0.33, 0.31, 1.0);
        } else {
            out.color = vec4f(0.082, 0.188, 0.204, 1.0);
        }
    } else {
        // Body fill
        out.color = vec4f(cr, cg, cb, alpha);
    }
    out.edge = in_vertex.edge_flag;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
