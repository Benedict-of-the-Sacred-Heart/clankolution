// crates/clank_app/assets/shaders/soil_step.wgsl
// GPU Soil Simulation & Direct Texture Generation
//
// Simulates renewal, decay, and direct dual-texture generation:
// 1. soil_data: Raw physics scalars (food, taint, scent) for hardware bilinear sampling
// 2. soil_display: Colormapped output for Bevy SoilSprite presentation

struct SoilCell {
    food_milli: atomic<i32>,
    taint_milli: atomic<i32>,
    scent_milli: atomic<i32>,
    fertility_milli: atomic<i32>,
}

struct SoilParams {
    renewal: f32,
    width: u32,
    height: u32,
    decay_rate: f32,
}

@group(0) @binding(0) var<storage, read_write> soil_buffer: array<SoilCell>;
@group(0) @binding(1) var soil_data: texture_storage_2d<rgba16float, write>;      // Raw physics [food, taint, scent, 1.0]
@group(0) @binding(2) var soil_display: texture_storage_2d<rgba16float, write>;   // Colormap display
@group(0) @binding(3) var<storage, read> bloom_table: array<f32>;
@group(0) @binding(4) var<uniform> params: SoilParams;

fn evaluate_soil_color(f: f32, t: f32, s: f32, coord: vec2u) -> vec4f {
    let inv18 = 1.0 / 1.8;
    let f_val = min(f * inv18, 1.0);
    var r = 9.5 + f_val * 42.0;
    var g = 22.5 + f_val * 61.0;
    var b = 25.5 + f_val * 44.0;

    if (t > 0.0) {
        let tc = min(t, 1.0);
        r += tc * 98.0;
        g -= tc * 13.0;
        b += tc * 23.0;
    }

    if (s > 0.0) {
        let sc = min(s, 1.0);
        r += sc * 27.0;
        g += sc * 20.0;
        b += sc * 33.0;
    }

    return vec4f(
        clamp(r / 255.0, 0.0, 1.0),
        clamp(g / 255.0, 0.0, 1.0),
        clamp(b / 255.0, 0.0, 1.0),
        1.0
    );
}

@compute @workgroup_size(8, 8)
fn soil_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let k = id.y * params.width + id.x;

    var f = f32(atomicLoad(&soil_buffer[k].food_milli)) * 0.001;
    var t = f32(atomicLoad(&soil_buffer[k].taint_milli)) * 0.001;
    var s = f32(atomicLoad(&soil_buffer[k].scent_milli)) * 0.001;

    // Environmental renewal & decay:
    f += params.renewal * bloom_table[k] * (1.0 - f / 1.7);
    f = clamp(f, 0.0, 2.5);
    if (t > 0.0) { t = max(0.0, t * (1.0 - params.decay_rate) - 0.0001); }
    if (s > 0.0) { s = s * 0.954; }

    // 1. Write back persistent atomic state for agent grazing:
    atomicStore(&soil_buffer[k].food_milli, i32(f * 1000.0));
    atomicStore(&soil_buffer[k].taint_milli, i32(t * 1000.0));
    atomicStore(&soil_buffer[k].scent_milli, i32(s * 1000.0));

    // 2. Output Raw Physics Snapshot for agent bilinear sensing:
    textureStore(soil_data, id.xy, vec4f(f, t, s, 1.0));

    // 3. Output Graded Colormap for Bevy screen presentation:
    let col = evaluate_soil_color(f, t, s, id.xy);
    textureStore(soil_display, id.xy, col);
}
