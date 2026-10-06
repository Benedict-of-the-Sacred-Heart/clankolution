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

@group(0) @binding(0) var<storage, read_write> soil_buffer: array<SoilCell>;
@group(0) @binding(1) var soil_data: texture_storage_2d<rgba16float, write>;      // Raw physics [food, taint, scent, 1.0]
@group(0) @binding(2) var soil_display: texture_storage_2d<rgba16float, write>;   // Colormap display
@group(0) @binding(3) var<storage, read> bloom_table: array<f32>;
@group(0) @binding(4) var<uniform> params: SoilParams;
@group(0) @binding(5) var<uniform> sim_params: GpuSimParams;

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

fn pcg_hash(id: u32, stream: u32, tick: u32) -> u32 {
    let state = id * 747796405u + stream * 2891336453u + tick * 1013904223u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_float(id: u32, stream: u32, tick: u32) -> f32 {
    return f32(pcg_hash(id, stream, tick)) / 4294967295.0;
}

@compute @workgroup_size(8, 8)
fn soil_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let k = id.y * params.width + id.x;

    let f_milli = atomicLoad(&soil_buffer[k].food_milli);
    var f = f32(f_milli) * 0.001;
    var t = f32(atomicLoad(&soil_buffer[k].taint_milli)) * 0.001;
    var s = f32(atomicLoad(&soil_buffer[k].scent_milli)) * 0.001;

    // Interactive tool deposit (Nourish = 1, Blight = 2) on sub_tick 0
    if (sim_params.sub_tick == 0u && (sim_params.tool_type == 1u || sim_params.tool_type == 2u)) {
        let inv_cw = 1.0 / 12.0;
        let inv_ch = 1.0 / 12.0;
        let tx = ((sim_params.tool_pos[0] % sim_params.world_size[0]) + sim_params.world_size[0]) % sim_params.world_size[0];
        let ty = ((sim_params.tool_pos[1] % sim_params.world_size[1]) + sim_params.world_size[1]) % sim_params.world_size[1];
        let cx = i32(floor(tx * inv_cw));
        let cy = i32(floor(ty * inv_ch));

        let cols = i32(params.width);
        let rows = i32(params.height);

        var dx = (i32(id.x) - cx) % cols;
        if (dx > cols / 2) { dx -= cols; }
        if (dx < -cols / 2) { dx += cols; }

        var dy = (i32(id.y) - cy) % rows;
        if (dy > rows / 2) { dy -= rows; }
        if (dy < -rows / 2) { dy += rows; }

        let rr = f32(dx * dx + dy * dy);

        if (sim_params.tool_type == 1u) {
            // Nourish: deposit(food, x, y, 0.28, 3)
            if (rr <= 9.5) {
                f = clamp(f + 0.28 / (1.0 + rr * 0.8), 0.0, 3.0);
            }
        } else if (sim_params.tool_type == 2u) {
            // Blight: deposit(taint, x, y, 0.38, 3); deposit(food, x, y, -0.14, 2)
            if (rr <= 9.5) {
                t = clamp(t + 0.38 / (1.0 + rr * 0.8), 0.0, 3.0);
            }
            if (rr <= 4.5) {
                f = clamp(f - 0.14 / (1.0 + rr * 0.8), 0.0, 3.0);
            }
        }
    }

    // Environmental renewal & decay (HTML lines 1249-1254 and clank_core line 128)
    f += params.renewal * bloom_table[k] * (1.0 - f / 1.7);
    let spawn_threshold = 0.00013 * params.renewal;
    let prng_seed = u32(f_milli) ^ (k * 1013904223u);
    if (pcg_float(k, prng_seed, 0u) < spawn_threshold) {
        f += 0.15 + 0.45 * pcg_float(k, prng_seed + 100u, 1u);
    }
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
