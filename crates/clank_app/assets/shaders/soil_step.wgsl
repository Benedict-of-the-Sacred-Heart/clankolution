// crates/clank_app/assets/shaders/soil_step.wgsl
// GPU Soil Simulation & Direct Texture Generation
//
// Simulates renewal, decay and the eclipse on the atomic soil buffer, which creatures read
// with point loads, and writes soil_display, the colormap for the Bevy SoilSprite.

// Fixed-point scale of SoilCell fields (micro-units). Must match
// SOIL_FIXED_SCALE in src/gpu/types.rs and SOIL_SCALE in agent_step.wgsl.
const SOIL_SCALE: f32 = 1000000.0;

struct SoilCell {
    food_micro: atomic<i32>,
    taint_micro: atomic<i32>,
    scent_micro: atomic<i32>,
    fertility_micro: atomic<i32>,
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
@group(0) @binding(2) var soil_display: texture_storage_2d<rgba8unorm, write>;    // Colormap display (copied into the soil sprite)
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

    // `generate_soil_rgba` stores `c.clamp(0.0, 255.0) as u8`: truncate, then normalize.
    return vec4f(floor(clamp(vec3f(r, g, b), vec3f(0.0), vec3f(255.0))) / 255.0, 1.0);
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

    var f = f32(atomicLoad(&soil_buffer[k].food_micro)) / SOIL_SCALE;
    var t = f32(atomicLoad(&soil_buffer[k].taint_micro)) / SOIL_SCALE;
    var s = f32(atomicLoad(&soil_buffer[k].scent_micro)) / SOIL_SCALE;

    // Interactive tool deposit (Nourish = 1, Blight = 2) on sub_tick 0
    if (sim_params.sub_tick == 0u && (sim_params.tool_type == 1u || sim_params.tool_type == 2u)) {
        let cell_w = sim_params.world_size[0] / f32(params.width);
        let cell_h = sim_params.world_size[1] / f32(params.height);
        let tx = ((sim_params.tool_pos[0] % sim_params.world_size[0]) + sim_params.world_size[0]) % sim_params.world_size[0];
        let ty = ((sim_params.tool_pos[1] % sim_params.world_size[1]) + sim_params.world_size[1]) % sim_params.world_size[1];
        let cx = min(i32(floor(tx / cell_w)), i32(params.width) - 1);
        let cy = min(i32(floor(ty / cell_h)), i32(params.height) - 1);

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
    // Spawn roll is drawn per cell per tick; it must not depend on the cell's own
    // food value, or a cell whose food stops changing re-rolls the same number forever.
    if (pcg_float(k, 0x50110001u, sim_params.tick) < spawn_threshold) {
        f += 0.15 + 0.45 * pcg_float(k, 0x50110002u, sim_params.tick);
    }
    f = clamp(f, 0.0, 2.5);
    if (t > 0.0) { t = max(0.0, t * (1.0 - params.decay_rate) - 0.0001); }
    if (s > 0.0) { s = s * 0.954; }

    // 1. Write back persistent atomic state for agent grazing:
    atomicStore(&soil_buffer[k].food_micro, i32(round(f * SOIL_SCALE)));
    atomicStore(&soil_buffer[k].taint_micro, i32(round(t * SOIL_SCALE)));
    atomicStore(&soil_buffer[k].scent_micro, i32(round(s * SOIL_SCALE)));

    // 2. Output Graded Colormap for Bevy screen presentation, from the stored (quantized)
    //    values so it shows exactly the soil the CPU would read back:
    let col = evaluate_soil_color(
        f32(i32(round(f * SOIL_SCALE))) / SOIL_SCALE,
        f32(i32(round(t * SOIL_SCALE))) / SOIL_SCALE,
        f32(i32(round(s * SOIL_SCALE))) / SOIL_SCALE,
        id.xy,
    );
    textureStore(soil_display, id.xy, col);
}

// Standalone tool deposit pass (Nourish = 1, Blight = 2) without advancing renewal/decay,
// used when multiple tool drops are queued in the same frame.
@compute @workgroup_size(8, 8)
fn soil_tool_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    if (sim_params.tool_type != 1u && sim_params.tool_type != 2u) { return; }
    let k = id.y * params.width + id.x;

    var f = f32(atomicLoad(&soil_buffer[k].food_micro)) / SOIL_SCALE;
    var t = f32(atomicLoad(&soil_buffer[k].taint_micro)) / SOIL_SCALE;
    let s = f32(atomicLoad(&soil_buffer[k].scent_micro)) / SOIL_SCALE;

    let cell_w = sim_params.world_size[0] / f32(params.width);
    let cell_h = sim_params.world_size[1] / f32(params.height);
    let tx = ((sim_params.tool_pos[0] % sim_params.world_size[0]) + sim_params.world_size[0]) % sim_params.world_size[0];
    let ty = ((sim_params.tool_pos[1] % sim_params.world_size[1]) + sim_params.world_size[1]) % sim_params.world_size[1];
    let cx = min(i32(floor(tx / cell_w)), i32(params.width) - 1);
    let cy = min(i32(floor(ty / cell_h)), i32(params.height) - 1);

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
        if (rr <= 9.5) {
            f = clamp(f + 0.28 / (1.0 + rr * 0.8), 0.0, 3.0);
        }
    } else if (sim_params.tool_type == 2u) {
        if (rr <= 9.5) {
            t = clamp(t + 0.38 / (1.0 + rr * 0.8), 0.0, 3.0);
        }
        if (rr <= 4.5) {
            f = clamp(f - 0.14 / (1.0 + rr * 0.8), 0.0, 3.0);
        }
    }

    let f_micro = i32(round(f * SOIL_SCALE));
    let t_micro = i32(round(t * SOIL_SCALE));
    atomicStore(&soil_buffer[k].food_micro, f_micro);
    atomicStore(&soil_buffer[k].taint_micro, t_micro);

    let col = evaluate_soil_color(
        f32(f_micro) / SOIL_SCALE,
        f32(t_micro) / SOIL_SCALE,
        s,
        id.xy,
    );
    textureStore(soil_display, id.xy, col);
}

const ECLIPSE_STREAM: u32 = 0xEC11u;

// Eclipse (after soil_main): while the countdown is running, on even ticks, 160 random cells
// lose food (×0.73) and gain taint (+0.14, clamped to [0, 2]). `sim_params.tick` is the tick
// before the reference's increment, so the reference's even tick is an odd one here. Two
// threads may pick the same cell, so each update is a compare-exchange loop.
@compute @workgroup_size(64)
fn eclipse_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= 160u || sim_params.eclipse == 0u || (sim_params.tick + 1u) % 2u != 0u) { return; }
    let k = pcg_hash(id.x, ECLIPSE_STREAM, sim_params.tick) % (params.width * params.height);

    var food = atomicLoad(&soil_buffer[k].food_micro);
    loop {
        let darkened = i32(f32(food) * 0.73);
        let cas = atomicCompareExchangeWeak(&soil_buffer[k].food_micro, food, darkened);
        if (cas.exchanged) { break; }
        food = cas.old_value;
    }
    var taint = atomicLoad(&soil_buffer[k].taint_micro);
    loop {
        let tainted = clamp(taint + i32(0.14 * SOIL_SCALE), 0, i32(2.0 * SOIL_SCALE));
        let cas = atomicCompareExchangeWeak(&soil_buffer[k].taint_micro, taint, tainted);
        if (cas.exchanged) { break; }
        taint = cas.old_value;
    }
}
