// crates/clank_app/assets/shaders/agent_step.wgsl
// GPU Agent Simulation Step Kernel
//
// 128B GpuAgentState + 352B GpuAgentGenome Struct-of-Arrays
// Branchless 88-Word Vectorized RNN with unpack4x8snorm
// Strict <= 8 Storage Buffer Limit for 100% WebGPU Portability

override ENABLE_BARNES_HUT: bool = false;
override ENABLE_EXPANDED_CORTEX: bool = false;
override ENABLE_SEXUAL_SELECTION: bool = false;

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
    packed_color: u32,
    visual_cache: u32,
}

struct GpuAgentGenome {
    packed_genes: array<u32, 88>,
}

struct GpuAgentAtomic {
    energy_milli: atomic<i32>,
    mate_claim: atomic<u32>,
    mate_energy_milli: atomic<u32>,
    dead_claimed: atomic<u32>,
}

struct SoilCell {
    food_milli: atomic<i32>,
    taint_milli: atomic<i32>,
    scent_milli: atomic<i32>,
    fertility_milli: atomic<i32>,
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

struct GpuTelemetry {
    population: atomic<u32>,
    kills: atomic<u32>,
    starvations: atomic<u32>,
    apex_record_milli: atomic<u32>,
    food_grazed_milli: atomic<u32>,
    sub_ticks_elapsed: atomic<u32>,
    apex_agent_id: atomic<u32>,
    freelist_top: atomic<u32>,
    birth_count: atomic<u32>,
    audio_voice_count: atomic<u32>,
    selected_agent_idx: u32,
    selected_agent_id: u32,
    total_births: atomic<u32>,
    total_deaths: atomic<u32>,
    max_generation: atomic<u32>,
    extinctions: atomic<u32>,
    lineage_counts: array<atomic<u32>, 16>,
}

struct BirthEvent {
    parent_a: u32,
    parent_b: u32,
    child_slot: u32,
    birth_tick: u32,
}

struct AudioVoice {
    pos: vec2f,
    event_type: u32,
    volume: f32,
}

struct ConsolidatedQueue {
    telemetry: GpuTelemetry,
    births: array<BirthEvent, 65536>,
    audio: array<AudioVoice, 256>,
}

// Exactly 8 storage buffers in Group 0
@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read> agent_genomes: array<GpuAgentGenome>;
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(3) var<storage, read_write> soil_buffer: array<SoilCell>;
@group(0) @binding(4) var<storage, read> spatial_keys: array<vec2u>;
@group(0) @binding(5) var<storage, read_write> cell_offsets: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(7) var<storage, read_write> queue_buffer: ConsolidatedQueue;

@group(1) @binding(0) var<uniform> params: GpuSimParams;
@group(1) @binding(1) var soil_data: texture_2d<f32>;
@group(1) @binding(2) var soil_sampler: sampler;

fn wrap_coords(p: vec2f) -> vec2f {
    return vec2f(
        p.x - params.world_size.x * floor(p.x / params.world_size.x),
        p.y - params.world_size.y * floor(p.y / params.world_size.y)
    );
}

fn sample_soil_probe(p: vec2f) -> vec4f {
    let uv = p / params.world_size;
    return textureSampleLevel(soil_data, soil_sampler, uv, 0.0);
}

fn get_gene(agent_idx: u32, p: u32) -> f32 {
    let word = agent_genomes[agent_idx].packed_genes[p >> 2u];
    let shift = (p & 3u) << 3u;
    let b = (i32(word >> shift) << 24) >> 24;
    return f32(b);
}

fn pcg_hash(id: u32, stream: u32, tick: u32) -> u32 {
    let state = id * 747796405u + stream * 2891336453u + tick * 1013904223u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg_float(id: u32, stream: u32, tick: u32) -> f32 {
    return f32(pcg_hash(id, stream, tick)) / 4294967295.0;
}

fn toroidal_aabb_dist_1d(p: f32, b_min: f32, b_max: f32, w: f32) -> f32 {
    if (p >= b_min && p <= b_max) { return 0.0; }
    let direct = select(p - b_max, b_min - p, p < b_min);
    let wrapped = select(w - p + b_min, w - b_max + p, p < b_min);
    return min(direct, wrapped);
}

fn distance_to_aabb(pos: vec2f, aabb_min: vec2f, aabb_max: vec2f) -> f32 {
    let dx = toroidal_aabb_dist_1d(pos.x, aabb_min.x, aabb_max.x, params.world_size.x);
    let dy = toroidal_aabb_dist_1d(pos.y, aabb_min.y, aabb_max.y, params.world_size.y);
    return sqrt(dx * dx + dy * dy);
}

fn toroidal_dist(p1: vec2f, p2: vec2f) -> f32 {
    let dx = abs(p1.x - p2.x);
    let x_dist = min(dx, params.world_size.x - dx);
    let dy = abs(p1.y - p2.y);
    let y_dist = min(dy, params.world_size.y - dy);
    return sqrt(x_dist * x_dist + y_dist * y_dist);
}

@compute @workgroup_size(64)
fn agent_main(@builtin(global_invocation_id) id: vec3u) {
    let agent_idx = id.x;
    if (agent_idx >= params.max_agents) { return; }

    // Top-of-Shader Tombstone Dead-Check Guard with dead_claimed CAS synchronization:
    let m_flags = agent_states[agent_idx].meta_flags;
    let is_dead = (m_flags & (1u << 13u)) != 0u;
    let dead_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed) != 0u;
    if (is_dead || dead_claimed) {
        if (!is_dead) {
            agent_states[agent_idx].meta_flags |= (1u << 13u);
            agent_states[agent_idx].angle_energy[1] = 0.0;
            agent_states[agent_idx].visual_cache = 0u;
        }
        return;
    }

    let pos = agent_states[agent_idx].pos_vel.xy;
    let vel = agent_states[agent_idx].pos_vel.zw;

    var angle = agent_states[agent_idx].angle_energy[0];
    var a_energy = agent_states[agent_idx].angle_energy[1];
    var a_feed = agent_states[agent_idx].angle_energy[2];
    var a_attack = agent_states[agent_idx].angle_energy[3];

    let tr0 = agent_states[agent_idx].traits[0][0]; // bulk
    let tr1 = agent_states[agent_idx].traits[0][1];
    let tr2 = agent_states[agent_idx].traits[0][2]; // sight
    let tr3 = agent_states[agent_idx].traits[0][3]; // armor
    let tr4 = agent_states[agent_idx].traits[1][0];
    let tr5 = agent_states[agent_idx].traits[1][1]; // carnivory

    let a_root = m_flags & 0x0Fu;
    var a_cooldown = (m_flags >> 4u) & 0x03u;
    var a_birth = (m_flags >> 6u) & 0x7Fu;
    var a_kills = (m_flags >> 14u) & 0x3FFFFu;

    var a_age = agent_states[agent_idx].age_gen & 0xFFFFu;
    let a_gen = agent_states[agent_idx].age_gen >> 16u;

    // Antennae feelers
    // HTML: reach = 19 + 38 * tr2 (sight trait)
    let reach = 19.0 + 38.0 * tr2;
    let fwd_term = vec2f(cos(angle), sin(angle)) * reach;

    let probe_here    = sample_soil_probe(pos);
    let probe_forward = sample_soil_probe(pos + fwd_term);
    let mid_antennae  = pos + fwd_term * 0.7;
    let off_antennae  = vec2f(-fwd_term.y, fwd_term.x) * 0.6;
    let probe_left    = sample_soil_probe(mid_antennae + off_antennae);
    let probe_right   = sample_soil_probe(mid_antennae - off_antennae);

    let here_food    = probe_here.r;
    let forward_food = probe_forward.r;
    let left_food    = probe_left.r;
    let right_food   = probe_right.r;
    let here_taint   = probe_here.g;
    let scent_diff   = probe_forward.b - probe_here.b;

    // Spatial Grid 9-Cell Moore Neighborhood Query (Exact match to clank_core & HTML near())
    var best_dist = 999999.0;
    var best_neighbor = 0xFFFFFFFFu;
    var density = 0.0;

    let sight_radius = 100.0 + 70.0 * tr2;

    if (params.agent_count > 1u) {
        let cell_w = params.world_size.x / 9.0;
        let cell_h = params.world_size.y / 6.0;
        let gx = i32(min(u32(max(0.0, pos.x) / cell_w), 8u));
        let gy = i32(min(u32(max(0.0, pos.y) / cell_h), 5u));

        for (var dy = -1; dy <= 1; dy++) {
            let n_gy = (gy + dy + 6) % 6;
            for (var dx = -1; dx <= 1; dx++) {
                let n_gx = (gx + dx + 9) % 9;
                let n_cell = u32(n_gy * 9 + n_gx);

                var curr = atomicLoad(&cell_offsets[n_cell]);
                var loop_count = 0u;
                while (curr != 0xFFFFFFFFu && loop_count < 128u) {
                    if (curr != agent_idx) {
                        let other_meta = agent_states[curr].meta_flags;
                        if ((other_meta & (1u << 13u)) == 0u) {
                            let other_pos = agent_states[curr].pos_vel.xy;
                            let d = toroidal_dist(pos, other_pos);
                            if (d <= 100.0) {
                                density += 1.0;
                            }
                            if (d < best_dist && d <= sight_radius) {
                                best_dist = d;
                                best_neighbor = curr;
                            }
                        }
                    }
                    curr = spatial_keys[curr].x;
                    loop_count += 1u;
                }
            }
        }
    }

    // 15 Canonical Sensory Inputs (matching clankolution.html lines 1334-1362)
    var sensory_inputs: array<f32, 15>;
    sensory_inputs[0] = clamp(a_energy / 75.0 - 1.0, -1.0, 1.0);
    sensory_inputs[1] = clamp(here_food - 1.0, -1.0, 1.0);
    sensory_inputs[2] = clamp(forward_food - left_food, -1.0, 1.0);
    sensory_inputs[3] = clamp(forward_food - right_food, -1.0, 1.0);
    sensory_inputs[4] = clamp(forward_food - here_food, -1.0, 1.0);
    sensory_inputs[5] = clamp(here_taint, 0.0, 1.0);
    sensory_inputs[6] = clamp(scent_diff, -1.0, 1.0);

    if (best_neighbor != 0xFFFFFFFFu) {
        let neighbor_pos = agent_states[best_neighbor].pos_vel.xy;
        var dx = neighbor_pos.x - pos.x;
        if (dx > params.world_size.x * 0.5) { dx -= params.world_size.x; }
        else if (dx < -params.world_size.x * 0.5) { dx += params.world_size.x; }

        var dy = neighbor_pos.y - pos.y;
        if (dy > params.world_size.y * 0.5) { dy -= params.world_size.y; }
        else if (dy < -params.world_size.y * 0.5) { dy += params.world_size.y; }

        let angle_to_neighbor = atan2(dy, dx);
        let bearing = angle_to_neighbor - angle;

        sensory_inputs[7] = sin(bearing);
        sensory_inputs[8] = cos(bearing);
        sensory_inputs[9] = clamp(1.0 - best_dist / (100.0 + 70.0 * tr2), -1.0, 1.0);
        sensory_inputs[14] = clamp(agent_states[best_neighbor].angle_energy[1] / 70.0 - 1.0, -1.0, 1.0);
    } else {
        sensory_inputs[7] = 0.0;
        sensory_inputs[8] = 1.0;
        sensory_inputs[9] = -1.0;
        sensory_inputs[14] = 0.0;
    }

    sensory_inputs[10] = clamp(density / 9.0, 0.0, 1.0);
    sensory_inputs[11] = clamp(f32(a_age) / 600.0, 0.0, 1.0);
    sensory_inputs[12] = sin(f32(params.tick) * 0.08 + f32(agent_states[agent_idx].id));
    sensory_inputs[13] = clamp(vel.x * cos(angle) + vel.y * sin(angle), -1.0, 1.0);

    // Canonical 326-Weight Elman RNN Forward Pass
    let h_prev = array<f32, 10>(
        agent_states[agent_idx].hidden[0].x, agent_states[agent_idx].hidden[0].y,
        agent_states[agent_idx].hidden[0].z, agent_states[agent_idx].hidden[0].w,
        agent_states[agent_idx].hidden[1].x, agent_states[agent_idx].hidden[1].y,
        agent_states[agent_idx].hidden[1].z, agent_states[agent_idx].hidden[1].w,
        agent_states[agent_idx].hidden_tail.x, agent_states[agent_idx].hidden_tail.y
    );

    let scale_h = 0.61 / 127.0;
    let scale_o = 0.66 / 127.0;

    var p = 0u;
    var new_h: array<f32, 10>;
    for (var j = 0u; j < 10u; j += 1u) {
        var s = 0.0;
        for (var k = 0u; k < 15u; k += 1u) {
            s += get_gene(agent_idx, p) * sensory_inputs[k];
            p += 1u;
        }
        for (var k = 0u; k < 10u; k += 1u) {
            s += get_gene(agent_idx, p) * h_prev[k];
            p += 1u;
        }
        s += get_gene(agent_idx, p); // bias
        p += 1u;
        new_h[j] = tanh(s * scale_h);
    }

    var out: array<f32, 6>;
    for (var j = 0u; j < 6u; j += 1u) {
        var s = 0.0;
        for (var k = 0u; k < 10u; k += 1u) {
            s += get_gene(agent_idx, p) * new_h[k];
            p += 1u;
        }
        s += get_gene(agent_idx, p); // bias
        p += 1u;
        out[j] = tanh(s * scale_o);
    }

    // Actuator 0: Steering (HTML: a.angle += o[0] * (0.11 + 0.09 * tr1))
    angle += out[0] * (0.11 + 0.09 * tr1);

    // Actuator 1: Thrust & Kinematics (HTML: thrust = (o[1] + 1) * 0.5; mot = 0.45 + 1.1 * tr1)
    let thrust = (out[1] + 1.0) * 0.5;
    let mot = 0.45 + 1.1 * tr1;
    let fwd_dir = vec2f(cos(angle), sin(angle));
    let thrust_force = fwd_dir * (thrust * mot * 0.22);
    let new_vel = (vel + thrust_force) * 0.89;
    var new_pos = wrap_coords(pos + new_vel);

    a_feed = max(0.0, out[2]);
    a_attack = max(0.0, out[3]);
    let a_signal = max(0.0, out[4]);
    var a_last_victim = u32(agent_states[agent_idx].traits[1][3]);

    // Soil Grazing
    let cell_w = params.world_size.x / f32(params.soil_grid.x);
    let cell_h = params.world_size.y / f32(params.soil_grid.y);
    let cx = min(u32(max(0.0, new_pos.x) / cell_w), params.soil_grid.x - 1u);
    let cy = min(u32(max(0.0, new_pos.y) / cell_h), params.soil_grid.y - 1u);
    let cell_idx = cy * params.soil_grid.x + cx;
    let intake_cap = (0.016 + 0.064 * a_feed) * (0.7 + tr4);
    let eaten_float = min(intake_cap, max(0.0, here_food));
    let eaten_milli = i32(eaten_float * 1000.0);
    atomicSub(&soil_buffer[cell_idx].food_milli, eaten_milli);
    atomicAdd(&queue_buffer.telemetry.food_grazed_milli, u32(max(0, eaten_milli)));

    // Scent emission (HTML line 1394)
    if (a_signal > 0.4) {
        let scent_milli = i32((a_signal - 0.4) * 0.035 * 1000.0);
        atomicAdd(&soil_buffer[cell_idx].scent_milli, scent_milli);
    }

    // Combat Resolution & Decisive Killer Attribution
    let contact_dist = 14.0 + 10.0 * tr0;
    if (best_neighbor != 0xFFFFFFFFu && best_dist < contact_dist && a_attack > 0.25 && a_cooldown == 0u) {
        let victim_idx = best_neighbor;
        let victim_tr3 = agent_states[victim_idx].traits[0][3];
        let damage = (0.5 + a_attack * 2.2) * params.hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3);
        let damage_milli = i32(damage * 1000.0);
        let old_energy_milli = atomicSub(&agent_atomics[victim_idx].energy_milli, damage_milli);

        a_cooldown = 3u;
        a_last_victim = agent_states[victim_idx].id;
        let combat_gain = damage * (0.1 + 0.55 * tr5);
        atomicAdd(&agent_atomics[agent_idx].energy_milli, i32(combat_gain * 1000.0));

        // Decisive killer attribution:
        if (old_energy_milli > 0 && old_energy_milli <= damage_milli) {
            a_kills += 1u;
            atomicAdd(&queue_buffer.telemetry.kills, 1u);
            atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
            let kill_bonus = min(9.0, 8.0 * tr5);
            atomicAdd(&agent_atomics[agent_idx].energy_milli, i32(kill_bonus * 1000.0));

            // Frustum-culled stochastic audio emission:
            let dx = abs(pos.x - params.camera_pos.x);
            let dist_x = min(dx, params.world_size.x - dx);
            let dy = abs(pos.y - params.camera_pos.y);
            let dist_y = min(dy, params.world_size.y - dy);
            let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
            if (in_view) {
                let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (params.world_size.x - 150.0), 0.0, 1.0);
                let kill_volume = mix(0.08, 1.0, zoom_factor);
                let density_filter = select(1u, 4u, params.agent_count > 10000u);
                if (pcg_hash(agent_idx, victim_idx, params.tick) % density_filter == 0u) {
                    let voice_slot = atomicAdd(&queue_buffer.telemetry.audio_voice_count, 1u);
                    if (voice_slot < 256u) {
                        queue_buffer.audio[voice_slot] = AudioVoice(pos, 1u /* EVENT_KILL */, kill_volume);
                    }
                }
            }

            // Atomic CAS death ownership
            let claim_death = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
            if (claim_death.exchanged) {
                let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
                if (free_slot < params.max_agents) {
                    freelist[free_slot] = victim_idx;
                }

                // Corpse deposition from victim
                let vpos = agent_states[victim_idx].pos_vel.xy;
                let vcx = min(u32(max(0.0, vpos.x) / cell_w), params.soil_grid.x - 1u);
                let vcy = min(u32(max(0.0, vpos.y) / cell_h), params.soil_grid.y - 1u);
                let v_cell = vcy * params.soil_grid.x + vcx;
                atomicAdd(&soil_buffer[v_cell].food_milli, 600);
                atomicAdd(&soil_buffer[v_cell].taint_milli, 100);
            }
        }
    }

    // Reproduction (sexual crossover if viable partner nearby, otherwise asexual)
    // HTML: a.energy > 58 + 12 * tr0 && a.age > 65 && a.birth === 0 && o[5] > -0.15 && agents.length < CAP
    let can_reproduce = (params.agent_count < params.max_capacity)
        && (out[5] > -0.15)
        && (a_energy > (58.0 + 12.0 * tr0))
        && (a_age > 65u)
        && (a_birth == 0u);

    if (can_reproduce) {
        var mate_partner = 0xFFFFFFFFu;
        if (best_neighbor != 0xFFFFFFFFu && best_dist < 18.0) {
            let partner_idx = best_neighbor;
            let partner_energy = agent_states[partner_idx].angle_energy[1];
            let partner_root = agent_states[partner_idx].meta_flags & 0x0Fu;
            if (partner_energy > 42.0 && partner_root != a_root) {
                if (ENABLE_SEXUAL_SELECTION) {
                    let my_energy_milli = u32(max(0.0, a_energy) * 1000.0);
                    let prev_bid = atomicMax(&agent_atomics[partner_idx].mate_energy_milli, my_energy_milli);
                    if (my_energy_milli > prev_bid) {
                        atomicStore(&agent_atomics[partner_idx].mate_claim, agent_idx + 1u);
                    }
                    mate_partner = partner_idx;
                } else if (pcg_float(agent_idx, 99u, params.tick) < 0.15) {
                    let partner_id = agent_states[partner_idx].id;
                    let my_id = agent_states[agent_idx].id;
                    if (partner_id > my_id) {
                        let claim = atomicCompareExchangeWeak(&agent_atomics[partner_idx].mate_claim, 0u, agent_idx + 1u);
                        if (claim.exchanged) {
                            mate_partner = partner_idx;
                        }
                    }
                }
            }
        }

        let queue_idx = atomicAdd(&queue_buffer.telemetry.birth_count, 1u);
        if (queue_idx < 65536u) {
            atomicSub(&agent_atomics[agent_idx].energy_milli, 24000);
            a_energy = max(0.0, a_energy - 24.0);
            if (mate_partner != 0xFFFFFFFFu) {
                atomicSub(&agent_atomics[mate_partner].energy_milli, 6000);
            }
            a_birth = 95u;
            a_cooldown = 10u;
            queue_buffer.births[queue_idx] = BirthEvent(agent_idx, mate_partner, 0xFFFFFFFFu, 0u);

            // Frustum-culled birth audio voice
            let dx = abs(pos.x - params.camera_pos.x);
            let dist_x = min(dx, params.world_size.x - dx);
            let dy = abs(pos.y - params.camera_pos.y);
            let dist_y = min(dy, params.world_size.y - dy);
            let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
            if (in_view) {
                let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (params.world_size.x - 150.0), 0.0, 1.0);
                let birth_volume = mix(0.08, 0.7, zoom_factor);
                let voice_slot = atomicAdd(&queue_buffer.telemetry.audio_voice_count, 1u);
                if (voice_slot < 256u) {
                    queue_buffer.audio[voice_slot] = AudioVoice(pos, 2u /* EVENT_BIRTH */, birth_volume);
                }
            }
        }
    }


    // Energy Gain and Basal/Thrust/Taint metabolic cost
    let energy_gain = eaten_float * (9.0 + 9.0 * tr4);
    let basal_cost = 0.10 + 0.12 * tr0 + 0.07 * tr1 + 0.035 * tr2 + 0.055 * tr3 + 0.035 * a_attack + 0.014 * a_signal;
    let thrust_cost = thrust * 0.06;
    let taint_cost = here_taint * (0.10 + 0.18 * (1.0 - tr3));
    let internal_delta_milli = i32((energy_gain - basal_cost - thrust_cost - taint_cost) * 1000.0);
    atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);


    // Strict Single-Writer Death Check (Energy depletion or Senescence at age > 2100)
    let already_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed);
    let current_energy_milli = atomicLoad(&agent_atomics[agent_idx].energy_milli);

    if (already_claimed != 0u || current_energy_milli <= 0 || a_age > 2100u) {
        if (already_claimed == 0u) {
            let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
            if (claim_death.exchanged) {
                atomicAdd(&queue_buffer.telemetry.starvations, 1u);
                atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
                let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
                if (free_slot < params.max_agents) {
                    freelist[free_slot] = agent_idx;
                }

                // Corpse deposition into soil
                let corpse_food = clamp(f32(max(0, current_energy_milli)) * 0.000016 + 0.6, 0.3, 2.0);
                atomicAdd(&soil_buffer[cell_idx].food_milli, i32(corpse_food * 1000.0));
                atomicAdd(&soil_buffer[cell_idx].taint_milli, 100);
            }
        }
        agent_states[agent_idx].meta_flags |= (1u << 13u);
        agent_states[agent_idx].angle_energy[1] = 0.0;
        agent_states[agent_idx].visual_cache = 0u;
        return;
    }

    a_energy = clamp(f32(current_energy_milli) * 0.001, 0.0, 110.0);
    atomicStore(&agent_atomics[agent_idx].energy_milli, i32(a_energy * 1000.0));
    atomicMax(&queue_buffer.telemetry.apex_record_milli, u32(a_energy * 1000.0));

    // Telemetry instantaneous population census on final sub-tick
    if (params.sub_tick == params.sub_ticks_per_frame - 1u) {
        atomicAdd(&queue_buffer.telemetry.population, 1u);
        atomicAdd(&queue_buffer.telemetry.lineage_counts[a_root % 16u], 1u);
    }

    // Age and cooldown updates
    a_age = min(a_age + 1u, 65535u);
    if (a_birth > 0u) { a_birth -= 1u; }
    if (a_cooldown > 0u) { a_cooldown -= 1u; }

    // Commit state
    agent_states[agent_idx].pos_vel = vec4f(new_pos.x, new_pos.y, new_vel.x, new_vel.y);
    agent_states[agent_idx].angle_energy = vec4f(angle, a_energy, a_feed, a_attack);
    agent_states[agent_idx].traits[1] = vec4f(tr4, tr5, a_signal, f32(a_last_victim));
    agent_states[agent_idx].hidden[0] = vec4f(new_h[0], new_h[1], new_h[2], new_h[3]);
    agent_states[agent_idx].hidden[1] = vec4f(new_h[4], new_h[5], new_h[6], new_h[7]);
    agent_states[agent_idx].hidden_tail = vec2f(new_h[8], new_h[9]);
    agent_states[agent_idx].meta_flags = (a_root & 0x0Fu) | (a_cooldown << 4u) | (a_birth << 6u) | (a_kills << 14u);
    agent_states[agent_idx].age_gen = (a_age & 0xFFFFu) | (a_gen << 16u);

    // Pack visual_cache
    let r_u8 = u32(clamp(tr0, 0.0, 1.0) * 255.0);
    let glow_u8 = u32(clamp(max(a_attack, select(0.0, 1.0, a_birth > 0u)), 0.0, 1.0) * 255.0);
    let e_u8 = u32(clamp(a_energy / 100.0, 0.0, 1.0) * 255.0);
    let vis_flags = select(0u, 1u << 24u, a_attack > 0.25) | select(0u, 1u << 25u, a_birth > 0u);
    agent_states[agent_idx].visual_cache = r_u8 | (glow_u8 << 8u) | (e_u8 << 16u) | vis_flags;
}
