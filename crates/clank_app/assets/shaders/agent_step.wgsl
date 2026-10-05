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
    pad: u32,
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
    _reserved0: array<u32, 4>,
    lineage_counts: array<atomic<u32>, 16>,
}

struct BirthEvent {
    parent_a: u32,
    parent_b: u32,
    child_slot: u32,
    pad: u32,
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
@group(0) @binding(5) var<storage, read> lbvh_nodes: array<GpuLbvhNode>;
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
    let fwd_reach = vec2f(cos(angle), sin(angle)) * (14.0 + 8.0 * tr0);
    let left_offset = vec2f(-sin(angle), cos(angle)) * (9.0 + 5.0 * tr0);

    let probe_here    = sample_soil_probe(pos);
    let probe_forward = sample_soil_probe(pos + fwd_reach);
    let probe_left    = sample_soil_probe(pos + fwd_reach * 0.7 + left_offset);
    let probe_right   = sample_soil_probe(pos + fwd_reach * 0.7 - left_offset);

    let here_food    = probe_here.r;
    let forward_food = probe_forward.r;
    let left_food    = probe_left.r;
    let right_food   = probe_right.r;
    let here_taint   = probe_here.g;
    let scent_diff   = probe_forward.b - probe_here.b;

    // LBVH Nearest Neighbor Search
    var best_dist = 999999.0;
    var best_neighbor = 0xFFFFFFFFu;
    var sensory_bearing = 0.0;
    var sensory_dist = 0.0;

    let sight_radius = 120.0 + 80.0 * tr2;

    if (params.agent_count > 1u) {
        var stack: array<u32, 32>;
        var stack_ptr = 0u;
        stack[0] = 0u; // Root
        stack_ptr = 1u;
        var search_r = sight_radius;

        while (stack_ptr > 0u) {
            stack_ptr -= 1u;
            let node_idx = stack[stack_ptr];
            let node = lbvh_nodes[node_idx];

            let box_d = distance_to_aabb(pos, node.aabb_min, node.aabb_max);
            if (box_d > search_r) { continue; }

            if (node.leaf_idx != 0xFFFFFFFFu) {
                let other_idx = node.leaf_idx;
                if (other_idx != agent_idx) {
                    let other_meta = agent_states[other_idx].meta_flags;
                    if ((other_meta & (1u << 13u)) == 0u) {
                        let other_pos = agent_states[other_idx].pos_vel.xy;
                        let d = toroidal_dist(pos, other_pos);
                        if (d < best_dist && d <= search_r) {
                            best_dist = d;
                            best_neighbor = other_idx;
                            search_r = d; // Dynamic radius shrinking
                        }
                    }
                }
            } else {
                if (stack_ptr < 30u) {
                    if (node.right_child != 0xFFFFFFFFu) {
                        stack[stack_ptr] = node.right_child;
                        stack_ptr += 1u;
                    }
                    if (node.left_child != 0xFFFFFFFFu) {
                        stack[stack_ptr] = node.left_child;
                        stack_ptr += 1u;
                    }
                }
            }
        }

        if (best_neighbor != 0xFFFFFFFFu) {
            let neighbor_pos = agent_states[best_neighbor].pos_vel.xy;
            var dx = neighbor_pos.x - pos.x;
            if (dx > params.world_size.x * 0.5) { dx -= params.world_size.x; }
            else if (dx < -params.world_size.x * 0.5) { dx += params.world_size.x; }

            var dy = neighbor_pos.y - pos.y;
            if (dy > params.world_size.y * 0.5) { dy -= params.world_size.y; }
            else if (dy < -params.world_size.y * 0.5) { dy += params.world_size.y; }

            let angle_to_neighbor = atan2(dy, dx);
            var bearing = angle_to_neighbor - angle;
            let pi = 3.14159265;
            if (bearing > pi) { bearing -= 2.0 * pi; }
            else if (bearing < -pi) { bearing += 2.0 * pi; }

            sensory_bearing = clamp(bearing / pi, -1.0, 1.0);
            sensory_dist = clamp(1.0 - best_dist / sight_radius, 0.0, 1.0);
        }
    }

    // Vectorized RNN Forward Pass
    var ins_hidden: array<vec4f, 7>;
    ins_hidden[0] = vec4f(here_food, clamp(forward_food - left_food, -1.0, 1.0), clamp(forward_food - right_food, -1.0, 1.0), clamp(forward_food - here_food, -1.0, 1.0));
    ins_hidden[1] = vec4f(clamp(here_taint, 0.0, 1.0), clamp(scent_diff, -1.0, 1.0), sensory_bearing, sensory_dist);
    ins_hidden[2] = vec4f(vel.x * 0.2, vel.y * 0.2, clamp(a_energy * 0.01, 0.0, 1.0), sin(f32(params.tick) * 0.05));

    ins_hidden[3] = agent_states[agent_idx].hidden[0];
    ins_hidden[4] = agent_states[agent_idx].hidden[1];
    ins_hidden[5] = vec4f(agent_states[agent_idx].hidden_tail.x, agent_states[agent_idx].hidden_tail.y, 0.0, 0.0);
    if (ENABLE_EXPANDED_CORTEX) {
        let grad_x = forward_food - here_food;
        let grad_y = left_food - right_food;
        ins_hidden[6] = vec4f(clamp(grad_x * 2.0, -1.0, 1.0), clamp(grad_y * 2.0, -1.0, 1.0), sensory_bearing, 1.0);
    } else {
        ins_hidden[6] = vec4f(0.0, 0.0, 0.0, 0.0);
    }

    var new_h: array<f32, 10>;
    for (var j = 0u; j < 10u; j += 1u) {
        var s = 0.0;
        let base_w = j * 7u;
        for (var k = 0u; k < 7u; k += 1u) {
            s += dot(unpack4x8snorm(agent_genomes[agent_idx].packed_genes[base_w + k]), ins_hidden[k]);
        }
        new_h[j] = tanh(s * 0.61);
    }

    var ins_output: array<vec4f, 3>;
    ins_output[0] = vec4f(new_h[0], new_h[1], new_h[2], new_h[3]);
    ins_output[1] = vec4f(new_h[4], new_h[5], new_h[6], new_h[7]);
    ins_output[2] = vec4f(new_h[8], new_h[9], 0.0, 0.0);

    var out: array<f32, 6>;
    for (var j = 0u; j < 6u; j += 1u) {
        var s = 0.0;
        let base_w = 70u + j * 3u;
        for (var k = 0u; k < 3u; k += 1u) {
            s += dot(unpack4x8snorm(agent_genomes[agent_idx].packed_genes[base_w + k]), ins_output[k]);
        }
        out[j] = tanh(s * 0.66);
    }

    // Kinematics & steering
    let steer = out[0] * 0.22;
    let thrust = clamp(out[1] * 0.5 + 0.5, 0.0, 1.0);
    angle += steer;

    let fwd_vec = vec2f(cos(angle), sin(angle));
    let thrust_force = fwd_vec * (thrust * (0.8 + 0.4 * tr1));
    var new_vel = (vel + thrust_force) * 0.88;

    if (ENABLE_BARNES_HUT && params.agent_count > 10u) {
        let root_node = lbvh_nodes[0];
        let macro_center = root_node.center_of_mass;
        var m_dx = macro_center.x - pos.x;
        if (m_dx > params.world_size.x * 0.5) { m_dx -= params.world_size.x; }
        else if (m_dx < -params.world_size.x * 0.5) { m_dx += params.world_size.x; }
        var m_dy = macro_center.y - pos.y;
        if (m_dy > params.world_size.y * 0.5) { m_dy -= params.world_size.y; }
        else if (m_dy < -params.world_size.y * 0.5) { m_dy += params.world_size.y; }
        let m_dist = max(length(vec2f(m_dx, m_dy)), 1.0);
        let macro_force = vec2f(m_dx, m_dy) / m_dist * 0.03;
        new_vel = (vel + thrust_force + macro_force) * 0.88;
    }

    var new_pos = wrap_coords(pos + new_vel);

    a_feed = clamp(out[2] * 0.5 + 0.5, 0.0, 1.0);
    a_attack = clamp(out[3] * 0.5 + 0.5, 0.0, 1.0);


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

    // Combat Resolution & Decisive Killer Attribution
    let contact_dist = 14.0 + 10.0 * tr0;
    if (best_neighbor != 0xFFFFFFFFu && best_dist < contact_dist && a_attack > 0.25 && a_cooldown == 0u) {
        let victim_idx = best_neighbor;
        let victim_tr3 = agent_states[victim_idx].traits[0][3];
        let damage = (0.5 + a_attack * 2.2) * params.hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3);
        let damage_milli = i32(damage * 1000.0);
        let old_energy_milli = atomicSub(&agent_atomics[victim_idx].energy_milli, damage_milli);

        a_cooldown = 3u;
        let combat_gain = damage * (0.1 + 0.55 * tr5);
        atomicAdd(&agent_atomics[agent_idx].energy_milli, i32(combat_gain * 1000.0));

        // Decisive killer attribution:
        if (old_energy_milli > 0 && old_energy_milli <= damage_milli) {
            a_kills += 1u;
            atomicAdd(&queue_buffer.telemetry.kills, 1u);
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
                freelist[free_slot] = victim_idx;
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
            let p_mate = pcg_float(agent_idx, 99u, params.tick);
            if (partner_energy > 42.0 && partner_root != a_root && p_mate < 0.15) {
                let claim = atomicCompareExchangeWeak(&agent_atomics[partner_idx].mate_claim, 0u, agent_idx + 1u);
                if (claim.exchanged) {
                    mate_partner = partner_idx;
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
    let basal_cost = 0.10 + 0.12 * tr0 + 0.07 * tr1 + 0.035 * tr2 + 0.055 * tr3 + 0.035 * a_attack;
    let thrust_cost = thrust * 0.06;
    let taint_cost = here_taint * (0.10 + 0.18 * (1.0 - tr3));
    let internal_delta_milli = i32((energy_gain - basal_cost - thrust_cost - taint_cost) * 1000.0);
    atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);


    // Strict Single-Writer Death Check
    let already_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed);
    let current_energy_milli = atomicLoad(&agent_atomics[agent_idx].energy_milli);

    if (already_claimed != 0u || current_energy_milli <= 0) {
        if (already_claimed == 0u) {
            let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
            if (claim_death.exchanged) {
                atomicAdd(&queue_buffer.telemetry.starvations, 1u);
                let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
                freelist[free_slot] = agent_idx;
            }
        }
        agent_states[agent_idx].meta_flags |= (1u << 13u);
        agent_states[agent_idx].angle_energy[1] = 0.0;
        agent_states[agent_idx].visual_cache = 0u;
        return;
    }

    a_energy = max(0.0, f32(current_energy_milli) * 0.001);
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
