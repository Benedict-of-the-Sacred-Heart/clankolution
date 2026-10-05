// crates/clank_app/assets/shaders/preamble_clear.wgsl
// GPU Simulation Preamble Clear Pass
//
// Dispatched before agent_step.wgsl to guarantee a clean GPU execution barrier.
// Separates sub-tick counters from frame-level aggregate counters.

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

struct GpuAgentAtomic {
    energy_milli: atomic<i32>,
    mate_claim: atomic<u32>,
    mate_energy_milli: atomic<u32>,
    dead_claimed: atomic<u32>,
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

@group(0) @binding(0) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(1) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(2) var<storage, read_write> cell_offsets: array<vec2u, 54>;
@group(1) @binding(0) var<uniform> params: GpuSimParams;

@compute @workgroup_size(64)
fn preamble_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < params.max_agents) {
        atomicStore(&agent_atomics[id.x].mate_claim, 0u);
        atomicStore(&agent_atomics[id.x].mate_energy_milli, 0u);
        atomicStore(&agent_atomics[id.x].dead_claimed, 0u);
    }

    if (id.x == 0u) {
        // Active queue depth reset (runs every sub-tick for birth processing):
        atomicStore(&queue_buffer.telemetry.birth_count, 0u);

        // Frame-level cumulative counter reset (strictly at frame start on sub-tick 0):
        if (params.sub_tick == 0u) {
            atomicStore(&queue_buffer.telemetry.audio_voice_count, 0u);
            atomicStore(&queue_buffer.telemetry.kills, 0u);
            atomicStore(&queue_buffer.telemetry.starvations, 0u);
            atomicStore(&queue_buffer.telemetry.apex_record_milli, 0u);
            atomicStore(&queue_buffer.telemetry.food_grazed_milli, 0u);
            if (params.tool_type == 0u) {
                queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
                queue_buffer.telemetry.selected_agent_id = 0u;
            }
        }
    }

    // Instantaneous census is cleared on final sub-tick before recounting:
    if (params.sub_tick == params.sub_ticks_per_frame - 1u) {
        if (id.x == 0u) {
            atomicStore(&queue_buffer.telemetry.population, 0u);
        }
        if (id.x < 16u) {
            atomicStore(&queue_buffer.telemetry.lineage_counts[id.x], 0u);
        }
    }

    if (id.x < 54u) {
        cell_offsets[id.x] = vec2u(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
}
