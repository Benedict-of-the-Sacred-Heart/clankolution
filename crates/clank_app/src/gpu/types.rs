//! GPU Compute Simulation Pipeline Types
//!
//! Provides std140/std430 compliant memory layouts for WebGPU compute shaders.
//! All structs are aligned to 16 bytes for universal cross-platform compatibility
//! across Metal, Vulkan, and DirectX 12.

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentState {
    // Cache Line 0 (64 bytes): Kinematics, Energy & Morphological Traits
    pub pos_vel: [f32; 4],      // 16 bytes: x, y, vx, vy (full 32-bit float precision)
    pub angle_energy: [f32; 4], // 16 bytes: angle, energy, feeding, attack (full 32-bit float precision)
    pub traits: [f32; 8],       // 32 bytes: tr[0..5], signal, last_victim (full 32-bit float precision)

    // Cache Line 1 (64 bytes): RNN Hidden States, Lossless Bit-Packed Metadata & Caches
    pub hidden: [f32; 10],      // 40 bytes: h[0..9] recurrent hidden states (full 32-bit float precision)
    pub id: u32,                // 4 bytes: full 32-bit unique creature ID (up to 4.29 billion)
    pub meta_flags: u32,        // 4 bytes: root (4b), cooldown (2b), birth (7b), dead (1b), kills (18b)
    pub age_gen: u32,           // 4 bytes: age (16b: 0..65,535), gen (16b: 0..65,535)
    pub morton_code: u32,       // 4 bytes: precomputed 32-bit Morton spatial hash key
    pub packed_color: u32,      // 4 bytes: rgba8unorm packed lineage color for direct GPU instancing
    pub visual_cache: u32,      // 4 bytes: packed rendering cache [radius_u8, glow_u8, energy_u8, flags_u8]
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentGenome {
    pub packed_genes: [u32; 88], // 352 bytes: 10 hidden * 7 vec4s (70 words) + 6 output * 3 vec4s (18 words)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentAtomic {
    pub energy_milli: i32,      // 4 bytes: atomic fixed-point millijoules
    pub mate_claim: u32,        // 4 bytes: atomic CAS mating winner agent_idx + 1 (0 = unclaimed)
    pub mate_energy_milli: u32, // 4 bytes: atomic highest bid in millijoules
    pub dead_claimed: u32,      // 4 bytes: atomic CAS death ownership (prevents double-free)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    // Chunk 0: Simulation Physics & Capacity Bounds (16B)
    pub tick: u32,                  // 4 bytes  (0..4)
    pub agent_count: u32,           // 4 bytes  (4..8)
    pub max_agents: u32,            // 4 bytes  (8..12)
    pub max_capacity: u32,          // 4 bytes  (12..16)

    // Chunk 1: Environmental Chemistry & Rates (16B)
    pub hostility: f32,             // 4 bytes  (16..20)
    pub mut_rate: f32,              // 4 bytes  (20..24)
    pub speed: f32,                 // 4 bytes  (24..28)
    pub renewal: f32,               // 4 bytes  (28..32)

    // Chunk 2: Multi-Tick Batching & Interactive Tools (16B)
    pub sub_tick: u32,              // 4 bytes  (32..36)
    pub sub_ticks_per_frame: u32,   // 4 bytes  (36..40)
    pub tool_type: u32,             // 4 bytes  (40..44)
    pub tool_radius: f32,           // 4 bytes  (44..48) repurposed from _pad0

    // Chunk 3: Tool & Camera Positions (16B)
    pub tool_pos: [f32; 2],         // 8 bytes  (48..56)
    pub camera_pos: [f32; 2],       // 8 bytes  (56..64)

    // Chunk 4: Camera Viewport & Arena Bounds (16B)
    pub camera_size: [f32; 2],      // 8 bytes  (64..72)
    pub world_size: [f32; 2],       // 8 bytes  (72..80)

    // Chunk 5: Soil Grid & Simulation State (16B)
    pub soil_grid: [u32; 2],        // 8 bytes  (80..88)
    pub eclipse: u32,               // 4 bytes  (88..92) repurposed from _pad1[0]
    pub epoch: u32,                 // 4 bytes  (92..96) repurposed from _pad1[1]
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BirthEvent {
    pub parent_a: u32,          // 4 bytes
    pub parent_b: u32,          // 4 bytes: 0xFFFFFFFFu if asexual virgin birth
    pub child_slot: u32,        // 4 bytes
    pub birth_tick: u32,        // 4 bytes: repurposed from pad
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AudioVoice {
    pub pos: [f32; 2],          // 8 bytes: arena world position
    pub event_type: u32,        // 4 bytes: 0 = bite/attack, 1 = kill, 2 = birth
    pub volume: f32,            // 4 bytes: zoom-modulated volume [0.08, 1.0]
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTelemetry {
    // Cache Line 0 (64 bytes): Engine Counters, Apex Records & Uncapped Specimen Picking
    pub population: u32,        // 4 bytes: active living agents (updated on final sub-tick)
    pub kills: u32,             // 4 bytes: total predatory kills this frame
    pub starvations: u32,       // 4 bytes: total starvation deaths this frame
    pub apex_record_milli: u32, // 4 bytes: highest energy recorded (atomicMax)
    pub food_grazed_milli: u32, // 4 bytes: total food consumed in millijoules
    pub sub_ticks_elapsed: u32, // 4 bytes: number of sub-ticks processed
    pub apex_agent_id: u32,     // 4 bytes: ID of apex creature
    pub freelist_top: u32,      // 4 bytes: atomic stack top for tombstone freelist
    pub birth_count: u32,       // 4 bytes: births queued this frame
    pub audio_voice_count: u32, // 4 bytes: audio events queued this frame
    pub selected_agent_idx: u32,// 4 bytes: full 32-bit slot index of picked creature (0xFFFFFFFF = none)
    pub selected_agent_id: u32, // 4 bytes: full 32-bit unique creature ID for identity guard
    pub total_births: u32,      // 4 bytes: lifetime births (repurposed from _reserved0[0])
    pub total_deaths: u32,      // 4 bytes: lifetime deaths (repurposed from _reserved0[1])
    pub max_generation: u32,    // 4 bytes: highest generation (repurposed from _reserved0[2])
    pub extinctions: u32,       // 4 bytes: total lineage extinctions (repurposed from _reserved0[3])

    // Cache Line 1 (64 bytes): 16-Lineage Real-Time Extinction Monitoring
    pub lineage_counts: [u32; 16], // 16 * 4B = 64 bytes (head counts for roots 0..15)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConsolidatedQueue {
    pub telemetry: GpuTelemetry,            // 128 bytes (Cache Lines 0 & 1)
    pub births: [BirthEvent; 65536],        // 1,048,576 bytes = 1,024 KB = 1 MB
    pub audio: [AudioVoice; 256],           // 4,096 bytes = 4 KB (1 memory page)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSoilCell {
    pub food_milli: i32,        // 4 bytes: atomic fixed-point millifood
    pub taint_milli: i32,       // 4 bytes: atomic fixed-point millitaint
    pub scent_milli: i32,       // 4 bytes: atomic fixed-point milliscent
    pub fertility_milli: i32,   // 4 bytes: repurposed from pad
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLbvhNode {
    pub aabb_min: [f32; 2],       // 8 bytes: bounding box min [x, y]
    pub aabb_max: [f32; 2],       // 8 bytes: bounding box max [x, y] (Total 16)
    pub center_of_mass: [f32; 2], // 8 bytes: weighted centroid [x, y]
    pub count: u32,               // 4 bytes: subtree creature count
    pub dominant_lineage: u32,    // 4 bytes: lineage root 0..15 with highest count in subtree (Total 32)
    pub left_child: u32,          // 4 bytes: child node index
    pub right_child: u32,         // 4 bytes: child node index
    pub parent: u32,              // 4 bytes: parent node index
    pub leaf_idx: u32,            // 4 bytes: leaf agent index (0..N-1), or 0xFFFFFFFF for internal nodes (Total 48)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDartInstance {
    pub pos_angle: [f32; 3],      // 12 bytes: x, y, angle
    pub pad0: f32,                // 4 bytes: armor trait
    pub vis_data: [u32; 2],       // 8 bytes: packed_color, visual_cache
    pub pad1: [u32; 2],           // 8 bytes: carnivory trait (f32 bits), sight trait (f32 bits)
}

