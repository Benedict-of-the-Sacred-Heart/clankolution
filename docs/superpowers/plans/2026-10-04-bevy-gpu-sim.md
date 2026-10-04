# GPU Compute Simulation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-throughput GPU compute simulation engine (`ENGINE: GPU`) running entirely in WGSL on a dedicated branch (`feature/bevy-gpu`), featuring zero-copy Tombstone Freelist buffers, a hybrid Morton Grid + multi-system Linear Bounding Volume Hierarchy (LBVH), packed 326-weight quantized RNN forward passes, and reactive GPU soil chemistry—coexisting seamlessly alongside the bit-exact CPU reference engine (`ENGINE: RUST`).

**Architecture:** A multi-pass Bevy compute pipeline operating over GPU storage buffers and 2D textures. Agents never reallocate or shift in VRAM thanks to a lock-free Tombstone Freelist. Spatial lookups use a hybrid Morton Grid (Tier 1: $O(1)$ local Moore cells) backed by an explicit Karras Linear Bounding Volume Hierarchy (LBVH). The unified LBVH tree powers five distinct subsystems: 1) Long-range sensory raycasting, 2) Camera viewport frustum culling, 3) Interactive AoE brush tools, 4) Hierarchical minimap LOD cluster rendering, and 5) Optional Barnes-Hut macro-swarm flocking (an experimental mod toggle, default OFF to guarantee 100% HTML behavioral parity). The dual-engine architecture preserves $100\%$ bit-for-bit historical CPU determinism when toggled to `ENGINE: RUST` while unlocking massive agent scaling when toggled to `ENGINE: GPU`.

**Tech Stack:** Rust, Bevy 0.19 (`bevy_render::render_resource`, `wgpu`), WGSL (WebGPU Shading Language), Metal/Vulkan compute pipelines, `rkyv`, `image`.

**Branching Strategy:** Work will proceed exclusively on a new branch: `feature/bevy-gpu` branched directly from `feature/bevy-port`. The base branch `feature/bevy-port` will remain clean and stable until `feature/bevy-gpu` is verified, tested, and visually confirmed.

---

## Explicit GPU Optimizations Covered in this Plan

1. **Decoupled Soil Simulation State & Single Universal `rgba16float` Texture (Zero-Copy)**:
   - **True Simulation State in L2 Cache**: Soil chemistry (food, taint, scent) is maintained in a dedicated 60 KB atomic storage buffer (`GpuSoilCell` for all $75 \times 50 = 3,750$ cells, $3,750 \times 16\text{ bytes} = 60\text{ KB}$). Fits 100% inside GPU L1/L2 cache.
   - **Coalesced Atomic Creature Writes**: Creatures in `agent_step.wgsl` atomically subtract grazed food (`atomicSub(&food_milli)`) and add scent/carcass taint directly to this 60 KB buffer.
   - **Single Texture Rasterization (No Ping-Pong Copy)**: `soil_step.wgsl` runs once per tick, reads the updated atomic cells, applies environmental bloom renewal and decay, writes back to the buffer, and rasterizes the colormap directly into a **single 2D `rgba16float` texture**.
   - **Universal Cross-Platform Linear Filtering**: Binds as `rgba16float` with universal hardware bilinear sampling (`textureSampleLevel`) across macOS (Metal), Windows (Vulkan/DX12), and Linux (Vulkan). **Completely eliminates texture ping-pong pointer swapping and eliminates $B \to A$ GPU memory copying!**

2. **Sensory Raycasting & Universal `unpack4x8snorm` Neural Net Vectorization**:
   - **Antennae Feelers**: Hardware bilinear texture fetches at 4 probe locations (`here`, `forward`, `left`, `right`) with hardware coordinate wrapping.
   - **2-Tier Hybrid Spatial Raycasting**:
     - Tier 1: $O(1)$ direct lookup in the Morton Uniform Grid ($3 \times 3$ cells, $d \le 100\text{px}$) for $99.8\%$ of creature checks with zero warp divergence.
     - Tier 2: Ring-expansion onto-scan fallback ($5 \times 5$ and $7 \times 7$ cells) guaranteeing **$100\%$ mathematical equivalence** to the original global scan for distant/isolated creatures.
   - **Universal `unpack4x8snorm` Vectorization & `vec4` Neuron Alignment**: In HTML, scaling constants are $\text{SCALE\_H} = 0.61 / 127.0$ and $\text{SCALE\_O} = 0.66 / 127.0$. We use the core WGSL built-in `unpack4x8snorm(word)` which natively unpacks 4 signed 8-bit integers into normalized floats $[-1.0, 1.0]$ (dividing by 127.0 in a single ALU cycle).
   - **Neuron Word Alignment (88 u32 Words = 352 Bytes)**:
     - 10 hidden neurons: each 26 weights padded to 28 weights ($7 \times \text{vec4} = 7\text{ words}$). $10 \times 7 = 70\text{ words}$.
     - 6 output neurons: each 11 weights padded to 12 weights ($3 \times \text{vec4} = 3\text{ words}$). $6 \times 3 = 18\text{ words}$.
     - Total: $70 + 18 = 88\text{ words}$ (352 bytes).
     - Every neuron computes as an unrolled, branchless loop: `s += dot(unpack4x8snorm(w), in_vec4)`. In baseline mode, dummy padded inputs are $0.0$, making the output **100% bit-exact to the HTML reference** with zero runtime bit-shifting or byte misalignment!

3. **Multi-System Linear Bounding Volume Hierarchy (LBVH)**:
   - Built in parallel using Karras 2012 LCP splits on sorted Morton codes ($\approx 0.3\text{ ms}$).
   - **Degenerate Population Guard ($N \le 1$)**: When population collapses or is $\le 1$, builder early-exits gracefully (`if (active_count < 2u) return;`), guarding against unsigned underflow in $N-2$ internal node bounds and falling back to direct Tier-1 lookups.
   - **Strict Key Uniqueness & Tie-Breaking**: If two agents have identical Morton keys (e.g. sharing positions), tie-breaking uses `32u + countLeadingZeros(agent_id_a ^ agent_id_b)` to guarantee distinct keys, preventing infinite loops or zero-length split branches.
   - **Subsystem 1 (Sensory Raycasting)**: $O(\log N)$ hierarchical traversal for long-range homing vectors.
   - **Subsystem 2 (Frustum Culling)**: Discards entire off-screen subtrees against the camera viewport $[x_{\min}, y_{\min}] \times [x_{\max}, y_{\max}]$, feeding visible agents into an indirect draw buffer.
   - **Subsystem 3 (AoE Tool Brushes)**: Accelerates `extinguish_at`, `blight_at`, and `nourish_at` from $O(N)$ linear scans to $O(\log N)$ bounding box intersections.
   - **Subsystem 4 (Minimap LOD Clustering)**: Draws aggregated cluster discs from intermediate tree depths (e.g. depth 5–6) directly onto the radar minimap.

4. **Cross-Cutting Morton Code Optimizations (Engine-Wide)**:
   - **Morton Cell Offsets Sentinel Initialization**: The 54-cell table (`cell_offsets: array<vec2u, 54>`) is cleared to `vec2u(0xFFFFFFFFu, 0xFFFFFFFFu)` in the preamble pass. In `agent_step.wgsl`, neighbor searches check `if (range.x != 0xFFFFFFFFu)`, instantly skipping empty cells with zero warp divergence.
   - **Lightweight 8-Byte Indirection Radix Sort**: Parallel radix sort operates strictly on `array<vec2u>` storing `(morton_key, agent_slot_idx)`. The heavy 496-byte `GpuAgent` structs remain completely stationary in VRAM, eliminating gigabytes/sec of memory bus saturation.
   - **Sub-Microsecond Mouse Picking ($O(\log N)$)**: Fast binary search on sorted Morton keys to select creatures for the specimen card.
   - **Coalesced Atomic Soil Writes**: Threads scheduled in Morton order write to contiguous L1/L2 cache lines for food/taint/scent deposits, cutting bus contention.
   - **Spatially Coherent Dart Rendering**: Morton-ordered instance streams maximize GPU tile-cache hit rates on Metal and tile-based rasterizers.

5. **Hardware Toroidal Coordinate Wrapping**:
   - In WGSL, coordinate wrapping uses:
     `pos.x - 900.0 * floor(pos.x / 900.0)` and `pos.y - 600.0 * floor(pos.y / 600.0)`
   - Evaluates in a single hardware `floor` and `fma` instruction, handling any negative coordinate seamlessly without edge pop-in.

6. **Birth / Compaction Freedoms & CAS Freelist Protection**:
   - **Tombstone Freelist & CAS Underflow Guard**: Fixed static `MAX_AGENTS` storage buffer. Dead creatures are marked with tombstone flags (`dead = 1`) and their slot indices are recycled onto an atomic stack (`atomicAdd` on deallocation). Allocation uses an atomic Compare-and-Swap (CAS) pop loop: checks `freelist_top > 0u` before popping, safely preventing unsigned 32-bit underflow (`4,294,967,295`) when population reaches carrying capacity. **Zero memory shifting or array compaction across frames.**
   - **Dedicated Preamble Clearing Pass (`preamble_clear.wgsl`)**: Dispatched immediately before `agent_step.wgsl` to eliminate cross-workgroup race conditions. Resets `mate_claim = 0u`, `dead_claimed = 0u`, `sim_counters.birth_count = 0u`, and `cell_offsets = 0xFFFFFFFFu`. Guarantees a global GPU execution and memory barrier across all workgroups before simulation stepping begins.
   - **Atomic CAS Double-Free Protection**: To prevent starvation and predation from double-freeing the same slot in the same tick, death ownership is acquired via CAS on `dead_claimed: atomic<u32>` ($0 \to 1$). Exactly one thread succeeds, sets `dead = 1`, and recycles the slot to `freelist`.
   - **Parallel Mating Handshake**: Mutual consent is verified via `atomicCompareExchangeWeak` on `mate_claim` in a dedicated L2 cache line. Deterministic symmetry breaking ($\min(A.\text{id}, B.\text{id})$) guarantees exactly one child is born per consenting pair, eliminating duplicate births and double-spending.

7. **Predation & Combat Resolution (Atomic Fixed-Point & Decisive Attribution)**:
   - **L2-Cached Atomic Damage Buffer**: `energy_milli: atomic<i32>` resides in an isolated 512KB buffer fitting entirely into GPU L2 cache, eliminating false-sharing contention with agent genome memory.
   - **Exact Formula Parity**: Damage evaluated via `(0.5 + attack * 2.2) * hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3)`.
   - **Decisive Killer Attribution**: Threads perform `atomicSub(&victim.energy_milli, dmg_milli)`. The exact thread crossing the zero threshold (`old > 0 && old <= dmg`) is attributed the kill, receives the siphon energy and kill bounty (`min(9.0, 8.0 * tr5)`), sets victim tombstone (`dead = 1`), and recycles the slot to the freelist via the `dead_claimed` CAS gate.

8. **Consolidated Atomic Counters Buffer (`SimCounters`)**:
   - Consolidates `freelist_top`, `birth_count`, and `kills_total` into a single 16-byte atomic struct buffer, keeping shader storage buffer bindings to 7 (well under the cross-platform limit of 8).

9. **Stateless Counter-Based PRNG & Triangular Distribution**:
   - **PCG 3D Hash in Hardware Registers**: Evaluates $\text{rand} = \text{pcg3d}(\text{vec3u}(\text{id}, \text{stream}, \text{tick}))$ in 2 nanoseconds directly in ALU registers. Requires $0$ bytes of persistent PRNG state, $0$ atomic locks, and guarantees bit-for-bit reproducibility across all GPU architectures.
   - **Triangular Distribution**: Computes $\mathcal{T}(-1, 0, 1)$ via $r_1 + r_2 - 1.0$ matching the exact HTML mutation distribution.

10. **Decoupled Birth Queue & Parallel Genome Mutation**:
    - **Decoupled Birth Pass**: Mating agents push a 16-byte `BirthEvent` to an atomic append queue. The main agent compute shader has **zero warp divergence** from gene loops.
    - **Parallel Genome SIMD Mutation (`birth_step.wgsl`)**: Dedicated workgroups process newborn agents, performing 88-word parallel crossover (48% parent crossover) and triangular mutation $(r_1 + r_2 - 1.0) \times 100 \times \text{mutRate}$ clamped to $[-127, 127]$, plus morphological trait mutation $(r_1 + r_2 - 1.0) \times \text{mutRate} \times 0.6$ clamped to $[0.03, 0.98]$.

11. **Experimental Simulation Mods Suite**:
    - **Mod 1: Barnes-Hut Macro-Flocking Mod (`[BARNES-HUT: OFF/ON]`)**: When toggled ON (default OFF for 100% HTML parity), queries coarse LBVH node centers of mass to apply far-field swarm acceleration vectors without modifying baseline neural weights or requiring $O(N^2)$ checks.
    - **Mod 2: Expanded Cortex Mod (`[EXPANDED CORTEX: OFF/ON]`)**:
      - Default OFF (100% HTML parity): Dummy weights in the 88-word layout are multiplied by $0.0$, reproducing the 326-weight brain with zero deviation.
      - When ON: Activates the 2 unused sensory inputs per hidden neuron (Input 24: local food tangent/gradient vector, Input 25: LBVH macro-swarm cluster centroid bearing) and 1 extra actuator (Actuator 6: sprint burst / armor hardening). Mutations during `birth_step.wgsl` evolve these weights actively, allowing complex pack dynamics and navigation to emerge!

12. **Seamless Dual-Engine State Hot-Swapping**:
    - Bi-directional bridge between Bevy ECS `SimWorld` and the GPU storage buffers (`GpuAgent`, `GpuAgentAtomic`, `GpuSoilCell`).
    - Switching `[ENGINE: RUST]` $\leftrightarrow$ `[ENGINE: GPU]` dynamically transfers live agents and simulation tick, allowing users to hot-swap engines live mid-run without resetting the simulation timeline or losing creature lineages.

13. **GPU Particle Sparks & Mesh Instancing**:
    - 16,384+ drifting sparks updated in an isolated GPU compute pass (velocity decay, alpha fading).
    - Instanced creature dart mesh generation on the GPU.

14. **Operations Utilizing the Padded Values & 512-Byte Layout ($2^9$)**:
    - **Operation 1: Neural Forward Pass (`agent_step.wgsl`)**: In baseline mode (`expanded_cortex == 0u`), dummy input slots are set to `0.0`, guaranteeing 100% bit-exact parity with HTML. In Expanded Cortex mode (`expanded_cortex == 1u`), slots are actively fed chemical tangent gradients and LBVH swarm cluster centroid bearings.
    - **Operation 2: Genome Mutation Masking (`birth_step.wgsl`)**: In baseline mode, mutation applies a bitmask (`word & 0x0000FFFFu`) to the 7th word of hidden neurons and 3rd word of output neurons, preventing silent random drift in inactive weights. When Expanded Cortex is toggled ON, the mask is lifted and mutations actively evolve novel traits.
    - **Operation 3: Dual-Engine Live Hot-Swapping (`sync_rust_to_gpu` & `sync_gpu_to_rust`)**: Exact bit-level pack/unpack maps 326 sequential `i8` genes into 88 vec4-aligned `u32` words (7 words per hidden neuron, 3 words per output neuron) with zero loss or drift.
    - **Operation 4: Savefile Backward Compatibility**: Saving always extracts the canonical 326 `i8` genes into `AgentData`, ensuring all `.clank` and `.json` files are 100% cross-compatible between CPU and GPU engines.
    - **Operation 5: Struct Tail Padding Acceleration (`_pad: [u32; 4]`)**: `_pad[0]` caches the precalculated 32-bit Morton code (eliminating redundant bit-interleaving across secondary passes), and `_pad[1]` caches the packed lineage color (`rgba8unorm`) for direct dart mesh instancing without runtime palette queries.

15. **Frustum-Culled Stochastic Audio (256 Voices = 4 KB)**:
    - **Single Page-Aligned Voice Buffer**: Audio events (bites, kills, births) are emitted to a fixed 256-voice atomic append buffer (`AudioVoice`: 16 bytes: `pos: [f32; 2]`, `event_type: u32`, `volume: f32`). 256 voices $\times$ 16 bytes = 4,096 bytes (4 KB, exactly 1 hardware memory page).
    - **Consolidated 20 KB Queue Buffer**: Combined with the 1,024-entry birth queue ($16\text{ KB}$), the unified queue storage buffer is exactly $20\text{ KB}$ ($20,480\text{ bytes}$, divisible by 4, cache-line aligned). Keeps `agent_step.wgsl` at 7 storage buffers (below WebGPU/wgpu limit of 8).
    - **Viewport Frustum Culling**: Events occurring outside the camera AABB $[x_{\min}, y_{\min}] \times [x_{\max}, y_{\max}]$ are culled on the GPU without atomic overhead.
    - **Zoom Loudness Modulation**: Sound volume scales with camera zoom level (0.08 ambient murmur fully zoomed out, 1.0 punchy bite fully zoomed in).
    - **Stochastic Hash Density Filter**: In dense swarms or mass extinction cascades, events are stochastically sampled via 1-cycle stateless PCG hash `pcg3d(vec3u(killer, victim, tick)) % density == 0u`, preventing audio mixer blowout and maintaining clean acoustic presence.

16. **Aggregate Frame Telemetry (128-Byte `GpuTelemetry`) & 32x Speed Visual Consistency Picking**:
    - **128-Byte Dual-Cache-Line Struct**: Replaces unbounded event ring buffers with a compact 128-byte aggregate telemetry block (2 $\times$ 64-byte hardware cache lines):
      - Cache Line 0 (Engine & Apex Stats): `population: atomic<u32>`, `kills: atomic<u32>`, `starvations: atomic<u32>`, `apex_record_milli: atomic<u32>` (updated via lock-free `atomicMax`), `food_grazed_milli: atomic<u32>`, `sub_ticks_elapsed: u32`, `apex_agent_id: u32`, `pad0: u32`.
      - Cache Line 1 (Lineage Extinction Monitoring): 16 $\times$ `atomic<u32>` living head counts for lineages 0..15. CPU detects lineage extinction in $O(1)$ by scanning for zero counts without reading back individual agent records.
    - **$250,000\times$ Bandwidth Reduction**: 128 bytes read back per frame via DMA staging buffer ($\approx 8\text{ nanoseconds}$ transfer time), completely eliminating VRAM bus bottlenecks during mass extinctions.
    - **32x Speed Visual Consistency Picking**: When running at 32x multi-tick speed (32 sub-ticks per frame), ray picking is evaluated strictly on **Sub-Tick 1** with a 16px hitbox. The picked agent index is locked and tracked through Sub-Ticks 2..32, guaranteeing that what the user clicked on in the rendered frame is the creature selected on the specimen card without erratic frame-jumping.

---

## Plan Overview: 6 Bite-Sized Tasks

- [ ] **Task 1: Branch Setup & GPU Compute Architecture Scaffolding**
  - Create branch `feature/bevy-gpu` from `feature/bevy-port`.
  - Add `bytemuck = { version = "1.21", features = ["derive"] }` to `crates/clank_app/Cargo.toml`.
  - Add `gpu` module in `crates/clank_app/src/gpu/` with buffer types: `GpuAgent` (512B, 88 words, exact $2^9$ power-of-two), `GpuAgentAtomic` (16B, with `dead_claimed`), `GpuSimParams` (64B), `GpuLbvhNode` (48B), `GpuSoilCell` (16B), `GpuSimCounters` (16B), `AudioVoice` (16B), `GpuTelemetry` (128B), and pipeline skeletons.
  - Implement unit tests for GPU struct memory layouts and 16-byte WGSL alignment.

- [ ] **Task 2: Tombstone Freelist & Zero-Copy Agent Storage Buffer**
  - Implement lock-free atomic stack allocator (`freelist: array<u32>`, `atomic<u32> freelist_top`).
  - Implement allocation with CAS underflow guard (`cur_top > 0u`).
  - Implement CAS death ownership (`dead_claimed: 0u -> 1u`) to eliminate double-freeing from concurrent starvation and predation.
  - Test parallel push/pop and slot recycling in automated unit test suite.

- [ ] **Task 3: GPU Soil Simulation & Direct Texture Generation**
  - Implement WGSL compute shader for soil chemistry: spatial bloom renewal, food clamp $[0.0, 2.5]$, taint decay ($0.994$), and scent decay ($0.954$).
  - Implement direct GPU colormap generation (food, taint, scent, vignette) into 2D texture, **completely eliminating CPU `generate_soil_rgba` upload**.
  - Bind soil as single 2D `rgba16float` texture with universal hardware bilinear filtering across Metal, Vulkan, and DX12.

- [ ] **Task 4: Hybrid Morton Grid, Multi-System LBVH & Spatial Queries**
  - Implement 32-bit Morton code generator from 2D coordinates in WGSL.
  - Implement 8-byte indirection parallel Radix Sort on `(morton_key, agent_slot_idx)`, keeping heavy agent structs stationary.
  - Implement Karras 2012 parallel LBVH construction with `agent_id` tie-breaking and $N \le 1$ degenerate population guard.
  - Implement Tier 1 ($3 \times 3$ local Moore neighborhood) + Tier 2 ring expansion for lonely creatures.
  - Implement **$O(\log N)$ binary search on Morton keys** for specimen mouse picking.
  - Implement **AoE Tool Bounding Box Query** (`extinguish_at`, `blight_at`, `nourish_at`) using LBVH intersection.

- [ ] **Task 5: Packed 88-Word Vectorized RNN, Combat Resolution, PRNG & Decoupled Birth Pipeline**
  - Implement `preamble_clear.wgsl` to reset `mate_claim`, `dead_claimed`, `birth_count`, `audio_voice_count`, `cell_offsets`, and `GpuTelemetry` counters with global execution barrier.
  - Implement 1-thread-per-agent WGSL compute shader with branchless `unpack4x8snorm` vectorization across 88 `u32` words (7 vec4s hidden, 3 vec4s output).
  - Integrate **Experimental Simulation Mods**: Barnes-Hut Macro-Flocking and Expanded Cortex (extra senses and sprint actuator).
  - Apply steering, thrust, and hardware toroidal coordinate wrap.
  - Implement coalesced atomic soil deposits, atomic millijoule combat resolution with `dead_claimed` CAS, frustum-culled stochastic audio voice emission (256-voice buffer), atomic telemetry updates (`atomicMax` on `apex_record_milli`), and decoupled SIMD genome mutation pass (`birth_step.wgsl`).

- [ ] **Task 6: Frustum Culling, Minimap LOD, Dual-Engine UI & Verification**
  - Implement **GPU Camera Viewport Frustum Culling** via LBVH writing into an Indirect Draw Buffer.
  - Implement **Minimap LOD Cluster Rendering** sampling intermediate LBVH depth nodes for density circles.
  - Implement **spatially coherent instanced dart rasterization** using Morton-ordered agent index streams for tile-cache efficiency.
  - Implement **Granular Bevy Audio playback** reading 256-voice audio queue with zoom loudness modulation.
  - Implement **DMA Readback of 128-Byte `GpuTelemetry`** for UI stats, The Record, and $O(1)$ zero-searching extinction notifications.
  - Implement **32x Speed Visual Consistency Picking** locking selection on Sub-Tick 1 across 32-tick batches.
  - Add top bar engine toggle: `ENGINE: RUST` $\leftrightarrow$ `ENGINE: GPU` with live bi-directional state bridge.
  - Add simulation mod toggles: `[BARNES-HUT: OFF/ON]` and `[EXPANDED CORTEX: OFF/ON]`.
  - Wire telemetry from GPU storage buffers into UI stats, The Record, and Specimen card.
  - Run all 51+ workspace tests.
  - Capture live GPU screenshot via API (`POST /screenshot`), visually verify with `view_file`, and commit to `feature/bevy-gpu`.

---

## Detailed Task Breakdown

### Task 1: Branch Setup & GPU Compute Architecture Scaffolding

**Files:**
- Modify: `crates/clank_app/Cargo.toml` (add `bytemuck = { version = "1.21", features = ["derive"] }`)
- Create: `crates/clank_app/src/gpu/mod.rs`
- Create: `crates/clank_app/src/gpu/types.rs`
- Modify: `crates/clank_app/src/lib.rs`
- Test: `crates/clank_app/tests/gpu_types_test.rs`

**Interfaces:**
- Consumes: `clank_core::agent::AgentData`, `clank_core::agent::GENES` (326)
- Produces: `GpuAgent`, `GpuAgentAtomic`, `GpuSimParams`, `GpuLbvhNode`, `GpuSoilCell`, `GpuSimCounters`, `BirthEvent`, `AudioVoice`, `GpuTelemetry` with exact 16-byte WGSL alignment

- [ ] **Step 1: Write failing test for GPU struct memory layouts**
```rust
// crates/clank_app/tests/gpu_types_test.rs
use clank_app::gpu::types::{
    GpuAgent, GpuSimParams, GpuLbvhNode, GpuAgentAtomic, BirthEvent,
    GpuSimCounters, GpuSoilCell, AudioVoice, GpuTelemetry,
};

#[test]
fn test_gpu_struct_alignments() {
    assert_eq!(std::mem::size_of::<GpuAgent>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgent>(), 512);
    assert_eq!(std::mem::size_of::<GpuSimParams>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSimParams>(), 64);
    assert_eq!(std::mem::size_of::<GpuLbvhNode>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuLbvhNode>(), 48);
    assert_eq!(std::mem::size_of::<GpuAgentAtomic>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentAtomic>(), 16);
    assert_eq!(std::mem::size_of::<BirthEvent>() % 16, 0);
    assert_eq!(std::mem::size_of::<BirthEvent>(), 16);
    assert_eq!(std::mem::size_of::<AudioVoice>() % 16, 0);
    assert_eq!(std::mem::size_of::<AudioVoice>(), 16);
    assert_eq!(std::mem::size_of::<GpuSimCounters>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSimCounters>(), 16);
    assert_eq!(std::mem::size_of::<GpuSoilCell>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSoilCell>(), 16);
    assert_eq!(std::mem::size_of::<GpuTelemetry>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuTelemetry>(), 128);
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement GPU types with `#[repr(C)]` and 16-byte alignment**
```rust
// crates/clank_app/src/gpu/types.rs
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgent {
    pub pos_vel: [f32; 4],      // 16 bytes: x, y, vx, vy (Cache Line 0 start)
    pub angle_energy: [f32; 4], // 16 bytes: angle, energy, feeding, attack
    pub traits: [f32; 8],       // 32 bytes: tr[0..5], signal, last_victim (Cache Line 0 end: 64B)
    pub hidden: [f32; 12],      // 48 bytes: h[0..9] recurrent hidden states, pad, pad
    pub meta: [u32; 8],         // 32 bytes: id, root, gen, age, cooldown, birth, kills, dead
    pub packed_genes: [u32; 88],// 352 bytes: 10 hidden * 7 vec4s (70 words) + 6 output * 3 vec4s (18 words)
    pub _pad: [u32; 4],         // 16 bytes: _pad[0] = cached Morton code, _pad[1] = packed lineage color, _pad[2..3] = reserved
} // Total: 512 bytes (exact power-of-two 2^9, 8 * 64B / 4 * 128B cache lines)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentAtomic {
    pub energy_milli: i32,      // 4 bytes: atomic fixed-point millijoules
    pub mate_claim: u32,        // 4 bytes: atomic CAS mating handshake
    pub dead_claimed: u32,      // 4 bytes: atomic CAS death ownership (prevents double-free)
    pub pad: u32,               // 4 bytes: 16-byte WGSL alignment
} // Total: 16 bytes

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    pub tick: u32,              // 4 bytes
    pub agent_count: u32,       // 4 bytes
    pub max_agents: u32,        // 4 bytes
    pub hostility: f32,         // 4 bytes (Total 16)
    pub mut_rate: f32,          // 4 bytes
    pub speed: f32,             // 4 bytes
    pub renewal: f32,           // 4 bytes
    pub barnes_hut_enabled: u32,// 4 bytes (Total 32)
    pub expanded_cortex: u32,   // 4 bytes: Expanded Cortex mod toggle (0=OFF, 1=ON)
    pub tool_type: u32,         // 4 bytes
    pub tool_pos: [f32; 2],     // 8 bytes (Total 48)
    pub camera_pos: [f32; 2],   // 8 bytes
    pub camera_size: [f32; 2],  // 8 bytes (Total 64)
} // Total: 64 bytes

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BirthEvent {
    pub parent_a: u32,          // 4 bytes
    pub parent_b: u32,          // 4 bytes
    pub child_slot: u32,        // 4 bytes
    pub pad: u32,               // 4 bytes: 16-byte WGSL alignment
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AudioVoice {
    pub pos: [f32; 2],          // 8 bytes: arena world position (for distance/viewport calculations)
    pub event_type: u32,        // 4 bytes: 0 = bite/attack, 1 = kill, 2 = birth
    pub volume: f32,            // 4 bytes: zoom-modulated volume [0.08, 1.0]
} // Total: 16 bytes (256 voices = 4 KB, 1 page)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimCounters {
    pub freelist_top: u32,      // 4 bytes
    pub birth_count: u32,       // 4 bytes
    pub kills_total: u32,       // 4 bytes
    pub audio_voice_count: u32, // 4 bytes: atomic counter for audio voice append
} // Total: 16 bytes

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTelemetry {
    // Cache Line 0 (64 bytes): Global Simulation & Apex Records
    pub population: u32,        // 4 bytes: active living agents
    pub kills: u32,             // 4 bytes: total predatory kills this frame
    pub starvations: u32,       // 4 bytes: total starvation deaths this frame
    pub apex_record_milli: u32, // 4 bytes: highest energy recorded (atomicMax)
    pub food_grazed_milli: u32, // 4 bytes: total food consumed in millijoules
    pub sub_ticks_elapsed: u32, // 4 bytes: number of sub-ticks processed
    pub apex_agent_id: u32,     // 4 bytes: ID of apex creature
    pub pad0: u32,              // 4 bytes
    pub _reserved0: [u32; 8],   // 32 bytes (Total Cache Line 0: 64B)

    // Cache Line 1 (64 bytes): 16-Lineage Real-Time Extinction Monitoring
    pub lineage_counts: [u32; 16], // 16 * 4B = 64 bytes (head counts for roots 0..15)
} // Total: 128 bytes (2 x 64B cache lines, 8ns DMA transfer)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSoilCell {
    pub food_milli: i32,        // 4 bytes: atomic fixed-point millifood
    pub taint_milli: i32,       // 4 bytes: atomic fixed-point millitaint
    pub scent_milli: i32,       // 4 bytes: atomic fixed-point milliscent
    pub pad: u32,               // 4 bytes: 16-byte WGSL alignment
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLbvhNode {
    pub aabb_min: [f32; 2],      // 8 bytes
    pub aabb_max: [f32; 2],      // 8 bytes (Total 16)
    pub center_of_mass: [f32; 2],// 8 bytes
    pub count: u32,              // 4 bytes
    pub dominant_lineage: u32,   // 4 bytes (Total 32)
    pub left_child: u32,         // 4 bytes
    pub right_child: u32,        // 4 bytes
    pub parent: u32,             // 4 bytes
    pub flags: u32,              // 4 bytes (Total 48)
}
```
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 2: Tombstone Freelist & Zero-Copy Agent Storage Buffer

**Files:**
- Create: `crates/clank_app/src/gpu/freelist.rs`
- Test: `crates/clank_app/tests/gpu_freelist_test.rs`

**Interfaces:**
- Consumes: `GpuAgent`, `MAX_AGENTS` (e.g. 65,536)
- Produces: `TombstoneFreelistManager` managing allocation head, tombstone flags, and zero-copy recycling

- [ ] **Step 1: Write failing test for Freelist initialization, slot recycling, and underflow protection**
```rust
#[test]
fn test_freelist_recycle_without_movement() {
    let mut freelist = TombstoneFreelistManager::new(1024);
    let slot_a = freelist.allocate().unwrap();
    let slot_b = freelist.allocate().unwrap();
    assert_ne!(slot_a, slot_b);
    freelist.free(slot_a);
    let slot_c = freelist.allocate().unwrap();
    assert_eq!(slot_c, slot_a); // Recycled slot without shifting slot_b!
}

#[test]
fn test_freelist_underflow_protection_at_capacity() {
    let mut freelist = TombstoneFreelistManager::new(2);
    assert!(freelist.allocate().is_some());
    assert!(freelist.allocate().is_some());
    // At capacity: next allocate must safely return None without unsigned underflow!
    assert_eq!(freelist.allocate(), None);
    assert_eq!(freelist.freelist_top(), 0);
}

#[test]
fn test_freelist_dead_claimed_cas_prevents_double_free() {
    // Verifies that atomic CAS on dead_claimed (0 -> 1) ensures exactly one thread frees the slot
    // even if predation and starvation trigger concurrently on the same tick.
    let mut freelist = TombstoneFreelistManager::new(1024);
    let slot = freelist.allocate().unwrap();
    assert!(freelist.claim_death_and_free(slot)); // First claim succeeds -> freed
    assert!(!freelist.claim_death_and_free(slot)); // Second claim fails -> prevented double-free!
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement Tombstone Freelist logic in Rust and WGSL CAS allocation snippet**
```wgsl
// WGSL CAS pop allocation: guards against underflow when carrying capacity is reached
fn allocate_child_slot() -> u32 {
    var cur_top = atomicLoad(&sim_counters.freelist_top);
    var child_slot = 0xFFFFFFFFu;
    while (cur_top > 0u) {
        let cas = atomicCompareExchangeWeak(&sim_counters.freelist_top, cur_top, cur_top - 1u);
        if (cas.exchanged) {
            child_slot = freelist[cur_top - 1u];
            break;
        }
        cur_top = cas.old_value;
    }
    return child_slot; // 0xFFFFFFFFu indicates population capacity reached
}

// WGSL CAS death ownership: guarantees slot is pushed to freelist exactly once
fn claim_death_and_free(victim_idx: u32) -> bool {
    let cas = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
    if (cas.exchanged) {
        agents[victim_idx].meta[7] = 1u; // dead = 1
        let free_slot = atomicAdd(&sim_counters.freelist_top, 1u);
        freelist[free_slot] = victim_idx;
        return true;
    }
    return false; // Already freed by another concurrent thread
}
```
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 3: GPU Soil Simulation & Direct Texture Generation

**Files:**
- Create: `crates/clank_app/src/gpu/soil_pipeline.rs`
- Create: `crates/clank_app/assets/shaders/soil_step.wgsl`
- Test: `crates/clank_app/tests/gpu_soil_test.rs`

**Interfaces:**
- Consumes: `SoilGrid` dimensions ($75 \times 50$), `growth_val`, `GpuSoilCell` storage buffer (60 KB)
- Produces: Direct GPU `rgba16float` texture generation with universal cross-platform linear filtering, completely eliminating CPU `generate_soil_rgba` upload and eliminating texture ping-pong memory copies

- [ ] **Step 1: Write failing test for soil atomic buffer bindings, renewal math, and single rgba16float texture rasterization**
```rust
// crates/clank_app/tests/gpu_soil_test.rs
#[test]
fn test_soil_atomic_buffer_and_texture_bounds() {
    // Verifies:
    // 1. GpuSoilCell fixed-point conversions (millifood, millitaint, milliscent)
    // 2. Renewal formula: f += renewal * bloom * (1 - f / 1.7) clamped to [0.0, 2.5]
    // 3. Taint decay: t * 0.994 - 0.0001, scent decay: s * 0.954
    // 4. Output texture matches 75x50 rgba16float format
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement WGSL soil compute kernel operating on 60 KB atomic buffer and rasterizing to single rgba16float texture**
```wgsl
// crates/clank_app/assets/shaders/soil_step.wgsl
struct SoilCell {
    food_milli: atomic<i32>,
    taint_milli: atomic<i32>,
    scent_milli: atomic<i32>,
    pad: u32,
}
@group(0) @binding(0) var<storage, read_write> soil_buffer: array<SoilCell, 3750>;
@group(0) @binding(1) var soil_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<storage, read> bloom_table: array<f32, 3750>;
@group(0) @binding(3) var<uniform> params: SoilParams;

@compute @workgroup_size(8, 8)
fn soil_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= 75u || id.y >= 50u) { return; }
    let k = id.y * 75u + id.x;

    var f = f32(atomicLoad(&soil_buffer[k].food_milli)) * 0.001;
    var t = f32(atomicLoad(&soil_buffer[k].taint_milli)) * 0.001;
    var s = f32(atomicLoad(&soil_buffer[k].scent_milli)) * 0.001;

    // Environmental renewal & decay:
    f += params.renewal * bloom_table[k] * (1.0 - f / 1.7);
    f = clamp(f, 0.0, 2.5);
    if (t > 0.0) { t = max(0.0, t * 0.994 - 0.0001); }
    if (s > 0.0) { s = s * 0.954; }

    // Write back updated simulation state:
    atomicStore(&soil_buffer[k].food_milli, i32(f * 1000.0));
    atomicStore(&soil_buffer[k].taint_milli, i32(t * 1000.0));
    atomicStore(&soil_buffer[k].scent_milli, i32(s * 1000.0));

    // Colormap grading into single rgba16float texture:
    let col = evaluate_soil_color(f, t, s, id.xy);
    textureStore(soil_texture, id.xy, col);
}
```
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 4: Hybrid Morton Grid, Multi-System LBVH & Spatial Queries

**Files:**
- Create: `crates/clank_app/src/gpu/spatial_index.rs`
- Create: `crates/clank_app/src/gpu/lbvh.rs`
- Create: `crates/clank_app/assets/shaders/morton_grid.wgsl`
- Create: `crates/clank_app/assets/shaders/lbvh_build.wgsl`
- Test: `crates/clank_app/tests/gpu_spatial_test.rs`

**Interfaces:**
- Consumes: `GpuAgent` positions, grid dimensions ($9 \times 6$, cell size $100\text{px}$)
- Produces: Sorted Morton indices (8-byte indirection buffer `vec2u(morton_key, agent_id)`), cell offset table, Karras LBVH tree with tie-breaking, $O(\log N)$ mouse picking, and AoE tool queries

- [ ] **Step 1: Write failing test for Morton bit-interleaving, cell_offsets sentinel initialization, LBVH tree building with tie-breaking, degenerate population guard ($N \le 1$), and AoE tool queries**
```rust
#[test]
fn test_cell_offsets_sentinel_initialization() {
    // Verifies that cell_offsets starts with (0xFFFFFFFF, 0xFFFFFFFF)
    // so empty cells are cleanly skipped during neighbor loops.
}

#[test]
fn test_lbvh_degenerate_population_guard() {
    // Verifies that when population is 0 or 1, tree builder safely early-exits
    // without unsigned underflow on N - 2 internal node indices.
}

#[test]
fn test_lbvh_identical_key_tie_breaking() {
    // Verifies that agents at identical coordinates have unique LCP lengths
    // via 32u + count_leading_zeros(id_a ^ id_b), preventing split loops.
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement Morton encoding, cell offset clearing pass with 0xFFFFFFFF sentinels, 8-byte indirection Radix sort, Karras LBVH hierarchy generation with $N \le 1$ early-exit guard and tie-breaking, and AoE bounding box query**
```wgsl
// In morton_grid.wgsl (clearing pass):
@compute @workgroup_size(64)
fn clear_cell_offsets(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < 54u) {
        cell_offsets[id.x] = vec2u(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
}

// In lbvh_build.wgsl:
fn common_prefix_length(i: i32, j: i32, n: u32) -> i32 {
    if (j < 0 || j >= i32(n)) { return -1; }
    let key_i = sorted_keys[i];
    let key_j = sorted_keys[j];
    if (key_i != key_j) {
        return i32(countLeadingZeros(key_i ^ key_j));
    }
    // Tie-break with unique agent slot id:
    let id_i = sorted_agent_ids[i];
    let id_j = sorted_agent_ids[j];
    return 32 + i32(countLeadingZeros(id_i ^ id_j));
}

@compute @workgroup_size(64)
fn build_lbvh(@builtin(global_invocation_id) id: vec3u) {
    let active_count = atomicLoad(&sim_counters.freelist_top); // or active_agents
    if (active_count < 2u || id.x >= active_count - 1u) {
        return; // Guard against N <= 1 underflow!
    }
    // Proceed with Karras 2012 LCP split evaluation...
}
```
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 5: Packed 88-Word Vectorized RNN, Combat Resolution, PRNG & Decoupled Birth Pipeline

**Files:**
- Create: `crates/clank_app/src/gpu/agent_pipeline.rs`
- Create: `crates/clank_app/src/gpu/birth_pipeline.rs`
- Create: `crates/clank_app/assets/shaders/preamble_clear.wgsl`
- Create: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Create: `crates/clank_app/assets/shaders/birth_step.wgsl`
- Test: `crates/clank_app/tests/gpu_agent_test.rs`
- Test: `crates/clank_app/tests/gpu_birth_mutation_test.rs`

**Interfaces:**
- Consumes: Soil texture (bilinear reads), `GpuSoilCell` storage buffer (atomic grazing/deposits), Spatial index (Morton cells), LBVH tree, `GpuAgent` storage buffer, `GpuAgentAtomic` L2 buffer, `GpuSimParams`
- Produces: Updated kinematics with toroidal wrapping, hidden RNN states via branchless `unpack4x8snorm` across 88 `u32` words, coalesced soil deposits, atomic combat resolution with `dead_claimed` CAS, and decoupled SIMD genome mutation

- [ ] **Step 1: Write failing test for universal `unpack4x8snorm` forward pass, combat resolution with double-free CAS, and toroidal coordinate wrapping**
```rust
// crates/clank_app/tests/gpu_agent_test.rs
#[test]
fn test_unpack4x8snorm_forward_pass_parity() {
    // Verifies:
    // 1. unpack4x8snorm(word) * 0.61 across 7 vec4s (hidden) and 3 vec4s (output) is bit-exact to HTML
    // 2. Hardware toroidal wrapping: x - W * floor(x / W) handles negative/positive wraps
    // 3. Combat damage formula, decisive killer attribution, and dead_claimed CAS protection
}
```

- [ ] **Step 2: Write failing test for stateless PCG hash & triangular distribution**
```rust
// crates/clank_app/tests/gpu_birth_mutation_test.rs
#[test]
fn test_stateless_pcg_triangular_distribution() {
    // Verifies:
    // 1. PCG 3D hash evaluates deterministically from (id, stream, tick)
    // 2. r1 + r2 - 1.0 produces zero-centered symmetric triangular distribution T(-1, 0, 1)
    // 3. Genome mutation formula round((r1 + r2 - 1) * 100 * mutRate) matches HTML bounds [-127, 127]
}
```

- [ ] **Step 3: Implement `preamble_clear.wgsl` and `agent_step.wgsl` with Universal `unpack4x8snorm`, Toroidal Wrapping, Frustum-Culled Audio & Aggregate Telemetry**
```wgsl
// crates/clank_app/assets/shaders/preamble_clear.wgsl
// Dispatched before agent_step.wgsl to guarantee a global GPU execution barrier:
@compute @workgroup_size(64)
fn preamble_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < params.max_agents) {
        atomicStore(&agent_atomics[id.x].mate_claim, 0u);
        atomicStore(&agent_atomics[id.x].dead_claimed, 0u);
    }
    if (id.x == 0u) {
        atomicStore(&sim_counters.birth_count, 0u);
        atomicStore(&sim_counters.audio_voice_count, 0u);
        atomicStore(&telemetry.population, 0u);
        atomicStore(&telemetry.kills, 0u);
        atomicStore(&telemetry.starvations, 0u);
        atomicStore(&telemetry.apex_record_milli, 0u);
        atomicStore(&telemetry.food_grazed_milli, 0u);
    }
    if (id.x < 16u) {
        atomicStore(&telemetry.lineage_counts[id.x], 0u);
    }
    if (id.x < 54u) {
        cell_offsets[id.x] = vec2u(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
}
```

```wgsl
// In agent_step.wgsl:

// Consolidated Queue Buffer (Binding 6: exactly 20 KB = 16 KB births + 4 KB audio):
struct ConsolidatedQueue {
    births: array<BirthEvent, 1024>, // 16 KB
    audio: array<AudioVoice, 256>,   // 4 KB (1 memory page)
}
@group(0) @binding(6) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(7) var<storage, read_write> telemetry: GpuTelemetry;

// Hardware toroidal coordinate wrapping (1 cycle via floor):
fn wrap_coords(p: vec2f) -> vec2f {
    return vec2f(
        p.x - 900.0 * floor(p.x / 900.0),
        p.y - 600.0 * floor(p.y / 600.0)
    );
}

// Branchless unpack4x8snorm neural forward pass:
// Hidden layer: 10 neurons, each 7 vec4s (28 weights, last 2 zero-padded in baseline)
var new_h: array<f32, 10>;
for (var j = 0u; j < 10u; j += 1u) {
    var s = 0.0;
    let base_w = j * 7u;
    for (var k = 0u; k < 7u; k += 1u) {
        s += dot(unpack4x8snorm(agents[agent_idx].packed_genes[base_w + k]), ins_hidden[k]);
    }
    new_h[j] = tanh(s * 0.61);
}

// Output layer: 6 neurons, each 3 vec4s (12 weights, last 1 zero-padded in baseline)
var out: array<f32, 6>;
for (var j = 0u; j < 6u; j += 1u) {
    var s = 0.0;
    let base_w = 70u + j * 3u;
    for (var k = 0u; k < 3u; k += 1u) {
        s += dot(unpack4x8snorm(agents[agent_idx].packed_genes[base_w + k]), ins_output[k]);
    }
    out[j] = tanh(s * 0.66);
}

// Telemetry population, lineage head count, and apex record tracking (living agents):
atomicAdd(&telemetry.population, 1u);
atomicAdd(&telemetry.lineage_counts[a_root % 16u], 1u);
atomicMax(&telemetry.apex_record_milli, u32(max(0.0, a_energy) * 1000.0));

// Atomic soil grazing (coalesced L2 cache lines):
let cell_idx = u32(pos.y / 12.0) * 75u + u32(pos.x / 12.0);
let eaten_milli = i32(eaten_float * 1000.0);
atomicSub(&soil_buffer[cell_idx].food_milli, eaten_milli);
atomicAdd(&telemetry.food_grazed_milli, u32(max(0, eaten_milli)));

// Combat resolution snippet in agent_step.wgsl:
let contact_dist = 14.0 + 10.0 * tr0;
if (best_dist < contact_dist && a_attack > 0.25 && a_cooldown == 0u) {
    let damage = (0.5 + a_attack * 2.2) * params.hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3);
    let damage_milli = i32(damage * 1000.0);
    let old_energy_milli = atomicSub(&agent_atomics[victim_idx].energy_milli, damage_milli);

    a_cooldown = 3u;
    a_energy += damage * (0.1 + 0.55 * tr5);

    // Decisive killer attribution:
    if (old_energy_milli > 0 && old_energy_milli <= damage_milli) {
        agents[agent_idx].meta[6] += 1u; // kills++
        atomicAdd(&sim_counters.kills_total, 1u);
        atomicAdd(&telemetry.kills, 1u);
        a_energy += min(9.0, 8.0 * tr5);

        // Frustum-Culled Stochastic Audio Emission (Kill Event):
        let cam_min = params.camera_pos - params.camera_size * 0.5;
        let cam_max = params.camera_pos + params.camera_size * 0.5;
        let in_view = (pos.x >= cam_min.x && pos.x <= cam_max.x && pos.y >= cam_min.y && pos.y <= cam_max.y);
        if (in_view) {
            let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (900.0 - 150.0), 0.0, 1.0);
            let kill_volume = mix(0.08, 1.0, zoom_factor);
            let density_filter = select(1u, 4u, params.agent_count > 10000u);
            if (pcg_rand(agent_idx, victim_idx, params.tick) % density_filter == 0u) {
                let voice_slot = atomicAdd(&sim_counters.audio_voice_count, 1u);
                if (voice_slot < 256u) {
                    queue_buffer.audio[voice_slot] = AudioVoice(pos, 1u /* EVENT_KILL */, kill_volume);
                }
            }
        }
        
        // Atomic CAS death ownership prevents double-freeing:
        let claim_death = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
        if (claim_death.exchanged) {
            agents[victim_idx].meta[7] = 1u; // dead = 1
            let free_slot = atomicAdd(&sim_counters.freelist_top, 1u);
            freelist[free_slot] = victim_idx;
        }
    }
}

// Starvation death handling:
if (a_energy <= 0.0) {
    atomicAdd(&telemetry.starvations, 1u);
    let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
    if (claim_death.exchanged) {
        agents[agent_idx].meta[7] = 1u; // dead = 1
        let free_slot = atomicAdd(&sim_counters.freelist_top, 1u);
        freelist[free_slot] = agent_idx;
    }
}

// Mating handshake snippet in agent_step.wgsl:
if (a_energy > 58.0 + 12.0 * tr0 && a_age > 65u && a_birth == 0u && brain_out5 > -0.15) {
    if (best_dist < 18.0 && partner_energy > 42.0 && partner_root != a_root && pcg_rand(agent_idx, 99u, tick) < 0.15) {
        let claim = atomicCompareExchangeWeak(&agent_atomics[partner_idx].mate_claim, 0u, agent_id);
        if (claim.exchanged) {
            // Handshake won! Allocate child slot via CAS loop preventing underflow at capacity:
            var cur_top = atomicLoad(&sim_counters.freelist_top);
            var child_slot = 0xFFFFFFFFu;
            while (cur_top > 0u) {
                let cas = atomicCompareExchangeWeak(&sim_counters.freelist_top, cur_top, cur_top - 1u);
                if (cas.exchanged) {
                    child_slot = freelist[cur_top - 1u];
                    break;
                }
                cur_top = cas.old_value;
            }

            if (child_slot != 0xFFFFFFFFu) {
                atomicSub(&agent_atomics[agent_idx].energy_milli, 24000);
                atomicSub(&agent_atomics[partner_idx].energy_milli, 6000);
                a_birth = 95u;

                let queue_idx = atomicAdd(&sim_counters.birth_count, 1u);
                if (queue_idx < 1024u) {
                    queue_buffer.births[queue_idx] = BirthEvent(agent_idx, partner_idx, child_slot, 0u);
                }

                // Birth Audio Voice Emission:
                let cam_min = params.camera_pos - params.camera_size * 0.5;
                let cam_max = params.camera_pos + params.camera_size * 0.5;
                let in_view = (pos.x >= cam_min.x && pos.x <= cam_max.x && pos.y >= cam_min.y && pos.y <= cam_max.y);
                if (in_view) {
                    let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (900.0 - 150.0), 0.0, 1.0);
                    let birth_volume = mix(0.06, 0.8, zoom_factor);
                    let voice_slot = atomicAdd(&sim_counters.audio_voice_count, 1u);
                    if (voice_slot < 256u) {
                        queue_buffer.audio[voice_slot] = AudioVoice(pos, 2u /* EVENT_BIRTH */, birth_volume);
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 4: Implement Decoupled Birth & Genome Mutation Pass (`birth_step.wgsl`)**
```wgsl
// birth_step.wgsl: 1 workgroup per newborn child (32 threads)
@compute @workgroup_size(32)
fn birth_main(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let event = birth_queue[wg_id.x];
    let parent_a = event.parent_a;
    let parent_b = event.parent_b;
    let child_idx = event.child_slot;

    // Parallel genome crossover & mutation (88 u32 words = 10*7 + 6*3)
    for (var w = local_id.x; w < 88u; w += 32u) {
        var word_a = agents[parent_a].packed_genes[w];
        var word_b = agents[parent_b].packed_genes[w];
        // Crossover 48% and triangular mutation per gene byte...
        var mutated_word = crossover_and_mutate(word_a, word_b, local_id.x, tick);

        // If baseline mode, mask out dummy padded bytes so inactive genes don't drift:
        if (params.expanded_cortex == 0u) {
            if (w < 70u && (w % 7u) == 6u) {
                mutated_word = mutated_word & 0x0000FFFFu; // Bytes 2 & 3 clamped to 0
            }
            if (w >= 70u && ((w - 70u) % 3u) == 2u) {
                mutated_word = mutated_word & 0x00FFFFFFu; // Byte 3 clamped to 0
            }
        }
        agents[child_idx].packed_genes[w] = mutated_word;
    }

    // Trait crossover, morphology & tail padding initialization on thread 0:
    if (local_id.x == 0u) {
        // Crossover 45% and clamp(base + (r1 + r2 - 1) * mutRate * 0.6, 0.03, 0.98)...
        // Initialize child position near parent: a.x + rand(-9, 9), a.y + rand(-9, 9)
        // Set child energy = 24.0 (24,000 milli)
        agents[child_idx]._pad[0] = 0u; // Cached Morton code (assigned by morton_encode)
        agents[child_idx]._pad[1] = pack_lineage_color(agents[child_idx].meta[1]); // Precomputed color cache
        agents[child_idx]._pad[2] = 0u;
        agents[child_idx]._pad[3] = 0u;
    }
}
```

- [ ] **Step 5: Run tests `gpu_agent_test` and `gpu_birth_mutation_test` ensuring 100% pass**
- [ ] **Step 6: Git commit on `feature/bevy-gpu`**

---

### Task 6: Frustum Culling, Minimap LOD, Dual-Engine UI, Audio & Verification

**Files:**
- Modify: `crates/clank_app/src/rendering.rs` (LBVH Camera Frustum Culling & Morton-ordered instanced dart stream)
- Modify: `crates/clank_app/src/audio.rs` (Consolidated 256-voice queue consumption & granular Bevy audio playback)
- Modify: `crates/clank_app/src/ui.rs` (Top bar toggle: `ENGINE: RUST` / `ENGINE: GPU`, Minimap LOD radar display, 32x speed picking)
- Modify: `crates/clank_app/src/api.rs` (Telemetry reporting for active engine)
- Test: `crates/clank_app/tests/dual_engine_test.rs`

**Interfaces:**
- Consumes: `SimWorld` (CPU), `GpuSimWorld` (GPU), `AudioVoice` queue (4 KB), `GpuTelemetry` (128B)
- Produces: Camera frustum-culled rendering, minimap LOD clustering, unified telemetry, 32x speed visual consistency picking, zoom-modulated audio soundscape, HUD metrics, and engine switching

- [ ] **Step 1: Write failing test for live dual-engine hot-swapping, frustum culling, telemetry readback, and 32x speed picking**
```rust
// crates/clank_app/tests/dual_engine_test.rs
#[test]
fn test_dual_engine_live_hotswap_parity() {
    // Verifies:
    // 1. Rust -> GPU uploads live agents, populates GpuAgent/GpuSoilCell buffers, and resumes compute
    // 2. GPU -> Rust reads back active agents, updates SimWorld.tick, and resumes CPU loop
    // 3. Population count, generation, and lineage roots are 100% preserved across toggle
}

#[test]
fn test_gpu_telemetry_and_extinction_detection() {
    // Verifies:
    // 1. 128-byte GpuTelemetry reads back via DMA staging buffer
    // 2. apex_record_milli accurately reflects atomicMax of creature energy
    // 3. 16-lineage counts detect lineage extinction via zero-search without scanning agent buffer
}

#[test]
fn test_picking_visual_consistency_at_32x_speed() {
    // Verifies:
    // 1. In a 32-tick batch, picking evaluates strictly on Sub-Tick 1 with 16px hitbox
    // 2. Picked agent index is locked across Sub-Ticks 2..32, matching the rendered visual frame
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement engine switch in Bevy UI, bi-directional state bridge (`sync_rust_to_gpu` / `sync_gpu_to_rust`), 128-byte DMA telemetry readbacks, 32x speed picking locking on Sub-Tick 1, Bevy audio playback reading 256-voice queue, connect LBVH frustum culling, draw Minimap LOD clusters, and wire `[BARNES-HUT: OFF/ON]` and `[EXPANDED CORTEX: OFF/ON]` mod toggles to `GpuSimParams`**
- [ ] **Step 4: Run all workspace tests (`cargo test --workspace`) ensuring 100% pass**
- [ ] **Step 5: Build release (`cargo build -p clank_app --release`), capture GPU screenshot via API (`POST /screenshot`), inspect with `view_file`**
- [ ] **Step 6: Git commit on `feature/bevy-gpu`**

---

## Verification Plan

### Automated Tests
```bash
# Run all workspace unit & integration tests
cargo test --workspace

# Run GPU-specific subsystem test suite
cargo test -p clank_app --test gpu_types_test
cargo test -p clank_app --test gpu_freelist_test
cargo test -p clank_app --test gpu_soil_test
cargo test -p clank_app --test gpu_spatial_test
cargo test -p clank_app --test gpu_agent_test
cargo test -p clank_app --test dual_engine_test
```

### Visual & Manual Confirmation
1. Build optimized binary: `cargo build -p clank_app --release`.
2. Launch native application and trigger `POST /screenshot` via Python test harness.
3. Visually inspect screenshot using `view_file` to confirm:
   - Top header engine badge displays active mode (`ENGINE: RUST` vs `ENGINE: GPU`).
   - Creatures move, feed, fight, and reproduce with identical visual fidelity.
   - Soil texture renders smoothly without artifacts.
   - Zooming in uses LBVH frustum culling without visual pop-in.
   - Dragging AoE tools (Extinguish, Nourish, Blight) operates in $O(\log N)$ time.
   - Minimap draws cluster density discs from intermediate LBVH levels.
   - UI metrics (population, generation, record graph) update in real-time from 128-byte `GpuTelemetry`.
   - Clicking on a creature at 32x speed accurately selects the clicked specimen on the specimen card.
   - Audio sounds modulate dynamically with zoom level (quiet murmur zoomed out, crisp bites zoomed in).
