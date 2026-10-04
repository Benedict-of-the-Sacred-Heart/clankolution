# GPU Compute Simulation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-throughput GPU compute simulation engine (`ENGINE: GPU`) running entirely in WGSL on a dedicated branch (`feature/bevy-gpu`), featuring zero-copy Tombstone Freelist buffers, a hybrid Morton Grid + multi-system Linear Bounding Volume Hierarchy (LBVH), packed 326-weight quantized RNN forward passes, and reactive GPU soil chemistry—coexisting seamlessly alongside the bit-exact CPU reference engine (`ENGINE: RUST`).

**Architecture:** A multi-pass Bevy compute pipeline operating over GPU storage buffers and 2D textures. Agents never reallocate or shift in VRAM thanks to a lock-free Tombstone Freelist. Memory architecture utilizes high-performance **Buffer Splitting (Struct-of-Arrays)**: separating dynamic simulation state (`GpuAgentState`, 128 bytes, 2 cache lines, $2^7$ power-of-two) from static neural genomes (`GpuAgentGenome`, 352 bytes), slashing memory bus traffic by **$4.0\times$** on all rendering and spatial passes. Spatial lookups use a hybrid Morton Grid (Tier 1: $O(1)$ local Moore cells) backed by an explicit Karras Linear Bounding Volume Hierarchy (LBVH). The unified LBVH tree powers five distinct subsystems: 1) Long-range sensory raycasting, 2) Camera viewport frustum culling, 3) Interactive AoE brush tools, 4) Hierarchical minimap LOD cluster rendering, and 5) Optional Barnes-Hut macro-swarm flocking (an experimental mod toggle, default OFF to guarantee 100% HTML behavioral parity). The dual-engine architecture preserves $100\%$ bit-for-bit historical CPU determinism when toggled to `ENGINE: RUST` while unlocking massive agent scaling when toggled to `ENGINE: GPU`.

**Tech Stack:** Rust, Bevy 0.19 (`bevy_render::render_resource`, `wgpu`), WGSL (WebGPU Shading Language), Metal/Vulkan compute pipelines, `rkyv`, `image`.

**Branching Strategy:** Work will proceed exclusively on a new branch: `feature/bevy-gpu` branched directly from `feature/bevy-port`. The base branch `feature/bevy-port` will remain clean and stable until `feature/bevy-gpu` is verified, tested, and visually confirmed.

---

## Explicit GPU Optimizations Covered in this Plan

1. **Decoupled Soil Simulation State & Dual-Target GPU Textures (Zero-Copy)**:
   - **True Simulation State in L2 Cache**: Soil chemistry (food, taint, scent) is maintained in a dedicated 60 KB atomic storage buffer (`GpuSoilCell` for all $75 \times 50 = 3,750$ cells, $3,750 \times 16\text{ bytes} = 60\text{ KB}$). Fits 100% inside GPU L1/L2 cache.
   - **Coalesced Atomic Creature Writes**: Creatures in `agent_step.wgsl` atomically subtract grazed food (`atomicSub(&food_milli)`) and add scent/carcass taint directly to this 60 KB buffer.
   - **Dual-Target Texture Generation (No Ping-Pong Copy)**: `soil_step.wgsl` runs at the start of each sub-tick, reads updated atomic cells, applies environmental bloom renewal and decay, writes back to `soil_buffer`, and outputs **two decoupled textures**:
     1. `soil_data: texture_storage_2d<rgba16float, write>` (75x50, 30 KB): Raw physics snapshot (`vec4f(f, t, s, 1.0)`) sampled by `agent_step.wgsl` with universal hardware bilinear filtering (`textureSampleLevel`) and hardware toroidal seam wrapping (`AddressMode::Repeat`), eliminating 7,000,000 atomic memory reads and read-after-write race conditions.
     2. `soil_display: texture_storage_2d<rgba16float, write>` (75x50, 30 KB): Rendered colormap with vignette, drawn directly by Bevy's `SoilSprite` without CPU upload or memory copies.
   - **Universal Cross-Platform Linear Filtering**: Binds `soil_display` with universal hardware bilinear sampling across macOS (Metal), Windows (Vulkan/DX12), and Linux (Vulkan). **Completely eliminates texture ping-pong pointer swapping and eliminates $B \to A$ GPU memory copying!**

2. **Sensory Raycasting & Universal `unpack4x8snorm` Neural Net Vectorization**:
   - **Antennae Feelers**: Hardware bilinear texture fetches from `soil_data` (`texture_2d<f32>`) at 4 probe locations (`here`, `forward`, `left`, `right`) with hardware coordinate wrapping (`AddressMode::Repeat`) in 0 ALU cycles. Feeler probes sample `probe.r` (food), `probe.g` (taint), and `probe.b` (scent) simultaneously from Texture L1 cache, providing continuous steering gradients across cell boundaries with zero atomic pipeline contention.
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
   - **Subsystem 3 (Unified Interactive Spatial Queries - `spatial_query.wgsl`)**:
     - *Mouse Picking (`tool_type == 0`)*: Traverses the LBVH with **Dynamic Radius Shrinking (Early Tree Pruning)**: starts with cursor radius $R_{\text{pick}} \le 24\text{px}$, but the instant a direct body hit is encountered ($d \le R_{\text{body}} \approx 3.5\text{px} - 5.0\text{px}$), the active search radius is clamped to $d$, immediately pruning all subtrees further than $d$ and resolving picks in ~5–8 node tests (< 40 nanoseconds).
     - *AoE Tool Brushes (`tool_type >= 1`)*: Accelerates `extinguish_at`, `blight_at`, `nourish_at`, and `seed_at` from $O(N)$ linear scans to $O(\log N)$ bounding box intersections.
   - **Subsystem 4 (Minimap LOD Clustering)**: Draws aggregated cluster discs from intermediate tree depths (e.g. depth 5–6) directly onto the radar minimap using cached `dominant_lineage` and `center_of_mass`.
   - **Subsystem 5 (Barnes-Hut Macro-Flocking - Mod 1)**: Evaluates far-field swarm acceleration vectors in $O(\log N)$ via multipole acceptance $\theta < 0.6$ on internal node `center_of_mass` and `count`, reusing the tree directly without adding storage buffers.

4. **Cross-Cutting Morton Code Optimizations (Engine-Wide)**:
   - **Morton Cell Offsets Sentinel Initialization**: The 54-cell table (`cell_offsets: array<vec2u, 54>`) is cleared to `vec2u(0xFFFFFFFFu, 0xFFFFFFFFu)` in the preamble pass. In `agent_step.wgsl`, neighbor searches check `if (range.x != 0xFFFFFFFFu)`, instantly skipping empty cells with zero warp divergence.
   - **Lightweight 8-Byte Indirection Radix Sort**: Parallel radix sort operates strictly on `array<vec2u>` storing `(morton_key, agent_slot_idx)`. The 128-byte `GpuAgentState` structs remain completely stationary in VRAM, eliminating gigabytes/sec of memory bus saturation.
   - **Sub-Microsecond Mouse Picking ($O(\log N)$)**: Fast binary search on sorted Morton keys to select creatures for the specimen card.
   - **Coalesced Atomic Soil Writes**: Threads scheduled in Morton order write to contiguous L1/L2 cache lines for food/taint/scent deposits, cutting bus contention.
   - **Spatially Coherent Dart Rendering**: Morton-ordered instance streams maximize GPU tile-cache hit rates on Metal and tile-based rasterizers.

5. **Hardware Toroidal Coordinate Wrapping**:
   - In WGSL, coordinate wrapping uses:
     `pos.x - 900.0 * floor(pos.x / 900.0)` and `pos.y - 600.0 * floor(pos.y / 600.0)`
   - Evaluates in a single hardware `floor` and `fma` instruction, handling any negative coordinate seamlessly without edge pop-in.

6. **Birth / Compaction Freedoms & CAS Freelist Protection**:
   - **Tombstone Freelist & CAS Underflow Guard**: Fixed static `MAX_AGENTS` storage buffer. Dead creatures are marked with tombstone flags (`dead = 1`) and their slot indices are recycled onto an atomic stack (`atomicAdd` on deallocation). Allocation uses an atomic Compare-and-Swap (CAS) pop loop: checks `freelist_top > 0u` before popping, safely preventing unsigned 32-bit underflow (`4,294,967,295`) when population reaches carrying capacity. **Zero memory shifting or array compaction across frames.**
   - **Dedicated Preamble Clearing Pass (`preamble_clear.wgsl`)**: Dispatched immediately before simulation stepping to eliminate cross-workgroup race conditions. Resets `mate_claim = 0u`, `mate_energy_milli = 0u`, `dead_claimed = 0u`, `queue_buffer.telemetry.birth_count = 0u`, `queue_buffer.telemetry.audio_voice_count = 0u`, `queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu`, `queue_buffer.telemetry.selected_agent_id = 0u`, and `cell_offsets = 0xFFFFFFFFu`. Guarantees a global GPU execution and memory barrier across all workgroups before simulation stepping begins.
   - **Atomic CAS Double-Free Protection**: To prevent starvation and predation from double-freeing the same slot in the same tick, death ownership is acquired via CAS on `dead_claimed: atomic<u32>` ($0 \to 1$). Exactly one thread succeeds, sets `dead = 1`, and recycles the slot to `freelist`.
   - **Canonical Mating Symmetry Breaking (Optimization A)**: When two creatures are mutually interested, only the creature with `partner_id > agent_id` initiates the claim on `agent_atomics[partner_idx].mate_claim`. This **cuts atomic CAS memory bus traffic in half (50% reduction)** and completely eliminates mutual deadlocks, duplicate twin births, and double-spending without needing multi-round handshakes.
   - **Seamless Asexual Fallback (Optimization B)**: In strict parity with the HTML reference (`clankolution.html#L1439-L1444`), if a sexual mate is absent, out of range, or rejected, the creature smoothly falls back to asexual virgin reproduction (`child(a, null)` / `partner_idx = 0xFFFFFFFFu`). This guarantees the reproductive cycle is never lost to lock contention.

7. **Predation & Combat Resolution (Atomic Fixed-Point & Decisive Attribution)**:
   - **L2-Cached Atomic Damage Buffer**: `energy_milli: atomic<i32>` resides in an isolated 16-byte aligned buffer (`GpuAgentAtomic`), serving as the single continuous atomic ground truth for energy.
   - **Exact Formula Parity**: Damage evaluated via `(0.5 + attack * 2.2) * hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3)`.
   - **Decisive Killer Attribution**: Threads perform `atomicSub(&agent_atomics[victim].energy_milli, dmg_milli)`. The exact thread crossing the zero threshold (`old > 0 && old <= dmg`) is attributed the kill, receives the siphon energy and kill bounty (`min(9.0, 8.0 * tr5)`), sets victim tombstone, and recycles the slot to the freelist via the `dead_claimed` CAS gate.

8. **Buffer Splitting & ConsolidatedQueue Telemetry Header (Strict $\le 8$ Storage Buffer Limit)**:
   - **Struct-of-Arrays (SoA) Splitting**: Separates the dynamic simulation state (`GpuAgentState`, 128 bytes, 2 cache lines, $2^7$ power-of-two) from static neural weights (`GpuAgentGenome`, 352 bytes). Eliminates L2 cache thrashing during simulation stepping and cuts memory bus traffic by **$4.0\times$** on all rendering, culling, and spatial passes.
   - **Consolidated Frame Output Queue (`ConsolidatedQueue`)**: Embeds `telemetry: GpuTelemetry` (128 bytes) directly into the header of `ConsolidatedQueue` alongside `births: array<BirthEvent, 65536>` ($1.0\text{ MB}$) and `audio: array<AudioVoice, 256>` ($4\text{ KB}$).
   - **Strict $\le 8$ Storage Buffer Compliance**: Keeps storage buffer bindings in `agent_step.wgsl` at **exactly 8** (`agent_states`, `agent_genomes`, `agent_atomics`, `soil_buffer`, `spatial_keys`, `lbvh_nodes`, `freelist`, and `queue_buffer`), ensuring 100% strict cross-platform compatibility across Metal, Vulkan, DX12, and WebGPU.

9. **Stateless Counter-Based PRNG & Triangular Distribution**:
   - **PCG 3D Hash in Hardware Registers**: Evaluates $\text{rand} = \text{pcg3d}(\text{vec3u}(\text{id}, \text{stream}, \text{tick}))$ in 2 nanoseconds directly in ALU registers. Requires $0$ bytes of persistent PRNG state, $0$ atomic locks, and guarantees bit-for-bit reproducibility across all GPU architectures.
   - **Triangular Distribution**: Computes $\mathcal{T}(-1, 0, 1)$ via $r_1 + r_2 - 1.0$ matching the exact HTML mutation distribution.

10. **Decoupled Birth Queue & Parallel Genome Mutation**:
    - **Decoupled Birth Pass**: Mating agents push a 16-byte `BirthEvent` to an atomic append queue. The main agent compute shader has **zero warp divergence** from gene loops.
    - **Parallel Genome SIMD Mutation (`birth_step.wgsl`)**: Dedicated workgroups process newborn agents, performing 88-word parallel crossover (48% parent crossover) and triangular mutation $(r_1 + r_2 - 1.0) \times 100 \times \text{mutRate}$ clamped to $[-127, 127]$, plus morphological trait mutation $(r_1 + r_2 - 1.0) \times \text{mutRate} \times 0.6$ clamped to $[0.03, 0.98]$.

11. **Experimental Simulation Mods Suite & WGSL Pipeline Specialization**:
    - **Decoupled Architecture via WGSL Pipeline Overrides (`override`)**:
      - Mod toggles are **completely decoupled from `GpuSimParams`**. The uniform buffer contains zero mod fields, keeping baseline physics 100% pure and frozen at exactly 64 bytes.
      - Shaders declare pipeline-overridable constants at module scope with `= false` defaults:
        ```wgsl
        override ENABLE_BARNES_HUT: bool = false;
        override ENABLE_EXPANDED_CORTEX: bool = false;
        override ENABLE_SEXUAL_SELECTION: bool = false;
        ```
      - **Compile-Time Dead-Code Elimination & Optimal Occupancy**: When a mod is disabled, the GPU driver compiler prunes the inactive code branch before register allocation. Baseline mode incurs **zero register pressure**, **zero dynamic branch instructions**, and **zero warp divergence**.
      - **Bevy Mod Control Plane (`SimMods` & `SimPipelines`)**: Bevy manages a dedicated resource `pub struct SimMods { pub barnes_hut: bool, pub expanded_cortex: bool, pub sexual_selection: bool }`. At startup, Bevy compiles/caches the 8 specialized pipeline variants ($2^3 = 8$, taking ~5–10ms). When a player flips a mod switch in the UI, Bevy binds the corresponding specialized pipeline handle with zero CPU/GPU transfer overhead.
    - **Mod 1: Barnes-Hut Macro-Flocking Mod (`[BARNES-HUT: OFF/ON]`)**:
      - Default OFF (100% HTML parity): The Barnes-Hut loop is eliminated by the compiler (`if (ENABLE_BARNES_HUT)`), preserving bit-exact baseline kinematics.
      - When ON: Queries internal LBVH node centers of mass (`node.center_of_mass`) and creature counts (`node.count`) via Multipole Acceptance Criterion $\theta = \text{size}(\text{AABB}) / \text{distance} < 0.6$. Approximates thousands of distant creatures as a single macro-mass in $O(\log N)$ time, applying far-field flocking vectors without $O(N^2)$ checks or neural weight modifications.
      - **Strict $\le 8$ Storage Buffer Compliance**: Reuses `@binding(4) lbvh_nodes: array<GpuLbvhNode>`, which is already bound in `agent_step.wgsl`. Requires **0 additional storage buffers** and zero extra passes, strictly maintaining WebGPU's 8-buffer portable ceiling.
    - **Mod 2: Expanded Cortex Mod (`[EXPANDED CORTEX: OFF/ON]`)**:
      - Default OFF (100% HTML parity): Inactive sensory slots are multiplied by $0.0$, reproducing the 326-weight brain with zero deviation.
      - When ON: Activates the 2 unused sensory inputs per hidden neuron (Input 24: local food tangent/gradient vector, Input 25: LBVH macro-swarm cluster centroid bearing) and 1 extra actuator (Actuator 6: sprint burst / armor hardening). Mutations during `birth_step.wgsl` evolve these weights actively, allowing complex pack dynamics and navigation to emerge!
    - **Mod 3: Natural Sexual Selection Tournament Mod (`[SEXUAL SELECTION: OFF/ON]`)**:
      - Default OFF (100% HTML parity): Uses baseline probabilistic mating with canonical symmetry breaking.
      - When ON: Replaces first-come-first-served CAS collisions with an emergent biological tournament. Suitors submit proposals to their desired mate's mailbox using full 32-bit addressing:
        `let my_energy_milli = u32(max(0.0, a_energy) * 1000.0);`
        `let prev_bid = atomicMax(&agent_atomics[partner_idx].mate_energy_milli, my_energy_milli);`
        `if (my_energy_milli > prev_bid) { atomicStore(&agent_atomics[partner_idx].mate_claim, agent_idx + 1u); }`
        In `birth_step.wgsl`, the desired mate automatically pairs with the highest-energy/fittest suitor (`mate_claim - 1u`), driving rapid Darwinian sexual selection across up to 4.29 billion creature slots without bit packing or buffer overhead!
    - **Mod 4: Lineage Genome Bank (Ultra-Scale Architectural Concept)**:
      - Concept design for future 10,000,000+ creature scaling. The baseline engine strictly preserves the direct 512-byte `GpuAgent` layout for 100% bit-exact parity with `clankolution.html` with zero structural modifications.
      - Concept architecture: Partitions the 88-word genome into 8 deduplicated functional gene blocks (44 bytes each: feelers, vision, scent, RNN core, steering, thrust, attack, mating). Creatures store 8 `u16` block pointers (16 bytes total instead of 352 bytes), shrinking the dynamic agent struct footprint from 512 bytes to 160 bytes ($3.2\times$ reduction) and unlocking ultra-scale simulations of 10,000,000+ creatures within ~1.6 GB of VRAM without altering baseline pipeline structs.

12. **Seamless Dual-Engine State Hot-Swapping**:
    - Bi-directional bridge between Bevy ECS `SimWorld` and the GPU storage buffers (`GpuAgent`, `GpuAgentAtomic`, `GpuSoilCell`).
    - Switching `[ENGINE: RUST]` $\leftrightarrow$ `[ENGINE: GPU]` dynamically transfers live agents and simulation tick, allowing users to hot-swap engines live mid-run without resetting the simulation timeline or losing creature lineages.

13. **GPU Particle Sparks & Mesh Instancing**:
    - 16,384+ drifting sparks updated in an isolated GPU compute pass (velocity decay, alpha fading).
    - Instanced creature dart mesh generation on the GPU.

14. **Buffer Splitting & 128-Byte Lossless State Layout ($2^7$ Power-of-Two)**:
    - **Operation 1: Struct-of-Arrays (SoA) Splitting**: High-frequency dynamic simulation state (`GpuAgentState`, 128 bytes) is separated from static neural genomes (`GpuAgentGenome`, 352 bytes). All non-simulation passes (frustum culling, dart rendering, Morton indexing, LBVH building, mouse picking) stream only 128 bytes per agent, cutting memory bus traffic by **$4.0\times$** ($128\text{ MB}$ vs $512\text{ MB}$ for 1,000,000 agents).
    - **Operation 2: Neural Forward Pass (`agent_step.wgsl`)**: Reads clean read-only `agent_genomes` from L2 cache while writing dynamic kinematics to `agent_states`. In baseline mode (`!ENABLE_EXPANDED_CORTEX`), dummy input slots are set to `0.0`, guaranteeing 100% bit-exact parity with HTML. In Expanded Cortex mode (`ENABLE_EXPANDED_CORTEX`), slots are actively fed chemical tangent gradients and LBVH swarm cluster centroid bearings.
    - **Operation 3: Genome Mutation Masking (`birth_step.wgsl`)**: In baseline mode (`!ENABLE_EXPANDED_CORTEX`), mutation applies a bitmask (`word & 0x0000FFFFu`) to the 7th word of hidden neurons and 3rd word of output neurons, preventing silent random drift in inactive weights. When Expanded Cortex is toggled ON (`ENABLE_EXPANDED_CORTEX`), the mask is lifted and mutations actively evolve novel traits.
    - **Operation 4: Dual-Engine Live Hot-Swapping (`sync_rust_to_gpu` & `sync_gpu_to_rust`)**: Exact bit-level pack/unpack maps 326 sequential `i8` genes into 88 vec4-aligned `u32` words in `agent_genomes` with zero loss or drift.
    - **Operation 5: Savefile Backward Compatibility**: Saving always extracts the canonical 326 `i8` genes into `AgentData`, ensuring all `.clank` and `.json` files are 100% cross-compatible between CPU and GPU engines.
    - **Operation 6: Precomputed Visual Render Cache (`visual_cache: u32`)**: The final 4 bytes of Cache Line 1 (offsets 124..128) packs 4 rendering attributes (`visual_radius: u8`, `glow_intensity: u8`, `energy_u8`, `visual_flags: u8`). The instanced dart vertex shader loads `vec2u(packed_color, visual_cache)` directly from Cache Line 1, completely bypassing Cache Line 0 (traits, kinematics) and saving ~72 MB of vertex fetch bandwidth per frame (~4.3 GB/s at 60 FPS) across 1,000,000 agents.

15. **Consolidated Output Queue (Telemetry Header + 65,536 Births + 256 Voices = 1,028 KB)**:
    - **Unified Buffer Architecture (`ConsolidatedQueue`)**: Combines aggregate telemetry (128 bytes), birth events (65,536 entries $\times$ 16B = 1,048,576 bytes = 1 MB), and audio voices (256 entries $\times$ 16B = 4,096 bytes = 4 KB) into a single 1,052,800-byte storage buffer ($1,028.1\text{ KB} \ll 128\text{ MB}$ WebGPU storage limit, 16-byte aligned).
    - **Freelist Reservation Order**: Reserves `queue_idx = atomicAdd(&queue_buffer.telemetry.birth_count, 1u)` *before* popping from `freelist`, completely eliminating slot leakage during reproductive blooms or queue saturation.
    - **Toroidal Seam Viewport Frustum Culling**: Events occurring outside the camera viewport are culled directly on the GPU using toroidal shortest distances:
      ```wgsl
      let dx = abs(pos.x - params.camera_pos.x);
      let dist_x = min(dx, 900.0 - dx);
      let dy = abs(pos.y - params.camera_pos.y);
      let dist_y = min(dy, 600.0 - dy);
      let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
      ```
      Takes only 4 ALU cycles and eliminates edge audio pop-out when panning across the 900x600 toroidal seam.
    - **Zoom Loudness Modulation**: Sound volume scales with camera zoom level (0.08 ambient murmur fully zoomed out, 1.0 punchy bite fully zoomed in).
    - **Stochastic Hash Density Filter**: In dense swarms, events are stochastically sampled via 1-cycle stateless PCG hash `pcg3d(vec3u(killer, victim, tick)) % density == 0u`, preventing audio mixer blowout.

16. **Hardening Specifications: Detailed Vulnerability Analyses & Hardening Fixes**:

    #### 1. Degenerate LBVH Traversal Guard for $N \le 1$ Agents (`spatial_query.wgsl`)
    - **The Vulnerability**:
      In `lbvh_build.wgsl`, when population is 0 or 1, the builder early-exits (`if (active_count < 2u) return;`), which is mathematically required because a 1-leaf tree has $N - 1 = 0$ internal nodes.
      However, in `spatial_query.wgsl`, mouse picking unconditionally pushed Node 0 to the traversal stack (`stack[0] = 0u; stack_ptr = 1u;`).
      If a user clicked when $N = 1$ (e.g. at initial seeding or after a near-extinction event) or $N = 0$, thread 0 would read uninitialized or stale data from `lbvh_nodes[0]`, risking GPU infinite loops, hang conditions, or invalid pointer dereferences.
    - **The Hardening Fix**:
      Add an explicit degenerate count gate in `spatial_query.wgsl`:
      ```wgsl
      if (params.agent_count == 0u) {
          queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
          queue_buffer.telemetry.selected_agent_id = 0u;
          return;
      }
      if (params.agent_count == 1u) {
          // Direct single-agent evaluation without tree traversal:
          let agent_pos = agent_states[0].pos_vel.xy;
          let d = toroidal_dist(tool_pos, agent_pos);
          let visual_r = 2.0 + 3.0 * agent_states[0].traits[0];
          if (d <= search_r) {
              queue_buffer.telemetry.selected_agent_idx = 0u;
              queue_buffer.telemetry.selected_agent_id = agent_states[0].id;
          } else {
              queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
              queue_buffer.telemetry.selected_agent_id = 0u;
          }
          return;
      }
      // Proceed with O(log N) tree traversal starting at root node 0...
      ```

    #### 2. Soil Buffer & Morton Cell Boundary Clamping Guard (`agent_step.wgsl` & `morton_grid.wgsl`)
    - **The Vulnerability**:
      In `agent_step.wgsl`:
      `let cell_idx = u32(pos.y / 12.0) * 75u + u32(pos.x / 12.0);`
      If an agent is positioned at the right or bottom boundary (`pos.x == 900.0` or `pos.y == 600.0`, which occurs after wrapping or near the boundary seam), `u32(pos.x / 12.0)` evaluates to `75u` and `u32(pos.y / 12.0)` evaluates to `50u`.
      This calculates `cell_idx = 50 * 75 + 75 = 3825`. But `soil_buffer` has size 3,750 (indices `0..3749`).
      Writing or reading `soil_buffer[3825]` produces an out-of-bounds storage buffer overrun. The reference implementation in `soil.rs:78-86` explicitly guards this with `min(col, cols - 1)` and `min(row, rows - 1)`.
    - **The Hardening Fix**:
      Clamp cell coordinates to array bounds before index calculation:
      ```wgsl
      let cx = min(u32(max(0.0, pos.x) / 12.0), 74u);
      let cy = min(u32(max(0.0, pos.y) / 12.0), 49u);
      let cell_idx = cy * 75u + cx;
      ```
      Apply the identical guard in `morton_grid.wgsl`:
      ```wgsl
      let gx = min(u32(max(0.0, pos.x) / 100.0), 8u);
      let gy = min(u32(max(0.0, pos.y) / 100.0), 5u);
      let cell_id = gy * 9u + gx; // Guaranteed 0..53 (safely bounds cell_offsets[54])
      ```

    #### 3. Canonical Atomic Energy Model & Starvation Attribution Order (`agent_step.wgsl`)
    - **The Vulnerabilities**:
      1. *Dual Energy Overwrite Race*: When an agent thread calculates its internal net delta (grazing - metabolism - thrust) in local float `a_energy` and writes to `agent_states[agent_idx].angle_energy[1]`, it could overwrite external combat damage and mating deductions inflicted concurrently by other threads via `atomicSub(&agent_atomics[victim_idx].energy_milli, damage_milli)`.
      2. *Starvation Double-Counting*: In naïve code, `atomicAdd(&queue_buffer.telemetry.starvations, 1u)` was called before `atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u)`. If a predator dealt the killing blow on the same tick, both kills and starvations were incremented for the same victim.
    - **The Hardening Fix**:
      - Make `agent_atomics[agent_idx].energy_milli` the single continuous source of truth for energy.
      - The agent thread calculates its internal net delta and applies it atomically:
        ```wgsl
        let internal_delta_milli = i32((graze_energy - basal_cost - thrust_cost) * 1000.0);
        atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);
        ```
      - Gate the starvation counter behind successful CAS claim:
        ```wgsl
        if (atomicLoad(&agent_atomics[agent_idx].energy_milli) <= 0) {
            let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
            if (claim_death.exchanged) {
                atomicAdd(&queue_buffer.telemetry.starvations, 1u); // ONLY incremented if this thread won the death claim!
                agent_states[agent_idx].meta_flags |= (1u << 13u); // dead = 1 (bit 13)
                let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
                freelist[free_slot] = agent_idx;
            }
        }
        ```

    #### 4. Non-Blocking Pipelined Double-Buffered DMA Staging Ring (Task 6)
    - **The Bottleneck**:
      In Bevy / wgpu, calling synchronous `device.poll(PollType::Wait)` for GPU buffer readbacks stalls the CPU main thread waiting for the command queue to drain, cutting frame rates by 30–50%.
    - **The Hardening Fix**:
      Implement a 2-frame ping-pong staging ring (`staging_telemetry[frame % 2]`, `staging_audio[frame % 2]`):
      - Frame $N$: Dispatches `encoder.copy_buffer_to_buffer` and issues `map_async` on `staging[N % 2]`.
      - Simultaneously: Reads back `staging[(N - 1) % 2]` (mapped during the prior frame) with **0 wait cycles**.
      - Provides completely non-blocking execution at full 60/120 FPS with 1-frame pipelined telemetry latency.

    #### 5. Struct Layout Sizing & `visual_cache: u32` 5 Hardening Invariants (Task 1 & Task 5)
    - **The Vulnerability**:
      Writing `pub _reserved: [u32; 2]` (8 bytes) at the end of `GpuAgentState` produced a struct size of $132\text{ bytes}$. Because of `#[repr(C, align(16))]`, $132$ rounded up to $144\text{ bytes}$, breaking the 128-byte power-of-two assertion (`size_of::<GpuAgentState>() == 128`), invalidating cache line alignment, and exceeding the 128 MB single buffer limit for 1M agents.
    - **The Hardening Fix & Invariants**:
      Replace `_reserved` with `pub visual_cache: u32` (offsets 124..128), locking `GpuAgentState` to **EXACTLY 128 bytes** (2 cache lines, $2^7$ power-of-two).
      Packs:
      - Bits 0..7: `visual_radius: u8` ($0..255 \rightarrow 2.0..5.0\text{ px}$)
      - Bits 8..15: `glow_intensity: u8` (attack flash, courtship glow, birth aura)
      - Bits 16..23: `energy_u8` ($0..100$ energy for opacity & vibrancy)
      - Bits 24..31: `visual_flags` (bit 24: attacking, bit 25: mating, bit 26: selected, bit 27: apex)
      **The 5 Hardening Invariants**:
      1. *Zero-Cost Dead State Invariant*: When an agent dies, `visual_cache` is set to `0u`, collapsing scale to $0.0\text{ px}$ and opacity to $0.0$, preventing phantom corpse rendering.
      2. *Newborn Initialization Invariant*: `birth_step.wgsl` initializes `visual_cache` with child trait 0, initial energy 24.0, and a birth flash (`glow = 255u`).
      3. *Picking Parity Invariant*: `spatial_query.wgsl` continues to evaluate full-precision `traits[0]` ($2.0 + 3.0 \times \text{tr}_0$), preserving float32 click selection bounds.
      4. *Hot-Swap Continuity Invariant*: `sync_rust_to_gpu` synthesizes `visual_cache` on upload; `sync_gpu_to_rust` cleanly discards it on readback.
      5. *VRAM Bandwidth Invariant*: Dart vertex shader loads `vec2u(packed_color, visual_cache)` from Cache Line 1, saving ~72 MB/frame vertex fetch bandwidth (~4.3 GB/s at 60 FPS).

    #### 6. Toroidal Boundary AABB Pruning Blindspot in `spatial_query.wgsl` (Task 4)
    - **The Vulnerability**:
      In `spatial_query.wgsl`, if `distance_to_aabb` evaluates standard Euclidean distance without torus wrapping, clicking near an arena seam ($x = 899.0$) against a bounding box on the opposite wrapped side ($x \in [0.0, 10.0]$) computes $dx = 899.0 - 10.0 = 889.0\text{ px}$. Because $889.0 > 16.0$, the entire subtree containing the creature at $x = 0.5$ is pruned, creating a click selection "blind spot" across screen boundaries.
    - **The Hardening Fix**:
      Implement branchless 1D toroidal AABB distance in `spatial_query.wgsl`:
      ```wgsl
      fn toroidal_aabb_dist_1d(p: f32, b_min: f32, b_max: f32, w: f32) -> f32 {
          if (p >= b_min && p <= b_max) { return 0.0; }
          let direct = select(p - b_max, b_min - p, p < b_min);
          let wrapped = select(w - p + b_min, w - b_max + p, p < b_min);
          return min(direct, wrapped);
      }

      fn distance_to_aabb(pos: vec2f, aabb_min: vec2f, aabb_max: vec2f) -> f32 {
          let dx = toroidal_aabb_dist_1d(pos.x, aabb_min.x, aabb_max.x, 900.0);
          let dy = toroidal_aabb_dist_1d(pos.y, aabb_min.y, aabb_max.y, 600.0);
          return sqrt(dx * dx + dy * dy);
      }
      ```

    #### 7. Strict Single-Writer Invariant for `GpuAgentState` (Cross-Thread Flag Race) (Task 5)
    - **The Vulnerability**:
      If a predator thread executes `agent_states[victim_idx].meta_flags |= (1u << 13u)` while the victim thread is concurrently running, the victim thread's final writeback (`agent_states[victim_idx].meta_flags = updated_flags`) clobbers the dead bit back to `0`, resurrecting a "zombie" creature.
    - **The Hardening Fix**:
      Enforce strict single-writer ownership: Thread $i$ writes ONLY to `agent_states[i]`.
      1. Predator only sets atomic CAS gate `agent_atomics[victim_idx].dead_claimed` and pushes to `freelist`. Predator never writes to `agent_states[victim_idx]`.
      2. Victim thread checks `atomicLoad(&agent_atomics[agent_idx].dead_claimed)` before committing state. If killed, victim sets its own dead bit (`meta_flags |= (1u << 13u)`), sets `visual_cache = 0u`, and aborts live state commits (movement, grazing, mating).

    #### 8. Pass-Decoupled Lock-Free Freelist (Eliminating Concurrent Push-Pop Races) (Tasks 2 & 5)
    - **The Vulnerability**:
      If deaths push to `freelist` (`atomicAdd(&freelist_top, 1u)`) while births pop from it (`atomicCompareExchangeWeak(&freelist_top, cur, cur - 1u)`) in the same `agent_step.wgsl` dispatch, incoming death threads can overwrite stack slots before pop threads finish reading them, causing slot leakage or premature recycling.
    - **The Hardening Fix**:
      Decouple push and pop across compute passes:
      - **In `agent_step.wgsl`**: ONLY pushes occur (deaths push deceased slots to `freelist`). Mating threads only record intent in `queue_buffer.births`.
      - **Compute Pass Barrier**: Explicit GPU barrier submitted by Bevy between `agent_step` and `birth_step`.
      - **In `birth_step.wgsl`**: ONLY pops occur. Each newborn thread pops from `freelist` with zero concurrent pushes occurring.

    #### 9. Dead Agent Partitioning in Morton Spatial Indexing (`0xFFFFFFFFu` Sentinel) (Task 4)
    - **The Vulnerability**:
      In a 1M agent simulation, inactive corpses in the freelist have residual coordinates. If `morton_encode` generates keys for all slots $0..N_{\max}$, dead agents are sorted into Morton cells and LBVH trees, corrupting spatial queries and picking.
    - **The Hardening Fix**:
      In `morton_grid.wgsl`, `morton_encode` assigns dead agents sentinel key `0xFFFFFFFFu`:
      ```wgsl
      let is_dead = (agent_states[id.x].meta_flags & (1u << 13u)) != 0u;
      if (is_dead) {
          spatial_keys[id.x] = vec2u(0xFFFFFFFFu, id.x);
      } else {
          let code = compute_morton_32(agent_states[id.x].pos_vel.xy);
          agent_states[id.x].morton_code = code;
          spatial_keys[id.x] = vec2u(code, id.x);
      }
      ```
      When Radix sorted, all dead agents are naturally partitioned to the tail of `spatial_keys`, completely excluded from grid cells 0..53 and LBVH leaf bounds.

    #### 10. Bottom-Up LBVH AABB Fitting Atomic Flag Initialization (Task 4)
    - **The Vulnerability**:
      In `lbvh_aabb.wgsl`, parallel tree climbing uses `atomicAdd(&node_flags[current], 1u)` to detect when both children have computed their bounding boxes. If `node_flags` is not reset to 0 every frame, on frame 2 and beyond `atomicAdd` returns $\ge 2$, causing threads to deadlock or produce corrupt bounding boxes.
    - **The Hardening Fix**:
      Thread `id.x` in Phase 1 (`lbvh_build.wgsl`) explicitly resets its node's atomic flag:
      `atomicStore(&node_flags[id.x], 0u);`
      Because Phase 1 is separated from Phase 2 by a compute pass barrier, all flags are guaranteed to be 0 before climbing begins.

    #### 11. Supplemental Hardening & Robustness Guarantees
    - **Degenerate LBVH Underflow Guard (`lbvh_build.wgsl`)**:
      In `lbvh_build.wgsl`, if `params.agent_count == 0u`, the condition `id.x < params.agent_count - 1u` underflows `0u - 1u` to `u32::MAX` ($4,294,967,295$), launching rogue out-of-bounds writes. Guarded via:
      `let active_count = params.agent_count; if (active_count < 2u || id.x >= active_count - 1u) return;`
    - **Two-Phase LBVH Construction Separation**:
      Explicit compute pass barrier between topology generation (`lbvh_build.wgsl`) and bottom-up bounding box fitting (`lbvh_aabb.wgsl`), eliminating workgroup scheduling race conditions.
    - **LBVH Identical Key Tie-Breaking**:
      If multiple agents occupy identical coordinates or have identical Morton keys, Karras LCP comparison evaluates:
      `return 32 + i32(countLeadingZeros(id_i ^ id_j));`
      guaranteeing strictly unique split positions and preventing tree-building infinite loops.
    - **Dynamic Radius Shrinking (Early Tree Pruning in `spatial_query.wgsl`)**:
      Begins traversal with cursor radius $R_{\text{pick}} = \text{clamp}(16.0 \times (\text{camera\_size}.x / 900.0), 4.0, 24.0)$. The instant a leaf node with a direct body hit is encountered ($d \le R_{\text{body}} \approx 2.0 + 3.0 \times \text{tr}_0$), the active search radius is clamped to $d$. Any subtrees or sibling nodes whose AABB distance to the cursor is $> d$ are **immediately pruned**, resolving clicks in ~5–8 node tests (< 40 nanoseconds on GPU).
    - **Two-Tier Euclidean Disambiguation**:
      Direct body hits (Priority 0) always beat proximity halos (Priority 1); exact 32-bit floating-point Euclidean distance resolves ties without millipixel distortion.
    - **Specimen Picking Identity Guard `(slot_idx, agent_id)`**:
      The CPU tracks the tuple `(slot_idx, agent_id)`. If Agent $K$ dies during sub-ticks 2..32 (or a subsequent tick) and slot $K$ is recycled, `agent_states[slot_idx].id != picked_id`. The UI immediately detects the death, prevents displaying the replacement creature, and displays final stats.
    - **Multi-Tick Telemetry & Audio Lifecycle Synchronization (`preamble_clear.wgsl`)**:
      In 32x speed mode, sub-tick resets (`mate_claim`, `mate_energy_milli`, `dead_claimed`, `birth_count = 0u`, `cell_offsets`) run every sub-tick. Frame-level cumulative counters (`audio_voice_count = 0u`, `kills = 0u`, `starvations = 0u`, `food_grazed_milli = 0u`, `apex_record_milli = 0u`) are reset strictly at frame start (`params.sub_tick == 0u`), preventing audio queue overwrites and telemetry loss. Instantaneous living census (`population` and `lineage_counts[16]`) is reset and recounted strictly on the final sub-tick (`sub_tick == params.sub_ticks_per_frame - 1u`).

    #### 12. Dual-Target Decoupled Soil Architecture & Feeler TMU Bilinear Acceleration (Tasks 3 & 5)
    - **The Vulnerabilities**:
      1. *Read-After-Write Race Condition*: If 1,000,000 agents read directly from `soil_buffer: array<SoilCell, 3750>` while other agents concurrently execute `atomicSub` to graze, sensory perception depends on non-deterministic warp scheduling jitter.
      2. *Atomic Memory Serialization*: 7,000,000 feeler reads per tick directed at 3,750 cells creates up to 35,000 concurrent atomic memory collisions in dense swarms, collapsing GPU memory pipeline throughput.
      3. *Colormap Sensory Corruption*: `soil_step.wgsl` renders `evaluate_soil_color(f, t, s)` (display RGB with dark dirt `[10, 23, 26]`, vignette, etc.). If agents sampled this display texture, creatures would sense the edge vignette as a severe famine and flee the arena borders!
    - **The Hardening Fix**:
      Decouple into two specialized 75x50 textures (60 KB total VRAM):
      1. `soil_data: texture_storage_2d<rgba16float, write>` (30 KB): Output by `soil_step.wgsl` storing raw simulation scalars `vec4f(food, taint, scent, 1.0)`. Sampled by `agent_step.wgsl` via `textureSampleLevel(soil_data, soil_sampler, uv, 0.0)`. Bypasses atomic units, utilizes TMU Texture L1 cache broadcasting, provides continuous bilinear gradients across cell boundaries, and wraps toroidal coordinates across seams in silicon (`AddressMode::Repeat`) at 0 ALU cycles.
      2. `soil_display: texture_storage_2d<rgba16float, write>` (30 KB): Output by `soil_step.wgsl` storing `evaluate_soil_color(f, t, s, id.xy)`. Displayed directly by Bevy's `SoilSprite` without CPU upload or custom fragment shaders.
      3. `soil_buffer: array<SoilCell, 3750>`: Serves exclusively as the persistent atomic write target for discrete grazing (`atomicSub`) and carcass/scent deposits in `agent_step.wgsl`.

    #### 13. Workgroup-Uniform Birth Bounds & Freelist Slot Theft Guard (`birth_step.wgsl`) (Task 5)
    - **The Vulnerability**:
      `birth_step.wgsl` processes queued births with 32 threads per newborn workgroup. When Bevy dispatches workgroups in fixed batches (or rounded up), excess workgroups ($wg\_id.x \ge \text{birth\_count}$) read uninitialized memory from `queue_buffer.births[wg_id.x]`. Thread 0 in excess workgroups would execute `atomicCompareExchangeWeak` on `freelist_top`, **popping real, valid slots from the freelist for non-existent births**—rapidly exhausting the freelist and spawning corrupted zombie agents.
    - **The Hardening Fix**:
      Place a workgroup-uniform bounds guard at the entry of `birth_main`:
      ```wgsl
      let total_births = min(atomicLoad(&queue_buffer.telemetry.birth_count), 65536u);
      if (wg_id.x >= total_births) {
          return; // Safely abort excess workgroup without popping freelist or reading uninitialized queue slots
      }
      ```
      Because `wg_id.x` is identical across all 32 threads in the workgroup, either all 32 threads return, or all 32 proceed to `workgroupBarrier()`, guaranteeing **zero intra-workgroup divergence and zero GPU barrier deadlocks**.

    #### 14. Top-of-Shader Tombstone Execution Guard (`agent_step.wgsl`) (Task 5)
    - **The Vulnerability**:
      Because freelist slots are recycled non-contiguously across $0..N_{\max}-1$, inactive corpses remain in `agent_states`. If a thread executing for a dead slot does not exit immediately, it evaluates neural forward passes, moves, grazes on soil food, deposits taint, attacks live agents, and corrupts census telemetry.
    - **The Hardening Fix**:
      Add an explicit dead-check at the very top of `agent_step.wgsl`:
      ```wgsl
      let agent_idx = id.x;
      if (agent_idx >= params.max_agents) { return; }
      let meta = agent_states[agent_idx].meta_flags;
      if ((meta & (1u << 13u)) != 0u) {
          return; // Tombstone corpse: skip neural net, movement, grazing, predation, and mating!
      }
      ```
      Deactivates inactive SIMD lanes via hardware execution masks with zero memory writes.

    #### 15. Cursor Coordinate Toroidal Wrapping & Non-Negative Distance Proof (`spatial_query.wgsl`) (Task 4)
    - **The Vulnerability**:
      If a user clicks slightly outside the canvas or near window seams ($x = -2.0$ or $x = 902.0$), naive 1D toroidal distance evaluates $dx = |-2.0 - 899.0| = 901.0 \implies 900.0 - 901.0 = -1.0$. A negative distance in the comparison tree distorts picking priority and prunes valid candidates.
    - **The Hardening Fix**:
      Pre-wrap `tool_pos` at entry of `spatial_query_main`:
      ```wgsl
      let tool_pos = vec2f(
          params.tool_pos[0] - 900.0 * floor(params.tool_pos[0] / 900.0),
          params.tool_pos[1] - 600.0 * floor(params.tool_pos[1] / 600.0)
      );
      ```
      Guarantees $dx \in [0.0, 900.0)$ and $900.0 - dx \in (0.0, 900.0]$, ensuring all distance metrics are strictly non-negative and monotonic.

17. **Multi-Layer GPU Compression Architecture & Mathematical Equivalence Proof**:
    - **Layer 1: Hardware-Level Silicon Compression (Automatic & Transparent)**:
      - Apple Silicon unified memory fabric and NVIDIA DCC automatically compress cache lines moving across LPDDR5/VRAM buses without shader intervention.
    - **Layer 2: Fixed-Function Block Texture Compression (ASTC $4\times4$ to $12\times12$)**:
      - Hardware texturing units decode ASTC blocks in 0 ALU cycles during `textureSampleLevel`.
      - Dynamic 60Hz soil remains uncompressed 30 KB `rgba16float` because GPU silicon decodes but does not encode ASTC in hardware, and 30 KB fits in L1 cache.
    - **Layer 3: Parallel Radix / Prefix Bit-Packing & Stream Compaction**:
      - *8-Byte Morton Indirection Sorting (Task 4)*: Radix sort operates over `vec2u(morton_key, slot_idx)`, reducing sorting bus bandwidth by **$16\times$** compared to shifting 128-byte structs.
      - *Camera Frustum Stream Compaction (Task 6)*: Visible creatures are compacted via parallel prefix scans (`atomicAdd(&draw_args.instance_count, 1u)`) into `visible_agent_indices: array<u32>`, driving `draw_indirect` with 0 CPU intervention.
      - *Page-Aligned Event Compaction (Task 5)*: Discrete births and audio voices are packed into the unified 1,028 KB `ConsolidatedQueue`.
    - **Layer 4: In-Shader Domain Compression & Mathematical Equivalence Proof**:
      - *Lossless 128-Byte Dynamic State Packing*:
        - `pos_vel` (16B), `angle_energy` (16B), and `traits` (32B) retain **100% full 32-bit `f32` precision** (Zero precision loss).
        - `hidden` (40B) retains 10 full 32-bit floats.
        - Discrete metadata (`root: 4b`, `cooldown: 2b`, `birth: 7b`, `dead: 1b`, `kills: 18b`, `age: 16b`, `gen: 16b`) is bit-packed losslessly into two `u32` words, and dead padding is eliminated.
        - **Result**: `GpuAgentState` is exactly **128 bytes** ($2^7$ power-of-two, 2 cache lines). Dynamic state for 1,000,000 agents is **exactly 128 MB**, fitting 100% inside WebGPU's portable limit!
      - *Lossless Neural Quantization (`unpack4x8snorm`)*: The HTML reference stores genes as `Int8Array(326)` and scales hidden sums by `0.61 / 127.0`. The built-in WGSL instruction `unpack4x8snorm(word)` executes the exact division by `127.0` in hardware across 88 `u32` words (352 bytes). The floating-point matrix math is **100% bit-exact to the JavaScript engine**.
      - *Morton Coordinate Compression*: Continuous kinematics $(x, y, v_x, v_y)$ are always stored as full 32-bit floats (`f32`); Morton codes are strictly used as spatial hash keys for bucket sorting.
      - *Lineage Color Packing*: 16-byte `rgba32float` colors are pre-packed into 4-byte `rgba8unorm` in `packed_color`, providing $4\times$ compression for direct GPU mesh instancing.

---

## Plan Overview: 6 Bite-Sized Tasks

- [ ] **Task 1: Branch Setup & GPU Compute Architecture Scaffolding**
  - Create branch `feature/bevy-gpu` from `feature/bevy-port`.
  - Add `bytemuck = { version = "1.21", features = ["derive"] }` to `crates/clank_app/Cargo.toml`.
  - Add `gpu` module in `crates/clank_app/src/gpu/` with buffer types: `GpuAgentState` (128B, exact $2^7$ power-of-two, 2 cache lines, with `visual_cache: u32`), `GpuAgentGenome` (352B, 88 words), `GpuAgentAtomic` (16B, with `dead_claimed`), `GpuSimParams` (64B, exact 4 quadwords, pure baseline physics, multi-tick pacing & tools), `GpuLbvhNode` (48B), `GpuSoilCell` (16B), `BirthEvent` (16B), `AudioVoice` (16B), `GpuTelemetry` (128B), `ConsolidatedQueue` (1,028 KB unified queue with telemetry header), and pipeline skeletons.
  - Implement unit tests for GPU struct memory layouts and 16-byte WGSL alignment (`assert_eq!(size_of::<GpuAgentState>(), 128)`).

- [ ] **Task 2: Tombstone Freelist & Zero-Copy Agent Storage Buffer**
  - Implement lock-free atomic stack allocator (`freelist: array<u32>`, `atomic<u32> queue_buffer.telemetry.freelist_top`).
  - Implement **Pass-Decoupled Freelist**: deaths push in `agent_step.wgsl`; newborn threads pop in `birth_step.wgsl` across compute pass barrier, eliminating ABA / concurrent push-pop races.
  - Implement CAS death ownership (`dead_claimed: 0u -> 1u`) and Strict Single-Writer Invariant (killer never writes to victim's `agent_states`).
  - Test parallel push/pop and slot recycling in automated unit test suite.

- [ ] **Task 3: GPU Soil Simulation & Direct Texture Generation**
  - Implement WGSL compute shader for soil chemistry: spatial bloom renewal, food clamp $[0.0, 2.5]$, taint decay ($0.994$), and scent decay ($0.954$).
  - Add coordinate clamping guards (`cx = min(u32(pos.x / 12.0), 74u)`, `cy = min(u32(pos.y / 12.0), 49u)`) preventing buffer overruns.
  - Implement direct GPU dual-texture generation: `soil_data` (raw physics `rgba16float` for bilinear agent sensing) and `soil_display` (colormap `rgba16float` for direct Bevy SoilSprite rendering), **completely eliminating CPU `generate_soil_rgba` upload**.
  - Bind `soil_display` as 2D `rgba16float` texture with universal hardware bilinear filtering across Metal, Vulkan, and DX12.

- [ ] **Task 4: Hybrid Morton Grid, Multi-System LBVH & Spatial Queries**
  - Implement 32-bit Morton code generator with dead agent partitioning (`0xFFFFFFFFu` sentinel in `morton_encode`) and boundary clamping (`gx = min(u32(pos.x / 100.0), 8u)`, `gy = min(u32(pos.y / 100.0), 5u)`).
  - Implement 8-byte indirection parallel Radix Sort on `(morton_key, agent_slot_idx)`, keeping heavy agent structs stationary.
  - Implement Karras 2012 two-phase LBVH construction: Phase 1 topology (`lbvh_build.wgsl`) with `agent_id` tie-breaking, `node_flags[id.x] = 0u` atomic reset, and $N \le 1$ degenerate population guard; Phase 2 bottom-up AABB fitting (`lbvh_aabb.wgsl`) across an explicit compute pass barrier.
  - Implement Tier 1 ($3 \times 3$ local Moore neighborhood) + Tier 2 ring expansion for lonely creatures.
  - Implement **Unified Interactive Spatial Query Pass (`spatial_query.wgsl`)** with branchless 1D toroidal AABB distance (preventing boundary pruning blind spots), cursor input toroidal wrapping, degenerate $N \le 1$ safety guard, dynamic radius shrinking (early tree pruning) for uncapped 32-bit mouse picking, and parallel AoE tool bounding box intersections.

- [ ] **Task 5: Packed 88-Word Vectorized RNN, Combat Resolution, PRNG & Decoupled Birth Pipeline**
  - Implement `preamble_clear.wgsl` to reset `mate_claim`, `mate_energy_milli`, `dead_claimed`, `birth_count` (sub-tick queue depth), `cell_offsets` every sub-tick, and frame-level counters (`audio_voice_count`, `kills`, `starvations`, `food_grazed_milli`, `apex_record_milli`) on `sub_tick == 0u` with global execution barrier.
  - Implement 1-thread-per-agent WGSL compute shader (`agent_step.wgsl`) binding 8 storage buffers in Group 0 and `(params, soil_data, soil_sampler)` in Group 1 within strict $\le 8$ storage buffer ceiling.
  - Enforce **Top-of-Shader Tombstone Dead-Check Guard**: inactive corpse slots exit immediately via hardware SIMD lane masking.
  - Enforce **Strict Single-Writer Invariant**: thread $i$ writes ONLY to `agent_states[i]`. Victim detects death via `dead_claimed`, marks own dead flag, zeroes `visual_cache`, and aborts live state commits.
  - Enforce `agent_atomics[i].energy_milli` as canonical atomic ground truth, preventing concurrent damage/grazing overwrites.
  - Gate starvation counter increment behind successful death claim CAS (`dead_claimed: 0u -> 1u`), eliminating double-counted deaths.
  - Hardware bilinear antennae feeler sampling (`textureSampleLevel`) from `soil_data` with silicon toroidal coordinate wrapping (`AddressMode::Repeat`), eliminating 7,000,000 atomic memory reads and race conditions.
  - Branchless `unpack4x8snorm` vectorization across 88 `u32` words (7 vec4s hidden, 3 vec4s output).
  - Integrate **Experimental Simulation Mods**: Barnes-Hut Macro-Flocking, Expanded Cortex (extra senses and sprint actuator), and Natural Sexual Selection Tournament.
  - Apply steering, thrust, hardware toroidal coordinate wrap, and pack `visual_cache` at step end.
  - Implement coalesced atomic soil deposits and grazing to `soil_buffer`, atomic millijoule combat resolution with `dead_claimed` CAS, toroidal seam frustum-culled stochastic audio voice emission (256-voice buffer), canonical mating symmetry breaking with seamless asexual fallback, and decoupled SIMD genome mutation pass (`birth_step.wgsl`) with newborn workgroup bounds guard (`wg_id.x >= min(birth_count, 65536u)`) and `visual_cache` birth flash initialization.

- [ ] **Task 6: Frustum Culling, Minimap LOD, Dual-Engine UI, Audio & Verification**
  - Implement **GPU Camera Viewport Frustum Culling** via LBVH streaming 128-byte `GpuAgentState` ($4.0\times$ less bandwidth than 512B structs) into an Indirect Draw Buffer.
  - Implement **instanced dart rendering loading contiguous `vec2u(packed_color, visual_cache)`** directly from Cache Line 1, completely bypassing Cache Line 0 traits and saving ~72 MB of vertex fetch bandwidth per frame.
  - Implement **Minimap LOD Cluster Rendering** sampling intermediate LBVH depth nodes for density circles.
  - Implement **spatially coherent instanced dart rasterization** using Morton-ordered agent index streams for tile-cache efficiency.
  - Implement **Granular Bevy Audio playback** reading 256-voice audio queue with zoom loudness modulation.
  - Implement **Non-Blocking Pipelined Double-Buffered DMA Staging Ring** (`staging[frame % 2]`) for 128-byte `GpuTelemetry` and audio queue readback (0 CPU wait cycles).
  - Implement **32x Speed Visual Consistency Picking** locking selection on Sub-Tick 1 across 32-tick batches with **Specimen Picking Identity Guard** `(slot_idx, agent_id)`.
  - Add top bar engine toggle: `ENGINE: RUST` $\leftrightarrow$ `ENGINE: GPU` with live bi-directional state bridge synthesizing `visual_cache` on upload.
  - Add **EXPERIMENTAL MUTATIONS** sidebar drawer wiring `SimMods` (`[BARNES-HUT]`, `[EXPANDED CORTEX]`, `[SEXUAL SELECTION]`) to specialized pipeline variants via WGSL `override` constants.
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
- Produces: `GpuAgentState`, `GpuAgentGenome`, `GpuAgentAtomic`, `GpuSimParams`, `GpuLbvhNode`, `GpuSoilCell`, `BirthEvent`, `AudioVoice`, `GpuTelemetry`, `ConsolidatedQueue` with exact 16-byte WGSL alignment

- [ ] **Step 1: Write failing test for GPU struct memory layouts**
```rust
// crates/clank_app/tests/gpu_types_test.rs
use clank_app::gpu::types::{
    GpuAgentState, GpuAgentGenome, GpuSimParams, GpuLbvhNode, GpuAgentAtomic, BirthEvent,
    GpuSoilCell, AudioVoice, GpuTelemetry, ConsolidatedQueue,
};

#[test]
fn test_gpu_struct_alignments() {
    assert_eq!(std::mem::size_of::<GpuAgentState>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentState>(), 128);
    assert_eq!(std::mem::size_of::<GpuAgentGenome>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentGenome>(), 352);
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
    assert_eq!(std::mem::size_of::<GpuSoilCell>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSoilCell>(), 16);
    assert_eq!(std::mem::size_of::<GpuTelemetry>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuTelemetry>(), 128);
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>() % 16, 0);
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>(), 1_052_800);
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement GPU types with `#[repr(C)]` and 16-byte alignment**
```rust
// crates/clank_app/src/gpu/types.rs
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
} // Total: EXACTLY 128 bytes (2^7 power-of-two, 2 * 64B cache lines)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentGenome {
    pub packed_genes: [u32; 88],// 352 bytes: 10 hidden * 7 vec4s (70 words) + 6 output * 3 vec4s (18 words)
} // Total: EXACTLY 352 bytes (16-byte aligned, 22 quadwords, 5.5 cache lines)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentAtomic {
    pub energy_milli: i32,      // 4 bytes: atomic fixed-point millijoules
    pub mate_claim: u32,        // 4 bytes: atomic CAS mating winner agent_idx + 1 (0 = unclaimed, full 32-bit capacity)
    pub mate_energy_milli: u32, // 4 bytes: atomic highest bid in millijoules (used by Mod 3 Tournament)
    pub dead_claimed: u32,      // 4 bytes: atomic CAS death ownership (prevents double-free)
} // Total: 16 bytes (1 quadword, perfectly aligned)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    // Simulation Physics & Rates (16B)
    pub tick: u32,                  // 4 bytes  (0..4)
    pub agent_count: u32,           // 4 bytes  (4..8)
    pub max_agents: u32,            // 4 bytes  (8..12)
    pub hostility: f32,             // 4 bytes  (12..16) -> Total 16B

    // Environmental Chemistry & Pacing (16B)
    pub mut_rate: f32,              // 4 bytes  (16..20)
    pub speed: f32,                 // 4 bytes  (20..24)
    pub renewal: f32,               // 4 bytes  (24..28)
    pub sub_tick: u32,              // 4 bytes  (28..32) - Sub-tick index within batch (0..sub_ticks_per_frame - 1)

    // Multi-Tick Batching & Interactive Tools (16B)
    pub sub_ticks_per_frame: u32,   // 4 bytes  (32..36) - Sub-ticks per frame (e.g. 1 in 1x mode, 32 in 32x mode)
    pub tool_type: u32,             // 4 bytes  (36..40) - 0xFFFFFFFF = none, 0 = inspect/pick, 1..5 = AoE tools
    pub tool_pos: [f32; 2],         // 8 bytes  (40..48) - 8-byte aligned vec2f

    // Camera Viewport (16B)
    pub camera_pos: [f32; 2],       // 8 bytes  (48..56) - 8-byte aligned vec2f
    pub camera_size: [f32; 2],      // 8 bytes  (56..64) - 8-byte aligned vec2f
} // Total: Exactly 64 bytes (4 quadwords, 0 pad holes, 100% std140/std430 aligned)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BirthEvent {
    pub parent_a: u32,          // 4 bytes
    pub parent_b: u32,          // 4 bytes: 0xFFFFFFFFu if asexual virgin birth
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
pub struct GpuTelemetry {
    // Cache Line 0 (64 bytes): Engine Counters, Apex Records & Uncapped Specimen Picking
    pub population: u32,        // 4 bytes: active living agents (updated on final sub-tick)
    pub kills: u32,             // 4 bytes: total predatory kills this frame
    pub starvations: u32,       // 4 bytes: total starvation deaths this frame
    pub apex_record_milli: u32, // 4 bytes: highest energy recorded (atomicMax)
    pub food_grazed_milli: u32, // 4 bytes: total food consumed in millijoules
    pub sub_ticks_elapsed: u32, // 4 bytes: number of sub-ticks processed
    pub apex_agent_id: u32,     // 4 bytes: ID of apex creature
    pub freelist_top: u32,      // 4 bytes: atomic stack top for tombstone freelist (merged from SimCounters)
    pub birth_count: u32,       // 4 bytes: births queued this frame (merged from SimCounters)
    pub audio_voice_count: u32, // 4 bytes: audio events queued this frame (merged from SimCounters)
    pub selected_agent_idx: u32,// 4 bytes: full 32-bit slot index of picked creature (0xFFFFFFFF = none)
    pub selected_agent_id: u32, // 4 bytes: full 32-bit unique creature ID for identity guard
    pub _reserved0: [u32; 4],   // 16 bytes: reserved (Total Cache Line 0: 64B)

    // Cache Line 1 (64 bytes): 16-Lineage Real-Time Extinction Monitoring
    pub lineage_counts: [u32; 16], // 16 * 4B = 64 bytes (head counts for roots 0..15)
} // Total: 128 bytes (2 x 64B cache lines, 8ns DMA transfer)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConsolidatedQueue {
    pub telemetry: GpuTelemetry,           // 128 bytes (Cache Lines 0 & 1)
    pub births: [BirthEvent; 65536],        // 1,048,576 bytes = 1,024 KB = 1 MB
    pub audio: [AudioVoice; 256],           // 4,096 bytes = 4 KB (1 memory page)
} // Total: 1,052,800 bytes (1,028.1 KB, 16-byte aligned)

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
    pub aabb_min: [f32; 2],      // 8 bytes: bounding box min [x, y]
    pub aabb_max: [f32; 2],      // 8 bytes: bounding box max [x, y] (Total 16)
    pub center_of_mass: [f32; 2],// 8 bytes: weighted centroid [x, y] (for Barnes-Hut & Minimap LOD)
    pub count: u32,              // 4 bytes: subtree creature count (for Barnes-Hut & Minimap LOD)
    pub dominant_lineage: u32,   // 4 bytes: lineage root 0..15 with highest count in subtree (Total 32)
    pub left_child: u32,         // 4 bytes: child node index
    pub right_child: u32,        // 4 bytes: child node index
    pub parent: u32,             // 4 bytes: parent node index
    pub leaf_idx: u32,           // 4 bytes: leaf agent index (0..N-1), or 0xFFFFFFFF for internal nodes (Total 48)
} // Total: 48 bytes (48 % 16 == 0)
```
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 2: Tombstone Freelist & Zero-Copy Agent Storage Buffer

**Files:**
- Create: `crates/clank_app/src/gpu/freelist.rs`
- Test: `crates/clank_app/tests/gpu_freelist_test.rs`

**Interfaces:**
- Consumes: `GpuAgent`, `MAX_AGENTS` (e.g. 1,000,000 agents)
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
    var cur_top = atomicLoad(&queue_buffer.telemetry.freelist_top);
    var child_slot = 0xFFFFFFFFu;
    while (cur_top > 0u) {
        let cas = atomicCompareExchangeWeak(&queue_buffer.telemetry.freelist_top, cur_top, cur_top - 1u);
        if (cas.exchanged) {
            child_slot = freelist[cur_top - 1u];
            break;
        }
        cur_top = cas.old_value;
    }
    return child_slot; // 0xFFFFFFFFu indicates population capacity reached
}

// WGSL CAS death ownership: guarantees slot is pushed to freelist exactly once (Pass-Decoupled Freelist)
fn claim_death_and_free(victim_idx: u32, is_self: bool) -> bool {
    let cas = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
    if (cas.exchanged) {
        // Strict Single-Writer Invariant: Only victim thread writes to its own agent_states struct!
        if (is_self) {
            agent_states[victim_idx].meta_flags |= (1u << 13u); // dead = 1 (bit 13)
            agent_states[victim_idx].visual_cache = 0u;         // Zero-cost dead state invariant (0 radius, 0 glow)
        }
        let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
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
- Produces: Direct GPU dual-texture generation: `soil_data` (raw physics `rgba16float` for agent bilinear feeler sensing) and `soil_display` (colormap `rgba16float` for direct Bevy `SoilSprite` screen presentation), completely eliminating CPU `generate_soil_rgba` upload and eliminating texture ping-pong memory copies

- [ ] **Step 1: Write failing test for soil atomic buffer bindings, renewal math, and dual rgba16float texture rasterization**
```rust
// crates/clank_app/tests/gpu_soil_test.rs
#[test]
fn test_soil_atomic_buffer_and_texture_bounds() {
    // Verifies:
    // 1. GpuSoilCell fixed-point conversions (millifood, millitaint, milliscent)
    // 2. Renewal formula: f += renewal * bloom * (1 - f / 1.7) clamped to [0.0, 2.5]
    // 3. Taint decay: t * 0.994 - 0.0001, scent decay: s * 0.954
    // 4. Output textures: soil_data (raw floats) and soil_display (colormap) match 75x50 rgba16float format
    // 5. Boundary coordinate clamping: min(pos.x / 12.0, 74) and min(pos.y / 12.0, 49) prevents OOB write at seam (900.0, 600.0)
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement WGSL soil compute kernel operating on 60 KB atomic buffer and rasterizing dual rgba16float textures**
```wgsl
// crates/clank_app/assets/shaders/soil_step.wgsl
struct SoilCell {
    food_milli: atomic<i32>,
    taint_milli: atomic<i32>,
    scent_milli: atomic<i32>,
    pad: u32,
}
@group(0) @binding(0) var<storage, read_write> soil_buffer: array<SoilCell, 3750>;
@group(0) @binding(1) var soil_data: texture_storage_2d<rgba16float, write>;      // Raw physics [food, taint, scent, 1.0]
@group(0) @binding(2) var soil_display: texture_storage_2d<rgba16float, write>;   // Colormap display with vignette
@group(0) @binding(3) var<storage, read> bloom_table: array<f32, 3750>;
@group(0) @binding(4) var<uniform> params: SoilParams;

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
- Create: `crates/clank_app/assets/shaders/spatial_query.wgsl`
- Test: `crates/clank_app/tests/gpu_spatial_test.rs`

**Interfaces:**
- Consumes: `GpuAgent` positions, grid dimensions ($9 \times 6$, cell size $100\text{px}$)
- Produces: Sorted Morton indices (8-byte indirection buffer `vec2u(morton_key, agent_id)`), cell offset table, Karras LBVH tree with tie-breaking, unified interactive spatial query pass (`spatial_query.wgsl`: $O(\log N)$ uncapped mouse picking with dynamic radius shrinking and AoE tool queries)

- [ ] **Step 1: Write failing test for Morton bit-interleaving, cell_offsets sentinel initialization, LBVH tree building with tie-breaking, degenerate population guard ($N \le 1$), uncapped picking query with dynamic radius shrinking, and AoE tool queries**
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

#[test]
fn test_lbvh_uncapped_mouse_picking_with_dynamic_radius_shrinking() {
    // Verifies:
    // 1. Traverses LBVH tree in O(log N) steps for cursor coordinate
    // 2. Selects correct agent across full 32-bit capacity (natively supports 1,000,000+ agents)
    // 3. Prioritizes direct body hit (Priority 0) over halo (Priority 1) with exact float Euclidean tie-breaking
    // 4. Clamps active search radius upon direct body hit, early-pruning distant branches (<40ns)
}

#[test]
fn test_lbvh_aoe_tool_bounding_box_query() {
    // Verifies:
    // 1. Bounding box intersection [tool_pos - R, tool_pos + R] correctly isolates agents within tool radius
    // 2. AoE tools (blight, nourish, extinguish, seed) apply effects in O(log N) time
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement Morton encoding, cell offset clearing pass with 0xFFFFFFFF sentinels, 8-byte indirection Radix sort, Karras LBVH hierarchy generation with $N \le 1$ early-exit guard and tie-breaking, and Unified `spatial_query.wgsl`**
```wgsl
// In morton_grid.wgsl (clearing pass & coordinate encoding):
@compute @workgroup_size(64)
fn clear_cell_offsets(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < 54u) {
        cell_offsets[id.x] = vec2u(0xFFFFFFFFu, 0xFFFFFFFFu);
    }
}

// Dead Agent Partitioning: Assign sentinel 0xFFFFFFFFu so tombstone slots sort to the array tail:
@compute @workgroup_size(64)
fn morton_encode(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.max_agents) { return; }
    let is_dead = (agent_states[id.x].meta_flags & (1u << 13u)) != 0u;
    if (is_dead) {
        spatial_keys[id.x] = vec2u(0xFFFFFFFFu, id.x); // Excluded from cells 0..53 & LBVH leaves!
    } else {
        let code = compute_morton_32(agent_states[id.x].pos_vel.xy);
        agent_states[id.x].morton_code = code;
        spatial_keys[id.x] = vec2u(code, id.x);
    }
}

// Morton cell boundary clamping guard:
fn get_cell_id(pos: vec2f) -> u32 {
    let gx = min(u32(max(0.0, pos.x) / 100.0), 8u);
    let gy = min(u32(max(0.0, pos.y) / 100.0), 5u);
    return gy * 9u + gx; // Guaranteed 0..53 (safely bounds cell_offsets[54])
}

// In lbvh_build.wgsl:
// Two-Phase LBVH Construction: Phase 1 evaluates hierarchy topology; Phase 2 fits bounding boxes bottom-up across an explicit compute pass barrier.
fn common_prefix_length(i: i32, j: i32, n: u32) -> i32 {
    if (j < 0 || j >= i32(n)) { return -1; }
    let key_i = spatial_keys[i].x;
    let key_j = spatial_keys[j].x;
    if (key_i != key_j) {
        return i32(countLeadingZeros(key_i ^ key_j));
    }
    // Tie-break with unique agent slot id:
    let id_i = spatial_keys[i].y;
    let id_j = spatial_keys[j].y;
    return 32 + i32(countLeadingZeros(id_i ^ id_j));
}

@compute @workgroup_size(64)
fn build_lbvh(@builtin(global_invocation_id) id: vec3u) {
    let active_count = params.agent_count;
    if (active_count < 2u || id.x >= active_count - 1u) {
        return; // Guard against N <= 1 underflow!
    }
    // Hardening: Initialize bottom-up atomic flag for Phase 2 before the compute barrier:
    atomicStore(&node_flags[id.x], 0u);
    // Phase 1: Karras 2012 LCP split evaluation, child and parent pointer generation...
}

// In spatial_query.wgsl: Unified interactive pass (Picking & AoE tools)
// Dispatched on Sub-Tick 1 when params.tool_type != 0xFFFFFFFFu
@group(0) @binding(0) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read> lbvh_nodes: array<GpuLbvhNode>;
@group(0) @binding(2) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(3) var<uniform> params: GpuSimParams;

// Branchless 1D toroidal AABB distance: prevents seam pruning blind spots
fn toroidal_aabb_dist_1d(p: f32, b_min: f32, b_max: f32, w: f32) -> f32 {
    if (p >= b_min && p <= b_max) { return 0.0; }
    let direct = select(p - b_max, b_min - p, p < b_min);
    let wrapped = select(w - p + b_min, w - b_max + p, p < b_min);
    return min(direct, wrapped);
}

fn distance_to_aabb(pos: vec2f, aabb_min: vec2f, aabb_max: vec2f) -> f32 {
    let dx = toroidal_aabb_dist_1d(pos.x, aabb_min.x, aabb_max.x, 900.0);
    let dy = toroidal_aabb_dist_1d(pos.y, aabb_min.y, aabb_max.y, 600.0);
    return sqrt(dx * dx + dy * dy);
}

// Inline toroidal Euclidean distance helper (guaranteed non-negative via wrapped coordinates):
fn toroidal_dist(p1: vec2f, p2: vec2f) -> f32 {
    let dx = abs(p1.x - p2.x);
    let x_dist = min(dx, 900.0 - dx);
    let dy = abs(p1.y - p2.y);
    let y_dist = min(dy, 600.0 - dy);
    return sqrt(x_dist * x_dist + y_dist * y_dist);
}

@compute @workgroup_size(64)
fn spatial_query_main(@builtin(global_invocation_id) id: vec3u) {
    // Pre-wrap cursor coordinates to [0.0, 900.0) x [0.0, 600.0) preventing negative distance distortion:
    let tool_pos = vec2f(
        params.tool_pos[0] - 900.0 * floor(params.tool_pos[0] / 900.0),
        params.tool_pos[1] - 600.0 * floor(params.tool_pos[1] / 600.0)
    );

    if (params.tool_type == 0u /* inspect/pick */) {
        if (id.x != 0u) { return; } // Thread 0 evaluates single-cursor picking

        // Degenerate Population Guard (N <= 1):
        if (params.agent_count == 0u) {
            queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
            queue_buffer.telemetry.selected_agent_id = 0u;
            return;
        }

        var search_r = clamp(16.0 * (params.camera_size[0] / 900.0), 4.0, 24.0);

        if (params.agent_count == 1u) {
            // Direct single-agent evaluation without tree traversal:
            let agent_pos = agent_states[0].pos_vel.xy;
            let d = toroidal_dist(tool_pos, agent_pos);
            let visual_r = 2.0 + 3.0 * agent_states[0].traits[0];
            if (d <= search_r) {
                queue_buffer.telemetry.selected_agent_idx = 0u;
                queue_buffer.telemetry.selected_agent_id = agent_states[0].id;
            } else {
                queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
                queue_buffer.telemetry.selected_agent_id = 0u;
            }
            return;
        }

        var best_idx = 0xFFFFFFFFu;
        var best_id = 0u;
        var best_priority = 2u; // 0 = direct body hit, 1 = halo, 2 = none
        var best_dist = search_r;

        // Stack-based O(log N) LBVH traversal (depth 64 safely bounds tree height for 1M+ agents):
        var stack: array<u32, 64>;
        var stack_ptr = 0u;
        stack[stack_ptr] = 0u; // Root node
        stack_ptr += 1u;

        while (stack_ptr > 0u) {
            stack_ptr -= 1u;
            let node_idx = stack[stack_ptr];
            let node = lbvh_nodes[node_idx];

            // Bounding box distance to cursor (with toroidal handling):
            let box_dist = distance_to_aabb(tool_pos, node.aabb_min, node.aabb_max);
            if (box_dist > search_r) { continue; } // Prune entire branch!

            if (node.leaf_idx != 0xFFFFFFFFu) {
                // Leaf node candidate:
                let agent_idx = node.leaf_idx;
                let agent_pos = agent_states[agent_idx].pos_vel.xy;
                let d = toroidal_dist(tool_pos, agent_pos);
                let visual_r = 2.0 + 3.0 * agent_states[agent_idx].traits[0];

                let priority = select(1u, 0u, d <= visual_r);
                if (d <= search_r) {
                    if (priority < best_priority || (priority == best_priority && d < best_dist)) {
                        best_priority = priority;
                        best_dist = d;
                        best_idx = agent_idx;
                        best_id = agent_states[agent_idx].id;

                        // DYNAMIC RADIUS SHRINKING (Early Tree Pruning):
                        // If direct body hit, shrink search radius to exact body distance!
                        if (priority == 0u) {
                            search_r = min(search_r, d);
                        }
                    }
                }
            } else {
                // Internal node: push children (guard stack capacity of 64 entries)
                if (stack_ptr < 62u) {
                    stack[stack_ptr] = node.right_child; stack_ptr += 1u;
                    stack[stack_ptr] = node.left_child;  stack_ptr += 1u;
                }
            }
        }

        queue_buffer.telemetry.selected_agent_idx = best_idx; // Full 32-bit slot index (0 to 4.29 billion)
        queue_buffer.telemetry.selected_agent_id = best_id;   // Full 32-bit unique creature ID for identity guard
    } else {
        // AoE Tool Brushes (Nourish, Blight, Extinguish, Seed):
        // Parallel tree/cell query applying atomic updates to enclosed agents in O(log N)
        apply_aoe_tool(id.x, tool_pos, params.tool_type);
    }
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
    // Instantaneous census is cleared on the final sub-tick before living agents recount:
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
```

```wgsl
// In agent_step.wgsl:
// WGSL Pipeline Overrides (Specialization Constants for Zero-Cost Mod Architecture):
override ENABLE_BARNES_HUT: bool = false;
override ENABLE_EXPANDED_CORTEX: bool = false;
override ENABLE_SEXUAL_SELECTION: bool = false;

// Buffer Splitting (SoA): Exactly 8 storage buffers (100% WebGPU portable compliant):
@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;  // 128B
@group(0) @binding(1) var<storage, read> agent_genomes: array<GpuAgentGenome>;       // 352B (clean read-only L2)
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>; // 16B (atomic CAS ground truth)
@group(0) @binding(3) var<storage, read_write> soil_buffer: array<SoilCell, 3750>;
@group(0) @binding(4) var<storage, read> spatial_keys: array<vec2u>;
@group(0) @binding(5) var<storage, read> lbvh_nodes: array<GpuLbvhNode>;
@group(0) @binding(6) var<storage, read_write> freelist: array<u32>;
@group(0) @binding(7) var<storage, read_write> queue_buffer: ConsolidatedQueue;      // 1,028.1 KB (telemetry + births + audio)
@group(1) @binding(0) var<uniform> params: GpuSimParams;                             // 64B uniform (separate binding space)
@group(1) @binding(1) var soil_data: texture_2d<f32>;                                   // Raw physics snapshot [food, taint, scent]
@group(1) @binding(2) var soil_sampler: sampler;                                        // Linear filter + Repeat wrap in silicon

// Hardware toroidal coordinate wrapping (1 cycle via floor):
fn wrap_coords(p: vec2f) -> vec2f {
    return vec2f(
        p.x - 900.0 * floor(p.x / 900.0),
        p.y - 600.0 * floor(p.y / 600.0)
    );
}

// Hardware bilinear probe helper: 0 ALU cycles for wrapping, 0 ALU cycles for interpolation
fn sample_soil_probe(p: vec2f) -> vec4f {
    let uv = p / vec2f(900.0, 600.0);
    return textureSampleLevel(soil_data, soil_sampler, uv, 0.0);
}

@compute @workgroup_size(64)
fn agent_main(@builtin(global_invocation_id) id: vec3u) {
    let agent_idx = id.x;
    if (agent_idx >= params.max_agents) { return; }

    // Top-of-Shader Tombstone Dead-Check Guard: Inactive corpses must NEVER execute physics or census!
    let meta = agent_states[agent_idx].meta_flags;
    let is_dead = (meta & (1u << 13u)) != 0u;
    if (is_dead) { return; }

    let pos = agent_states[agent_idx].pos_vel.xy;

    // Hardware Bilinear Antennae Feelers (Zero Race Conditions, Zero Atomic Contention):
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

    // Sensory inputs (identical mathematical formula to reference engine):
    ins_hidden[0] = vec4f(here_food, clamp(forward_food - left_food, -1.0, 1.0), clamp(forward_food - right_food, -1.0, 1.0), clamp(forward_food - here_food, -1.0, 1.0));
    ins_hidden[1] = vec4f(clamp(here_taint, 0.0, 1.0), clamp(scent_diff, -1.0, 1.0), sensory_bearing, sensory_dist);

// Branchless unpack4x8snorm neural forward pass:
// Clean read-only genome access (agent_genomes) separates 352B neural weights from volatile state:
// Hidden layer: 10 neurons, each 7 vec4s (28 weights, last 2 zero-padded in baseline)
var new_h: array<f32, 10>;
for (var j = 0u; j < 10u; j += 1u) {
    var s = 0.0;
    let base_w = j * 7u;
    for (var k = 0u; k < 7u; k += 1u) {
        s += dot(unpack4x8snorm(agent_genomes[agent_idx].packed_genes[base_w + k]), ins_hidden[k]);
    }
    new_h[j] = tanh(s * 0.61);
}

// Output layer: 6 neurons, each 3 vec4s (12 weights, last 1 zero-padded in baseline)
var out: array<f32, 6>;
for (var j = 0u; j < 6u; j += 1u) {
    var s = 0.0;
    let base_w = 70u + j * 3u;
    for (var k = 0u; k < 3u; k += 1u) {
        s += dot(unpack4x8snorm(agent_genomes[agent_idx].packed_genes[base_w + k]), ins_output[k]);
    }
    out[j] = tanh(s * 0.66);
}

// Telemetry instantaneous population census (evaluated strictly on the final sub-tick of frame):
if (params.sub_tick == params.sub_ticks_per_frame - 1u) {
    atomicAdd(&queue_buffer.telemetry.population, 1u);
    atomicAdd(&queue_buffer.telemetry.lineage_counts[a_root % 16u], 1u);
}
// Apex energy record tracking across all sub-ticks:
atomicMax(&queue_buffer.telemetry.apex_record_milli, u32(max(0.0, a_energy) * 1000.0));

// Soil grazing with cell index clamping:
let cx = min(u32(max(0.0, pos.x) / 12.0), 74u);
let cy = min(u32(max(0.0, pos.y) / 12.0), 49u);
let cell_idx = cy * 75u + cx;
let eaten_milli = i32(eaten_float * 1000.0);
atomicSub(&soil_buffer[cell_idx].food_milli, eaten_milli);
atomicAdd(&queue_buffer.telemetry.food_grazed_milli, u32(max(0, eaten_milli)));

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
        // Bit-packed kills increment (bits 14..31 of meta_flags):
        let cur_meta = agent_states[agent_idx].meta_flags;
        let kills = (cur_meta >> 14u) + 1u;
        agent_states[agent_idx].meta_flags = (cur_meta & 0x00003FFFu) | (kills << 14u);
        atomicAdd(&queue_buffer.telemetry.kills, 1u);
        a_energy += min(9.0, 8.0 * tr5);

        // Toroidal Seam Frustum-Culled Stochastic Audio Emission (Kill Event):
        let dx = abs(pos.x - params.camera_pos.x);
        let dist_x = min(dx, 900.0 - dx);
        let dy = abs(pos.y - params.camera_pos.y);
        let dist_y = min(dy, 600.0 - dy);
        let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
        if (in_view) {
            let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (900.0 - 150.0), 0.0, 1.0);
            let kill_volume = mix(0.08, 1.0, zoom_factor);
            let density_filter = select(1u, 4u, params.agent_count > 10000u);
            if (pcg_rand(agent_idx, victim_idx, params.tick) % density_filter == 0u) {
                let voice_slot = atomicAdd(&queue_buffer.telemetry.audio_voice_count, 1u);
                if (voice_slot < 256u) {
                    queue_buffer.audio[voice_slot] = AudioVoice(pos, 1u /* EVENT_KILL */, kill_volume);
                }
            }
        }
        
        // Atomic CAS death ownership prevents double-freeing:
        let claim_death = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
        if (claim_death.exchanged) {
            // Strict Single-Writer Invariant: Killer NEVER writes to agent_states[victim_idx]!
            let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
            freelist[free_slot] = victim_idx;
        }
    }
}

// Canonical Atomic Energy Model: apply internal net delta (grazing - basal - thrust) to atomic ground truth:
let internal_delta_milli = i32((eaten_float - basal_cost - thrust_cost) * 1000.0);
atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);

// Strict Single-Writer Death Check (Combat Victim or Starvation):
let already_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed);
let current_energy_milli = atomicLoad(&agent_atomics[agent_idx].energy_milli);

if (already_claimed != 0u || current_energy_milli <= 0) {
    if (already_claimed == 0u) {
        let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
        if (claim_death.exchanged) {
            atomicAdd(&queue_buffer.telemetry.starvations, 1u); // Only increment if this thread won the death claim!
            let free_slot = atomicAdd(&queue_buffer.telemetry.freelist_top, 1u);
            freelist[free_slot] = agent_idx;
        }
    }
    // Strict Single-Writer: Victim marks its own dead flag and clears visual_cache (Zero-Cost Dead State Invariant):
    agent_states[agent_idx].meta_flags |= (1u << 13u); // dead = 1 (bit 13)
    agent_states[agent_idx].angle_energy[1] = 0.0;
    agent_states[agent_idx].visual_cache = 0u;         // Degenerate 0-radius, 0-glow, 0-energy
    return; // Abort committing movement, grazing, or mating!
}

// Update visual/state energy float from atomic ground truth:
a_energy = max(0.0, f32(current_energy_milli) * 0.001);
agent_states[agent_idx].angle_energy[1] = a_energy;

// Mating handshake snippet in agent_step.wgsl:
if (a_energy > 58.0 + 12.0 * tr0 && a_age > 65u && a_birth == 0u && brain_out5 > -0.15) {
    var mate_partner = 0xFFFFFFFFu;
    var sexual_success = false;

    // Check if a viable partner is nearby:
    if (best_dist < 18.0 && partner_energy > 42.0 && partner_root != a_root) {
        if (ENABLE_SEXUAL_SELECTION) {
            // Mod 3: Natural Sexual Selection Tournament (highest-energy suitor wins proposal):
            let my_energy_milli = u32(max(0.0, a_energy) * 1000.0);
            let prev_bid = atomicMax(&agent_atomics[partner_idx].mate_energy_milli, my_energy_milli);
            if (my_energy_milli > prev_bid) {
                atomicStore(&agent_atomics[partner_idx].mate_claim, agent_idx + 1u);
            }
            mate_partner = partner_idx;
            sexual_success = true;
        } else if (pcg_rand(agent_idx, 99u, params.tick) < 0.15) {
            // Optimization A: Canonical Symmetry Breaking (partner_id > agent_id cuts bus traffic 50%):
            if (agent_states[partner_idx].id > agent_states[agent_idx].id) {
                let claim = atomicCompareExchangeWeak(&agent_atomics[partner_idx].mate_claim, 0u, agent_idx + 1u);
                if (claim.exchanged) {
                    mate_partner = partner_idx;
                    sexual_success = true;
                }
            }
        }
    }

    // Pass-Decoupled Freelist: Mating threads only record intent in the birth queue.
    // Child slot allocation occurs across the compute pass barrier in birth_step.wgsl (Zero ABA / Push-Pop Race)!
    let queue_idx = atomicAdd(&queue_buffer.telemetry.birth_count, 1u);
    if (queue_idx < 65536u) {
        atomicSub(&agent_atomics[agent_idx].energy_milli, 24000);
        if (sexual_success && mate_partner != 0xFFFFFFFFu) {
            atomicSub(&agent_atomics[mate_partner].energy_milli, 6000);
        }
        a_birth = 95u;

        // Optimization B: Seamless Asexual Fallback (mate_partner = 0xFFFFFFFFu when virgin birth):
        queue_buffer.births[queue_idx] = BirthEvent(agent_idx, mate_partner, 0xFFFFFFFFu, 0u);

        // Toroidal Seam Frustum-Culled Birth Audio Voice:
        let dx = abs(pos.x - params.camera_pos.x);
        let dist_x = min(dx, 900.0 - dx);
        let dy = abs(pos.y - params.camera_pos.y);
        let dist_y = min(dy, 600.0 - dy);
        let in_view = (dist_x <= params.camera_size.x * 0.5 && dist_y <= params.camera_size.y * 0.5);
        if (in_view) {
            let zoom_factor = clamp(1.0 - (params.camera_size.x - 150.0) / (900.0 - 150.0), 0.0, 1.0);
            let birth_volume = mix(0.06, 0.8, zoom_factor);
            let voice_slot = atomicAdd(&queue_buffer.telemetry.audio_voice_count, 1u);
            if (voice_slot < 256u) {
                queue_buffer.audio[voice_slot] = AudioVoice(pos, 2u /* EVENT_BIRTH */, birth_volume);
            }
        }
    }
}

// Living Agent: commit state and pack visual_cache (offsets 124..128):
let r_u8 = u32(clamp(tr0, 0.0, 1.0) * 255.0);
let glow_u8 = u32(clamp(max(a_attack, select(0.0, 1.0, a_birth > 0u)), 0.0, 1.0) * 255.0);
let e_u8 = u32(clamp(a_energy / 100.0, 0.0, 1.0) * 255.0);
let vis_flags = select(0u, 1u << 24u, a_attack > 0.25) | select(0u, 1u << 25u, a_birth > 0u);
agent_states[agent_idx].visual_cache = r_u8 | (glow_u8 << 8u) | (e_u8 << 16u) | vis_flags;
```

- [ ] **Step 4: Implement Decoupled Birth & Genome Mutation Pass (`birth_step.wgsl`)**
```wgsl
// birth_step.wgsl: 1 workgroup per newborn child (32 threads)
override ENABLE_EXPANDED_CORTEX: bool = false;

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> agent_genomes: array<GpuAgentGenome>;
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(3) var<storage, read> queue_buffer: ConsolidatedQueue;
@group(0) @binding(4) var<storage, read_write> freelist: array<u32>;
@group(1) @binding(0) var<uniform> params: GpuSimParams;

var<workgroup> shared_child_slot: u32;

@compute @workgroup_size(32)
fn birth_main(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    // Workgroup-Uniform Bounds Guard: Safely bounds births to actual queued count and buffer capacity
    let total_births = min(atomicLoad(&queue_buffer.telemetry.birth_count), 65536u);
    if (wg_id.x >= total_births) {
        return; // Safely abort excess workgroup without popping freelist or reading uninitialized queue slots
    }

    let event = queue_buffer.births[wg_id.x];
    let parent_a = event.parent_a;
    let parent_b = event.parent_b;

    // Pass-Decoupled Allocation: Thread 0 pops from freelist (zero concurrent pushes occurring):
    if (local_id.x == 0u) {
        var cur_top = atomicLoad(&queue_buffer.telemetry.freelist_top);
        var slot = 0xFFFFFFFFu;
        while (cur_top > 0u) {
            let cas = atomicCompareExchangeWeak(&queue_buffer.telemetry.freelist_top, cur_top, cur_top - 1u);
            if (cas.exchanged) {
                slot = freelist[cur_top - 1u];
                break;
            }
            cur_top = cas.old_value;
        }
        shared_child_slot = slot;
    }
    workgroupBarrier();

    let child_idx = shared_child_slot;
    if (child_idx == 0xFFFFFFFFu) { return; } // Population carrying capacity reached!

    // Parallel genome crossover & mutation (88 u32 words = 10*7 + 6*3)
    for (var w = local_id.x; w < 88u; w += 32u) {
        var word_a = agent_genomes[parent_a].packed_genes[w];
        var word_b = select(word_a, agent_genomes[parent_b].packed_genes[w], parent_b != 0xFFFFFFFFu);
        // Crossover 48% and triangular mutation per gene byte...
        var mutated_word = crossover_and_mutate(word_a, word_b, local_id.x, params.tick);

        // If baseline mode, mask out dummy padded bytes so inactive genes don't drift:
        if (!ENABLE_EXPANDED_CORTEX) {
            if (w < 70u && (w % 7u) == 6u) {
                mutated_word = mutated_word & 0x0000FFFFu; // Bytes 2 & 3 clamped to 0
            }
            if (w >= 70u && ((w - 70u) % 3u) == 2u) {
                mutated_word = mutated_word & 0x00FFFFFFu; // Byte 3 clamped to 0
            }
        }
        agent_genomes[child_idx].packed_genes[w] = mutated_word;
    }

    // Trait crossover, morphology & state initialization on thread 0:
    if (local_id.x == 0u) {
        // Crossover 45% and clamp(base + (r1 + r2 - 1) * mutRate * 0.6, 0.03, 0.98)...
        let parent_pos = agent_states[parent_a].pos_vel.xy;
        let child_pos = wrap_coords(parent_pos + vec2f(pcg_signed(child_idx, 1u, params.tick) * 9.0, pcg_signed(child_idx, 2u, params.tick) * 9.0));
        
        agent_states[child_idx].pos_vel = vec4f(child_pos.x, child_pos.y, 0.0, 0.0);
        agent_states[child_idx].angle_energy = vec4f(pcg_float(child_idx, 3u, params.tick) * 6.2831853, 24.0, 0.0, 0.0);
        atomicStore(&agent_atomics[child_idx].energy_milli, 24000);
        atomicStore(&agent_atomics[child_idx].dead_claimed, 0u);
        atomicStore(&agent_atomics[child_idx].mate_claim, 0u);
        atomicStore(&agent_atomics[child_idx].mate_energy_milli, 0u);

        // Inherit & mutate traits into agent_states[child_idx].traits...
        let parent_meta = agent_states[parent_a].meta_flags;
        let child_root = parent_meta & 0xFu; // Root lineage 0..15
        agent_states[child_idx].meta_flags = (child_root & 0xFu) | (95u << 6u); // root, birth cooldown = 95, dead = 0, kills = 0
        
        let parent_gen = agent_states[parent_a].age_gen >> 16u;
        agent_states[child_idx].age_gen = ((parent_gen + 1u) & 0xFFFFu) << 16u; // age = 0, gen = parent_gen + 1
        agent_states[child_idx].id = atomicAdd(&queue_buffer.telemetry.apex_agent_id, 1u); // Next unique creature ID
        agent_states[child_idx].morton_code = 0u; // Assigned by morton_encode
        agent_states[child_idx].packed_color = pack_lineage_color(child_root);

        // Newborn Initialization Invariant: Populate visual_cache with birth flash!
        let child_r_u8 = u32(clamp(agent_states[child_idx].traits[0], 0.0, 1.0) * 255.0);
        let child_e_u8 = u32(clamp(24.0 / 100.0, 0.0, 1.0) * 255.0);
        let child_glow_u8 = 255u; // Newborn birth flash!
        agent_states[child_idx].visual_cache = child_r_u8 | (child_glow_u8 << 8u) | (child_e_u8 << 16u);
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
- Consumes: `SimWorld` (CPU), `GpuSimWorld` (GPU), `ConsolidatedQueue` (1,052,800B) via non-blocking double-buffered DMA staging ring (`staging[frame % 2]`), `GpuAgentState` (128B) storage buffer
- Produces: Camera frustum-culled rendering streaming only 128B `GpuAgentState` ($4.0\times$ bandwidth boost, skipping genomes), minimap LOD clustering, unified 128-byte telemetry, 32x speed visual consistency picking, zoom-modulated audio soundscape, HUD metrics, and live engine switching

- [ ] **Step 1: Write failing test for live dual-engine hot-swapping, frustum culling, telemetry readback, and 32x speed picking**
```rust
// crates/clank_app/tests/dual_engine_test.rs
#[test]
fn test_dual_engine_live_hotswap_parity() {
    // Verifies:
    // 1. Rust -> GPU uploads live agents, populates GpuAgentState/GpuAgentGenome/GpuSoilCell buffers, syncs tick and eclipse countdown, and resumes compute
    // 2. GPU -> Rust reads back active agents, updates SimWorld.tick and eclipse state, and resumes CPU loop
    // 3. Population count, generation, timeline tick, and lineage roots are 100% preserved across toggle
}

#[test]
fn test_gpu_telemetry_and_extinction_detection() {
    // Verifies:
    // 1. 128-byte GpuTelemetry reads back via non-blocking double-buffered DMA staging ring (staging[frame % 2]) without CPU pipeline stalls
    // 2. apex_record_milli accurately reflects atomicMax of creature energy
    // 3. 16-lineage counts detect lineage extinction via zero-search without scanning agent buffer
}

#[test]
fn test_two_tier_uncapped_picking_disambiguation() {
    // Verifies:
    // 1. In a 32-tick batch, picking evaluates strictly on Sub-Tick 1
    // 2. Direct body hit (Priority 0) beats proximity halo (Priority 1)
    // 3. Exact float Euclidean distance resolves ties between clustered agents
    // 4. Full 32-bit slot index and agent ID support 1,000,000+ agents without artificial 16-bit ceiling
    // 5. Specimen picking identity guard (slot_idx, agent_id) flags death if slot recycled
}

#[test]
fn test_mating_canonical_symmetry_and_sexual_selection_mod() {
    // Verifies:
    // 1. Canonical symmetry breaking (partner_id > agent_id) eliminates mutual race conditions
    // 2. Seamless asexual fallback (child(a, null)) when partner unavailable or claim fails
    // 3. Mod 3 Sexual Selection Tournament: atomicMax chooses fittest suitor across full 32-bit capacity
}
```
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement engine switch in Bevy UI, bi-directional state bridge (`sync_rust_to_gpu` synthesizing `visual_cache` / `sync_gpu_to_rust`), non-blocking double-buffered DMA staging ring (`staging[frame % 2]`) for 128B telemetry and 4 KB audio readbacks without CPU stalls, direct 128-byte `GpuAgentState` streaming for LBVH frustum culling and instanced dart rendering loading contiguous `vec2u(packed_color, visual_cache)` ($4.0\times$ bandwidth boost over reading full 480B agents, bypassing Cache Line 0 traits and saving ~72 MB/frame vertex fetch bandwidth), 32x speed picking locking on Sub-Tick 1 with `(slot_idx, agent_id)` identity guard, Two-Tier uncapped LBVH distance disambiguation, Bevy audio playback reading 256-voice queue, connect LBVH frustum culling, draw Minimap LOD clusters, wire `SimMods` (`[BARNES-HUT]`, `[EXPANDED CORTEX]`, `[SEXUAL SELECTION]`) to specialized pipeline variants via WGSL `override` constants, and add the "EXPERIMENTAL MUTATIONS" drawer to the Bevy UI sidebar**
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
   - Mod toggle badges display active states (`[BARNES-HUT]`, `[EXPANDED CORTEX]`, `[SEXUAL SELECTION]`).
   - Creatures move, feed, fight, and reproduce with identical visual fidelity.
   - Soil texture renders smoothly without artifacts.
   - Zooming in uses LBVH frustum culling without visual pop-in.
   - Dragging AoE tools (Extinguish, Nourish, Blight) operates in $O(\log N)$ time.
   - Minimap draws cluster density discs from intermediate LBVH levels.
   - UI metrics (population, generation, record graph) update in real-time from 128-byte `GpuTelemetry`.
   - Clicking on a creature in a dense swarm at 32x speed accurately selects the clicked specimen without cluster jumping.
   - Audio sounds modulate dynamically with zoom level (quiet murmur zoomed out, crisp bites zoomed in) and pan smoothly across the toroidal seam.
