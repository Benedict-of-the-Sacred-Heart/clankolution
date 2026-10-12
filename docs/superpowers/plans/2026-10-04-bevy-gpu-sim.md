# GPU Compute Simulation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline. Per `AGENTS.md` §5, never dispatch subagents to generate code. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-throughput GPU compute simulation engine (`ENGINE: GPU`) running entirely in WGSL on a dedicated branch (`feature/bevy-gpu`), featuring zero-copy Tombstone Freelist buffers, a hybrid Morton Grid (the planned Linear Bounding Volume Hierarchy was dropped on 2026-10-09), packed 326-weight quantized RNN forward passes, and reactive GPU soil chemistry—coexisting alongside the deterministic CPU reference engine (`ENGINE: RUST`). The GPU engine targets **statistical behavioral parity** with the reference (see *Parity target* in the Verification Plan), not bit-exact parity.

**Architecture:** A multi-pass Bevy compute pipeline operating over GPU storage buffers and 2D textures. Agents never reallocate or shift in VRAM thanks to a lock-free Tombstone Freelist. Memory architecture utilizes high-performance **Buffer Splitting (Struct-of-Arrays)**: separating dynamic simulation state (`GpuAgentState`, 128 bytes, 2 cache lines, $2^7$ power-of-two) from static neural genomes (`GpuAgentGenome`, 352 bytes), slashing memory bus traffic by **$4.0\times$** on all rendering and spatial passes. Spatial lookups use a hybrid Morton Grid (Tier 1: $O(1)$ local Moore cells, Tier 2: a scan of the live keys). A Karras Linear Bounding Volume Hierarchy (LBVH) was planned to power these subsystems, and was removed on 2026-10-09 (section 3): 1) Long-range sensory raycasting (dropped: the grid tiers cover sensing), 2) Camera viewport frustum culling (dropped: the per-slot cull stays, see Subsystem 2), 3) Mouse picking and interactive AoE brush tools (picking reads the grid; the AoE tools check each creature, see Subsystem 3), and 4) Hierarchical minimap LOD cluster rendering (deferred: the minimap was removed from the UI on 2026-10-06 because the whole 900×600 world is on screen). The dual-engine architecture preserves bit-for-bit CPU determinism when toggled to `ENGINE: RUST` while unlocking massive agent scaling when toggled to `ENGINE: GPU`. The GPU engine is parallel and f32, so it is neither bit-exact nor run-to-run deterministic against the CPU engine; it must match the reference within the tolerances defined in the Verification Plan.

**Tech Stack:** Rust, Bevy 0.19 (`bevy_render::render_resource`, `wgpu`), WGSL (WebGPU Shading Language), Metal/Vulkan compute pipelines, `rkyv`, `image`.

**Branching Strategy:** Work will proceed exclusively on a new branch: `feature/bevy-gpu` branched directly from `feature/bevy-port`. The base branch `feature/bevy-port` will remain clean and stable until `feature/bevy-gpu` is verified, tested, and visually confirmed.

---

## Explicit GPU Optimizations Covered in this Plan

1. **Atomic Soil Simulation State & GPU Colormap**:
   - **True Simulation State in L2 Cache**: Soil chemistry (food, taint, scent) is maintained in a dedicated 60 KB atomic storage buffer (`GpuSoilCell` for all $75 \times 50 = 3,750$ cells, $3,750 \times 16\text{ bytes} = 60\text{ KB}$). Fits 100% inside GPU L1/L2 cache.
   - **Fixed-Point Micro-Units**: Each field is an `i32` at scale `SOIL_FIXED_SCALE` = 1,000,000. Bloom regrowth adds as little as ~1e-5 food per tick; the earlier milli-units (`i32(f * 1000)`) truncated that to zero every tick, so regrowth stalled (measured: 2,861 total food after 600 empty ticks vs 3,944 in the reference).
   - **Coalesced Atomic Creature Writes**: Creatures in `agent_step.wgsl` atomically subtract grazed food (`atomicSub(&food_micro)`) and add scent/carcass taint directly to this 60 KB buffer.
   - **Point Reads for Sensing**: Creatures read single cells with `atomicLoad`, exactly as the reference reads `food[k]`. Bilinear sampling of a soil texture would change behavior, so there is no `soil_data` texture.
   - **GPU Colormap**: `soil_step.wgsl` runs at the start of each sub-tick, applies regrowth and decay, writes back to `soil_buffer`, and writes the colormap `soil_display` as `rgba8unorm` with the CPU colormap's truncation. Task 8 copies it GPU→GPU into the soil sprite's texture, with no CPU upload.

2. **Sensory Raycasting & Packed Neural Net**:
   - **Antennae Feelers**: Point `atomicLoad` reads of `soil_buffer` at 4 probe locations (`here`, `forward`, `left`, `right`), wrapped toroidally and clamped into the grid, as in the reference.
   - **2-Tier Hybrid Spatial Raycasting** (must reproduce the reference `near()`: nearest by squared toroidal distance with strict `<`, skipping self; `density` counts neighbours with `d² < 10000`):
     - Tier 1: direct lookup in the uniform grid ($3 \times 3$ cells of 100 px around the creature). Every creature within 100 px is in these cells, so density always comes from Tier 1. No per-cell cap.
     - Tier 2: when the nearest Tier 1 squared distance is ≥ 10000 (nothing within 100 px), scan all live creatures (every sorted live key, not slots `< agent_count`). This equals the reference's global scan.
   - **Linear Genome Packing**: The 326 `i8` genes are packed in order, 4 per `u32` (82 used words of 88; gene `p` is byte `p % 4` of word `p / 4`, sign-extended). The forward pass follows the reference `brain()` order exactly (`s += w[p++] * in[i] / 127`, then `tanh(s * .61)` hidden and `tanh(s * .66)` output). The earlier "7 vec4s per hidden neuron / 3 per output" padded layout is withdrawn. The f32 result is not bit-exact to the f64 reference (measured median output error 6e-8).

3. **Multi-System Linear Bounding Volume Hierarchy (LBVH)**:
   - **Removed on 2026-10-09.** Only picking still used the tree, yet it was rebuilt every sub-tick: two more dispatches over every slot, and a walk whose unbounded first version froze the development machine. Picking now reads the grid (Subsystem 3), so `lbvh_build.wgsl`, `lbvh_aabb.wgsl`, `gpu/lbvh.rs`, `GpuLbvhNode` and the node buffers are gone. The bullets below are the original design, kept for the record.
   - Built in parallel using Karras 2012 LCP splits on sorted Morton codes ($\approx 0.3\text{ ms}$).
   - **Degenerate Population Guard ($N \le 1$)**: When population collapses or is $\le 1$, builder early-exits gracefully (`if (active_count < 2u) return;`), guarding against unsigned underflow in $N-2$ internal node bounds and falling back to direct Tier-1 lookups.
   - **Strict Key Uniqueness & Tie-Breaking**: If two agents have identical Morton keys (e.g. sharing positions), tie-breaking uses `32u + countLeadingZeros(agent_id_a ^ agent_id_b)` to guarantee distinct keys, preventing infinite loops or zero-length split branches.
   - ~~**Subsystem 1 (Sensory Raycasting)**: $O(\log N)$ hierarchical traversal for long-range homing vectors.~~ Dropped on 2026-10-09. Sensing uses the grid tiers in section 2 (Tier 1 cells, Tier 2 scan of the live keys), which reproduce the reference `near()`.
   - ~~**Subsystem 2 (Frustum Culling)**: Discards entire off-screen subtrees against the camera viewport $[x_{\min}, y_{\min}] \times [x_{\max}, y_{\max}]$, feeding visible agents into an indirect draw buffer.~~ Dropped on 2026-10-09. `creature_render.wgsl` `cull_main` checks every slot, one thread each. At 65,000 creatures on the FirePro D500 a cull costs 0.40 ms in full view and 0.27 ms zoomed to 150×100. Of that, 0.21–0.23 ms is fixed pass cost that remains with nothing to check. So a tree could save at most the checking itself: nothing in full view, where every creature is visible, and 0.06 ms zoomed. A tree cull needs a second pass and an indirect-args copy, which likely cost more. It would also draw from a tree that is one sub-tick stale and misses the final sub-tick's newborns, and the Rust engine mode has no tree at all.
   - **Subsystem 3 (Unified Interactive Spatial Queries - `spatial_query.wgsl`)**:
     - *Mouse Picking (`tool_type == 0`)*: As the Rust engine's `rendering::find_agent_at_position`, so both engines pick the same creature: the nearest live creature whose body radius (`2.3 + 4.5·size`) plus `tool_radius` (25 px) reaches the cursor, by plain distance, the lower id on a tie. That reach is under one 100 px grid cell, so one thread checks the cursor's 3×3 cells (each count clamped to `max_agents`), read from the grid the sort built this sub-tick. Replaced on 2026-10-09: the LBVH walk with a zoom-dependent 4–24 px radius and body-hit priority.
     - *AoE Tool Brushes (`tool_type >= 1`)*: `extinguish_at`, `blight_at`, `nourish_at` and `seed_at` run one thread per slot, each creature checking its own distance to the brush. Dropped on 2026-10-09: the $O(\log N)$ tree walk. A brush runs once per frame on sub-tick 0, and the per-slot check costs no more than the per-slot creature cull (Subsystem 2).
   - **Subsystem 4 (Minimap LOD Clustering)** *(deferred with the minimap)*: Draws aggregated cluster discs from intermediate tree depths (e.g. depth 5–6) directly onto the radar minimap using cached `dominant_lineage` and `center_of_mass`.

4. **Cross-Cutting Morton Code Optimizations (Engine-Wide)**:
   - **Grid Cell Table Sentinels**: The 54-cell table of `(start, count)` pairs is stored in the `spatial_keys` buffer after the `N` sorted keys (`[0, N)` sorted keys, `[N, N + 54)` cell table, `[N + 54, 2N + 54)` `cell_slots`, `N = params.max_agents`). The grid scan rewrites all 54 entries every sub-tick, writing empty cells as `vec2u(0xFFFFFFFFu, 0xFFFFFFFFu)`, so nothing else clears it. In `agent_step.wgsl`, neighbor searches check `if (range.x != 0xFFFFFFFFu)`, instantly skipping empty cells with zero warp divergence. Morton order does not keep a 100 px grid cell's creatures contiguous (cell edges are not power-of-two Morton boundaries), so the table is filled by a count pass, a 54-entry prefix sum, and a scatter of live slots into `cell_slots`; each `(start, count)` indexes `cell_slots`. There is no per-cell cap (the per-cell linked list capped at 128 is replaced).
   - **Lightweight 8-Byte Indirection Radix Sort**: Parallel radix sort operates strictly on `array<vec2u>` storing `(morton_key, agent_slot_idx)`. The 128-byte `GpuAgentState` structs remain completely stationary in VRAM, eliminating gigabytes/sec of memory bus saturation. It is a multi-pass LSD sort (4 stable passes of 8 bits over the Morton code, keys starting in slot order, so equal codes stay in slot order). Single-pass decoupled look-back scans can hang on Metal and are not used. Each pass is two dispatches: a count of each 256-key workgroup's digits, and a scatter whose workgroups work out their own offsets from those counts (no separate scan dispatches), ranking a key among its workgroup's earlier keys with the same digit by counting bits in a per-digit 256-bit mask (a loop over all earlier threads cost about 60 µs per dispatch on the FirePro D500). Pass 0 also builds the grid cell table from the unsorted keys and counts `live_count` (reset by the preamble). The sort pipelines skip zero-initialising workgroup memory, which cost about 120 µs per dispatch; every workgroup variable is written before it is read. Measured on the D500: about 0.2 ms of GPU time per sort at 340 slots and 0.33 ms at 10,000. Morton codes quantize with `floor` in both WGSL and `spatial_index::compute_morton_32_with_size` (the Rust helper currently uses `.round()`; they must agree).
   - ~~**Sub-Microsecond Mouse Picking ($O(\log N)$)**: Fast binary search on sorted Morton keys to select creatures for the specimen card.~~ Dropped on 2026-10-09: picking is the grid lookup in Subsystem 3.
   - **Coalesced Atomic Soil Writes**: Threads scheduled in Morton order write to contiguous L1/L2 cache lines for food/taint/scent deposits, cutting bus contention. The agent pass's thread `i` takes slot `spatial_keys[i].y` (the sort's first `max_agents` keys hold every slot once, live creatures first by position), so neighbouring threads also read neighbouring creatures and cell lists. Measured on the D500 (32 sub-ticks from reference founders): 65,000 creatures 510 → 128 ms per tick, 16,000 creatures 16.4 → 11.9 ms.
   - **Spatially Coherent Dart Rendering**: Morton-ordered instance streams maximize GPU tile-cache hit rates on Metal and tile-based rasterizers.

5. **Hardware Toroidal Coordinate Wrapping**:
   - In WGSL, coordinate wrapping uses:
     `pos.x - 900.0 * floor(pos.x / 900.0)` and `pos.y - 600.0 * floor(pos.y / 600.0)`
   - Evaluates in a single hardware `floor` and `fma` instruction, handling any negative coordinate seamlessly without edge pop-in.

6. **Birth / Compaction Freedoms & CAS Freelist Protection**:
   - **Tombstone Freelist & CAS Underflow Guard**: Fixed static `MAX_AGENTS` storage buffer. Dead creatures are marked with tombstone flags (`dead = 1`) and their slot indices are recycled onto an atomic stack (`atomicAdd` on deallocation). Allocation uses an atomic Compare-and-Swap (CAS) pop loop: checks `freelist_top > 0u` before popping, safely preventing unsigned 32-bit underflow (`4,294,967,295`) when population reaches carrying capacity. **Zero memory shifting or array compaction across frames.**
   - **Dedicated Preamble Clearing Pass (`preamble_clear.wgsl`)**: Dispatched immediately before simulation stepping to eliminate cross-workgroup race conditions. Resets `mate_claim = 0u`, `mate_energy_milli = 0u`, `queue_buffer.telemetry.birth_count = 0u`, `queue_buffer.telemetry.audio_voice_count = 0u`, `queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu`, `queue_buffer.telemetry.selected_agent_id = 0u`, and `queue_buffer.telemetry.live_count = 0u` (the spatial sort's first pass counts the live keys into it). (It does not touch the grid cell table, which the spatial sort rebuilds.) Guarantees a global GPU execution and memory barrier across all workgroups before simulation stepping begins.
   - **Atomic CAS Double-Free Protection**: To prevent starvation and predation from double-freeing the same slot in the same tick, death ownership is acquired via CAS on `dead_claimed: atomic<u32>` ($0 \to 1$). Exactly one thread succeeds, sets `dead = 1`, recycles the slot to `freelist`, and is the only thread that counts the death (`kills` or `starvations`, and `total_deaths`). `dead_claimed` is persistent: it is cleared only when `birth_step.wgsl` reuses the slot, never by the preamble.
   - **Mating Without Exclusive Claims** *(replaces "Canonical Mating Symmetry Breaking")*: The reference lets several suitors mate with the same creature in one tick (`child(a, mate)` with `mate = best.b`, each costing the mate 6 energy). A suitor therefore takes its mate with an `atomicSub` of 6000 from the mate's `energy_milli`, with no claim. The earlier `partner_id > agent_id` rule halved the sexual birth rate relative to the reference and is withdrawn.
   - **Seamless Asexual Fallback (Optimization B)**: In strict parity with the HTML reference (`clankolution.html#L1439-L1444`), if a sexual mate is absent, out of range, or rejected, the creature smoothly falls back to asexual virgin reproduction (`child(a, null)` / `partner_idx = 0xFFFFFFFFu`). This guarantees the reproductive cycle is never lost to lock contention.

7. **Predation & Combat Resolution (Atomic Fixed-Point & Decisive Attribution)**:
   - **L2-Cached Atomic Damage Buffer**: `energy_milli: atomic<i32>` resides in an isolated 16-byte aligned buffer (`GpuAgentAtomic`), serving as the single continuous atomic ground truth for energy. Every change is atomic; the upper clamp is `atomicMin(&energy_milli, 110000)`. Nothing plain-stores energy after other threads may have written it (a final load-clamp-store overwrote concurrent damage and mating costs).
   - **Exact Formula Parity**: Damage evaluated via `(0.5 + attack * 2.2) * hostility * (0.8 + tr0) * (1.0 - 0.65 * victim_tr3)`.
   - **Decisive Killer Attribution**: Threads perform `atomicSub(&agent_atomics[victim].energy_milli, dmg_milli)`. The exact thread crossing the zero threshold (`old > 0 && old <= dmg`) is attributed the kill, receives the siphon energy and kill bounty (`min(9.0, 8.0 * tr5)`), sets victim tombstone, and recycles the slot to the freelist via the `dead_claimed` CAS gate.

8. **Buffer Splitting & ConsolidatedQueue Telemetry Header (Strict $\le 8$ Storage Buffer Limit)**:
   - **Struct-of-Arrays (SoA) Splitting**: Separates the dynamic simulation state (`GpuAgentState`, 128 bytes, 2 cache lines, $2^7$ power-of-two) from static neural weights (`GpuAgentGenome`, 352 bytes). Eliminates L2 cache thrashing during simulation stepping and cuts memory bus traffic by **$4.0\times$** on all rendering, culling, and spatial passes.
   - **Consolidated Frame Output Queue (`ConsolidatedQueue`)**: Embeds `telemetry: GpuTelemetry` (128 bytes) directly into the header of `ConsolidatedQueue` alongside `births: array<BirthEvent, 65536>` ($1.0\text{ MB}$) and `audio: array<AudioVoice, 256>` ($4\text{ KB}$).
   - **Strict $\le 8$ Storage Buffer Compliance**: Keeps storage buffer bindings in `agent_step.wgsl` at **7** (`agent_states`, `agent_genomes`, `agent_atomics`, `soil_buffer`, `spatial_keys` (sorted keys, grid cell table and `cell_slots` in one buffer), `freelist` (the stack in `[0, max_agents)` and the pending-free list in `[max_agents, 2·max_agents)`), and `queue_buffer`), ensuring 100% strict cross-platform compatibility across Metal, Vulkan, DX12, and WebGPU.

9. **Stateless Counter-Based PRNG & Triangular Distribution**:
   - **PCG 3D Hash in Hardware Registers**: Evaluates $\text{rand} = \text{pcg3d}(\text{vec3u}(\text{id}, \text{stream}, \text{tick}))$ in 2 nanoseconds directly in ALU registers. Requires $0$ bytes of persistent PRNG state, $0$ atomic locks, and guarantees bit-for-bit reproducibility across all GPU architectures. Every use draws from its own stream constant (crossover, mutation, trait crossover, trait mutation, child position, child angle, soil spawn, mate roll, audio filter), so no two draws for the same gene or creature are correlated. The current shaders use a linear `pcg_hash(id, stream, tick)` mix whose streams collide (the crossover roll equals the mutation roll for the same gene); it is replaced by `pcg3d`.
   - **Triangular Distribution**: Computes $\mathcal{T}(-1, 0, 1)$ via $r_1 + r_2 - 1.0$ matching the exact HTML mutation distribution.

10. **Decoupled Birth Queue & Parallel Genome Mutation**:
    - **Decoupled Birth Pass**: Mating agents push a 16-byte `BirthEvent` to an atomic append queue. The main agent compute shader has **zero warp divergence** from gene loops.
    - **Atomic Capacity Reservation**: A birth first reserves a slot by decrementing an available-slot counter (seeded each sub-tick from `freelist_top`, refused at 0). Only a birth that reserves a slot charges energy (parent 24, mate 6) and sets the 95-tick countdown. The cap therefore holds across every sub-tick of a frame, and no parent pays for a birth that does not happen. (The reference checks `agents.length < CAP` where `agents` still includes creatures that died this tick.)
    - **Whole-Queue Processing**: The birth pass is dispatched indirectly with one workgroup per queued birth, not a fixed 64 workgroups.
    - **Parallel Genome SIMD Mutation (`birth_step.wgsl`)**: Dedicated workgroups process newborn agents, performing 82-word parallel crossover (48% parent crossover) and triangular mutation $(r_1 + r_2 - 1.0) \times 100 \times \text{mutRate}$ clamped to $[-127, 127]$, plus morphological trait mutation $(r_1 + r_2 - 1.0) \times \text{mutRate} \times 0.6$ clamped to $[0.03, 0.98]$.

11. **Experimental simulation mods: removed (2026-10-07).** Barnes-Hut flocking, Expanded Cortex and the Sexual Selection tournament (and the Lineage Genome Bank concept) are out of the game until the base simulation matches the reference. There are no `override` constants, pipeline variants, mod toggles or mod drawer; each pass has one pipeline.

12. **Seamless Dual-Engine State Hot-Swapping**:
    - Bi-directional bridge between Bevy ECS `SimWorld` and the GPU storage buffers (`GpuAgentState`, `GpuAgentGenome`, `GpuAgentAtomic`, `GpuSoilCell`).
    - Switching `[ENGINE: RUST]` $\leftrightarrow$ `[ENGINE: GPU]` dynamically transfers live agents, soil, the simulation tick and the eclipse countdown, allowing users to hot-swap engines live mid-run without resetting the simulation timeline or losing creature lineages. Lineage roots cross the bridge at full width (the 72 founding lineages stay 72).

13. **GPU Particle Sparks & Mesh Instancing**:
    - 16,384+ drifting sparks updated in an isolated GPU compute pass (velocity decay, alpha fading).
    - Instanced creature dart mesh generation on the GPU.

14. **Buffer Splitting & 128-Byte Lossless State Layout ($2^7$ Power-of-Two)**:
    - **Operation 1: Struct-of-Arrays (SoA) Splitting**: High-frequency dynamic simulation state (`GpuAgentState`, 128 bytes) is separated from static neural genomes (`GpuAgentGenome`, 352 bytes). All non-simulation passes (frustum culling, dart rendering, Morton indexing, LBVH building, mouse picking) stream only 128 bytes per agent, cutting memory bus traffic by **$4.0\times$** ($128\text{ MB}$ vs $512\text{ MB}$ for 1,000,000 agents).
    - **Operation 2: Neural Forward Pass (`agent_step.wgsl`)**: Reads clean read-only `agent_genomes` from L2 cache while writing dynamic kinematics to `agent_states`. Only the 326 reference genes are read.
    - **Operation 3: Genome Mutation Masking (`birth_step.wgsl`)**: The spare bytes after gene 325 (the last two bytes of word 81 and words 82..87) stay zero, preventing silent random drift in unused weights.
    - **Operation 4: Dual-Engine Live Hot-Swapping (`sync_rust_to_gpu` & `sync_gpu_to_rust`)**: Exact bit-level pack/unpack maps 326 sequential `i8` genes into the first 82 `u32` words of `agent_genomes` (linear order, 4 genes per word) with zero loss or drift.
    - **Operation 5: Savefile Backward Compatibility**: Saving always extracts the canonical 326 `i8` genes into `AgentData`, ensuring all `.clank` and `.json` files are 100% cross-compatible between CPU and GPU engines.
    - **Operation 6: Precomputed Visual Render Cache (`visual_cache: u32`)**: The final 4 bytes of Cache Line 1 (offsets 124..128) pack the rendering attributes, identically in `agent_step.wgsl`, `birth_step.wgsl` and `bridge::pack_agent_state`: bits 0..7 radius (`tr0 × 255`), bits 8..15 signal (`signal × 255`), bits 16..23 energy (`energy / 100 × 255`), bit 24 attacking (`attack > 0.5`, clankolution.html line 58), bits 25..31 birth countdown. The vertex shader loads `vec2u(root, visual_cache)` directly from Cache Line 1 and looks the lineage colour up as `PALETTE[root % 8]`, completely bypassing Cache Line 0 (traits, kinematics) and saving ~72 MB of vertex fetch bandwidth per frame (~4.3 GB/s at 60 FPS) across 1,000,000 agents.

15. **Consolidated Output Queue (Telemetry Header + 65,536 Births + 256 Voices = 1,028 KB)**:
    - **Unified Buffer Architecture (`ConsolidatedQueue`)**: Combines aggregate telemetry (128 bytes), birth events (65,536 entries $\times$ 16B = 1,048,576 bytes = 1 MB), audio voices (256 entries $\times$ 16B = 4,096 bytes = 4 KB) and chronicle events (64 entries $\times$ 16B = 1,024 bytes) into a single 1,053,824-byte storage buffer ($1,029.1\text{ KB} \ll 128\text{ MB}$ WebGPU storage limit, 16-byte aligned).
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
    *Obsolete since 2026-10-09: the LBVH was removed (section 3).*
    - **The Vulnerability**:
      In `lbvh_build.wgsl`, when population is 0 or 1, the builder early-exits (`if (active_count < 2u) return;`), which is mathematically required because a 1-leaf tree has $N - 1 = 0$ internal nodes.
      However, in `spatial_query.wgsl`, mouse picking unconditionally pushed Node 0 to the traversal stack (`stack[0] = 0u; stack_ptr = 1u;`).
      If a user clicked when $N = 1$ (e.g. at initial seeding or after a near-extinction event) or $N = 0$, thread 0 would read uninitialized or stale data from `lbvh_nodes[0]`, risking GPU infinite loops, hang conditions, or invalid pointer dereferences.
    - **The Hardening Fix**:
      Add an explicit degenerate count gate in `spatial_query.wgsl`. The count is `live_count`, the number of live keys at the head of the sorted `spatial_keys` (written by the sort), not `params.agent_count` (last frame's census). With recycled slots the single live creature can be in any slot; it is `spatial_keys[0].y`:
      ```wgsl
      if (live_count == 0u) {
          queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
          queue_buffer.telemetry.selected_agent_id = 0u;
          return;
      }
      if (live_count == 1u) {
          // Direct single-agent evaluation without tree traversal:
          let only = spatial_keys[0].y;
          let agent_pos = agent_states[only].pos_vel.xy;
          let d = toroidal_dist(tool_pos, agent_pos);
          if (d <= search_r) {
              queue_buffer.telemetry.selected_agent_idx = only;
              queue_buffer.telemetry.selected_agent_id = agent_states[only].id;
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
      let cell_id = gy * 9u + gx; // Guaranteed 0..53 (safely bounds the 54-entry cell table)
      ```

    #### 3. Canonical Atomic Energy Model & Starvation Attribution Order (`agent_step.wgsl`)
    - **The Vulnerabilities**:
      1. *Dual Energy Overwrite Race*: When an agent thread calculates its internal net delta (grazing - metabolism - thrust) in local float `a_energy` and writes to `agent_states[agent_idx].angle_energy[1]`, it could overwrite external combat damage and mating deductions inflicted concurrently by other threads via `atomicSub(&agent_atomics[victim_idx].energy_milli, damage_milli)`.
      2. *Starvation Double-Counting*: In naïve code, `atomicAdd(&queue_buffer.telemetry.starvations, 1u)` was called before `atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u)`. If a predator dealt the killing blow on the same tick, both kills and starvations were incremented for the same victim.
      3. *Lifetime Death Double-Counting* (measured: over 800 ticks 91 creatures died but `total_deaths` rose by 93): the killer path adds to `total_deaths` unconditionally, and the victim's own starvation CAS also counts.
      4. *Final Clamp Overwrite*: the agent pass ends with a plain `atomicStore(energy, clamp(atomicLoad(energy)))`, which discards any damage or mating cost another thread applied between the load and the store.
    - **The Hardening Fix**:
      - Make `agent_atomics[agent_idx].energy_milli` the single continuous source of truth for energy.
      - The agent thread calculates its internal net delta and applies it atomically, then clamps with an atomic:
        ```wgsl
        let internal_delta_milli = i32((graze_energy - basal_cost - thrust_cost) * 1000.0);
        atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);
        atomicMin(&agent_atomics[agent_idx].energy_milli, 110000);
        ```
      - Gate every death counter (`starvations` or `kills`, and `total_deaths`) behind the successful CAS claim, so each death is counted by exactly one thread:
        ```wgsl
        if (atomicLoad(&agent_atomics[agent_idx].energy_milli) <= 0) {
            let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
            if (claim_death.exchanged) {
                atomicAdd(&queue_buffer.telemetry.starvations, 1u); // ONLY incremented if this thread won the death claim!
                atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
                agent_states[agent_idx].meta_flags |= (1u << 13u); // dead = 1 (bit 13)
                push_pending_free(agent_idx); // recycled after the birth pass (#8)
            }
        }
        ```

    #### 4. Non-Blocking Pipelined Staging Ring (Task 8)
    - **The Bottleneck**:
      In Bevy / wgpu, calling synchronous `device.poll(PollType::Wait)` for GPU buffer readbacks stalls the CPU main thread waiting for the command queue to drain, cutting frame rates by 30–50%.
    - **The Hardening Fix** (implemented by Task 8 Step 2, which supersedes the 2-deep design that was here):
      A 3-deep staging ring for telemetry, audio voices and the selected creature:
      - Frame $N$: copies into a free staging entry and issues `map_async`.
      - Completed entries are consumed with `poll(PollType::Poll)`, never waiting.
      - The only frame-path wait left is back-pressure, when three frames are already in flight.

    #### 5. Struct Layout Sizing & `visual_cache: u32` 5 Hardening Invariants (Task 1 & Task 5)
    - **The Vulnerability**:
      Writing `pub _reserved: [u32; 2]` (8 bytes) at the end of `GpuAgentState` produced a struct size of $132\text{ bytes}$. Because of `#[repr(C, align(16))]`, $132$ rounded up to $144\text{ bytes}$, breaking the 128-byte power-of-two assertion (`size_of::<GpuAgentState>() == 128`), invalidating cache line alignment, and exceeding the 128 MB single buffer limit for 1M agents.
    - **The Hardening Fix & Invariants**:
      Replace `_reserved` with `pub visual_cache: u32` (offsets 124..128), locking `GpuAgentState` to **EXACTLY 128 bytes** (2 cache lines, $2^7$ power-of-two).
      Packs (identically in `agent_step.wgsl`, `birth_step.wgsl` and `bridge::pack_agent_state`, the last of which feeds Rust-mode rendering):
      - Bits 0..7: radius (`tr0 × 255`)
      - Bits 8..15: signal (`signal × 255`; antennae are drawn at `r × (1.1 + signal)`)
      - Bits 16..23: energy (`energy / 100 × 255`, for opacity & vibrancy)
      - Bit 24: attacking (`attack > 0.5`, clankolution.html line 58)
      - Bits 25..31: birth countdown (the halo radius and alpha follow it)
      **The 5 Hardening Invariants**:
      1. *Zero-Cost Dead State Invariant*: When an agent dies, `visual_cache` is set to `0u`, collapsing scale to $0.0\text{ px}$ and opacity to $0.0$, preventing phantom corpse rendering.
      2. *Newborn Initialization Invariant*: `birth_step.wgsl` initializes `visual_cache` with child trait 0, initial energy 24.0, signal 0 and birth countdown 0.
      3. *Picking Parity Invariant*: `spatial_query.wgsl` continues to evaluate full-precision `traits[0]` ($2.0 + 3.0 \times \text{tr}_0$), preserving float32 click selection bounds.
      4. *Hot-Swap Continuity Invariant*: `sync_rust_to_gpu` synthesizes `visual_cache` on upload; `sync_gpu_to_rust` cleanly discards it on readback.
      5. *VRAM Bandwidth Invariant*: Dart vertex shader loads `vec2u(root, visual_cache)` from Cache Line 1, saving ~72 MB/frame vertex fetch bandwidth (~4.3 GB/s at 60 FPS).

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
      1. Predator only sets atomic CAS gate `agent_atomics[victim_idx].dead_claimed` and pushes the slot to the pending-free list (#8). Predator never writes to `agent_states[victim_idx]`.
      2. Victim thread checks `atomicLoad(&agent_atomics[agent_idx].dead_claimed)` before committing state. If killed, victim sets its own dead bit (`meta_flags |= (1u << 13u)`), sets `visual_cache = 0u`, and aborts live state commits (movement, grazing, mating).

    #### 8. Pass-Decoupled Lock-Free Freelist (Eliminating Concurrent Push-Pop Races) (Tasks 2 & 5)
    - **The Vulnerability**:
      If deaths push to `freelist` (`atomicAdd(&freelist_top, 1u)`) while births pop from it (`atomicCompareExchangeWeak(&freelist_top, cur, cur - 1u)`) in the same `agent_step.wgsl` dispatch, incoming death threads can overwrite stack slots before pop threads finish reading them, causing slot leakage or premature recycling.
    - **The Hardening Fix**:
      Decouple push and pop across compute passes:
      - **In `agent_step.wgsl`**: ONLY pushes occur (deaths push deceased slots to the pending-free list, `freelist[max_agents + atomicAdd(&pending_free_count, 1u)]`). Mating threads only record intent in `queue_buffer.births`.
      - **Compute Pass Barrier**: Explicit GPU barrier submitted by Bevy between `agent_step` and `birth_step`.
      - **In `birth_step.wgsl`**: ONLY pops occur. Each newborn thread pops from `freelist` with zero concurrent pushes occurring.
      - **Merge after births**: a pass after `birth_step` moves the pending-free list onto the stack and zeroes `pending_free_count`. Deaths therefore become reusable only on the next sub-tick, so a newborn is never written into the slot of a creature that died this sub-tick while that creature's genome may still be read as a parent (pushing straight onto the stack allowed exactly that).

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
          // A live creature exactly on the far corner would get the all-ones code (the wrap
          // `p - w * floor(p / w)` can land on the edge); keep live codes below the sentinel.
          let code = min(compute_morton_32(agent_states[id.x].pos_vel.xy), 0xFFFFFFFEu);
          agent_states[id.x].morton_code = code;
          spatial_keys[id.x] = vec2u(code, id.x);
      }
      ```
      When Radix sorted, all dead agents are naturally partitioned to the tail of `spatial_keys`, completely excluded from grid cells 0..53 and LBVH leaf bounds. The sort also writes `live_count` (the number of non-sentinel keys), which the LBVH build, picking and Tier 2 neighbour search use as $N$. Without the sort the LBVH is built over unsorted slots `0..agent_count` (dead slots included, high live slots left out) and comes out cyclic (measured: 30 of 30 trees invalid).

    #### 10. Bottom-Up LBVH AABB Fitting Atomic Flag Initialization (Task 4)
    *Obsolete since 2026-10-09: the LBVH was removed (section 3).*
    - **The Vulnerability**:
      In `lbvh_aabb.wgsl`, parallel tree climbing uses `atomicAdd(&node_flags[current], 1u)` to detect when both children have computed their bounding boxes. If `node_flags` is not reset to 0 every frame, on frame 2 and beyond `atomicAdd` returns $\ge 2$, causing threads to deadlock or produce corrupt bounding boxes. Even with the flags reset, the second thread to arrive then reads the sibling's box with plain loads, and that box was written by another workgroup: WGSL has no device-scope fence, so nothing orders those writes before the read. Measured: 83 of 300 consecutive trees had a box equal to the previous tick's child box.
    - **The Hardening Fix**:
      Phase 1 (`lbvh_build.wgsl`) initialises every internal node's box to empty (min `3e38`, max `0`) and its `count` to `last - first + 1`. Phase 2 reads no child boxes: each live creature writes its leaf, then climbs to the root, growing each ancestor's box with `atomicMin`/`atomicMax` on the box words (the node buffer is bound as `array<atomic<u32>>`, 12 words per node). Box coordinates are clamped to be non-negative, so `u32` order is float order. A thread stops at the first node where it changes nothing: whichever thread set that node's extreme values climbs on with them. The result does not depend on arrival order (300 of 300 trees exact after the change). The climb is capped at 66 steps (a Karras tree over 32-bit keys with a 32-bit tie-break is at most 65 levels deep). Internal nodes' `center_of_mass` and `dominant_lineage` are left at 0; only the deferred minimap would use them.

    #### 11. Supplemental Hardening & Robustness Guarantees
    - **Degenerate LBVH Underflow Guard (`lbvh_build.wgsl`)**: *(obsolete since 2026-10-09: the LBVH was removed)*
      In `lbvh_build.wgsl`, if the count is `0u`, the condition `id.x < count - 1u` underflows `0u - 1u` to `u32::MAX` ($4,294,967,295$), launching rogue out-of-bounds writes. Guarded via (with `live_count` from the sort, #9):
      `let active_count = live_count; if (active_count < 2u || id.x >= active_count - 1u) return;`
    - **Two-Phase LBVH Construction Separation**: *(obsolete since 2026-10-09: the LBVH was removed)*
      Topology generation (`lbvh_build.wgsl`) and bottom-up bounding box fitting (`lbvh_aabb.wgsl`) are separate dispatches, each its own WebGPU synchronization scope, eliminating workgroup scheduling race conditions.
    - **LBVH Identical Key Tie-Breaking**: *(obsolete since 2026-10-09: the LBVH was removed)*
      If multiple agents occupy identical coordinates or have identical Morton keys, Karras LCP comparison evaluates:
      `return 32 + i32(countLeadingZeros(id_i ^ id_j));`
      guaranteeing strictly unique split positions and preventing tree-building infinite loops.
    - **Dynamic Radius Shrinking (Early Tree Pruning in `spatial_query.wgsl`)**: *(obsolete since 2026-10-09: the LBVH was removed)*
      Begins traversal with cursor radius $R_{\text{pick}} = \text{clamp}(16.0 \times (\text{camera\_size}.x / 900.0), 4.0, 24.0)$. The instant a leaf node with a direct body hit is encountered ($d \le R_{\text{body}} \approx 2.0 + 3.0 \times \text{tr}_0$), the active search radius is clamped to $d$. Any subtrees or sibling nodes whose AABB distance to the cursor is $> d$ are **immediately pruned**, resolving clicks in ~5–8 node tests (< 40 nanoseconds on GPU).
    - **Two-Tier Euclidean Disambiguation**:
      Direct body hits (Priority 0) always beat proximity halos (Priority 1); exact 32-bit floating-point Euclidean distance resolves ties without millipixel distortion.
    - **Specimen Picking Identity Guard `(slot_idx, agent_id)`**:
      The CPU tracks the tuple `(slot_idx, agent_id)`. If Agent $K$ dies during sub-ticks 2..32 (or a subsequent tick) and slot $K$ is recycled, `agent_states[slot_idx].id != picked_id`. The UI immediately detects the death, prevents displaying the replacement creature, and displays final stats.
    - **Multi-Tick Telemetry & Audio Lifecycle Synchronization (`preamble_clear.wgsl`)**:
      In 32x speed mode, sub-tick resets (`mate_claim`, `mate_energy_milli`, `birth_count = 0u`; `dead_claimed` is persistent) run every sub-tick. Frame-level cumulative counters (`audio_voice_count = 0u`, `kills = 0u`, `starvations = 0u`, `food_grazed_milli = 0u`, `apex_record_milli = 0u`) are reset strictly at frame start (`params.sub_tick == 0u`), preventing audio queue overwrites and telemetry loss. Instantaneous living census (`population`, `max_generation`, `living_lineages`) is reset and recounted strictly on the final sub-tick (`sub_tick == params.sub_ticks_per_frame - 1u`) by a census pass that runs after the birth pass, so the population includes that sub-tick's newborns (counting in the agent pass lagged one birth pass behind). `living_lineages` counts distinct full-width roots: each live creature inserts `root + 1` by CAS into a cleared open-addressed table sized to a power of two of at least twice the slot count (every creature may head its own lineage, so the table never fills and probe chains stay short — a fixed 4096-entry table capped the count at 4096 and made every later insert probe the whole table), and each insert of a new root adds 1. The 16-bucket `lineage_counts` cannot represent full-width roots and is removed.

    #### 12. Soil Sensing Reads the Simulation Buffer, Never the Colormap (Tasks 3 & 5)
    - **The Vulnerabilities**:
      1. *Colormap Sensory Corruption*: `soil_step.wgsl` renders `evaluate_soil_color(f, t, s)` (display RGB with dark dirt `[10, 23, 26]`, vignette, etc.). If agents sampled this display texture, creatures would sense the edge vignette as a severe famine and flee the arena borders!
      2. *Behavior Drift from Filtering*: The reference reads single cells (`food[k]`). Bilinear feelers (an earlier design here, with a `soil_data` texture) would blend neighbouring cells and change steering.
    - **The Hardening Fix**:
      1. `soil_buffer: array<SoilCell, 3750>`: the one soil state. Feelers read it with point `atomicLoad`s at the clamped cell index (#2); grazing (`atomicSub`) and carcass/scent deposits write it. Reads race with concurrent grazing within a tick; that is accepted under statistical parity.
      2. `soil_display: texture_storage_2d<rgba8unorm, write>`: Output by `soil_step.wgsl` storing `evaluate_soil_color(f, t, s, id.xy)` with the CPU colormap's truncation, copied GPU→GPU into the soil sprite (Task 8 Step 4). Creatures never read it.

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
      The pass is dispatched with `dispatch_workgroups_indirect`, from an argument record set to `min(birth_count, 65536)` after the agent pass, so every queued birth gets a workgroup. (A fixed dispatch of 64 workgroups dropped every birth past the 64th in a sub-tick.)

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
      - The dynamic 60Hz soil colormap remains an uncompressed 15 KB `rgba8unorm` texture because GPU silicon decodes but does not encode ASTC in hardware, and 15 KB fits in L1 cache.
    - **Layer 3: Parallel Radix / Prefix Bit-Packing & Stream Compaction**:
      - *8-Byte Morton Indirection Sorting (Task 4)*: Radix sort operates over `vec2u(morton_key, slot_idx)`, reducing sorting bus bandwidth by **$16\times$** compared to shifting 128-byte structs.
      - *Camera Frustum Stream Compaction (Task 6)*: Visible creatures are compacted via parallel prefix scans (`atomicAdd(&draw_args.instance_count, 1u)`) into `visible_agent_indices: array<u32>`, driving `draw_indirect` with 0 CPU intervention.
      - *Page-Aligned Event Compaction (Task 5)*: Discrete births and audio voices are packed into the unified 1,028 KB `ConsolidatedQueue`.
    - **Layer 4: In-Shader Domain Compression & Mathematical Equivalence Proof**:
      - *Lossless 128-Byte Dynamic State Packing*:
        - `pos_vel` (16B), `angle_energy` (16B), and `traits` (32B) retain **100% full 32-bit `f32` precision** (Zero precision loss).
        - `hidden` (40B) retains 10 full 32-bit floats.
        - Discrete metadata (`cooldown: 2b`, `birth: 7b`, `dead: 1b`, `kills: 18b` in `meta_flags`, bits 0..3 unused; `age: 16b`, `gen: 16b` in `age_gen`) is bit-packed losslessly into two `u32` words, and dead padding is eliminated.
        - The lineage `root` is a full `u32` field. An earlier 4-bit root in `meta_flags` merged the 72 founding lineages into 16, which broke the different-lineage mating rule, lineage colours and the lineage count.
        - **Result**: `GpuAgentState` is exactly **128 bytes** ($2^7$ power-of-two, 2 cache lines). Dynamic state for 1,000,000 agents is **exactly 128 MB**, fitting 100% inside WebGPU's portable limit!
      - *Lossless Gene Storage*: The HTML reference stores genes as `Int8Array(326)` and scales sums by `1 / 127` then `0.61` (hidden) or `0.66` (output). The GPU stores the same 326 bytes losslessly in 82 `u32` words; the f32 arithmetic differs from the f64 reference only by rounding (measured median output error 6e-8), not bit-exact.
      - *Morton Coordinate Compression*: Continuous kinematics $(x, y, v_x, v_y)$ are always stored as full 32-bit floats (`f32`); Morton codes are strictly used as spatial hash keys for bucket sorting.
      - *Lineage Color Lookup*: A creature's colour is purely `PALETTE[root % 8]`, so it is not stored; the render shader looks it up from `root` (the `root` field takes the 4 bytes that `packed_color` used).

---

## Plan Overview: Tasks 1–6 and 8

The remediation steps that used to be Task 7 now sit in the tasks they belong to; Task 8 keeps its number. Every task is verified by tests that dispatch the real WGSL through `GpuComputeDriver` (shared harness `crates/clank_app/tests/support/gpu_harness.rs`) and compare against `clank_core::World` or an exact invariant. Tests of CPU mirror code (`gpu::lbvh::LbvhTree`, `gpu::spatial_index`, `gpu::agent_pipeline`, `gpu::soil_pipeline` arithmetic) are not evidence that the shaders work; Tasks 3–6 had been marked complete on such tests.

Ticks verified on 2026-10-11 at `f681c8d`: each ticked task's items were checked in the code, every test this plan names exists (only the four struck LBVH tests are gone), and every `clank_app` test file passes run one at a time (53 files, 193 tests, 0 failed, 0 ignored; `engine_benchmark`, `pipeline_timing_bench` and the compile-only `tdd_gpu_api_and_biology_parity_test` left out). The "failed first" evidence for each test is in the Verification Matrix.

- [x] **Task 1: Branch Setup & GPU Compute Architecture Scaffolding**
  - Create branch `feature/bevy-gpu` from `feature/bevy-port`.
  - Add `bytemuck = { version = "1.21", features = ["derive"] }` to `crates/clank_app/Cargo.toml`.
  - Add `gpu` module in `crates/clank_app/src/gpu/` with buffer types: `GpuAgentState` (128B, exact $2^7$ power-of-two, 2 cache lines, with `visual_cache: u32`), `GpuAgentGenome` (352B, 88 words, 82 used), `GpuAgentAtomic` (16B, with `dead_claimed`), `GpuSimParams` (96B, 6 quadwords: physics, rates, multi-tick pacing, tools, camera, world size, soil grid, eclipse), `GpuLbvhNode` (48B), `GpuSoilCell` (16B), `BirthEvent` (16B), `AudioVoice` (16B), `GpuTelemetry` (128B), `ConsolidatedQueue` (1,028 KB unified queue with telemetry header), and pipeline skeletons.
  - Implement unit tests for GPU struct memory layouts and 16-byte WGSL alignment (`assert_eq!(size_of::<GpuAgentState>(), 128)`), and check every WGSL struct declaration against its Rust struct field by field (naga parse of each shader).
  - Give the lineage root a full `u32` field (`root`, replacing `packed_color`) so 72+ lineages survive.

- [x] **Task 2: Tombstone Freelist & Zero-Copy Agent Storage Buffer**
  - Implement lock-free atomic stack allocator (`freelist: array<u32>`, `atomic<u32> queue_buffer.telemetry.freelist_top`).
  - Implement **Pass-Decoupled Freelist**: deaths push in `agent_step.wgsl`; newborn threads pop in `birth_step.wgsl` across compute pass barrier, eliminating ABA / concurrent push-pop races.
  - Implement CAS death ownership (`dead_claimed: 0u -> 1u`) and Strict Single-Writer Invariant (killer never writes to victim's `agent_states`). Each death is counted once, by the CAS winner.
  - Deaths go to the pending-free list and are merged onto the stack after the birth pass.
  - Changing the population cap rebuilds the freelist on the GPU (`capacity_step.wgsl`): creatures in slots at or above the new cap are tombstoned, and the dead slots below it are pushed. (Play-test bug: in GPU mode a raised cap never took effect, because the freelist still held only the old cap's slots.)
  - Test slot accounting, single counting, recycling, tombstones and the cap change on the real shaders.

- [x] **Task 3: GPU Soil Simulation & Direct Texture Generation**
  - Implement WGSL compute shader for soil chemistry: spatial bloom renewal, random food spawns, food clamp $[0.0, 2.5]$, taint decay ($0.994$), and scent decay ($0.954$), on micro-unit fixed point.
  - Roll the spawn per cell per tick (`pcg3d(k, SPAWN_STREAM, tick)`), never from the cell's food value. (The old roll was seeded from the food value with a constant tick, so a stalled cell re-rolled the same failing number forever and the GPU max cell never left its initial 1.613.)
  - Implement the eclipse on the GPU, and count `world.eclipse` down in GPU mode.
  - Add coordinate clamping guards (`cx = min(u32(pos.x / 12.0), 74u)`, `cy = min(u32(pos.y / 12.0), 49u)`) preventing buffer overruns.
  - Write the colormap `soil_display` (`rgba8unorm`) on the GPU for Task 8's GPU→GPU copy, **completely eliminating CPU `generate_soil_rgba` upload** in GPU mode. Creatures read `soil_buffer` with point reads; there is no `soil_data` texture.

- [x] **Task 4: Hybrid Morton Grid, Multi-System LBVH & Spatial Queries**
  - Implement 32-bit Morton code generator with dead agent partitioning (`0xFFFFFFFFu` sentinel in `morton_encode`; live codes capped at `0xFFFFFFFEu`), `floor` quantization identical to the Rust helper, and boundary clamping (`gx = min(u32(pos.x / 100.0), 8u)`, `gy = min(u32(pos.y / 100.0), 5u)`).
  - Implement 8-byte indirection parallel Radix Sort on `(morton_key, agent_slot_idx)`, keeping heavy agent structs stationary. Before this task no sort existed: `spatial_keys` was a per-cell linked list.
  - Build the grid cell table (count, prefix sum, scatter into `cell_slots`), with no per-cell cap, in the sort's first pass.
  - ~~Implement Karras 2012 two-phase LBVH construction over the `live_count` sorted keys: Phase 1 topology (`lbvh_build.wgsl`) with slot tie-breaking, empty internal boxes, and $N \le 1$ degenerate population guard; Phase 2 bottom-up AABB fitting (`lbvh_aabb.wgsl`) in the next dispatch, by an `atomicMin`/`atomicMax` climb that reads no sibling boxes (#10).~~ Removed on 2026-10-09 (section 3).
  - Implement Tier 1 ($3 \times 3$ local Moore neighborhood) + Tier 2 global scan when nothing is within 100 px, matching the reference `near()` (nearest and density `d² < 10000`).
  - Implement **Unified Interactive Spatial Query Pass (`spatial_query.wgsl`)**: picking from the grid's 3×3 cells around the cursor, as the Rust engine picks (Subsystem 3), and AoE tools checked by one thread per slot.
  - The AoE tools match the reference: Extinguish kills by plain (not toroidal) distance `< 23`; Seed Life adds two creatures within ±11 px when under the cap.

- [x] **Task 5: Packed 88-Word Vectorized RNN, Combat Resolution, PRNG & Decoupled Birth Pipeline**
  - Implement `preamble_clear.wgsl` to reset `mate_claim`, `mate_energy_milli`, `birth_count` (sub-tick queue depth) every sub-tick, and frame-level counters (`audio_voice_count`, `kills`, `starvations`, `food_grazed_milli`, `apex_record_milli`) on `sub_tick == 0u` with global execution barrier.
  - Implement 1-thread-per-agent WGSL compute shader (`agent_step.wgsl`) binding 7 storage buffers in Group 0 and `params` in Group 1 within strict $\le 8$ storage buffer ceiling.
  - Feed the oscillator input the number of the tick being computed (the reference `evolve` increments `tick` before the agents run; the shader currently uses the tick before it, one behind).
  - Enforce **Top-of-Shader Tombstone Dead-Check Guard**: inactive corpse slots exit immediately via hardware SIMD lane masking.
  - Enforce **Strict Single-Writer Invariant**: thread $i$ writes ONLY to `agent_states[i]`. Victim detects death via `dead_claimed`, marks own dead flag, zeroes `visual_cache`, and aborts live state commits.
  - Enforce `agent_atomics[i].energy_milli` as canonical atomic ground truth, preventing concurrent damage/grazing overwrites; clamp with `atomicMin`, never a final load-clamp-store.
  - Gate every death counter behind successful death claim CAS (`dead_claimed: 0u -> 1u`), eliminating double-counted deaths.
  - Point `atomicLoad` antennae feeler reads from `soil_buffer` at toroidally wrapped, clamped cells, as the reference reads single cells.
  - Forward pass over the 326 linearly packed genes in reference `brain()` order.
  - Apply steering, thrust, hardware toroidal coordinate wrap, and pack `visual_cache` at step end.
  - Implement coalesced atomic soil deposits (each clamping the cell to [0, 3]; corpses deposit at the victim's final position) and grazing to `soil_buffer`, atomic millijoule combat resolution with `dead_claimed` CAS, toroidal seam frustum-culled stochastic audio voice emission (256-voice buffer), reference mating (several suitors may share a mate) with seamless asexual fallback, and decoupled SIMD genome mutation pass (`birth_step.wgsl`) with newborn workgroup bounds guard (`wg_id.x >= min(birth_count, 65536u)`) and `visual_cache` newborn initialization.
  - Births reserve capacity atomically (charged only when a slot is reserved; the cap holds across sub-ticks), and the birth pass is dispatched indirectly over the whole queue.
  - Draw every random number from its own `pcg3d(id, stream, tick)` stream.
  - Give new agents (population-collapse replenishment and Seed Life) reference founder genomes: genes uniform in [−58, 58], each trait uniform in its reference range ([0.25, 0.8], [0.3, 0.8], [0.3, 0.8], [0.25, 0.75], [0.2, 0.8], [0.2, 0.8]), energy uniform in [34, 52], a new root each, after every lineage of the uploaded world (`next_agent_root`, carried back to `World::roots` on download), and ids after the GPU's newborns (`apex_agent_id`). (Today they get a fixed gene pattern, traits of 0.5 and energy 43.)
  - Census pass after births: population, max generation, living lineages.
  - Run the agent pass's threads in Morton order (thread `i` takes slot `spatial_keys[i].y`; see Cross-Cutting Morton Code Optimizations).

- [ ] **Task 6: Frustum Culling, Minimap LOD, Dual-Engine UI, Audio & Verification**
  - *Verified 2026-10-11: not done.* Open: the Morton-ordered draw (Step 3) and the on-screen check with the user (Step 5).
  - Implement **GPU Camera Viewport Frustum Culling** streaming 128-byte `GpuAgentState` ($4.0\times$ less bandwidth than 512B structs) into an Indirect Draw Buffer. One thread per slot, not via the LBVH (Subsystem 2 explains why).
  - Implement **instanced dart rendering loading contiguous `vec2u(root, visual_cache)`** directly from Cache Line 1, completely bypassing Cache Line 0 traits and saving ~72 MB of vertex fetch bandwidth per frame.
  - ~~Implement **Minimap LOD Cluster Rendering** sampling intermediate LBVH depth nodes for density circles.~~ Deferred: the minimap was removed from the UI on 2026-10-06.
  - Implement **spatially coherent instanced dart rasterization** using Morton-ordered agent index streams for tile-cache efficiency.
  - Implement **Granular Bevy Audio playback** reading 256-voice audio queue with zoom loudness modulation.
  - Implement **Non-Blocking Pipelined Staging Ring** for 128-byte `GpuTelemetry` and audio queue readback (0 CPU wait cycles); built by Task 8 Step 2 as a 3-deep ring.
  - Implement **32x Speed Visual Consistency Picking** locking selection on Sub-Tick 1 across 32-tick batches with **Specimen Picking Identity Guard** `(slot_idx, agent_id)`.
  - Add top bar engine toggle: `ENGINE: RUST` $\leftrightarrow$ `ENGINE: GPU` with live bi-directional state bridge synthesizing `visual_cache` on upload, packed exactly as `agent_step.wgsl` packs it (covered by `visual_cache_packing_test`; play-test fix 2026-10-06, Rust-mode halos, borders and antennae had differed).
  - Trails are per tick, not per rendered frame: `trail_step.wgsl` appends each creature's position after the agent pass into `agent_trails_buf`, owned by creature id (play-test fix 2026-10-06: trails stretched with speed and stitched reused ids; covered by `gpu_trail_and_reset_test`).
  - NEW WORLD and loads replace the GPU world: `SimWorld::gpu_resync_pending` is set by `reset()` and the loaders and handled before the pause check (never flushed back), and `upload_state` zeroes lifetime telemetry (play-test fix 2026-10-06; covered by `gpu_trail_and_reset_test`).
  - Wire telemetry from GPU storage buffers into UI stats, The Record, and Specimen card. In GPU mode the GENERATION and LINEAGES boxes read `max_generation` and `living_lineages` from telemetry, not the stale CPU `World`. (Play-test bug: in GPU mode they never changed.)
  - GPU chronicle: the agent pass queues "Lineage R has taken N lives." on a creature's every 8th kill and the birth pass "Generation G opens in lineage R." for 8% of births at a positive multiple of 12 generations (its own `pcg3d` stream), into `ConsolidatedQueue::chronicle` (64 per frame, `chronicle_count` reset at frame start). The frame readback carries them and `apply_frame_readback` adds them to `world.events`, which the UI drains. (Play-test bug 2026-10-07: in GPU mode the chronicle never updated.)
  - Long-run statistical parity gate against the reference run (Verification Plan tolerances).
  - Run the targeted GPU test suite listed in the Verification Plan (never `cargo test --workspace`).

  - Run the desktop app in GPU mode and confirm with the user on screen before committing.

- [ ] **Task 8: Zero-Copy Presentation (one device, GPU-resident rendering, no per-frame blocking readbacks)**
  - *Verified 2026-10-11: Steps 1–4 done.* Open: the on-screen check with the user (Step 5) and deleting the old copy path (Step 6: nothing deleted yet).
  - Build the compute driver on Bevy's `RenderDevice`; cull, gather and draw creatures from the simulation buffers with `draw_indirect` in the `Transparent2d` phase; render both engines through the same GPU renderer; copy the soil colormap GPU→GPU; replace per-frame blocking readbacks with an async ring.

---

## Detailed Task Breakdown

### Task 1: Branch Setup & GPU Compute Architecture Scaffolding

**Files:**
- Modify: `crates/clank_app/Cargo.toml` (add `bytemuck = { version = "1.21", features = ["derive"] }`)
- Create: `crates/clank_app/src/gpu/mod.rs`
- Create: `crates/clank_app/src/gpu/types.rs`
- Modify: `crates/clank_app/src/lib.rs`
- Modify: `crates/clank_app/src/gpu/bridge.rs` (pack/unpack the full-width `root`) and every shader declaring `GpuAgentState` (`root` replaces `packed_color`; `meta_flags` bits 0..3 unused)
- Test: `crates/clank_app/tests/gpu_types_test.rs` (Rust sizes)
- Test: `crates/clank_app/tests/plan_t1_layouts_test.rs` (every WGSL struct in every shader, parsed with naga, against its Rust struct: size, and at each Rust field offset the same field name and scalar kind)
- Test: `crates/clank_app/tests/plan_t1_layouts_test.rs::test_72_lineages_survive_the_engine_round_trip` (Rust → GPU → Rust keeps every creature's id, root, generation and position; 72 founding roots stay 72)

**Interfaces:**
- Consumes: `clank_core::agent::AgentData`, `clank_core::agent::GENES` (326)
- Produces: `GpuAgentState`, `GpuAgentGenome`, `GpuAgentAtomic`, `GpuSimParams`, `GpuLbvhNode` (removed on 2026-10-09 with the LBVH), `GpuSoilCell`, `BirthEvent`, `AudioVoice`, `GpuTelemetry`, `ConsolidatedQueue` with exact 16-byte WGSL alignment

- [x] **Step 1: Write failing test for GPU struct memory layouts** (sizes below; the shader-vs-Rust field check is `plan_t1_layouts_test`)
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
    assert_eq!(std::mem::size_of::<GpuSimParams>(), 96);
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
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>(), 1_053_824);
}
```
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement GPU types with `#[repr(C)]` and 16-byte alignment**
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
    pub meta_flags: u32,        // 4 bytes: unused (4b), cooldown (2b), birth (7b), dead (1b), kills (18b)
    pub age_gen: u32,           // 4 bytes: age (16b: 0..65,535), gen (16b: 0..65,535)
    pub morton_code: u32,       // 4 bytes: precomputed 32-bit Morton spatial hash key
    pub root: u32,              // 4 bytes: full-width lineage root (colour = PALETTE[root % 8])
    pub visual_cache: u32,      // 4 bytes: radius (0..7), signal (8..15), energy (16..23), attacking (24), birth countdown (25..31)
} // Total: EXACTLY 128 bytes (2^7 power-of-two, 2 * 64B cache lines)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentGenome {
    pub packed_genes: [u32; 88],// 352 bytes: 326 i8 genes in order, 4 per word (words 0..81 used, 82..87 zero)
} // Total: EXACTLY 352 bytes (16-byte aligned, 22 quadwords, 5.5 cache lines)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAgentAtomic {
    pub energy_milli: i32,      // 4 bytes: atomic fixed-point millijoules
    pub mate_claim: u32,        // 4 bytes: atomic CAS mating winner agent_idx + 1 (0 = unclaimed, full 32-bit capacity)
    pub mate_energy_milli: u32, // 4 bytes: unused since the mods were removed (keeps the struct at 16 bytes)
    pub dead_claimed: u32,      // 4 bytes: atomic CAS death ownership (prevents double-free)
} // Total: 16 bytes (1 quadword, perfectly aligned)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    // Simulation Physics & Capacity Bounds (16B)
    pub tick: u32,                  // 4 bytes  (0..4)
    pub agent_count: u32,           // 4 bytes  (4..8)
    pub max_agents: u32,            // 4 bytes  (8..12)   - allocated slots
    pub max_capacity: u32,          // 4 bytes  (12..16)  - population cap (slots >= cap are never alive)

    // Environmental Chemistry & Rates (16B)
    pub hostility: f32,             // 4 bytes  (16..20)
    pub mut_rate: f32,              // 4 bytes  (20..24)
    pub speed: f32,                 // 4 bytes  (24..28)
    pub renewal: f32,               // 4 bytes  (28..32)

    // Multi-Tick Batching & Interactive Tools (16B)
    pub sub_tick: u32,              // 4 bytes  (32..36) - Sub-tick index within batch (0..sub_ticks_per_frame - 1)
    pub sub_ticks_per_frame: u32,   // 4 bytes  (36..40) - Sub-ticks per frame (e.g. 1 in 1x mode, 32 in 32x mode)
    pub tool_type: u32,             // 4 bytes  (40..44) - 0xFFFFFFFF = none, 0 = inspect/pick, 1..5 = AoE tools
    pub tool_radius: f32,           // 4 bytes  (44..48)

    // Tool & Camera Positions (16B)
    pub tool_pos: [f32; 2],         // 8 bytes  (48..56)
    pub camera_pos: [f32; 2],       // 8 bytes  (56..64)

    // Camera Viewport & Arena Bounds (16B)
    pub camera_size: [f32; 2],      // 8 bytes  (64..72)
    pub world_size: [f32; 2],       // 8 bytes  (72..80)

    // Soil Grid & Simulation State (16B)
    pub soil_grid: [u32; 2],        // 8 bytes  (80..88)
    pub eclipse: u32,               // 4 bytes  (88..92) - eclipse countdown
    pub epoch: u32,                 // 4 bytes  (92..96)
} // Total: Exactly 96 bytes (6 quadwords, 0 pad holes, std140/std430 aligned)

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
    pub total_births: u32,      // 4 bytes: lifetime births
    pub total_deaths: u32,      // 4 bytes: lifetime deaths (counted once, by the death CAS winner)
    pub max_generation: u32,    // 4 bytes: census: highest living generation
    pub living_lineages: u32,   // 4 bytes: census: distinct living roots (Total Cache Line 0: 64B)

    // Cache Line 1 (64 bytes): sort/merge counters and reserve
    pub live_count: u32,        // 4 bytes: live keys at the head of the sorted spatial_keys
    pub pending_free_count: u32,// 4 bytes: deaths awaiting the post-birth merge
    pub available_slots: u32,   // 4 bytes: birth capacity left this sub-tick
    pub chronicle_count: u32,   // 4 bytes: chronicle events queued this frame
    pub _reserved1: [u32; 12],  // 48 bytes
} // Total: 128 bytes (2 x 64B cache lines, 8ns DMA transfer)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConsolidatedQueue {
    pub telemetry: GpuTelemetry,           // 128 bytes (Cache Lines 0 & 1)
    pub births: [BirthEvent; 65536],        // 1,048,576 bytes = 1,024 KB = 1 MB
    pub audio: [AudioVoice; 256],           // 4,096 bytes = 4 KB (1 memory page)
    pub chronicle: [GpuChronicleEvent; 64], // 1,024 bytes: (tick, type, p1, p2), as clank_core::SimEvent
} // Total: 1,053,824 bytes (1,029.1 KB, 16-byte aligned)

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSoilCell {
    pub food_micro: i32,        // 4 bytes: atomic fixed-point food (SOIL_FIXED_SCALE = 1e6)
    pub taint_micro: i32,       // 4 bytes: atomic fixed-point taint (SOIL_FIXED_SCALE)
    pub scent_micro: i32,       // 4 bytes: atomic fixed-point scent (SOIL_FIXED_SCALE)
    pub pad: u32,               // 4 bytes: 16-byte WGSL alignment
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLbvhNode {
    pub aabb_min: [f32; 2],      // 8 bytes: bounding box min [x, y]
    pub aabb_max: [f32; 2],      // 8 bytes: bounding box max [x, y] (Total 16)
    pub center_of_mass: [f32; 2],// 8 bytes: weighted centroid [x, y] (for Minimap LOD)
    pub count: u32,              // 4 bytes: subtree creature count (for Minimap LOD)
    pub dominant_lineage: u32,   // 4 bytes: lineage root with the highest count in the subtree (Total 32)
    pub left_child: u32,         // 4 bytes: child node index
    pub right_child: u32,        // 4 bytes: child node index
    pub parent: u32,             // 4 bytes: parent node index
    pub leaf_idx: u32,           // 4 bytes: leaf agent index (0..N-1), or 0xFFFFFFFF for internal nodes (Total 48)
} // Total: 48 bytes (48 % 16 == 0)
```
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 2: Tombstone Freelist & Zero-Copy Agent Storage Buffer

**Files:**
- Create: `crates/clank_app/src/gpu/freelist.rs`
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`, `birth_step.wgsl` (death claim, pending-free push, pop), a merge pass after `birth_step`
- Create: `crates/clank_app/assets/shaders/capacity_step.wgsl`; Modify: `crates/clank_app/src/gpu/compute_driver.rs` (`apply_capacity` before each frame's dispatch when `max_capacity` changed)
- Test: `crates/clank_app/tests/gpu_freelist_test.rs` (CPU mirror; not evidence for the shaders)
- Test: `crates/clank_app/tests/plan_t2_freelist_test.rs` (real shaders):
  - `test_every_slot_is_alive_or_free_exactly_once`: at 6 checkpoints over 600 ticks, no slot is on the freelist twice, no live slot is on it, and live + `freelist_top` == cap;
  - `test_each_death_is_counted_exactly_once`: over 800 ticks, the creatures that disappear equal the rise in `total_deaths` (allowing the final tick's pending kills);
  - `test_dead_slots_are_recycled_by_births`: founder slots are reused by newborns, and no slot at or above the cap is alive;
  - `test_dead_creatures_are_tombstoned`: a slot claimed dead has its dead bit set and `visual_cache == 0` within one tick.
  - `test_a_slot_freed_this_sub_tick_is_not_reused_until_the_next`: 48 parents give birth in the pass where they die of old age; each lineage gets exactly one child;
  - `test_changing_the_cap_loses_no_slots`: 64 victims killed by attackers in later slots, then a cap change; every slot under the cap is still alive or free (the rebuild treats a claimed slot as free before its dead bit is set).
- Test: `crates/clank_app/tests/gpu_capacity_test.rs` (real shaders): raising the cap from 80 to 500 frees slots up to the new cap; lowering it removes creatures over it.

**Interfaces:**
- Consumes: `GpuAgentState`, `MAX_AGENTS` (e.g. 1,000,000 agents)
- Produces: `TombstoneFreelistManager` managing allocation head, tombstone flags, and zero-copy recycling

- [x] **Step 1: Write failing test for Freelist initialization, slot recycling, and underflow protection** (the CPU mirror below, plus the real-shader tests listed under **Files**)
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
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement Tombstone Freelist logic in Rust and WGSL CAS allocation snippet**
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

// WGSL CAS death ownership: guarantees the slot is recycled and the death counted exactly once.
// The slot goes to the pending-free list; the merge pass after birth_step moves it onto the stack.
fn claim_death_and_free(victim_idx: u32, is_self: bool) -> bool {
    let cas = atomicCompareExchangeWeak(&agent_atomics[victim_idx].dead_claimed, 0u, 1u);
    if (cas.exchanged) {
        // Strict Single-Writer Invariant: Only victim thread writes to its own agent_states struct!
        if (is_self) {
            agent_states[victim_idx].meta_flags |= (1u << 13u); // dead = 1 (bit 13)
            agent_states[victim_idx].visual_cache = 0u;         // Zero-cost dead state invariant (0 radius, 0 glow)
        }
        atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
        let pending = atomicAdd(&queue_buffer.telemetry.pending_free_count, 1u);
        freelist[params.max_agents + pending] = victim_idx;
        return true;
    }
    return false; // Already freed by another concurrent thread
}
```
- [x] **Step 4: Run test to verify it passes** (`gpu_freelist_test`, `plan_t2_freelist_test`, `gpu_capacity_test`)
- [x] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 3: GPU Soil Simulation & Direct Texture Generation

**Files:**
- Create: `crates/clank_app/src/gpu/soil_pipeline.rs`
- Create: `crates/clank_app/assets/shaders/soil_step.wgsl`
- Modify: `crates/clank_app/src/sim.rs` (GPU mode counts `world.eclipse` down with the ticks dispatched)
- Test: `crates/clank_app/tests/gpu_soil_test.rs` (CPU mirror; not evidence for the shaders)
- Test: `crates/clank_app/tests/plan_t3_soil_test.rs` (real shader against `SoilGrid::step_soil` and the `World` tools):
  - one soil tick matches the reference cell by cell (2e-6), excluding at most 4 cells where either engine rolled a random spawn;
  - food above 2.5 is clamped to 2.5;
  - a creature at the far corner grazes the corner cell or the cell across the seam, and nothing else;
  - Nourish and Blight deposit exactly like `nourish_at` / `blight_at`;
  - with zero creatures, over 600 ticks, total food stays within 3% of the reference at ticks 100..600, and the GPU max cell rises above its initial value (spawns fire). Also `gpu_soil_parity_test`;
  - with `eclipse > 0`, total food falls and taint rises at the reference rate (160 random cells on even ticks: food ×0.73, taint +0.14 clamped to 2) (`test_an_eclipse_darkens_the_soil_at_the_reference_rate`), and the countdown decrements once per tick in GPU mode (`test_gpu_mode_counts_the_eclipse_down`, through the app's systems);
  - an eclipse with 2 ticks left in an 8-sub-tick frame does one round of damage (`test_an_eclipse_ends_mid_frame`), and none on the reference's odd ticks (`test_an_eclipse_skips_odd_ticks`).

**Interfaces:**
- Consumes: `SoilGrid` dimensions ($75 \times 50$), `growth_val`, `GpuSoilCell` storage buffer (60 KB, micro-units)
- Produces: the soil state creatures read with point `atomicLoad`s, and the colormap `soil_display` (`rgba8unorm`, copied GPU→GPU into the Bevy `SoilSprite` by Task 8), completely eliminating CPU `generate_soil_rgba` upload in GPU mode

- [x] **Step 1: Write failing test for soil atomic buffer bindings, renewal math, spawns, eclipse and the colormap** (the CPU mirror below, plus the real-shader tests listed under **Files**)
```rust
// crates/clank_app/tests/gpu_soil_test.rs
#[test]
fn test_soil_atomic_buffer_and_texture_bounds() {
    // Verifies:
    // 1. GpuSoilCell fixed-point conversions (micro-units, SOIL_FIXED_SCALE = 1e6)
    // 2. Renewal formula: f += renewal * (.0008 + .0022 * bloom) * (1 - f / 1.7), clamped to [0.0, 2.5]
    // 3. Taint decay: t * 0.994 - 0.0001, scent decay: s * 0.954
    // 4. Output texture: soil_display (colormap) is 75x50 rgba8unorm
    // 5. Boundary coordinate clamping: min(pos.x / 12.0, 74) and min(pos.y / 12.0, 49) prevents OOB write at seam (900.0, 600.0)
}
```
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement WGSL soil compute kernel operating on the 60 KB atomic buffer and writing the colormap**
```wgsl
// crates/clank_app/assets/shaders/soil_step.wgsl
const SOIL_SCALE: f32 = 1000000.0; // must equal SOIL_FIXED_SCALE in types.rs
struct SoilCell {
    food_micro: atomic<i32>,
    taint_micro: atomic<i32>,
    scent_micro: atomic<i32>,
    pad: u32,
}
@group(0) @binding(0) var<storage, read_write> soil_buffer: array<SoilCell, 3750>;
@group(0) @binding(1) var soil_display: texture_storage_2d<rgba8unorm, write>;    // Colormap display with vignette
@group(0) @binding(2) var<storage, read> regrowth_table: array<f32, 3750>;     // .0008 + .0022 * bloom(x, y)
@group(0) @binding(3) var<uniform> params: GpuSimParams;

@compute @workgroup_size(8, 8)
fn soil_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= 75u || id.y >= 50u) { return; }
    let k = id.y * 75u + id.x;

    var f = f32(atomicLoad(&soil_buffer[k].food_micro)) / SOIL_SCALE;
    var t = f32(atomicLoad(&soil_buffer[k].taint_micro)) / SOIL_SCALE;
    var s = f32(atomicLoad(&soil_buffer[k].scent_micro)) / SOIL_SCALE;

    // Environmental renewal, random spawn (rolled per cell per tick) & decay:
    f += params.renewal * regrowth_table[k] * (1.0 - f / 1.7);
    if (pcg3d_float(k, SPAWN_STREAM, params.tick) < 0.00013 * params.renewal) {
        f += 0.15 + 0.45 * pcg3d_float(k, SPAWN_AMOUNT_STREAM, params.tick);
    }
    f = clamp(f, 0.0, 2.5);
    t = max(0.0, t * 0.994 - 0.0001);
    s = s * 0.954;

    // 1. Write back persistent atomic state for agent grazing and sensing:
    atomicStore(&soil_buffer[k].food_micro, i32(round(f * SOIL_SCALE)));
    atomicStore(&soil_buffer[k].taint_micro, i32(round(t * SOIL_SCALE)));
    atomicStore(&soil_buffer[k].scent_micro, i32(round(s * SOIL_SCALE)));

    // 2. Output Graded Colormap for Bevy screen presentation:
    let col = evaluate_soil_color(f, t, s, id.xy);
    textureStore(soil_display, id.xy, col);
}

// Eclipse (after soil_main, while params.eclipse > 0, on even ticks): 160 threads each pick a
// random cell k = pcg3d(i, ECLIPSE_STREAM, tick) % 3750 and apply food *= .73,
// taint = clamp(taint + .14, 0, 2) with atomic compare-exchange loops.
```
- [x] **Step 4: Run test to verify it passes** (`gpu_soil_test`, `plan_t3_soil_test`, `gpu_soil_parity_test`)
- [x] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 4: Hybrid Morton Grid, Multi-System LBVH & Spatial Queries

**Files:**
- Create: `crates/clank_app/src/gpu/spatial_index.rs`
- ~~Create: `crates/clank_app/src/gpu/lbvh.rs`~~ (removed on 2026-10-09 with the LBVH)
- Create: `crates/clank_app/assets/shaders/morton_grid.wgsl`
- ~~Create: `crates/clank_app/assets/shaders/lbvh_build.wgsl`~~ (removed on 2026-10-09)
- Create: `crates/clank_app/assets/shaders/spatial_query.wgsl`
- Create: `crates/clank_app/assets/shaders/radix_sort.wgsl` (count and scatter per pass, the scatter computing its own offsets; multi-pass, no decoupled look-back; pass 0 also builds the grid table)
- Create: `crates/clank_app/src/gpu/radix_sort.rs` (`SpatialSort`: pipelines, scratch and histogram buffers; `encode(cpass, max_agents)` records the 8 dispatches into the sub-tick's compute pass, for `params.max_agents` slots; `spatial_keys_size(max_agents)`)
- ~~Modify: `crates/clank_app/assets/shaders/lbvh_aabb.wgsl` (`atomicMin`/`atomicMax` climb, #10)~~ (removed on 2026-10-09)
- Modify: `crates/clank_app/assets/shaders/preamble_clear.wgsl` (drop the `cell_offsets` clear)
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl` (Tier 1 / Tier 2 neighbour search over `cell_slots` and the sorted keys)
- Modify: `crates/clank_app/src/gpu/compute_driver.rs` (`seed_agents_gpu` places two creatures per Seed Life click, within ±11 px, while under the cap; the old body becomes `place_founders_gpu`, which `replenish_agents_gpu` keeps calling with one creature per position; drop `cell_offsets_buf` and the linked-list pipelines; the sort covers `params.max_agents` slots, which can be smaller than the driver's, and its workgroups past them return; record each sub-tick as one compute pass, since each dispatch is its own WebGPU synchronization scope and a separate pass costs the CPU about 0.22 ms to record in a debug build against about 0.04 ms for a dispatch; `active_slots(max_cap)` = max(cap, uploaded creatures), at most the driver's slots, bounds every live creature, and the capacity pass always covers every slot)
- Modify: `crates/clank_app/src/sim.rs` (app frames set `params.max_agents` to `driver.active_slots(max_cap)`, not the driver's 65,536 slots)
- Modify: `crates/clank_app/assets/shaders/preamble_clear.wgsl` (reset `live_count` for the sort's count)
- Test: `crates/clank_app/tests/gpu_spatial_test.rs` (CPU mirror; not evidence for the shaders)
- Test: `crates/clank_app/tests/plan_t4_spatial_test.rs` (real shaders; picks go through the harness's `Gpu::pick`):
  - `test_live_creatures_carry_the_morton_code_of_their_position` (same `floor` quantization as the Rust helper);
  - `test_spatial_keys_are_sorted_morton_slot_pairs_with_dead_slots_last`;
  - `test_the_sort_holds_when_params_cover_fewer_slots_than_the_driver` (a 65536-slot driver with `params.max_agents` 340, as in the app);
  - `test_the_grid_table_holds_every_live_creature_once` (300 creatures in one cell, after 3 earlier ticks: every live creature listed once, in its cell, no dead slots);
  - `test_a_creature_on_the_far_corner_stays_in_the_index` (a creature at exactly (900, 600) is counted live and listed in the grid table);
  - ~~`test_lbvh_is_a_valid_tree_over_the_live_creatures`, `test_lbvh_leaves_are_in_morton_order`, `test_lbvh_boxes_contain_their_children`, `test_lbvh_of_one_creature_is_its_leaf`~~ (removed on 2026-10-09 with the LBVH);
  - `test_picking_returns_the_creature_under_the_cursor`;
  - `test_picking_selects_what_the_rust_engine_selects` (12 cursors: on a creature, 8 px beside one, 26 and 29 px away, in empty space, at the world's left edge; each must match `find_agent_at_position`);
  - `test_extinguish_tool_kills_what_the_reference_kills` (cursors in the middle, at a side seam and at a corner);
  - `test_seed_tool_adds_two_creatures_at_the_cursor`;
  - `test_nearest_neighbour_matches_near`: clustered and isolated scenarios, including a nearest neighbour beyond 100 px, one in a high slot, and 196 creatures in one cell with the nearest holding the cell's highest Morton code; a creature with a different-root neighbour at distance `d` mates only when `d < 18`, as the reference decides from `near()`;
  - `test_density_counts_neighbours_within_100px`: more than 128 creatures in one cell; GPU behaviour depending on density matches the reference.

**Interfaces:**
- Consumes: `GpuAgentState` positions, grid dimensions ($9 \times 6$, cell size $100\text{px}$)
- Produces: Sorted Morton indices (8-byte indirection buffer `vec2u(morton_key, agent_slot)`), `live_count`, grid cell table and `cell_slots`, unified interactive spatial query pass (`spatial_query.wgsl`: picking from the cursor's 3×3 grid cells, and AoE tool queries). The Karras LBVH it first produced was removed on 2026-10-09 (section 3).

- [x] **Step 1: Write failing test for Morton bit-interleaving, empty-cell sentinels, LBVH tree building with tie-breaking, degenerate population guard ($N \le 1$), uncapped picking query with dynamic radius shrinking, and AoE tool queries** (CPU mirror outlines below, plus the real-shader tests listed under **Files**)
```rust
#[test]
fn test_empty_cells_hold_the_sentinel() {
    // Verifies that empty cells in the grid cell table are (0xFFFFFFFF, 0xFFFFFFFF)
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
    // 2. AoE tools (blight, nourish, extinguish, seed) apply their effects (one thread per slot)
}
```
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement Morton encoding, the grid cell table with 0xFFFFFFFF sentinels for empty cells, 8-byte indirection Radix sort, Karras LBVH hierarchy generation with $N \le 1$ early-exit guard and tie-breaking, and Unified `spatial_query.wgsl`** (the LBVH sketches below are the original design; the LBVH was removed on 2026-10-09 and picking reads the grid, section 3)
```wgsl
// In morton_grid.wgsl (coordinate encoding):
// Dead Agent Partitioning: Assign sentinel 0xFFFFFFFFu so tombstone slots sort to the array tail:
@compute @workgroup_size(64)
fn morton_encode(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= params.max_agents) { return; }
    let is_dead = (agent_states[id.x].meta_flags & (1u << 13u)) != 0u;
    if (is_dead) {
        spatial_keys[id.x] = vec2u(0xFFFFFFFFu, id.x); // Excluded from cells 0..53 & LBVH leaves!
    } else {
        let code = min(compute_morton_32(agent_states[id.x].pos_vel.xy), 0xFFFFFFFEu); // stays below the sentinel
        agent_states[id.x].morton_code = code;
        spatial_keys[id.x] = vec2u(code, id.x);
    }
}

// Morton cell boundary clamping guard:
fn get_cell_id(pos: vec2f) -> u32 {
    let gx = min(u32(max(0.0, pos.x) / 100.0), 8u);
    let gy = min(u32(max(0.0, pos.y) / 100.0), 5u);
    return gy * 9u + gx; // Guaranteed 0..53 (safely bounds the 54-entry cell table)
}

// In radix_sort.wgsl: 4 stable LSD passes of 8 bits over spatial_keys[i].x, two dispatches each:
// count per workgroup, then scatter (each workgroup sums the counts it needs and ranks its keys
// with per-digit thread bitmasks). Keys start in slot order, so equal Morton codes stay in slot
// order. Pass 0 also builds the grid cell table: its count adds live keys per cell and to
// live_count, its scatter does a 54-entry exclusive scan into the cell table (start, count) at
// spatial_keys[N..N + 54) and places each live slot into cell_slots at spatial_keys[N + 54..].
// No per-cell cap. Workgroups past params.max_agents return at once.

// In lbvh_build.wgsl:
// Two-Phase LBVH Construction: Phase 1 evaluates hierarchy topology; Phase 2 fits bounding boxes bottom-up in the next dispatch.
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
    let active_count = atomicLoad(&queue_buffer.telemetry.live_count); // live keys at the head of the sorted spatial_keys
    if (active_count < 2u || id.x >= active_count - 1u) {
        return; // Guard against N <= 1 underflow!
    }
    // Phase 2 grows this node's box by atomicMin/atomicMax (#10): start it empty.
    // aabb_min = (3e38, 3e38), aabb_max = (0, 0), count = last - first + 1.
    // Phase 1: Karras 2012 LCP split evaluation, child and parent pointer generation...
}

// In spatial_query.wgsl: Unified interactive pass (Picking & AoE tools)
// Dispatched on Sub-Tick 1 when params.tool_type != 0xFFFFFFFFu
@group(0) @binding(0) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read> lbvh_nodes: array<GpuLbvhNode>;
@group(0) @binding(2) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(0) @binding(3) var<uniform> params: GpuSimParams;
@group(0) @binding(4) var<storage, read> spatial_keys: array<vec2u>; // sorted (morton, slot), live keys first

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

        // Degenerate Population Guard (N <= 1), on the live count of this tick's sorted keys:
        let live_count = atomicLoad(&queue_buffer.telemetry.live_count);
        if (live_count == 0u) {
            queue_buffer.telemetry.selected_agent_idx = 0xFFFFFFFFu;
            queue_buffer.telemetry.selected_agent_id = 0u;
            return;
        }

        var search_r = clamp(16.0 * (params.camera_size[0] / 900.0), 4.0, 24.0);

        if (live_count == 1u) {
            // Direct single-agent evaluation without tree traversal (the one live creature may be in any slot):
            let only = spatial_keys[0].y;
            let agent_pos = agent_states[only].pos_vel.xy;
            let d = toroidal_dist(tool_pos, agent_pos);
            if (d <= search_r) {
                queue_buffer.telemetry.selected_agent_idx = only;
                queue_buffer.telemetry.selected_agent_id = agent_states[only].id;
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

        // Bounded walk: a valid tree has 2n - 1 nodes, so more visits than that means a corrupt
        // tree. Stop rather than loop (an unbounded walk on a cyclic tree froze the whole machine).
        var visits = 0u;
        let max_visits = 2u * live_count;
        while (stack_ptr > 0u && visits < max_visits) {
            visits += 1u;
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
        // Parallel tree/cell query applying atomic updates to enclosed agents in O(log N).
        // Reference semantics: Nourish deposit(food, .28, r3); Blight deposit(taint, .38, r3) and
        // deposit(food, -.14, r2); Extinguish kills every live creature with PLAIN (not toroidal)
        // distance < 23 to the unwrapped cursor, through the dead_claimed CAS, depositing the
        // corpse as die() does. Seed Life (CPU-driven seed_agents_gpu) adds two agents within
        // +-11 px while the population is below the cap.
        apply_aoe_tool(id.x, tool_pos, params.tool_type);
    }
}
```
- [x] **Step 4: Run test to verify it passes** (`gpu_spatial_test`, `plan_t4_spatial_test`)
- [x] **Step 5: Git commit on `feature/bevy-gpu`**

---

### Task 5: Packed 88-Word Vectorized RNN, Combat Resolution, PRNG & Decoupled Birth Pipeline

**Files:**
- Create: `crates/clank_app/src/gpu/agent_pipeline.rs`
- Create: `crates/clank_app/src/gpu/birth_pipeline.rs`
- Create: `crates/clank_app/assets/shaders/preamble_clear.wgsl`
- Create: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Create: `crates/clank_app/assets/shaders/birth_step.wgsl`
- Create: `crates/clank_app/assets/shaders/birth_args.wgsl` (indirect birth dispatch arguments; `BirthDispatch` in `birth_pipeline.rs`)
- Create: `crates/clank_app/assets/shaders/census.wgsl` and `crates/clank_app/src/gpu/census_pipeline.rs` (census pass after `birth_step`, on the frame's final sub-tick) and the pending-free merge (Task 2)
- Modify: `crates/clank_app/src/gpu/compute_driver.rs` (indirect birth dispatch; `replenish_agents_gpu` / `seed_agents_gpu` founder genomes)
- Test: `crates/clank_app/tests/gpu_agent_test.rs` (CPU mirror; not evidence for the shaders)
- Test: `crates/clank_app/tests/gpu_birth_mutation_test.rs` (CPU mirror; not evidence for the shaders)
- Test: `crates/clank_app/tests/plan_t5_agent_test.rs` (real shaders, one tick against `World::evolve` in controlled scenarios, tolerances from the Verification Plan):
  - `test_a_lone_creature_takes_the_reference_tick`: 8 creatures with random genomes, traits and hidden state, including ones crossing each seam;
  - `test_a_kill_matches_the_reference`: victim dies, kill counted and credited, attacker energy and cooldown match;
  - `test_an_asexual_birth_matches_the_reference`: child energy, generation, lineage, placement; parent energy and countdown;
  - `test_dead_slots_are_not_modified`;
  - `test_a_kill_emits_an_audio_voice`;
  - `test_visual_cache_is_packed_like_the_bridge`;
  - `test_damage_is_kept_when_the_victim_runs_after_the_attacker`: the victim's own pass must not overwrite the attacker's `atomicSub`;
  - `test_parents_pay_only_for_births_that_get_a_slot`: at the cap, ready parents keep their energy;
  - `test_births_refused_at_the_cap_are_not_queued_and_spend_no_id`: 10 ready parents, room for 5: 5 births queued, 5 ids spent, each child of a charged parent;
  - `test_more_than_64_births_in_one_sub_tick_all_land`;
  - `test_the_cap_holds_across_sub_ticks`: 32 sub-ticks per frame never put more live creatures than the cap;
  - `test_crossover_and_mutation_draws_are_uncorrelated`: sexual births from parents with all genes +50 and −50; which parent a gene came from is independent of whether it mutated;
  - `test_new_agents_get_reference_founder_genomes`: genes spread over [−58, 58], each trait over its reference range, energy in [34, 52];
  - `test_sexual_births_happen_at_the_reference_rate`: 450 pairs of a ready parent and an able mate of another lineage 10px apart (genes +50 and −50, so a child carrying both signs is a sexual birth; the reference keeps no record of which births were sexual, so a long run cannot measure this): every parent gives birth, and the sexual share is within 7 points of the reference's (about 15%). `test_crossover_and_mutation_draws_are_uncorrelated` uses the same lattice;
  - `test_deposits_keep_cells_within_0_and_3`;
  - `test_corpse_food_lands_where_the_victim_died`;
  - `test_the_census_counts_newborns_and_not_the_dead`: after a tick with 40 births and 5 starvations, population, max generation and living lineages equal the reference's;
  - `test_the_census_counts_every_lineage_in_a_large_world`: 6000 founders, each its own lineage, on 8192 slots: `living_lineages` equals the live creatures' distinct roots;
  - `test_every_creature_in_a_crowd_takes_one_step`: with the agent pass in Morton order, each of 300 creatures takes exactly one step (age +1; energy change as the reference's, except near-tie neighbours).

**Interfaces:**
- Consumes: `GpuSoilCell` storage buffer (point reads, atomic grazing/deposits), Spatial index (sorted keys, grid table, `cell_slots`), `GpuAgentState` (128B) and `GpuAgentGenome` (352B) storage buffers, `GpuAgentAtomic` L2 buffer, `GpuSimParams`
- Produces: Updated kinematics with toroidal wrapping, hidden RNN states from the 326 linearly packed genes, coalesced soil deposits, atomic combat resolution with `dead_claimed` CAS, decoupled SIMD genome mutation, and the census (population, max generation, living lineages)

- [x] **Step 1: Write failing test for the forward pass, combat resolution with double-free CAS, and toroidal coordinate wrapping** (the CPU mirror below, plus the real-shader tests listed under **Files**)
```rust
// crates/clank_app/tests/gpu_agent_test.rs
#[test]
fn test_forward_pass_parity() {
    // Verifies:
    // 1. The forward pass over the 326 linearly packed genes (s += w[p++] * in[i] / 127, tanh(s * .61) / tanh(s * .66)) matches the reference within 1e-4
    // 2. Hardware toroidal wrapping: x - W * floor(x / W) handles negative/positive wraps
    // 3. Combat damage formula, decisive killer attribution, and dead_claimed CAS protection
}
```

- [x] **Step 2: Write failing test for stateless PCG hash & triangular distribution**
```rust
// crates/clank_app/tests/gpu_birth_mutation_test.rs
#[test]
fn test_stateless_pcg_triangular_distribution() {
    // Verifies:
    // 1. PCG 3D hash evaluates deterministically from (id, stream, tick), and distinct streams are uncorrelated
    // 2. r1 + r2 - 1.0 produces zero-centered symmetric triangular distribution T(-1, 0, 1)
    // 3. Genome mutation formula round((r1 + r2 - 1) * 100 * mutRate) matches HTML bounds [-127, 127]
}
```

- [x] **Step 3: Implement `preamble_clear.wgsl` and `agent_step.wgsl` with the Linear-Genome Forward Pass, Toroidal Wrapping, Frustum-Culled Audio & Aggregate Telemetry**
```wgsl
// crates/clank_app/assets/shaders/preamble_clear.wgsl
// Dispatched before agent_step.wgsl to guarantee a global GPU execution barrier:
@group(0) @binding(0) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(1) var<storage, read_write> queue_buffer: ConsolidatedQueue;
@group(1) @binding(0) var<uniform> params: GpuSimParams;

@compute @workgroup_size(64)
fn preamble_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < params.max_agents) {
        atomicStore(&agent_atomics[id.x].mate_claim, 0u);
        atomicStore(&agent_atomics[id.x].mate_energy_milli, 0u);
        // dead_claimed is persistent: only birth_step.wgsl clears it when it reuses the slot.
    }
    if (id.x == 0u) {
        // Active queue depth reset (runs every sub-tick for birth processing):
        atomicStore(&queue_buffer.telemetry.birth_count, 0u);
        // The spatial sort's first pass counts the live keys into it.
        atomicStore(&queue_buffer.telemetry.live_count, 0u);
        // Birth capacity for this sub-tick: the slots on the stack after the previous merge.
        atomicStore(&queue_buffer.telemetry.available_slots, atomicLoad(&queue_buffer.telemetry.freelist_top));

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
    // The instantaneous census (population, max_generation, living_lineages and its root
    // table) is cleared and recounted by the census pass (census.wgsl) after the births of
    // the final sub-tick.
}
```

```wgsl
// In agent_step.wgsl:
// Buffer Splitting (SoA): 7 storage buffers (100% WebGPU portable compliant):
@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;  // 128B
@group(0) @binding(1) var<storage, read> agent_genomes: array<GpuAgentGenome>;       // 352B (clean read-only L2)
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>; // 16B (atomic CAS ground truth)
@group(0) @binding(3) var<storage, read_write> soil_buffer: array<SoilCell, 3750>;
@group(0) @binding(4) var<storage, read> spatial_keys: array<vec2u>;                 // [sorted keys][54 cell ranges][cell_slots]
@group(0) @binding(6) var<storage, read_write> freelist: array<u32>;                 // [stack][pending-free list]
@group(0) @binding(7) var<storage, read_write> queue_buffer: ConsolidatedQueue;      // 1,028.1 KB (telemetry + births + audio)
@group(1) @binding(0) var<uniform> params: GpuSimParams;                             // 96B uniform (separate binding space)

// Hardware toroidal coordinate wrapping (1 cycle via floor):
fn wrap_coords(p: vec2f) -> vec2f {
    return vec2f(
        p.x - 900.0 * floor(p.x / 900.0),
        p.y - 600.0 * floor(p.y / 600.0)
    );
}

// Point probe helper, as the reference reads single cells: wrap, clamp into the grid, atomicLoad.
fn sample_soil_probe(p: vec2f) -> vec4f {
    let q = wrap_coords(p);
    let cx = min(u32(q.x / 12.0), 74u);
    let cy = min(u32(q.y / 12.0), 49u);
    let k = cy * 75u + cx;
    return vec4f(
        f32(atomicLoad(&soil_buffer[k].food_micro)) / SOIL_SCALE,
        f32(atomicLoad(&soil_buffer[k].taint_micro)) / SOIL_SCALE,
        f32(atomicLoad(&soil_buffer[k].scent_micro)) / SOIL_SCALE,
        1.0
    );
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

    // Antennae Feelers (point reads of soil_buffer):
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
    // ... remaining reference inputs. The oscillator uses the number of the tick being computed
    // (the reference evolve() does tick++ before the agents run): sin(f32(step_tick) * 0.08 + f32(id)),
    // with step_tick = params.tick + 1u (the driver writes each sub-tick's own tick into params.tick).

// Forward pass over the 326 linearly packed genes, in reference brain() order:
// gene p is byte p % 4 of word p / 4, sign-extended. N = 15 inputs, Q = 10 hidden, O = 6 outputs.
fn gene(agent_idx: u32, p: u32) -> f32 {
    let word = agent_genomes[agent_idx].packed_genes[p / 4u];
    return f32(extractBits(i32(word), (p % 4u) * 8u, 8u)) / 127.0;
}
var p = 0u;
var new_h: array<f32, 10>;
for (var j = 0u; j < 10u; j += 1u) {
    var s = 0.0;
    for (var i = 0u; i < 15u; i += 1u) { s += gene(agent_idx, p) * ins[i]; p += 1u; }
    for (var k = 0u; k < 10u; k += 1u) { s += gene(agent_idx, p) * h[k]; p += 1u; }
    s += gene(agent_idx, p); p += 1u;
    new_h[j] = tanh(s * 0.61);
}
var out: array<f32, 6>;
for (var j = 0u; j < 6u; j += 1u) {
    var s = 0.0;
    for (var k = 0u; k < 10u; k += 1u) { s += gene(agent_idx, p) * new_h[k]; p += 1u; }
    s += gene(agent_idx, p); p += 1u;
    out[j] = tanh(s * 0.66);
}

// The population census is counted by the census pass after births (not here), so it includes newborns.
// Apex energy record tracking across all sub-ticks:
atomicMax(&queue_buffer.telemetry.apex_record_milli, u32(max(0.0, a_energy) * 1000.0));

// Soil grazing with cell index clamping:
let cx = min(u32(max(0.0, pos.x) / 12.0), 74u);
let cy = min(u32(max(0.0, pos.y) / 12.0), 49u);
let cell_idx = cy * 75u + cx;
let eaten_micro = i32(round(eaten_float * SOIL_SCALE));
atomicSub(&soil_buffer[cell_idx].food_micro, eaten_micro);
atomicAdd(&queue_buffer.telemetry.food_grazed_milli, u32(max(0, eaten_micro / 1000)));

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
            if (pcg3d(agent_idx, AUDIO_STREAM ^ victim_idx, step_tick) % density_filter == 0u) {
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
            // Only the CAS winner counts the death; the slot is recycled after the birth pass.
            atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
            let pending = atomicAdd(&queue_buffer.telemetry.pending_free_count, 1u);
            freelist[params.max_agents + pending] = victim_idx;
        }
    }
}

// Canonical Atomic Energy Model: apply internal net delta (grazing - basal - thrust) to atomic ground truth,
// then clamp atomically. Never plain-store energy: other threads may have subtracted damage or mating costs.
let internal_delta_milli = i32((eaten_float - basal_cost - thrust_cost) * 1000.0);
atomicAdd(&agent_atomics[agent_idx].energy_milli, internal_delta_milli);
atomicMin(&agent_atomics[agent_idx].energy_milli, 110000);

// Strict Single-Writer Death Check (Combat Victim or Starvation):
let already_claimed = atomicLoad(&agent_atomics[agent_idx].dead_claimed);
let current_energy_milli = atomicLoad(&agent_atomics[agent_idx].energy_milli);

if (already_claimed != 0u || current_energy_milli <= 0) {
    if (already_claimed == 0u) {
        let claim_death = atomicCompareExchangeWeak(&agent_atomics[agent_idx].dead_claimed, 0u, 1u);
        if (claim_death.exchanged) {
            atomicAdd(&queue_buffer.telemetry.starvations, 1u); // Only increment if this thread won the death claim!
            atomicAdd(&queue_buffer.telemetry.total_deaths, 1u);
            let pending = atomicAdd(&queue_buffer.telemetry.pending_free_count, 1u);
            freelist[params.max_agents + pending] = agent_idx;
        }
    }
    // Corpse (die()): food clamp(energy * .016 + .6, .3, 2) at radius 2 and taint .1 at radius 1, at this
    // creature's position after its last move; every deposit clamps the cell to [0, 3].
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
        if (pcg3d_float(agent_idx, MATE_ROLL_STREAM, step_tick) < 0.15) {
            // Reference mating: no exclusive claim. Several suitors may take the same mate in one
            // tick, each costing it 6 energy (the old partner_id > agent_id rule halved the rate).
            mate_partner = partner_idx;
            sexual_success = true;
        }
    }

    // Atomic capacity reservation: only a birth that gets a slot proceeds and charges energy,
    // so the cap holds across sub-ticks and nobody pays for a birth that cannot happen.
    var reserved = false;
    var avail = atomicLoad(&queue_buffer.telemetry.available_slots);
    while (avail > 0u) {
        let r = atomicCompareExchangeWeak(&queue_buffer.telemetry.available_slots, avail, avail - 1u);
        if (r.exchanged) { reserved = true; break; }
        avail = r.old_value;
    }

    // Pass-Decoupled Freelist: Mating threads only record intent in the birth queue.
    // Child slot allocation occurs across the compute pass barrier in birth_step.wgsl (Zero ABA / Push-Pop Race)!
    // Not select(): WGSL evaluates both of its arguments, so the atomicAdd would queue births
    // that got no slot (fixed 2026-10-10, see the Verification Plan).
    var queue_idx = 0xFFFFFFFFu;
    if (reserved) { queue_idx = atomicAdd(&queue_buffer.telemetry.birth_count, 1u); }
    if (reserved && queue_idx < 65536u) {
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

// Living Agent: commit state and pack visual_cache (offsets 124..128), exactly as bridge::pack_agent_state:
let r_u8 = u32(clamp(tr0, 0.0, 1.0) * 255.0);
let signal_u8 = u32(clamp(a_signal, 0.0, 1.0) * 255.0);
let e_u8 = u32(clamp(a_energy / 100.0, 0.0, 1.0) * 255.0);
let vis_flags = select(0u, 1u << 24u, a_attack > 0.5) | ((a_birth & 0x7Fu) << 25u);
agent_states[agent_idx].visual_cache = r_u8 | (signal_u8 << 8u) | (e_u8 << 16u) | vis_flags;
```

- [x] **Step 4: Implement Decoupled Birth & Genome Mutation Pass (`birth_step.wgsl`)**
```wgsl
// birth_step.wgsl: 1 workgroup per newborn child (32 threads)

@group(0) @binding(0) var<storage, read_write> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> agent_genomes: array<GpuAgentGenome>;
@group(0) @binding(2) var<storage, read_write> agent_atomics: array<GpuAgentAtomic>;
@group(0) @binding(3) var<storage, read> queue_buffer: ConsolidatedQueue;
@group(0) @binding(4) var<storage, read_write> freelist: array<u32>;
@group(1) @binding(0) var<uniform> params: GpuSimParams;

var<workgroup> shared_child_slot: u32;
var<workgroup> shared_child_id: u32;

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
        // Only a birth with a slot spends an id (as the reference's nextId++). Not select(),
        // which evaluates both arguments.
        shared_child_id = 0u;
        if (slot != 0xFFFFFFFFu) { shared_child_id = atomicAdd(&queue_buffer.telemetry.apex_agent_id, 1u); }
    }
    workgroupBarrier();

    let child_idx = shared_child_slot;
    let child_id = shared_child_id;
    let step_tick = params.tick + 1u;
    // Cannot happen when the agent pass reserved capacity (available_slots), kept as a guard:
    if (child_idx == 0xFFFFFFFFu) { return; }

    // Parallel genome crossover & mutation (326 genes linearly packed in words 0..81; 82..87 spare)
    for (var w = local_id.x; w < 88u; w += 32u) {
        var word_a = agent_genomes[parent_a].packed_genes[w];
        var word_b = select(word_a, agent_genomes[parent_b].packed_genes[w], parent_b != 0xFFFFFFFFu);
        // Per gene byte: crossover (48% from the mate) and triangular mutation, each from its own
        // pcg3d stream keyed by (child id, gene index), so the two draws are uncorrelated.
        var mutated_word = crossover_and_mutate(word_a, word_b, child_id, w, step_tick);

        // Keep the spare bytes after gene 325 zero so unused genes don't drift:
        if (w == 81u) {
            mutated_word = mutated_word & 0x0000FFFFu; // genes 324, 325; bytes 2 & 3 spare
        }
        if (w >= 82u) {
            mutated_word = 0u;
        }
        agent_genomes[child_idx].packed_genes[w] = mutated_word;
    }

    // Trait crossover, morphology & state initialization on thread 0:
    if (local_id.x == 0u) {
        // Crossover 45% and clamp(base + (r1 + r2 - 1) * mutRate * 0.6, 0.03, 0.98)...
        let parent_pos = agent_states[parent_a].pos_vel.xy;
        let child_pos = wrap_coords(parent_pos + vec2f(pcg3d_signed(child_id, CHILD_X_STREAM, step_tick) * 9.0, pcg3d_signed(child_id, CHILD_Y_STREAM, step_tick) * 9.0));
        
        agent_states[child_idx].pos_vel = vec4f(child_pos.x, child_pos.y, 0.0, 0.0);
        agent_states[child_idx].angle_energy = vec4f(pcg3d_float(child_id, CHILD_ANGLE_STREAM, step_tick) * 6.2831853, 24.0, 0.0, 0.0);
        atomicStore(&agent_atomics[child_idx].energy_milli, 24000);
        atomicStore(&agent_atomics[child_idx].dead_claimed, 0u);
        atomicStore(&agent_atomics[child_idx].mate_claim, 0u);
        atomicStore(&agent_atomics[child_idx].mate_energy_milli, 0u);

        // Inherit & mutate traits into agent_states[child_idx].traits; hidden state and signal start at 0...
        agent_states[child_idx].root = agent_states[parent_a].root; // full-width lineage root
        agent_states[child_idx].meta_flags = 0u; // cooldown 0, birth countdown 0 (as agent() in the reference), dead 0, kills 0

        // gen = max(parent gen, mate gen) + 1, as agent() in the reference:
        var parent_gen = agent_states[parent_a].age_gen >> 16u;
        if (parent_b != 0xFFFFFFFFu) { parent_gen = max(parent_gen, agent_states[parent_b].age_gen >> 16u); }
        agent_states[child_idx].age_gen = ((parent_gen + 1u) & 0xFFFFu) << 16u; // age = 0
        agent_states[child_idx].id = child_id; // from atomicAdd(&queue_buffer.telemetry.apex_agent_id, 1u)
        agent_states[child_idx].morton_code = 0u; // Assigned by morton_encode

        // Newborn Initialization Invariant: visual_cache as bridge::pack_agent_state packs a newborn
        // (signal 0, not attacking, birth countdown 0):
        let child_r_u8 = u32(clamp(agent_states[child_idx].traits[0], 0.0, 1.0) * 255.0);
        let child_e_u8 = u32(clamp(24.0 / 100.0, 0.0, 1.0) * 255.0);
        agent_states[child_idx].visual_cache = child_r_u8 | (child_e_u8 << 16u);
    }
}
// After this pass: the merge pass moves the pending-free list onto the stack (Task 2), then the census
// pass counts population, max generation and living lineages on the final sub-tick.
```

- [x] **Step 5: Run tests `gpu_agent_test`, `gpu_birth_mutation_test` and `plan_t5_agent_test` ensuring 100% pass**
- [x] **Step 6: Git commit on `feature/bevy-gpu`**

---

### Task 6: Frustum Culling, Minimap LOD, Dual-Engine UI, Audio & Verification

**Files:**
- Modify: `crates/clank_app/src/rendering.rs` (camera frustum culling, per slot; no LBVH cull or Morton-ordered draw, see Subsystem 2)
- Create: `crates/clank_app/src/audio.rs` (Consolidated 256-voice queue consumption & granular Bevy audio playback)
- Modify: `crates/clank_app/src/lib.rs` (register `pub mod audio;`)
- Modify: `crates/clank_app/src/ui.rs` (Top bar toggle: `ENGINE: RUST` / `ENGINE: GPU`, 32x speed picking, GPU-mode GENERATION / LINEAGES from telemetry; the minimap radar was removed on 2026-10-06)
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`, `birth_step.wgsl`, `preamble_clear.wgsl`, `spatial_query.wgsl`, `src/gpu/types.rs`, `src/gpu/frame_readback.rs` (the chronicle queue and its readback)
- Modify: `crates/clank_app/src/sim.rs` (`apply_frame_readback` stores `max_generation` and `living_lineages` and adds the frame's chronicle events to `world.events`; `SimWorld::max_generation()` and `SimWorld::living_lineages()` return them in GPU mode and the CPU world's values in Rust mode, as `active_population()` does, and the UI reads them; GPU mode counts `world.eclipse` down)
- Modify: `crates/clank_app/src/api.rs` (Telemetry reporting for active engine)
- Test: `crates/clank_app/tests/dual_engine_test.rs` (bridge-level; not evidence for the shaders)
- Test: `crates/clank_app/tests/plan_t6_integration_test.rs` (real shaders):
  - `test_gpu_mode_stats_follow_the_gpu`: after GPU ticks with births, the GENERATION and LINEAGES values the UI shows equal the live GPU creatures' maximum generation and distinct roots;
  - `test_telemetry_census_matches_the_live_creatures`: `population` equals the live slots after the tick, newborns included, and `apex_record_milli` equals the highest live energy within a tick's lag;
  - `test_audio_voices_reach_the_playback_queue`: kill and birth voices in view arrive in `AudioVoiceQueue` through `apply_frame_readback`; voices out of view do not;
  - `test_a_kill_milestone_is_chronicled_like_the_reference`: an attacker's 8th kill gives the reference's event (tick, type 3, lineage, 8);
  - `test_generation_milestones_are_chronicled_at_the_reference_rate`: 450 births of generation 12 give 16–60 events (the reference expects about 36), each naming generation 12 and the parent's lineage;
  - `test_gpu_events_reach_the_chronicle`: in the GPU-mode app, the kill milestone reaches `world.events`;
  - `test_long_run_matches_the_reference_statistically`: from the same seed-42 world, population, cumulative births, kills and total soil food are within ±20% of the reference at ticks 200, 400, 600, 800 and 1000 (or within 3√n when that is larger: kills stay in single digits for hundreds of ticks). Also strengthens `long_run_behavior_test`.
- Test: `crates/clank_app/tests/gpu_capacity_test.rs` (the capacity slider, Task 2)

**Interfaces:**
- Consumes: `SimWorld` (CPU), `GpuSimWorld` (GPU), `ConsolidatedQueue` (1,053,824B) via the non-blocking staging ring (Task 8 Step 2), `GpuAgentState` (128B) storage buffer
- Produces: Camera frustum-culled rendering streaming only 128B `GpuAgentState` ($4.0\times$ bandwidth boost, skipping genomes), unified 128-byte telemetry, 32x speed visual consistency picking, zoom-modulated audio soundscape, HUD metrics, and live engine switching

- [x] **Step 1: Write failing test for live dual-engine hot-swapping, frustum culling, telemetry readback, and 32x speed picking**
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
    // 1. 128-byte GpuTelemetry reads back via the non-blocking staging ring (Task 8 Step 2) without CPU pipeline stalls
    // 2. apex_record_milli accurately reflects atomicMax of creature energy
    // 3. max_generation and living_lineages (distinct full-width roots) match the live creatures without scanning the agent buffer on the CPU
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
fn test_reference_mating() {
    // Verifies:
    // 1. Reference mating: several suitors may share one mate in a tick, each costing it 6 energy
    // 2. Seamless asexual fallback (child(a, null)) when no partner qualifies
}
```
- [x] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement engine switch in Bevy UI, bi-directional state bridge (`sync_rust_to_gpu` synthesizing `visual_cache` / `sync_gpu_to_rust`), the non-blocking staging ring (Task 8 Step 2) for 128B telemetry and 4 KB audio readbacks without CPU stalls, GPU-mode GENERATION / LINEAGES from `max_generation` / `living_lineages`, direct 128-byte `GpuAgentState` streaming for per-slot frustum culling and instanced dart rendering loading contiguous `vec2u(root, visual_cache)` ($4.0\times$ bandwidth boost over reading full 480B agents, bypassing Cache Line 0 traits and saving ~72 MB/frame vertex fetch bandwidth), 32x speed picking locking on Sub-Tick 1 with `(slot_idx, agent_id)` identity guard, Two-Tier uncapped LBVH distance disambiguation, Bevy audio playback reading 256-voice queue, connect frustum culling**
  - *Open (verified 2026-10-11):* built except the Morton-ordered draw: `creature_render.wgsl` culls and draws in slot order. The "Two-Tier uncapped LBVH distance disambiguation" went with the LBVH; picking reads the grid (Task 4).
- [ ] **Step 4: Run all workspace tests (`cargo test --workspace`) ensuring 100% pass** *(withdrawn: forbidden by `AGENTS.md` §10; these tests did not exercise the shaders)*
- [ ] **Step 5: Run the debug build (`cargo run -p clank_app`; never `--release` to mask a debug failure) in GPU mode and confirm with the user on screen (Visual & Manual Confirmation checklist); a `POST /screenshot` capture may supplement but does not replace the live check**
  - *Open:* screenshots through the API covered Rust/GPU parity, speeds 1 and 32, picking and NEW WORLD (2026-10-10). Still to confirm with the user on screen: zoom/pan pop-in, dragging the AoE tools, zoom-modulated and seam-panned audio, the cap change, dense-swarm picking at 32x, trail length at 1x vs 32x.
- [ ] **Step 6: Git commit on `feature/bevy-gpu`**

---

### Task 8: Zero-Copy Presentation (one device, GPU-resident rendering, no per-frame blocking readbacks)

**Why:** The plan promised zero-copy rendering, but the app never had it. `GpuComputeDriver::create_for_world` requests its **own** `wgpu::Device`, so Bevy's renderer cannot bind the simulation buffers. Every GPU-mode frame therefore makes these blocking `poll(wait_indefinitely)` round trips:
- the cull count, the visible darts and their trails (`update_agent_mesh_system`);
- the soil colormap every second tick (`copy_soil_display_rgba`);
- telemetry, audio voices and the selected creature (`sim_step_system`).

The CPU then rebuilds four Bevy meshes from those copies. That CPU shadow of GPU state caused the speed-dependent tails (trails sampled per frame). It also leaves Rust mode and GPU mode drawn by different code paths (`generate_*_from_gpu_states` vs `generate_*_from_instances`).

**Target architecture:**
1. **One device.** In the app the driver is built from Bevy's `RenderDevice`/`RenderQueue` (both are inserted into the main world by `bevy_render`). Headless tests keep a standalone device.
2. **Simulation state stays on the GPU; presentation reads it in place.** A new `CreatureRenderer` (raw `wgpu`, so headless tests can drive it) owns only presentation buffers. Each frame, on the render thread, it:
   - culls the driver's `agent_states_buf` against the arena camera;
   - gathers the matching `agent_trails_buf` entries;
   - writes four indirect draw-argument records.

   Four layers are then drawn with `draw_indirect` from `Transparent2d` phase items. Their sort keys put them between the vignette (z = -5) and the particles (z = 5): glow -2.2, trails -2.05, body -2.0, outline -1.9.
3. **Both engines use the same renderer.** In Rust mode the CPU world is written into the same agent and trail buffers each frame (CPU→GPU writes are inherent to a CPU simulation). The two engines can then only differ in simulation, never in drawing.
4. **Soil stays on the GPU.** `soil_step.wgsl` writes `soil_display` as `rgba8unorm` with the CPU colormap's truncation. In GPU mode the render thread copies it GPU→GPU into the soil sprite's `Rgba8UnormSrgb` texture (copy-compatible formats). The sprite, the vignette and the z-order are unchanged.
5. **Non-blocking readback ring.** Telemetry, audio voices and the selected slot are copied into a 3-deep staging ring with `map_async`, and consumed one frame later with `poll(PollType::Poll)`. The only frame-path wait left is back-pressure, when three frames are already in flight.

**Bevy 0.19 facts this task relies on (verified in `~/.cargo/registry/src/*/bevy_render-0.19.1`, `bevy_core_pipeline-0.19.1`):**
- **No `render_graph` module.** Per-camera rendering is the `Core2d` schedule, with sets `Core2dSystems::{Prepass, MainPass, EarlyPostProcess, PostProcess}`. Systems in it take `ViewQuery<D, F>` and `RenderContext` (`ctx.command_encoder()`, `ctx.begin_tracked_render_pass`), as `main_transparent_pass_2d` does.
- **Device resources in both worlds.** `RenderDevice` and `RenderQueue(Arc<WgpuWrapper<Queue>>)` are inserted into both worlds (`settings.rs::unpack_into`). `RenderDevice::from(wgpu::Device)` and `WgpuWrapper::new` exist, and `wgpu::Device`/`wgpu::Queue` are `Clone`.
- **Adapter limits.** The default `WgpuSettingsPriority::Functionality` requests the adapter's limits, so the existing compute pipelines (8 storage buffers in `agent_step`) fit on Bevy's device.
- **Phase items.** `Transparent2d { sort_key, entity, pipeline, draw_function, batch_range, extracted_index, extra_index, indexed }` is added with `ViewSortedRenderPhases::<Transparent2d>::get_mut(&view.retained_view_entity)?.add_transient(..)`.
  - **`SortedRenderPhase::render_range` skips items with an empty `batch_range`**, so creature items use `0..1`.
  - `CachedRenderPipelineId::INVALID` is valid for items whose render command sets its own pipeline.
- **Wrappers.** `From<wgpu::{Buffer, BindGroup, RenderPipeline, Texture, TextureView}>` exist for the Bevy wrappers, and `TrackedRenderPass::wgpu_pass()` exposes the raw pass. One `draw_layer(&mut wgpu::RenderPass)` therefore serves both the Bevy phase and offscreen tests.
- **Pass compatibility.** Transparent-pass pipelines must match the pass:
  - color format `ExtractedView::target_format`;
  - `Msaa::samples()`;
  - depth `CORE_2D_DEPTH_FORMAT` (`Depth32Float`), `depth_write_enabled: Some(false)`, `depth_compare: Some(CompareFunction::GreaterEqual)`, as `bevy_sprite_render`'s `SpritePipeline` uses.
- **Image usages.** Bevy `Image` textures carry `TEXTURE_BINDING | COPY_DST | COPY_SRC`.

**Files:**
- Create: `crates/clank_app/assets/shaders/creature_render.wgsl`. Cull/gather/argument compute plus the four layer vertex shaders and one fragment shader.
- Create: `crates/clank_app/src/gpu/creature_render.rs`. `CreatureRenderer`, `CreatureSource`, `CreatureRenderParams`, `CreatureLayerKind`, `CreatureDrawKey`.
- Create: `crates/clank_app/src/creature_pass.rs`. Bevy integration: extraction, `Transparent2d` queueing, render command, `Core2d` cull and soil-copy system.
- Create: `crates/clank_app/tests/support/cpu_creature_mesh.rs`. A frozen copy of today's CPU mesh generators: the visual reference for parity tests.
- Create tests:
  - `crates/clank_app/tests/gpu_shared_device_test.rs`
  - `crates/clank_app/tests/gpu_frame_readback_test.rs`
  - `crates/clank_app/tests/creature_render_test.rs`
  - `crates/clank_app/tests/gpu_soil_display_test.rs`
- Modify:
  - `crates/clank_app/src/gpu/compute_driver.rs`: shared-device constructor, readback ring and counter, rgba8 soil display; drop the cull, instance and render code.
  - `crates/clank_app/src/gpu/mod.rs`, `crates/clank_app/src/lib.rs`, `crates/clank_app/src/main.rs`.
  - `crates/clank_app/src/sim.rs`: driver creation, readback ring consumption.
  - `crates/clank_app/src/rendering.rs`: remove the CPU agent meshes and `update_agent_mesh_system`; GPU-mode soil no longer touches the `Image`.
  - `crates/clank_app/src/gpu/bridge.rs`: `render_snapshot`.
  - `crates/clank_app/assets/shaders/soil_step.wgsl`: `rgba8unorm` display.
  - `crates/clank_app/assets/shaders/spatial_query.wgsl`: drop `frustum_cull_*`.
  - `crates/clank_app/assets/shaders/trail_step.wgsl`: drop `trail_gather_main`.
- Delete: `crates/clank_app/assets/shaders/dart_instanced.wgsl` and its pipeline, `render_darts_instanced`, `view_uniform_buf`, `dart_template_buf`.
- Migrate tests:
  - `gpu_culling_test`, `instanced_rendering_test`, `gpu_trail_and_reset_test` (trail part), `rendering_test`, `tdd_behavior_and_graphics_parity_test` (mesh-layer test) move to `CreatureRenderer`.
  - `gpu_tools_and_indirection_test` (indirect-args parts) moves to `CreatureRenderer`.
  - `soil_rendering_test` moves to rgba8.

**Interfaces (names later steps rely on):**
```rust
// gpu/compute_driver.rs
impl GpuComputeDriver {
    pub fn request_standalone_device() -> Option<(wgpu::Device, wgpu::Queue)>;
    pub fn from_render_device(device: &RenderDevice, queue: &RenderQueue, soil_cols: u32, soil_rows: u32, max_agents: u32) -> Self;
    pub fn blocking_readback_waits(&self) -> u64;               // every blocking GPU wait adds one
    pub fn request_frame_readback(&self, selected_slot: Option<u32>, picking: bool);
    pub fn poll_frame_readbacks(&self) -> Vec<FrameReadback>;   // completed frames, oldest first, never waits
    pub fn pending_frame_readbacks(&self) -> usize;             // requested, not yet returned
    pub fn creature_source(&self) -> CreatureSource;            // states, atomics, trails, generation
    pub fn upload_render_snapshot(&self, states: &[GpuAgentState], trails: &[GpuAgentTrail]) -> u32; // returns slot_count
    // GPU mode culls slot_count = max_agents (as dispatch_culling did); no high-water tracking.
}
// gpu/frame_readback.rs
pub struct FrameReadback { pub telemetry: GpuTelemetry, pub audio: Vec<AudioVoice>, pub selected: Option<SelectedSlotReadback>, pub picking: bool }
pub struct SelectedSlotReadback { pub slot: u32, pub state: GpuAgentState, pub genome: GpuAgentGenome, pub atomic: GpuAgentAtomic, pub trail: GpuAgentTrail }

// gpu/creature_render.rs
#[derive(Resource, Clone)] pub struct CreatureSource { pub agent_states: wgpu::Buffer, pub agent_atomics: wgpu::Buffer, pub agent_trails: wgpu::Buffer,
    pub soil_display: Option<wgpu::Texture>, pub slot_count: u32, pub world_size: [f32; 2], pub generation: u64 }
#[repr(C, align(16))] #[derive(Clone, Copy, Pod, Zeroable)] pub struct CreatureRenderParams {
    pub clip_from_world: [[f32; 4]; 4], pub camera_pos: [f32; 2], pub camera_size: [f32; 2], pub world_size: [f32; 2], pub slot_count: u32, pub _pad: u32 } // 96 B
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)] pub enum CreatureLayerKind { Glow, Trail, Body, Outline }
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)] pub struct CreatureDrawKey { pub color_format: wgpu::TextureFormat, pub samples: u32, pub depth: Option<wgpu::TextureFormat> }
impl CreatureRenderer {
    pub fn new(device: &wgpu::Device, max_instances: u32) -> Self;
    pub fn encode_cull(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, source: &CreatureSource, params: &CreatureRenderParams);
    pub fn ensure_draw_pipelines(&mut self, device: &wgpu::Device, key: CreatureDrawKey);
    pub fn draw_layer(&self, pass: &mut wgpu::RenderPass<'_>, key: CreatureDrawKey, layer: CreatureLayerKind);
    pub fn readback_visible(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (Vec<GpuDartInstance>, Vec<GpuAgentTrail>); // diagnostics/tests only
    pub fn readback_draw_args(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [[u32; 4]; 4];                   // diagnostics/tests only
}
impl CreatureLayerKind { pub const ALL: [Self; 4]; pub fn vertices_per_instance(self) -> u32; pub fn sort_z(self) -> f32; }
```

- [x] **Step 1: One device for simulation and rendering**
  - **Failing test** `gpu_shared_device_test.rs`:
    ```rust
    use std::sync::Arc;
    use bevy::prelude::*;
    use bevy::render::renderer::{RenderDevice, RenderQueue, WgpuWrapper};
    use clank_app::gpu::compute_driver::GpuComputeDriver;
    use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};

    /// Binding a driver buffer through Bevy's device panics (wgpu: resource from another device)
    /// unless the driver was built on that device.
    fn bind_with(render_device: &RenderDevice, buffer: &wgpu::Buffer) -> Option<wgpu::Error> {
        let device = render_device.wgpu_device();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }],
        });
        let _ = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }] });
        bevy::tasks::block_on(scope.pop())
    }

    #[test]
    fn test_app_builds_gpu_driver_on_bevy_render_device() {
        let Some((device, queue)) = GpuComputeDriver::request_standalone_device() else { return };
        let render_device = RenderDevice::from(device);
        let render_queue = RenderQueue(Arc::new(WgpuWrapper::new(queue)));
        let mut app = App::new();
        app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin)
            .insert_resource(render_device.clone()).insert_resource(render_queue);
        app.world_mut().resource_mut::<SimWorld>().active_engine = ActiveEngine::Gpu;
        app.update();
        let gpu = app.world().resource::<GpuDriverResource>();
        let driver = gpu.driver.as_ref().expect("driver created");
        assert!(bind_with(&render_device, &driver.agent_states_buf).is_none(), "driver buffers live on Bevy's device");
    }
    ```
  - Run `cargo test -p clank_app --test gpu_shared_device_test`. Expected: compile error (`request_standalone_device` missing). After stubbing that helper, expect a cross-device validation error or panic.
  - **Implement:**
    - Split adapter/device creation out of `create_for_world` into `request_standalone_device()`. It prefers a card that drives no display (Metal `isHeadless`, through `device_is_headless`): headless work on the display's card freezes the screen while it runs (2026-10-08: a 65,000-creature probe hitched the whole Mac Pro, whose second D500 sat idle). Test: `gpu_standalone_device_test`.
    - Add `from_render_device`:
      ```rust
      pub fn from_render_device(device: &RenderDevice, queue: &RenderQueue, soil_cols: u32, soil_rows: u32, max_agents: u32) -> Self {
          let device = device.wgpu_device().clone();
          let queue = wgpu::Queue::clone(&queue.0);
          Self::new_with_grid(Arc::new(device), Arc::new(queue), max_agents, soil_cols, soil_rows)
      }
      ```
    - In `sim.rs`, add an `ensure_gpu_driver` system (runs before `sim_step_system` in both engine modes). It (re)creates the driver when it is missing or the soil grid changed:
      - with `from_render_device` when `Option<Res<RenderDevice>>` and `Option<Res<RenderQueue>>` are present;
      - otherwise with `create_for_world`.

      Remove the creation code from the GPU branch of `sim_step_system`.
  - Run the test (PASS) plus `gpu_trail_and_reset_test`, `sim_test`, `dual_engine_test`. Commit `feat(gpu): build the compute driver on Bevy's render device`.

- [x] **Step 2: Non-blocking frame readback ring and readback accounting**
  - **As built (2026-10-06)** — differences from the text below:
    - The counter counts blocking *waits*, not bytes (`blocking_readback_waits`). All 17 `poll(wait_indefinitely)` sites in `compute_driver.rs` go through one private `wait_blocking()`, and a grep for `wait_indefinitely` finds only that helper.
    - The ring lives in `gpu/frame_readback.rs` (`ReadbackRing`, 4816-byte buffers). `upload_state` bumps its epoch, so frames from a replaced world are dropped as they land.
    - `request_frame_readback` takes `picking`, so a picking answer is applied only from the frame that ran the query. When the picked id changes, the cache is cleared until that creature's snapshot arrives.
    - Frames are applied before the pause check (so the ring drains while paused) and after `flush_gpu_to_rust` when switching to the Rust engine.
    - Agent replenishment drains the ring right after `replenish_agents_gpu`, which already waits. Without that, pre-seed frames landing afterwards reset `gpu_population` and seeded again (3 creatures → 27 instead of 15). Covered by `test_agent_replenishment_seeds_once_despite_lagged_frames`.
    - Extra tests: `test_ring_returns_each_frame_once_in_order` (driver level: 30 requests → 30 frames, in order, the last one equal to a synchronous read), plus the agent replenishment test above. Red-checked by mutation: a blocking telemetry read in the frame path fails the first test (40 waits); returning each frame twice fails the ordering test (60 ≠ 30).
    - **Residual:** `GpuSimParams.agent_count` (reproduction cap) comes from the newest landed frame, so it now lags by up to `RING_DEPTH` frames instead of one. The fix is to have the passes read the population from `queue_buffer`; this is not done yet.
  - **Failing test** `gpu_frame_readback_test.rs`:
    ```rust
    use bevy::prelude::*;
    use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};

    fn blocking_bytes(app: &App) -> u64 {
        app.world().resource::<GpuDriverResource>().driver.as_ref().unwrap().blocking_readback_bytes()
    }

    /// A running GPU-mode frame never stalls the CPU on a readback: telemetry, audio and the
    /// selected creature arrive through the async ring at most a frame or two later.
    #[test]
    fn test_gpu_frames_make_no_blocking_readbacks() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
        app.world_mut().resource_mut::<SimWorld>().active_engine = ActiveEngine::Gpu;
        app.update();
        if app.world().resource::<GpuDriverResource>().driver.is_none() { return; }
        for _ in 0..5 { app.update(); }

        let first_id = app.world().resource::<SimWorld>().world.agents[0].id;
        {
            let mut sim = app.world_mut().resource_mut::<SimWorld>();
            sim.selected_agent_slot = Some(0);
            sim.selected_agent_id = Some(first_id);
        }
        let before = blocking_bytes(&app);
        for _ in 0..40 { app.update(); }
        assert_eq!(blocking_bytes(&app) - before, 0, "GPU-mode frames must not block on readbacks");

        let sim = app.world().resource::<SimWorld>();
        assert!(sim.gpu_population > 0, "population arrives through the ring");
        assert!(sim.world.births >= 72, "birth totals arrive through the ring");
        let cache = sim.selected_agent_cache.as_ref().expect("selected creature arrives through the ring");
        assert_eq!(cache.id, first_id);
    }
    ```
  - Run it. Expected: compile error (`blocking_readback_bytes` missing). With only the counter added, the assertion fails: `readback_telemetry` and `readback_agent_slot` block every frame.
  - **Implement:**
    - Add `blocking_readback_bytes: AtomicU64` and one private `read_buffer_blocking(&self, src: &wgpu::Buffer, offset: u64, len: u64) -> Vec<u8>`. Route every `readback_*`, `copy_soil_display_rgba` and the freelist readbacks for new agents through it, so the counter is complete by construction.
    - Add the ring:
      - three `FRAME_READBACK_BYTES` (128 telemetry + 4096 audio + 128 state + 352 genome + 16 atomic + 96 trail = 4816, rounded up to 4864) `MAP_READ | COPY_DST` buffers behind a `Mutex<[RingSlot; 3]>`;
      - `RingSlot { state: Free | Pending { selected: Option<u32>, ready: Arc<AtomicBool> } }`.
    - `request_frame_readback`:
      - takes the next `Free` slot, or, when all three are pending, waits on the oldest (back-pressure; counted as blocking);
      - encodes `copy_buffer_to_buffer` for telemetry (queue offset 0, 128 B), audio (queue offset `128 + 65536 * 16`, 4096 B), and the selected slot's state, genome, atomic and trail;
      - submits, then calls `map_async(.., move |r| { if r.is_ok() { ready.store(true) } })`.
    - `poll_frame_readbacks` calls `device.poll(wgpu::PollType::Poll)`, then drains `Pending` slots whose `ready` is set, **in request order**, into `FrameReadback`s and unmaps them.
    - In `sim_step_system` GPU branch:
      - after `dispatch_sub_ticks`, call `request_frame_readback(sim.selected_agent_slot)`;
      - then, for each `poll_frame_readbacks()` result in order, apply exactly what the blocking code applied (kills `+=`, births `= gpu_births_base + total_births`, population, audio sparks and voices, picking, selection cache with identity guard and trail). Each frame's per-frame `kills` is applied once because every ring slot is consumed once.
    - `upload_world_to_gpu` drains and discards in-flight ring slots, so telemetry from a replaced world is never applied.
  - Run the test (PASS), then `gpu_trail_and_reset_test`, `picking_test`, `dual_engine_test`, `long_run_behavior_test`. Commit `perf(gpu): consume telemetry, audio and selection through an async readback ring`.

- [x] **Step 3: `CreatureRenderer`: GPU-resident cull, gather and four-layer draw (headless)**
  - **As built (2026-10-06)** — differences from the text below:
    - **Interfaces.** `creature_source()` takes no arguments; the slot count travels in `CreatureRenderParams`. `render_slot_count` was dropped: GPU mode culls all `max_agents` slots, as `dispatch_culling` already did. `upload_render_snapshot` also clears `agent_atomics` for slots `0..n`, so stale death claims from an earlier GPU session can't hide creatures. It is only valid while `!is_initialized()`.
    - **Bridge.** It gained `pack_agent_state(a, world_size)`, extracted from `sync_rust_to_gpu` and shared with `render_snapshot`.
    - **Shader.** The draw bindings are 8–10, so the compute (0–7) and draw groups share one module without binding collisions. The layer vertex counts are written explicitly in `finalize_args_main` rather than through a dynamically indexed `const` array.
    - **Parity test.** It uses the Rust-mode path (`render_snapshot` → `upload_render_snapshot`) and sets birth, attack, signal and sight on 12 creatures so halos, red borders and antennae appear.
      - Measured difference from the frozen CPU meshes: **0.0000%**, pixel-identical.
      - The planned 0.2% bound let 5 of 9 visible defects through. The test now uses **0.01%**: every one of these mutations fails it — halo growth or alpha, trail alpha, trail wrap, antenna length, border colour, notch depth, body alpha, glow radius, rotation sign. The smallest changes about 200 pixels.
      - The cull tests catch mutations of the owner reset, `dead_claimed`, `slot_count`, the toroidal test and the args count.
    - **Found on the way.** The Rust bridge packed `visual_cache` unlike the GPU engine (see the play-test defects above); fixed first, with `visual_cache_packing_test`.
  - **Frozen reference.** Copy today's `generate_glow_mesh_from_instances`, `generate_swarm_trails_from_gpu` (with `push_trail_segments`), `generate_dart_mesh_from_instances` and `generate_outline_mesh_from_instances` verbatim into `tests/support/cpu_creature_mesh.rs`. Each test that uses them includes it with `#[path = "support/cpu_creature_mesh.rs"] mod cpu_creature_mesh;`.
  - **Failing test** `creature_render_test.rs`. It covers culling, trails, indirect args and pixel parity against the frozen CPU meshes:
    ```rust
    #[path = "support/cpu_creature_mesh.rs"]
    mod cpu_creature_mesh;
    use clank_app::gpu::bridge::sync_rust_to_gpu;
    use clank_app::gpu::compute_driver::GpuComputeDriver;
    use clank_app::gpu::creature_render::*;
    use clank_app::sim::SimWorld;

    const W: u32 = 900;
    const H: u32 = 600;
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    fn ortho(world: [f32; 2]) -> [[f32; 4]; 4] {
        bevy::math::Mat4::orthographic_rh(0.0, world[0], 0.0, world[1], -1000.0, 1000.0).to_cols_array_2d()
    }
    // Standard offscreen helpers (bodies written in this step, no simulation logic in them):
    fn target(device: &wgpu::Device) -> wgpu::Texture { /* W×H FORMAT texture, RENDER_ATTACHMENT | COPY_SRC */ }
    fn clear_pass(view: &wgpu::TextureView) -> wgpu::RenderPassDescriptor<'_> { /* one color attachment, LoadOp::Clear(opaque soil-ish color), no depth */ }
    fn read_rgba(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> { /* padded copy_texture_to_buffer + blocking map */ }
    /// Draws CPU-built [pos.xyz, rgba] vertex lists with the same blend state Bevy's ColorMaterial
    /// uses (vertex color × white, `BlendState::ALPHA_BLENDING`) in the given layer order.
    fn draw_reference(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, clip: [[f32; 4]; 4],
                      layers: &[(wgpu::PrimitiveTopology, Vec<[f32; 3]>, Vec<[f32; 4]>)]) { /* minimal pos+color pipeline */ }

    fn differing_pixel_fraction(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len() / 4;
        let diff = (0..n).filter(|i| (0..4).any(|c| (a[i * 4 + c] as i32 - b[i * 4 + c] as i32).abs() > 2)).count();
        diff as f64 / n as f64
    }

    #[test]
    fn test_gpu_layers_match_the_cpu_meshes_pixel_for_pixel() {
        let Some(driver) = GpuComputeDriver::create_for_world(75, 50, 4096) else { return };
        let mut sim = SimWorld::new(42);
        sim.step(30); // trails, attacks, births and halos present
        let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
        params.max_agents = driver.max_agents;
        driver.upload_state(&states, &genomes, &atomics, &soil, &params);
        let trails = clank_app::gpu::bridge::render_snapshot(&sim).1;
        let slot_count = driver.upload_render_snapshot(&states, &trails);

        let (device, queue) = (driver.device.as_ref(), driver.queue.as_ref());
        let mut renderer = CreatureRenderer::new(device, driver.max_agents);
        let world = [W as f32, H as f32];
        let rp = CreatureRenderParams { clip_from_world: ortho(world), camera_pos: [450.0, 300.0], camera_size: world,
                                        world_size: world, slot_count, _pad: 0 };
        let source = driver.creature_source(slot_count, world, false);
        let key = CreatureDrawKey { color_format: FORMAT, samples: 1, depth: None };
        renderer.ensure_draw_pipelines(device, key);

        let gpu_tex = target(device);
        let mut enc = device.create_command_encoder(&Default::default());
        renderer.encode_cull(device, queue, &mut enc, &source, &rp);
        {
            let view = gpu_tex.create_view(&Default::default());
            let mut pass = enc.begin_render_pass(&clear_pass(&view));
            for layer in CreatureLayerKind::ALL { renderer.draw_layer(&mut pass, key, layer); }
        }
        queue.submit([enc.finish()]);

        let (instances, gathered) = renderer.readback_visible(device, queue);
        assert_eq!(instances.len(), sim.world.agents.len(), "whole world in view: every creature visible");
        for (inst, t) in instances.iter().zip(&gathered) {
            assert_eq!(t.owner_id & 0x00FF_FFFF, inst.pad1[0] & 0x00FF_FFFF, "gathered trail belongs to the dart");
        }

        let (mut gp, mut gc, mut tp, mut tc, mut bp, mut bc, mut op, mut oc) = Default::default();
        cpu_creature_mesh::generate_glow_mesh_from_instances(&instances, H as f32, &mut gp, &mut gc);
        cpu_creature_mesh::generate_swarm_trails_from_gpu(&instances, &gathered, W as f32, H as f32, &mut tp, &mut tc);
        cpu_creature_mesh::generate_dart_mesh_from_instances(&instances, H as f32, &mut bp, &mut bc);
        cpu_creature_mesh::generate_outline_mesh_from_instances(&instances, H as f32, &mut op, &mut oc);
        let ref_tex = target(device);
        draw_reference(device, queue, &ref_tex, rp.clip_from_world, &[
            (wgpu::PrimitiveTopology::TriangleList, gp, gc), (wgpu::PrimitiveTopology::LineList, tp, tc),
            (wgpu::PrimitiveTopology::TriangleList, bp, bc), (wgpu::PrimitiveTopology::LineList, op, oc)]);

        let frac = differing_pixel_fraction(&read_rgba(device, queue, &gpu_tex), &read_rgba(device, queue, &ref_tex));
        assert!(frac < 0.002, "GPU layers differ from the CPU meshes on {:.3}% of pixels", frac * 100.0);
    }

    #[test]
    fn test_cull_respects_camera_dead_flags_and_writes_indirect_args() {
        // Port gpu_culling_test's scenarios: off-screen, toroidal-edge, dead-flag, visual_cache == 0
        // and dead_claimed agents. Assert the visible id set, and that draw_args[i] ==
        // (vertices_per_instance, visible, 0, 0) for each layer.
    }
    ```
    Write the last test in full by porting the existing `gpu_culling_test.rs` scenarios onto `encode_cull` and `readback_visible`, plus an assertion on `CreatureRenderer::readback_draw_args()`.
  - Run `cargo test -p clank_app --test creature_render_test`. Expected: compile error (module missing).
  - **Implement** `creature_render.wgsl`. Structs `GpuAgentState`, `GpuAgentTrail` and `DartInstance` are copied from `trail_step.wgsl`/`spatial_query.wgsl`. `AgentAtomicView` is the read-only view of `GpuAgentAtomic` with plain `u32` fields.
    ```wgsl
    struct RenderParams { clip_from_world: mat4x4f, camera_pos: vec2f, camera_size: vec2f, world_size: vec2f, slot_count: u32, pad: u32 }
    struct CullCount { count: atomic<u32>, pad0: u32, pad1: u32, pad2: u32 }
    struct DrawArgs { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 }

    // Compute group (CreatureRenderer::cull_layout)
    @group(0) @binding(0) var<uniform> params: RenderParams;
    @group(0) @binding(1) var<storage, read> agent_states: array<GpuAgentState>;
    @group(0) @binding(2) var<storage, read> agent_atomics: array<AgentAtomicView>;
    @group(0) @binding(3) var<storage, read> agent_trails: array<GpuAgentTrail>;
    @group(0) @binding(4) var<storage, read_write> cull_count: CullCount;
    @group(0) @binding(5) var<storage, read_write> instances: array<DartInstance>;
    @group(0) @binding(6) var<storage, read_write> instance_trails: array<GpuAgentTrail>;
    @group(0) @binding(7) var<storage, read_write> draw_args: array<DrawArgs, 4>;

    @compute @workgroup_size(64)
    fn cull_main(@builtin(global_invocation_id) id: vec3u) {
        let idx = id.x;
        if (idx >= params.slot_count) { return; }
        let s = agent_states[idx];
        if ((s.meta_flags & (1u << 13u)) != 0u || s.visual_cache == 0u || agent_atomics[idx].dead_claimed != 0u) { return; }
        let dx = abs(s.pos_vel.x - params.camera_pos.x);
        let dy = abs(s.pos_vel.y - params.camera_pos.y);
        let half = params.camera_size * 0.5 + vec2f(24.0);
        if (min(dx, params.world_size.x - dx) > half.x || min(dy, params.world_size.y - dy) > half.y) { return; }
        let slot = atomicAdd(&cull_count.count, 1u);
        if (slot >= arrayLength(&instances)) { return; }
        let carn_u8 = u32(clamp(s.traits[1].y, 0.0, 1.0) * 255.0);
        instances[slot] = DartInstance(vec3f(s.pos_vel.xy, s.angle_energy.x), s.traits[0].w,
                                       vec2u(PALETTE_RGBA8[s.root % 8u], s.visual_cache), // lineage colour from the full-width root (Task 1)
                                       vec2u((carn_u8 << 24u) | (s.id & 0x00FFFFFFu), bitcast<u32>(s.traits[0].z)));
        var t = agent_trails[idx];
        if (t.owner_id != s.id) { t.count = 0u; }
        instance_trails[slot] = t;
    }

    const LAYER_VERTS = array<u32, 4>(60u, 16u, 6u, 44u); // Glow, Trail, Body, Outline

    @compute @workgroup_size(1)
    fn finalize_args_main() {
        let n = min(atomicLoad(&cull_count.count), arrayLength(&instances));
        for (var i = 0u; i < 4u; i++) { draw_args[i] = DrawArgs(LAYER_VERTS[i], n, 0u, 0u); }
    }

    // Draw group (CreatureRenderer::draw_layout), VERTEX | FRAGMENT
    @group(0) @binding(0) var<uniform> view_params: RenderParams;
    @group(0) @binding(1) var<storage, read> draw_instances: array<DartInstance>;
    @group(0) @binding(2) var<storage, read> draw_trails: array<GpuAgentTrail>;

    struct VOut { @builtin(position) clip: vec4f, @location(0) color: vec4f }
    const OFF = vec4f(-2.0, -2.0, 0.5, 1.0); // outside the clip volume: the primitive is discarded

    fn to_bevy(p: vec2f) -> vec2f { return vec2f(p.x, view_params.world_size.y - p.y); }
    fn clip_at(p: vec2f, z: f32) -> vec4f { return view_params.clip_from_world * vec4f(p, z, 1.0); }
    // glam Mat2::from_angle(-angle) * v, as the CPU meshes use
    fn rot(angle: f32, v: vec2f) -> vec2f { let a = -angle; return vec2f(cos(a) * v.x - sin(a) * v.y, sin(a) * v.x + cos(a) * v.y); }
    fn lineage_rgb(pc: u32) -> vec3f { return vec3f(f32(pc & 0xFFu), f32((pc >> 8u) & 0xFFu), f32((pc >> 16u) & 0xFFu)) / 255.0; }
    fn radius(vc: u32) -> f32 { return 2.3 + f32(vc & 0xFFu) / 255.0 * 4.5; }
    fn signal(vc: u32) -> f32 { return f32((vc >> 8u) & 0xFFu) / 255.0; }
    // Color::srgba_u8(.., (alpha * 255) as u8): alpha is quantized to 1/255
    fn body_alpha(vc: u32) -> f32 { let a = clamp(0.55 + f32((vc >> 16u) & 0xFFu) / 255.0 * 100.0 / 160.0, 0.55, 1.0); return floor(a * 255.0) / 255.0; }
    fn dart_point(inst: DartInstance, k: u32) -> vec2f { // 0 nose, 1 right wing, 2 rear notch, 3 left wing
        let r = radius(inst.vis_data.y);
        let armor = inst.pad0;
        let carn = f32(inst.pad1.x >> 24u) / 255.0;
        var local = vec2f(r * 1.5, 0.0);
        if (k == 1u) { local = vec2f(-r * 0.75, r * (0.5 + armor * 0.45)); }
        if (k == 2u) { local = vec2f(-r * (0.45 + carn), 0.0); }
        if (k == 3u) { local = vec2f(-r * 0.75, -r * (0.5 + armor * 0.45)); }
        return to_bevy(inst.pos_angle.xy) + rot(inst.pos_angle.z, local);
    }

    @vertex fn vs_body(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
        let inst = draw_instances[i];
        let order = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u); // [nose,right,rear], [nose,rear,left]
        return VOut(clip_at(dart_point(inst, order[v]), -2.0), vec4f(lineage_rgb(inst.vis_data.x), body_alpha(inst.vis_data.y)));
    }

    @vertex fn vs_glow(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
        let inst = draw_instances[i];
        let sig = signal(inst.vis_data.y);
        let glow_r = radius(inst.vis_data.y) + 5.0 + 8.0 * sig;
        let rgb = lineage_rgb(inst.vis_data.x);
        let center = to_bevy(inst.pos_angle.xy);
        let seg = v / 3u;
        let corner = v % 3u;
        if (corner == 0u) { return VOut(clip_at(center, -2.2), vec4f(rgb, clamp(0.20 + 0.18 * sig, 0.12, 0.40))); }
        let theta = f32(seg + corner - 1u) * 6.28318530718 / 20.0;
        return VOut(clip_at(center + vec2f(cos(theta), sin(theta)) * glow_r, -2.2), vec4f(rgb, 0.0));
    }

    @vertex fn vs_trail(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
        let inst = draw_instances[i];
        let t = draw_trails[i];
        let seg = v / 2u;
        if (t.count < 2u || seg + 1u >= t.count) { return VOut(OFF, vec4f(0.0)); }
        let p1 = to_bevy(t.pts[seg]);
        let p2 = to_bevy(t.pts[seg + 1u]);
        if (abs(p1.x - p2.x) >= view_params.world_size.x * 0.5 || abs(p1.y - p2.y) >= view_params.world_size.y * 0.5) {
            return VOut(OFF, vec4f(0.0));
        }
        let alpha = clamp(0.18 + 0.32 * (f32(seg + 1u) / f32(t.count - 1u)), 0.15, 0.50);
        return VOut(clip_at(select(p1, p2, (v & 1u) == 1u), -2.05), vec4f(lineage_rgb(inst.vis_data.x), alpha));
    }

    @vertex fn vs_outline(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VOut {
        let inst = draw_instances[i];
        let vc = inst.vis_data.y;
        let r = radius(vc);
        let rgb = lineage_rgb(inst.vis_data.x);
        let end = v & 1u;
        if (v < 8u) { // 4 perimeter edges
            let k = (v / 2u + end) % 4u;
            var border = vec4f(0.082, 0.188, 0.204, 1.0);
            if ((vc & (1u << 24u)) != 0u) { border = vec4f(1.0, 0.33, 0.31, 1.0); }
            return VOut(clip_at(dart_point(inst, k), -1.9), border);
        }
        if (v < 12u) { // antennae when sight > 0.56
            let sight = bitcast<f32>(inst.pad1.y);
            if (sight <= 0.56) { return VOut(OFF, vec4f(0.0)); }
            let side = select(1.0, -1.0, v >= 10u);
            var local = vec2f(-r * 0.3, side * r * 0.6);
            if (end == 1u) { local = vec2f(-r * (1.5 + sight), side * r * (1.1 + signal(vc))); }
            return VOut(clip_at(to_bevy(inst.pos_angle.xy) + rot(inst.pos_angle.z, local), -1.9), vec4f(rgb, 0.55));
        }
        let birth = f32((vc >> 25u) & 0x7Fu); // 16-segment birth halo
        if (birth <= 0.0) { return VOut(OFF, vec4f(0.0)); }
        let halo_r = r + 3.0 + max(95.0 - birth, 0.0) * 0.12;
        let theta = f32((v - 12u) / 2u + end) * 6.28318530718 / 16.0;
        let p = to_bevy(inst.pos_angle.xy) + vec2f(cos(theta), sin(theta)) * halo_r;
        return VOut(clip_at(p, -1.8), vec4f(rgb, clamp(birth / 120.0, 0.0, 1.0)));
    }

    @fragment fn fs_color(in: VOut) -> @location(0) vec4f { return in.color; }
    ```
  - **Implement** `creature_render.rs`:
    - **Buffers:** `params_buf` (uniform, 96 B), `cull_count_buf` (16 B storage), `instances_buf` (`max_instances * 32`), `instance_trails_buf` (`max_instances * 96`), `draw_args_buf` (64 B, `STORAGE | INDIRECT | COPY_SRC`).
    - **Compute:** pipelines `cull_main` and `finalize_args_main`, with an 8-entry compute layout.
    - **Draw:** a 3-entry `VERTEX | FRAGMENT` draw layout and draw bind group.
    - **`encode_cull`:**
      - writes `params` with `queue.write_buffer`;
      - rebuilds the compute bind group when `source.generation` changes;
      - calls `encoder.clear_buffer(&cull_count_buf, 0, None)`;
      - dispatches `cull_main` over `ceil(slot_count / 64)` workgroups, then a new pass for `finalize_args_main`.
    - **`ensure_draw_pipelines(key)`:** builds four pipelines into `HashMap<CreatureDrawKey, [wgpu::RenderPipeline; 4]>`:
      - topology TriangleList for Glow/Body and LineList for Trail/Outline;
      - target `key.color_format` with `BlendState::ALPHA_BLENDING`;
      - `multisample.count = key.samples`;
      - `depth_stencil` only when `key.depth` is `Some`: `depth_write_enabled: Some(false)`, `depth_compare: Some(CompareFunction::GreaterEqual)`.
    - **`draw_layer`:** sets the pipeline and draw bind group, then calls `draw_indirect(&draw_args_buf, layer as u64 * 16)`.
    - **`CreatureLayerKind`:** `sort_z` = -2.2 / -2.05 / -2.0 / -1.9; `vertices_per_instance` = 60 / 16 / 6 / 44.
  - **Implement** in the driver and bridge:
    - `GpuComputeDriver::creature_source`, and `generation` (a process-wide `AtomicU64` assigned in `new_with_grid`).
    - `render_slot_count()`: `max(max_capacity_at_last_upload, states_at_last_upload).min(max_agents)`, stored at `upload_state`.
    - `bridge::render_snapshot(sim) -> (Vec<GpuAgentState>, Vec<GpuAgentTrail>)`. It reuses `sync_rust_to_gpu`'s per-agent packing (extract it into `pack_agent_state(a, sim)`) and fills trails from `trail_x/trail_y/trail_count` with `owner_id = a.id`.
    - `upload_render_snapshot`: writes states, then trails, and returns `states.len()`.
  - Run `cargo test -p clank_app --test creature_render_test` (PASS). If the parity fraction is too high, debug with systematic-debugging; never loosen the 0.2% bound to make it pass. Commit `feat(render): GPU-resident creature cull, trail gather and four-layer indirect draw`.

- [x] **Step 4: Soil colormap written as `rgba8unorm` for GPU→GPU copies**
  - **As built (2026-10-06):**
    - The red run failed for the intended reason: wgpu `TextureFormatsNotCopyCompatible { src_format: Rgba16Float, dst_format: Rgba8UnormSrgb }`.
    - After the change, **0 of 3750** texels differ from `generate_soil_rgba` on GPU-ticked soil with seeded taint and scent.
    - The ±1 / 0.2% tolerance is kept only for f32-vs-f64 boundary cases. A round-instead-of-truncate mutation fails it (3444 texels).
    - `soil_rendering_test` passes unchanged; `copy_soil_display_rgba` now copies rows directly.
  - **Failing test** `gpu_soil_display_test.rs`:
    - step 50 soil-only ticks on the driver;
    - read the soil back through `sync_gpu_to_rust`, then compute `generate_soil_rgba` on the CPU;
    - `copy_texture_to_texture` the driver's `soil_display_tex` into a fresh `Rgba8UnormSrgb` texture (the soil sprite's format) and read it back.

    Assert that no channel differs by more than 1 and at most 0.2% of texels differ at all. Today the format is `rgba16float`, so the copy fails validation: wrong format class.
  - **Implement:**
    - Make `soil_display` `rgba8unorm` in `soil_step.wgsl`.
    - Compute the color from the quantized food, taint and scent (`f32(i32(round(f * SOIL_SCALE))) / SOIL_SCALE`), and store `floor(clamp(c, 0, 255)) / 255` to match the CPU's `as u8` truncation.
    - Update the texture descriptor, bind layout and `copy_soil_display_rgba` (now 4 bytes per texel).
    - Add `pub fn soil_display_texture(&self) -> &wgpu::Texture`.

    Migrate `soil_rendering_test`. Commit `feat(gpu): write the soil colormap as rgba8 for direct texture copies`.

- [ ] **Step 5: Bevy integration (`creature_pass.rs`) and removal of the CPU agent meshes**
  - **Implemented 2026-10-06; awaiting the live on-screen check with the user (not ticked until then).** Differences from the text below:
    - **Which buffers are presented** depends on who holds the world, not on the engine mode. If `driver.is_initialized()`, the GPU buffers are drawn (`slot_count = max_agents`) and the GPU soil is copied. Otherwise the Rust snapshot is uploaded. This covers the paused-after-switching-to-Rust case, where the GPU still holds the world until the flush.
    - **Layer items** use four plain main-world entities, keyed `(Entity::PLACEHOLDER, MainEntity)` with `extracted_index: usize::MAX`, as Bevy's mesh2d items are. Sprite batching and mesh2d batching both skip such items (verified in the 0.19 source). No `ExtractComponent` is needed for the layers; `MainCamera` is extracted.
    - **Cull and soil copy** are encoded in `RenderSystems::Prepare` and submitted on `RenderQueue` ahead of the render graph, instead of a `Core2d` node. `DrawCreatureLayer` draws through `pass.wgpu_pass()`, which resets Bevy's tracked state, so the next sprite rebinds its pipeline.
    - **Upload also presents.** `upload_world_to_gpu` writes the CPU soil colormap into `soil_display` (it needed `COPY_DST`) and carries the creatures' trails across. Before this, NEW WORLD or an engine switch while paused showed black soil, because only a GPU soil tick writes `soil_display`.
    - **Tests.**
      - `creature_pass_render_test` renders the real `ClankSimPlugin + ClankRenderPlugin` headlessly into an image. In the Rust test, ≥99% of changed pixels must lie within 40px of a creature. In the cross-engine test, the same paused world in the GPU engine must match the Rust frame: measured **0 of 540,000** pixels differ.
      - Mutations that fail the cross-engine test: no soil colormap on upload (516,789 pixels), no trail carry-over (313).
      - Bevy's render error handler turns any wgpu validation error into a missing screenshot, so the test also guards against those.
    - **Live app (debug build), driven through the HTTP API** while the screen was locked. It went Rust → GPU → speed 32 → NEW WORLD running → NEW WORLD paused → Rust, stayed up, and logged no validation errors. On-screen appearance is not yet verified.
    - **Removed.** `update_agent_mesh_system`, `setup_agent_rendering`, the four `Agent*MeshResource`s, every `generate_*_from_{instances,gpu_states,sim}` and `extract_gpu_agent_render_data`. The CPU-mesh assertions in `rendering_test`, `tdd_behavior_and_graphics_parity_test` and `pipeline_timing_bench` moved to `CreatureLayerKind` (the pixels are covered by `creature_render_test`).
    - **Frame-path readbacks.** The grep finds none in `rendering.rs`, `ui.rs` or `creature_pass.rs`. `sim.rs` keeps only `flush_gpu_to_rust` (engine switch); `api.rs` reads states for HTTP requests.
  - **Main world:**
    - `CreatureLayer(CreatureLayerKind)` components: four entities spawned at `Startup` with `SyncToRenderWorld`, extracted with `ExtractComponentPlugin`.
    - `MainCamera` derives `Clone` + `ExtractComponent`.
    - `publish_creature_source` runs after `sim_step_system`:
      - GPU mode with an initialized driver: `creature_source(driver.render_slot_count(), world, true)`;
      - Rust mode: `upload_render_snapshot(render_snapshot(&sim))`, then `creature_source(n, world, false)`;
      - no driver: remove the resource.
    - `CreatureSource` is extracted with `ExtractResourcePlugin`, and `SoilTextureHandle` is extracted too.
  - **Render world:**
    - `CreatureRenderState { renderer: CreatureRenderer }`, initialized in `Plugin::finish` from `RenderDevice` with capacity 65536.
    - In `Core2d`, `creature_cull_and_soil_copy.before(Core2dSystems::MainPass)`, which takes `ViewQuery<&ExtractedView, With<MainCamera>>`:
      - computes `clip_from_world = view.clip_from_world.unwrap_or(view.clip_from_view * view.world_from_view.to_matrix().inverse())`;
      - takes `camera_pos` (sim coordinates) from `bevy_to_sim_coord(view.world_from_view.translation().truncate(), world_h)`;
      - takes `camera_size` from the viewport size times the orthographic scale (`clip_from_view` x/y scale), so it matches today's culling;
      - calls `encode_cull` on `ctx.command_encoder()`;
      - when `source.soil_display` is `Some`, also `copy_texture_to_texture` into `RenderAssets<GpuImage>`'s soil texture, if the sizes match.
    - `queue_creature_layers` in `RenderSystems::Queue`, for views with `MainCamera`:
      - `ensure_draw_pipelines` with `CreatureDrawKey { color_format: view.target_format, samples: msaa.samples(), depth: Some(CORE_2D_DEPTH_FORMAT) }`;
      - `add_transient(Transparent2d { sort_key: FloatOrd(kind.sort_z()), entity: (render_entity, main_entity), pipeline: CachedRenderPipelineId::INVALID, draw_function, batch_range: 0..1, extracted_index: 0, extra_index: PhaseItemExtraIndex::None, indexed: false })` per layer entity.
    - `DrawCreatureLayer` (`RenderCommand<Transparent2d>`):
      - `Param = SRes<CreatureRenderState>`, `ViewQuery = (&ExtractedView, &Msaa)`, `ItemQuery = &CreatureLayer`;
      - calls `state.renderer.draw_layer(pass.wgpu_pass(), key, layer.0)`;
      - registered with `add_render_command::<Transparent2d, DrawCreatureLayer>()`.
  - **Remove:**
    - `update_agent_mesh_system`, the four `Agent*MeshResource`s and their `Mesh2d` entities, and every `generate_*_from_{instances,gpu_states,sim}` function from `rendering.rs` (the frozen copies live in `tests/support`).
    - `update_soil_texture_system` stops calling `images.get_mut` in GPU mode, except to resize. A `get_mut` marks the asset changed, and the CPU upload would overwrite the GPU copy.
  - **Verify:**
    - `cargo build -p clank_app` and every `clank_app` test file individually.
    - `grep -n "readback_\|copy_soil_display_rgba\|dispatch_culling" crates/clank_app/src/{rendering,sim,ui,creature_pass}.rs` lists only the readback-ring calls in `sim.rs`.
  - **Live check with the user** (`cargo run -p clank_app`, debug build):
    - Rust and GPU modes look identical;
    - trail length doesn't change between 1x and 32x;
    - soil, vignette, particles and the selection reticle layer correctly;
    - zoom/pan culling has no pop-in;
    - New World works while running and paused.

    Commit only after the user confirms: `feat(render): draw creatures and soil straight from GPU buffers in both engines`.

- [ ] **Step 6: Delete the superseded copy paths and correct the record**
  - **Delete:**
    - `dart_instanced.wgsl`, `dart_render_pipeline`, `render_darts_instanced`, `dart_render_bind_group`, `view_uniform_buf`, `dart_template_buf`, `DartVertex`;
    - the driver's `visible_instances_buf`, `cull_output_buf`, `dart_instances_buf`, `dart_trails_buf` and their staging buffers, plus `dispatch_culling`, `readback_dart_instances`, `readback_dart_trails`, `readback_visible_instances`, `readback_cull_args`;
    - `frustum_cull_clear` and `frustum_cull_main` (`spatial_query.wgsl`) and `trail_gather_main` (`trail_step.wgsl`);
    - the now-unused spatial-query bindings 4–6.
  - **Migrate** the remaining tests (`gpu_culling_test`, `instanced_rendering_test`, the trail half of `gpu_trail_and_reset_test`, `gpu_tools_and_indirection_test` indirect parts, `tdd_behavior_and_graphics_parity_test` mesh layers, `rendering_test`) to `CreatureRenderer`. Each keeps its original assertion intent.
  - **Update this plan:**
    - record in the Verification Plan that readbacks are no longer blocking per frame;
    - replace "zero-copy" claims with the measured facts (blocking bytes per frame = 0 in steady state; one shared device);
    - mark Task 8 done.
  - Run every `clank_app` test file individually, `cargo fmt --all`, `git diff --check`. Commit `refactor(gpu): remove the CPU-readback rendering path`.

**Out of scope (recorded, not hidden):**
- Agent replenishment (`replenish_agents_gpu`) still reads the freelist back synchronously. It runs only when the GPU population collapses below 15.
- The UI's food history still reads the CPU soil copy, which is stale in GPU mode.
- The selected creature panel is up to two frames behind the GPU.

---

## Verification Plan

### Parity target
"100% bit-exact parity" is withdrawn everywhere in this document. f32 vs f64, a different PRNG, and parallel vs sequential update order make it unattainable. GPU runs are also not reproducible run to run (same seed: 127 vs 129 population at tick 500), because creatures read neighbours' positions and energy while those neighbours' threads write them; that is accepted for this plan. Acceptance is:
- **Single-tick lockstep** in controlled scenarios (`plan_t5_agent_test`): position within 1e-3 px (toroidal), velocity and angle within 1e-4, energy within 2e-3, hidden state within 1e-4, feeding/attack/signal within 1e-4, age exact.
- **Soil**: one tick cell by cell within 2e-6 (spawn cells excluded); 600 empty ticks with total food within 3% of the reference at ticks 100..600.
- **Long run** (`plan_t6_integration_test`, `long_run_behavior_test`): population, cumulative births, kills and total soil food within ±20% of the reference at ticks 200..1000 (or 3√n for small counts).
- **Exact invariants**: struct layouts, slot accounting, single death counting, tombstones, tool effects.

### Rules for every step
Follow `superpowers:systematic-debugging` and `superpowers:test-driven-development`: the new test must fail on the current shader for the intended reason before production code changes. One step per commit. Run `cargo fmt --all -- --check` and `git diff --check`. Run only the targeted tests named in the step plus the existing GPU behavior tests (`long_run_behavior_test`, `multi_tick_parity_test`). Never dispatch anything that can loop without bound on bad data: prove such defects from CPU readbacks (the harness refuses picks until 100 consecutive trees read back valid).

### Automated Tests
```bash
# Never run `cargo test --workspace` (AGENTS.md §10). Run targeted tests only.

# Tests that dispatch the real WGSL shaders against the reference engine or an exact invariant (authoritative)
cargo test -p clank_app --test plan_t1_layouts_test
cargo test -p clank_app --test plan_t2_freelist_test
cargo test -p clank_app --test gpu_capacity_test
cargo test -p clank_app --test plan_t3_soil_test
cargo test -p clank_app --test gpu_soil_parity_test -- --nocapture
cargo test -p clank_app --test plan_t4_spatial_test
cargo test -p clank_app --test plan_t5_agent_test
cargo test -p clank_app --test plan_t6_integration_test
cargo test -p clank_app --test visual_cache_packing_test
cargo test -p clank_app --test gpu_trail_and_reset_test
cargo test -p clank_app --test long_run_behavior_test -- --nocapture
cargo test -p clank_app --test multi_tick_parity_test -- --nocapture

# Task 8 (presentation)
cargo test -p clank_app --test gpu_shared_device_test
cargo test -p clank_app --test gpu_standalone_device_test
cargo test -p clank_app --test gpu_frame_readback_test
cargo test -p clank_app --test creature_render_test
cargo test -p clank_app --test gpu_soil_display_test
cargo test -p clank_app --test creature_pass_render_test

# Layout and CPU-mirror tests (necessary but NOT evidence of shader correctness)
cargo test -p clank_app --test gpu_types_test
cargo test -p clank_app --test gpu_freelist_test
cargo test -p clank_app --test gpu_soil_test
cargo test -p clank_app --test gpu_spatial_test
cargo test -p clank_app --test gpu_agent_test
cargo test -p clank_app --test dual_engine_test
```

### Verification Matrix (measured 2026-10-07)
Re-run 2026-10-11 at `f681c8d`: every file below passes (53 files, 193 tests, 0 failed, 0 ignored). The "uncommitted" notes below describe the tree when first measured; that work is now committed.
Measured on the working tree, which holds uncommitted work not yet in any commit: the capacity pass (`capacity_step.wgsl`), the soil micro-unit and spawn fixes, trails, NEW WORLD and the Task 8 presentation work. "Mutation-checked" means the test was re-run with the production code deliberately broken and failed.

| Task | Test | Result | Why it fails / evidence |
|---|---|---|---|
| 1 | `plan_t1_layouts_test` (2) | pass | |
| 1 | `plan_t1::test_72_lineages_survive_the_engine_round_trip` | pass (Task 1 implemented, uncommitted) | failed first: 72 lineages came back as 16. Mutation-checked (masking the root to 4 bits on unpack fails) |
| 2 | `plan_t2::test_each_death_is_counted_exactly_once` | pass (Task 2 implemented, uncommitted) | failed first: 106 died, `total_deaths` rose by 110 (the killer counted before its claim). Mutation-checked (counting every decisive hit fails) |
| 2 | `plan_t2::test_a_slot_freed_this_sub_tick_is_not_reused_until_the_next` | pass (Task 2 implemented, uncommitted) | failed first: lineages 1, 11, 17, 21 got two or three children, 12, 16, 19… none. Mutation-checked (pushing deaths straight onto the stack fails) |
| 2 | `plan_t2::test_changing_the_cap_loses_no_slots` | pass (Task 2 implemented, uncommitted) | failed first: 64 live + 213 free != cap 341 (all 64 victim slots lost); passes without the cap change |
| 2 | other `plan_t2` (3), `gpu_capacity_test` (2) | pass | a merge that copies nothing fails slot accounting, recycling and single counting |
| 3 | `plan_t3` tick, clamp, corner, Nourish, Blight (5) | pass | |
| 3 | `plan_t3::test_600_soil_ticks_track_the_reference_food_total` | pass | mutation-checked (regrowth 1.7 → 1.9 fails) |
| 3 | `plan_t3::test_an_eclipse_darkens_the_soil_at_the_reference_rate` | pass (Task 3 implemented, uncommitted) | failed first: no eclipse on the GPU, food lost 0 vs 178. Mutation-checked (food ×0.83 fails) |
| 3 | `plan_t3::test_gpu_mode_counts_the_eclipse_down` | pass (Task 3 implemented, uncommitted) | failed first: 59 after 22 ticks. Mutation-checked (removing the countdown in `sim.rs` fails) |
| 3 | `plan_t3::test_an_eclipse_ends_mid_frame` | pass | mutation-checked (no countdown between sub-ticks: taint +87.8 vs +21.5) |
| 3 | `plan_t3::test_an_eclipse_skips_odd_ticks` | pass | mutation-checked (the wrong tick parity adds 22.4 taint) |
| 3 | `gpu_soil_parity_test` | pass | |
| 4 | `plan_t4` Morton, sort, tree, leaves, boxes (5) | Morton and sort pass; the 4 tree tests were removed with the LBVH on 2026-10-09 | failed first: keys unsorted, Morton rounds instead of floors, 30 of 30 trees invalid. Mutation-checked (rounding fails; a sort ignoring the top byte fails; a box climb stopping at the parent fails; dropping the preamble's `live_count` reset fails the tree tests). The flag-based box fitting left 83 of 300 trees with stale boxes; the atomic climb gives 0 of 300 |
| 4 | `plan_t4::test_the_sort_holds_when_params_cover_fewer_slots_than_the_driver` | pass | mutation-checked (dropping the workgroup guards fails 2 of 2 runs: workgroups past `params.max_agents` overwrote the histogram; in the app kills fell to 3–9 against the reference's 28) |
| 4 | `plan_t4::test_the_grid_table_holds_every_live_creature_once` | pass | mutation-checked (a 128-entry cap in the table fails; cell counters not cleared between sub-ticks fail) |
| 4 | `gpu_capacity_test::test_active_slots_cover_the_cap_and_the_uploaded_creatures` | pass | failed first: `active_slots(50)` was 50 with 72 creatures uploaded |
| 4 | `gpu_capacity_test::test_lowering_the_cap_removes_creatures_born_above_the_new_frame_size` | pass | failed first: creatures above the new cap stayed alive when the capacity pass covered only the frame's slots |
| 4 | `plan_t4::test_a_creature_on_the_far_corner_stays_in_the_index` | pass | failed first: live count 2 of 3 (the corner creature's all-ones code sorted among the dead) |
| 4 | `plan_t4::test_picking_returns_the_creature_under_the_cursor` | pass | runs since 2026-10-09: the grid pick has only bounded loops |
| 4 | `plan_t4::test_picking_selects_what_the_rust_engine_selects` | pass | failed first: the old harness would not dispatch a pick (its tree check counted the tick's newborns). Mutation-checked (no 25 px margin fails; only the cursor's own cell fails; not preferring the nearest fails) |
| 4 | `plan_t4::test_nearest_neighbour_matches_near` | pass | failed first: 3 of 7 scenarios (a neighbour above the live count missed; no global scan when the 3×3 cells hold only a creature beyond 100 px; the 128-per-cell cap). Mutation-checked (no global scan fails; a 128-per-cell search cap fails the 196-creature scenario added for it, 3 of 3 runs) |
| 4 | `plan_t4::test_density_counts_neighbours_within_100px` | pass | failed first: a creature exactly 100 px away was counted. Mutation-checked (`<=` fails) |
| 4 | `plan_t4` Extinguish, Seed | pass | failed first: toroidal erase distance; Seed added 1 creature, not 2. Mutation-checked (toroidal distance fails; one creature per click fails) |
| 5 | `plan_t5` kill, asexual birth, tombstone, kill voice, `visual_cache` (5) | pass | |
| 5 | `plan_t5::test_a_lone_creature_takes_the_reference_tick` | pass | was: the oscillator read `params.tick`; the reference has already incremented its tick (now `step_tick = params.tick + 1u`) |
| 5 | `plan_t5::test_damage_is_kept_when_the_victim_runs_after_the_attacker` | pass | was: 64 of 64 victims lost the damage to the victim's final `atomicStore` (now `atomicMin` to the 110 ceiling) |
| 5 | `plan_t5::test_parents_pay_only_for_births_that_get_a_slot` | pass | was: cap 15, 5 births, 10 parents charged (now births reserve `available_slots` before charging) |
| 5 | `plan_t5::test_births_refused_at_the_cap_are_not_queued_and_spend_no_id` | pass | was: (10 queued, 10 ids) for 5 births. The reservation and the id were taken inside `select()`, which evaluates both arguments, so every refused birth was queued anyway; its birth-pass workgroup read a queue entry not written this sub-tick and spent an id. Seen in the app 2026-10-10: creature id 454,337 after 2,409 births at the cap. Mutation-checked: the old `select` in `agent_step` alone gives (10, 5). The `birth_step` half cannot be reached once refused births are not queued; it is kept as a guard |
| 5 | `plan_t5::test_more_than_64_births_in_one_sub_tick_all_land` | pass | was: 64 of 100 (now dispatched indirectly, one workgroup per queued birth: `birth_args.wgsl`) |
| 5 | `plan_t5::test_the_cap_holds_across_sub_ticks` | pass | mutation-checked: with all three cap guards removed (freelist sized to the cap at upload, `apply_capacity`, the birth-pass check), 214 live against a cap of 200 |
| 5 | `plan_t5::test_crossover_and_mutation_draws_are_uncorrelated` | pass | was blocked by the 64-birth limit; each draw now has its own `pcg3d` stream |
| 5 | `plan_t5::test_sexual_births_happen_at_the_reference_rate` | pass | was: 64 of 450 births (the 64-birth limit) |
| 5 | `plan_t5::test_new_agents_get_reference_founder_genomes` | pass | was: traits all 0.5, energy always 43 (the test first expected traits over [0, 1]; the reference ranges are narrower) |
| 5 | `plan_t5::test_deposits_keep_cells_within_0_and_3` | pass | was: corpse food reached 20–32 in a cell; mutation-checked: without the scent cap, scent reaches 6.1 |
| 5 | `plan_t5::test_the_census_counts_newborns_and_not_the_dead` | pass | was: (population, max generation, lineages) = (40, 5, 0) against (80, 5, 13): counted before the births, no lineage count |
| 5 | `plan_t5::test_the_census_counts_every_lineage_in_a_large_world` | pass | was: (population, lineages) = (6000, 4096) against (6000, 6000): the fixed 4096-entry root table filled; every later creature probed all 8192 entries, so a speed-1 tick of 16,000 founders cost 46.9 ms against 17.5 ms |
| 5 | `plan_t5::test_every_creature_in_a_crowd_takes_one_step` | pass (guards the Morton-ordered agent pass; passes on slot order too) | mutation-checked: threads taking `spatial_keys[2i]` leave 148 of 299 unstepped; `spatial_keys[i / 2]` steps each creature twice and 274 change energy unlike the reference. One creature (id 53) differs for the accepted reason: two neighbours 37.18 and 37.00/37.62 px away, the reference having moved one of them first |
| 8 | `gpu_standalone_device_test` | pass | was: the standalone device was on the display's D500 while the second, headless D500 sat idle |
| 5 | `plan_t5::test_corpse_food_lands_where_the_victim_died` | pass (known difference) | the corpse lands in the reference's cells; the GPU victim may still eat one meal (0.019 food here) in the tick it is killed, which the test allows in that one cell. The reference victim, hit by a creature that moves before it, dies at the top of its own turn without moving or eating; on the GPU both run at once. Accepted as small (2026-10-07); making it exact needs a two-step agent pass (attacks first, then survivors move and eat), left for after the plan |
| 6 | `plan_t6::test_gpu_mode_stats_follow_the_gpu` | pass | was: did not compile (`SimWorld::max_generation()` / `living_lineages()` did not exist) |
| 6 | `plan_t6::test_telemetry_census_matches_the_live_creatures` | pass | was: `population` left out that tick's newborns (fixed by Task 5's census pass) |
| 6 | `plan_t6::test_a_kill_milestone_is_chronicled_like_the_reference` | pass | was: did not compile (no GPU chronicle) |
| 6 | `plan_t6::test_generation_milestones_are_chronicled_at_the_reference_rate` | pass | was: did not compile (no GPU chronicle); with no events the count (0) is outside 16–60 |
| 6 | `plan_t6::test_gpu_events_reach_the_chronicle` | pass | was: did not compile (no GPU chronicle) |
| 6 | `plan_t6::test_audio_voices_reach_the_playback_queue` | pass | mutation-checked (everything in view fails) |
| 6 | `plan_t6::test_long_run_matches_the_reference_statistically` | pass | mutation-checked (base metabolism 0.10 → 0.25 fails) |
| 8 | `gpu_shared_device_test` (3), `gpu_frame_readback_test` (4), `creature_render_test` (3), `gpu_soil_display_test` (1), `creature_pass_render_test` (2) | pass | |
| 8 | `gpu_frame_readback_test::test_gpu_frames_make_no_blocking_readbacks` | pass (5 of 6 runs in parallel with its 3 sibling tests; 1 wait in the other) | failed after Task 4's first version: 11–13 real 23–156 ms stalls in 40 frames, with its siblings in parallel (pre-Task-4 code: 6 of 6 pass). Causes found by measurement on the D500, debug build: the sort cost about 1 ms of GPU per sub-tick (workgroup-memory zero-fill and a 255-step rank loop in each scatter); app frames covered all 65,536 driver slots; and each sub-tick recorded about 12 compute passes at about 0.22 ms of CPU each. After the fixes, a speed-32 frame costs about 2.35 ms per sub-tick (before Task 4: about 2.8 ms; first Task 4 version: about 4.3 ms), and GPU wait per single-sub-tick frame is back to the pre-Task-4 1.7–1.8 ms. The remaining rare wait comes from the CPU now being faster than before Task 4, so four parallel apps load the GPU harder |
| 6 | `engine_switch_test` (4) | pass | was: an engine switch while paused handed nothing over (the panel showed population 74 and lineages 0 until the first tick; Rust-mode picks used the world from before the GPU ran); GPU ids continued from 10,000 (a 1,000 floor and a driver counter that NEW WORLD never reset), unlike the reference's 1, 2, 3...; ids the GPU issued to creatures already gone were forgotten on switching back. Mutation-checked: without raising the counter to the world's `next_id`, id 72 is issued again |
| 6 | `gpu_frame_readback_test::test_a_selected_creature_that_dies_is_deselected` | pass | was: the panel kept showing the dead creature, or the newborn now in its slot (seen in the app 2026-10-10: specimen 1906 on the panel with 1389 selected). Now deselected, as the reference's `selected = null` |
| 6 | `api_server_test::test_state_generation_comes_from_the_engine_that_owns_the_world` | pass | was: `/state` generation 0 in GPU mode (read from the CPU copy, which GPU mode does not update, and counting dead creatures) |

The `plan_t6` file compiles since 2026-10-07; before that, its results other than the stats test were measured from a copy of the file without that test.

Found while writing these tests, not yet covered by any task: when many creatures graze one cell in the same tick, the GPU feeds all of them from the same food (60 creatures fed from 0.33 food; the reference runs out after 29).

Found in the app on 2026-10-10, not yet covered by any task: in GPU mode the tools (picking, Nourish, Blight, Extinguish) run inside the next tick, so they do nothing while paused and act once the world runs; the reference applies them at once, paused or not.

### Visual & Manual Confirmation
1. Build and run the debug binary (`cargo run -p clank_app`); debug builds must work, so never use `--release` to mask a debug failure.
2. Confirm with the user on screen, live (a `POST /screenshot` capture may supplement this, never replace it):
   - Top header engine badge displays active mode (`ENGINE: RUST` vs `ENGINE: GPU`).
   - Creatures move, feed, fight, and reproduce with identical visual fidelity.
   - Soil texture renders smoothly without artifacts.
   - Zooming in culls creatures without visual pop-in at the edges.
   - Dragging AoE tools (Extinguish, Nourish, Blight) stays smooth.
   - Trails are the same length at 1x and 32x speed, in both engines.
   - NEW WORLD in GPU mode (running and paused) replaces the creatures immediately.
   - UI metrics (population, generation, lineages, record graph) update in real-time from 128-byte `GpuTelemetry`.
   - Changing the population cap in GPU mode takes effect at once, without freezing the computer.
   - Clicking on a creature in a dense swarm at 32x speed accurately selects the clicked specimen without cluster jumping.
   - Audio sounds modulate dynamically with zoom level (quiet murmur zoomed out, crisp bites zoomed in) and pan smoothly across the toroidal seam.
