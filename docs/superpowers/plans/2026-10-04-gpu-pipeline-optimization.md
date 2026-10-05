# Full GPU Offload & Indirection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offload the entire simulation and rendering pipeline to the GPU using lightweight indirection buffers, zero-copy VRAM persistence, and GPU-driven instanced rendering to achieve 60+ FPS at 20,000+ agents.

**Architecture:**
1. **Persistent VRAM Residency:** Keep agent states, genomes, atomics, and soil permanently in GPU VRAM across ticks. Eliminate per-frame state re-upload (`sync_rust_to_gpu`) and eliminate all heavy per-frame readbacks of genomes, atomics, and soil.
2. **GPU Indirection & Frustum Culling:** Use the 8-byte Morton spatial indirection keys `(morton, slot_idx)` in `morton_grid.wgsl` and LBVH culling to output visible living agent slot indices into a GPU indirection buffer (`visible_instances_buf: array<u32>`) and indirect draw parameters directly on the GPU.
3. **GPU-Driven Instanced Rendering (`dart_instanced.wgsl`):** Draw a single 6-vertex dart template instanced across visible agents, reading transforms, colors, and visual caches directly from `agent_states_buf` in VRAM. Completely eliminate CPU vertex generation (no 80,000 vertices allocated or uploaded on CPU).
4. **Direct GPU Soil Display Texture:** Render the `soil_display` storage texture produced by `soil_step.wgsl` directly on the arena background quad without CPU pixel rasterization.
5. **Strict 4-Byte / 16-Byte Divisibility & Functional Padding:** Enforce that all structs, array strides, and buffer copies are strictly divisible by 4 and 16 (`std140`/`std430`). Repurpose all dummy padding into active simulation fields (`tool_radius`, `eclipse`, `epoch`, `birth_tick`, `fertility_milli`, `decay_rate`, `total_births`, `total_deaths`, `max_generation`, `extinctions`).
6. **Zero-Stall Telemetry & Direct Spore Seeding:** Read back only the 128-byte `GpuTelemetry` struct for UI counters. When population collapses (< 15), inject newborn spores directly into VRAM freelist slots via `queue.write_buffer`.
7. **On-Demand CPU Sync:** Synchronize state back to the CPU (`sync_gpu_to_rust`) only when switching engines (`ENGINE: RUST`), saving to disk (`/persist`), or inspecting a clicked specimen.

**Tech Stack:** Rust, Bevy 0.19.1, wgpu 29.0.4, WGSL, bytemuck.

**Spec:** Original GPU pipeline specification in `docs/superpowers/plans/2026-10-04-bevy-gpu-sim.md` and bit-exact mechanics in `clankolution.html`.

---

## Global Constraints
- **Zero CPU Per-Agent Hot-Paths:** The main loop must perform zero per-agent work on the CPU during normal GPU simulation ticks. No CPU vertex generation, no CPU world serialization.
- **4-Byte / 16-Byte Divisibility Invariant:** Every struct size, array stride, buffer offset, and copy length must be strictly divisible by 4. Uniform buffers must be aligned to and sized in 16-byte chunks.
- **Zero Dead Padding Invariant:** No struct shall contain dummy zero padding where real simulation parameters or metrics can be stored.
- **GPU Indirection Invariant:** Agents remain stationary in VRAM; spatial indexing, sorting, culling, and rendering operate strictly through 8-byte indirection keys `(morton, slot_idx)`.
- **Dual-Engine Integrity:** Hot-swapping between `ENGINE: RUST` and `ENGINE: GPU` must remain completely seamless and preserve exact agent counts, positions, energies, and traits.
- **Carrying Capacity:** `max_capacity` enforcement must remain strictly active on GPU.

---

## Padding Repurposing Mapping

| Struct | Old Padding Field | Repurposed Functional Field | Purpose | Size / Offset |
| :--- | :--- | :--- | :--- | :--- |
| **`GpuSimParams`** | `_pad0: u32` | `tool_radius: f32` | Dynamic brush radius for tools (replaces hardcoded 45.0 in shaders) | 4B (offset 44..48) |
| **`GpuSimParams`** | `_pad1[0]: u32` | `eclipse: u32` | Solar eclipse duration & environmental darkness | 4B (offset 88..92) |
| **`GpuSimParams`** | `_pad1[1]: u32` | `epoch: u32` | Current evolutionary epoch counter | 4B (offset 92..96) |
| **`BirthEvent`** | `pad: u32` | `birth_tick: u32` | Simulation tick of conception for age auditing | 4B (offset 12..16) |
| **`GpuSoilCell`** | `pad: u32` | `fertility_milli: i32` | Regional soil fertility / micro-climate modifier | 4B (offset 12..16) |
| **`SoilParams`** | `pad: u32` | `decay_rate: f32` | Dynamic taint and scent decay multiplier | 4B (offset 12..16) |
| **`GpuTelemetry`** | `_reserved0[0]: u32`| `total_births: u32` | Cumulative lifetime creature births | 4B (offset 48..52) |
| **`GpuTelemetry`** | `_reserved0[1]: u32`| `total_deaths: u32` | Cumulative lifetime creature deaths | 4B (offset 52..56) |
| **`GpuTelemetry`** | `_reserved0[2]: u32`| `max_generation: u32`| Highest generation index reached in arena | 4B (offset 56..60) |
| **`GpuTelemetry`** | `_reserved0[3]: u32`| `extinctions: u32` | Count of lineages currently extinct | 4B (offset 60..64) |

---

## Review Focus
1. **Shader Struct Alignment Synchronization:** When updating `GpuSimParams`, `GpuSoilCell`, `BirthEvent`, and `GpuTelemetry` in Rust, all 8 WGSL shaders (`agent_step.wgsl`, `birth_step.wgsl`, `soil_step.wgsl`, `preamble_clear.wgsl`, `morton_grid.wgsl`, `lbvh_build.wgsl`, `lbvh_aabb.wgsl`, `spatial_query.wgsl`) must match exact field names and 4-byte types.
2. **Indirect Draw / Instance Buffer Bounds:** The GPU frustum culling pass must clamp `visible_count` to `max_capacity` to prevent out-of-bounds draws.
3. **Zero-Cost Dead Culling in Vertex Shader:** `dart_instanced.wgsl` must collapse dead agents (`visual_cache == 0u`) to degenerate clip coords `vec4f(0.0, 0.0, -100.0, 1.0)`.
4. **Direct Spore Seeding In VRAM:** Seeding spores on GPU must pop freelist slots and write directly via `queue.write_buffer` without touching CPU `sim.world.agents`.
5. **Lossless Engine Hot-Swap:** When switching `ENGINE: GPU` to `ENGINE: RUST`, drain all GPU queues before calling `sync_gpu_to_rust`.

---

### Task 1: 4-Byte Divisibility Enforcement & Functional Padding Repurposing
**Files:**
- Modify: `crates/clank_app/src/gpu/types.rs`
- Modify: `crates/clank_app/src/gpu/bridge.rs`
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/birth_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/soil_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/preamble_clear.wgsl`
- Modify: `crates/clank_app/assets/shaders/morton_grid.wgsl`
- Modify: `crates/clank_app/assets/shaders/lbvh_build.wgsl`
- Modify: `crates/clank_app/assets/shaders/lbvh_aabb.wgsl`
- Modify: `crates/clank_app/assets/shaders/spatial_query.wgsl`
- Test: `crates/clank_app/tests/gpu_types_test.rs`

**Interfaces:**
- Consumes: `GpuSimParams`, `GpuSoilCell`, `BirthEvent`, `SoilParams`, `GpuTelemetry`
- Produces: Zero-pad, 4-byte/16-byte validated struct definitions and shader uniforms

- [x] **Step 1: Write tests in `tests/gpu_types_test.rs` verifying 4-byte/16-byte divisibility and functional field offsets**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Update Rust structs in `crates/clank_app/src/gpu/types.rs`**
  - In `GpuSimParams`: replace `_pad0` with `tool_radius: f32`, `_pad1` with `eclipse: u32, epoch: u32`.
  - In `BirthEvent`: replace `pad: u32` with `birth_tick: u32`.
  - In `GpuSoilCell`: replace `pad: u32` with `fertility_milli: i32`.
  - In `SoilParams`: replace `pad: u32` with `decay_rate: f32`.
  - In `GpuTelemetry`: replace `_reserved0` with `total_births`, `total_deaths`, `max_generation`, `extinctions`.
- [x] **Step 4: Update all 8 WGSL shader definitions to match the exact new field names**
- [x] **Step 5: Update `bridge.rs` and `compute_driver.rs` to populate functional values**
- [x] **Step 6: Run `cargo test --package clank_app --test gpu_types_test` to verify 100% pass**
- [x] **Step 7: Git commit**

---

### Task 2: Persistent VRAM Simulation Loop (Zero Readbacks)
**Files:**
- Modify: `crates/clank_app/src/sim.rs`
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Test: `crates/clank_app/tests/gpu_persistence_test.rs`

**Interfaces:**
- Consumes: `GpuComputeDriver`, `GpuSimParams`, `GpuTelemetry`
- Produces: `driver.update_params(&params)`, `driver.readback_telemetry()`, `driver.is_initialized()`

- [x] **Step 1: Write test in `tests/gpu_persistence_test.rs` verifying multi-frame simulation without full re-upload**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Implement `update_params` in `compute_driver.rs` to write only `sim_params_buf`**
- [x] **Step 4: Refactor `sim_step_system` in `sim.rs` to persist state on GPU across frames**
  - On switch to GPU or first frame: call `upload_state` once.
  - On normal ticks: call `driver.update_params(&params)` and `driver.dispatch_sub_ticks(steps, &params)`.
  - Read back ONLY `telemetry` (128 bytes) + active audio events.
  - Eliminate per-frame calls to `readback_agent_states`, `readback_agent_genomes`, `readback_atomics`, `readback_soil`, and `sync_rust_to_gpu`.
- [x] **Step 5: Run test to verify `gpu_persistence_test.rs` passes**
- [x] **Step 6: Git commit**

---

### Task 3: GPU Indirection & Frustum Culling Compute Pass
**Files:**
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/assets/shaders/spatial_query.wgsl`
- Test: `crates/clank_app/tests/gpu_culling_test.rs`

**Interfaces:**
- Consumes: `spatial_keys: array<vec2u>`, `agent_states: array<GpuAgentState>`, camera viewport
- Produces: `visible_instances_buf: array<u32>`, `visible_count: u32`

- [x] **Step 1: Write test in `tests/gpu_culling_test.rs` verifying GPU frustum culling outputs visible living slot indices**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Add visible instances indirection buffer to `compute_driver.rs`**
- [x] **Step 4: Implement frustum culling entry point in `spatial_query.wgsl` writing visible slot indices**
- [x] **Step 5: Run test to verify `gpu_culling_test.rs` passes**
- [x] **Step 6: Git commit**

---

### Task 4: GPU Instanced Rendering Pipeline (`dart_instanced.wgsl`)
**Files:**
- Modify: `crates/clank_app/src/rendering.rs`
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Test: `crates/clank_app/tests/instanced_rendering_test.rs`

**Interfaces:**
- Consumes: `agent_states_buf`, `visible_instances_buf`
- Produces: Direct GPU instanced dart rendering, zero CPU vertex generation

- [x] **Step 1: Write test in `tests/instanced_rendering_test.rs` verifying GPU instance stream feeds template mesh**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Connect `dart_instanced.wgsl` to render agents directly from `agent_states_buf`**
  - Draw 1 base dart template instanced across visible agents.
  - Zero CPU vertex generation (eliminate `generate_dart_mesh_from_gpu_states` and `generate_outline_mesh_from_gpu_states`).
- [x] **Step 4: Run test to verify `instanced_rendering_test.rs` passes**
- [x] **Step 5: Git commit**

---

### Task 5: Direct GPU Soil Display Texture
**Files:**
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/src/rendering.rs`
- Test: `crates/clank_app/tests/soil_rendering_test.rs`

**Interfaces:**
- Consumes: `soil_display` storage texture from `soil_step.wgsl`
- Produces: Zero-copy direct arena background texture

- [x] **Step 1: Write test in `tests/soil_rendering_test.rs` verifying `soil_display` texture output**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Bind `soil_display` directly to the `SoilSprite` background quad**
  - Eliminate CPU `generate_soil_rgba`.
- [x] **Step 4: Run test to verify passes**
- [x] **Step 5: Git commit**

---

### Task 6: Direct GPU Spore Seeding & Lossless Hot-Swap
**Files:**
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/src/sim.rs`
- Modify: `crates/clank_app/src/ui.rs`
- Test: `crates/clank_app/tests/gpu_spore_test.rs`

**Interfaces:**
- Consumes: `telemetry.population`
- Produces: `driver.seed_spores_gpu(&[(x, y)])`, `flush_gpu_to_rust(&mut sim)`

- [x] **Step 1: Write test in `tests/gpu_spore_test.rs` verifying spores spawn directly in GPU buffers**
- [x] **Step 2: Implement `seed_spores_gpu` in `compute_driver.rs`**
  - Writes new `GpuAgentState` and `GpuAgentGenome` directly to VRAM buffers via `queue.write_buffer`.
- [x] **Step 3: Implement `flush_gpu_to_rust` with queue barrier and full 4-buffer readback**
- [x] **Step 4: Connect `flush_gpu_to_rust` to engine switch toggle, `/persist`, and `/reset`**
- [x] **Step 5: Run test to verify `gpu_spore_test.rs` passes**
- [x] **Step 6: Git commit**

---

### Task 7: End-to-End Benchmarking & Verification
**Files:**
- Test: `cargo test --workspace`
- Test: `cargo bench --bench gpu_microbench`
- Artifact: `artifacts/gpu_performance_benchmark.md`
- Screenshot: `artifacts/live_render_gpu_optimized_fps.png`

- [x] **Step 1: Run full workspace test suite (`cargo test --workspace`) and verify 100% pass**
- [x] **Step 2: Launch release build and benchmark 10,000 and 20,000 agents in GPU mode**
- [x] **Step 3: Verify framerate exceeds 60 FPS smoothly with zero CPU stalls**
- [x] **Step 4: Test bidirectional engine hot-swapping preserves population, kills, and specimen data**
- [x] **Step 5: Capture screenshot and generate performance comparison report**
- [x] **Step 6: Git commit and summarize results**
