# GPU Compute Node, Instanced Dart Rendering & Minimap Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 16x simulation speed bottleneck by:
1. Executing simulation sub-ticks in parallel on GPU silicon via real WGPU `dispatch_workgroups` calls (<1 ms instead of 40 ms CPU loop).
2. Streaming `vec2u(packed_color, visual_cache)` into an instanced GPU mesh draw call, completely eliminating ~22,500 immediate-mode CPU gizmo lines per frame.
3. Caching the simulation's LBVH tree for radar minimap clustering, eliminating redundant per-frame CPU tree construction inside egui.

**Architecture:**
- **GPU Compute Node (`GpuComputeDriver`)**: Owns WGPU compute pipelines (`soil_step.wgsl`, `morton_grid.wgsl`, `lbvh_build.wgsl`, `agent_step.wgsl`, `birth_step.wgsl`) and VRAM storage buffers. Dispatches multi-tick batches via `wgpu::ComputePass::dispatch_workgroups` using Bevy's `RenderDevice` and `RenderQueue`.
- **Instanced Dart & Gizmo Mesh**: Dart body and dark teal / combat red outlines rendered in a single instanced GPU mesh draw call. Vertex shader loads `vec2u(packed_color, visual_cache)` to unpack radius, glow, energy, and combat flags, executing the zero-cost dead check (`visual_cache == 0u` collapses scale to 0) and avoiding all 22,500 CPU gizmo line draw calls.
- **Minimap Cluster Cache**: `MinimapCache` resource stores pre-extracted clusters computed during simulation step, reducing egui radar minimap rendering to a trivial iteration with zero allocations.

**Tech Stack:** Rust, Bevy 0.19.1 (`bevy_render`, `bevy_sprite_render`), `wgpu 29.0.4`, `bytemuck 1.21`, WebGPU WGSL.

**Spec:** `docs/superpowers/plans/2026-10-04-bevy-gpu-sim.md`

## Global Constraints
- Target frame rate: >60 FPS at 16x simulation speed with 1,000+ agents.
- WebGPU portable limits: strictly $\le 8$ storage buffers in Bind Group 0.
- Struct alignment: 16-byte aligned `GpuSimParams`, `GpuAgentState`, `GpuAgentGenome`, `GpuTelemetry`.
- Single-writer invariant: compute threads write only to their own state struct slot.
- Zero-cost dead state invariant: `visual_cache == 0u` triggers instant GPU-side culling.
- Strict TDD: Write failing unit/integration tests before writing implementation code.

## Review Focus
1. **GPU Staging Readback & Zero CPU Stalls**: Telemetry and state transfers between GPU and CPU must use non-blocking asynchronous mapping or staging buffers to prevent pipeline bubbles.
2. **Toroidal Coordinates in Vertex Shader**: Instance translations must respect simulation boundary wrap and camera conversion coordinates.
3. **Dead Agent Invalidation**: If an agent dies during GPU compute, its `visual_cache` must be 0 and instantly collapse in the instanced draw call without lingering corpses.
4. **Portability of Storage Buffers**: No shader may exceed 8 storage buffer bindings in Group 0.
5. **Fallback to CPU Mode**: Toggling `ENGINE: RUST` must continue to function bit-exactly without panic.

---

### Task 1: Minimap LBVH Cluster Cache

**Files:**
- Create: `crates/clank_app/tests/minimap_cache_test.rs`
- Modify: `crates/clank_app/src/ui.rs`
- Modify: `crates/clank_app/src/sim.rs`

- [ ] **Step 1: Write failing test for minimap cluster caching**
  Create `crates/clank_app/tests/minimap_cache_test.rs` verifying that `MinimapCache` accurately holds pre-extracted clusters and that `render_radar_minimap` renders from the cache without calling `LbvhTree::build` on every frame.
- [ ] **Step 2: Run test to confirm it fails**
  Run `cargo test -p clank_app --test minimap_cache_test`.
- [ ] **Step 3: Implement `MinimapCache` resource and update logic**
  - Add `MinimapCache` resource containing `pub clusters: Vec<MinimapCluster>`.
  - In `sim_step_system`, update `MinimapCache` from living agents or simulation LBVH tree once per step.
  - In `crates/clank_app/src/ui.rs`, refactor `render_radar_minimap` to read from `MinimapCache` instead of allocating `living_states` and building `LbvhTree` from scratch inside egui.
- [ ] **Step 4: Run test to confirm it passes**
  Run `cargo test -p clank_app --test minimap_cache_test`.
- [ ] **Step 5: Git commit**
  `git commit -am "perf(ui): cache LBVH minimap clusters and eliminate per-frame tree rebuilds in egui"`

---

### Task 2: Instanced GPU Rendering for Darts & Outlines with `vec2u(packed_color, visual_cache)`

**Files:**
- Create: `crates/clank_app/assets/shaders/dart_instanced.wgsl`
- Create: `crates/clank_app/tests/instanced_rendering_test.rs`
- Modify: `crates/clank_app/src/rendering.rs`

- [ ] **Step 1: Write failing test for visual cache unpacking and instanced mesh generation**
  Create `crates/clank_app/tests/instanced_rendering_test.rs` verifying:
  - Zero-cost dead check: `visual_cache == 0` generates 0 vertices / collapsed instance.
  - Direct unpacking of `radius`, `glow`, `energy`, and `combat` flag from `visual_cache` and `packed_color`.
  - Generation of combined dart body and perimeter outline without calling immediate-mode `Gizmos`.
- [ ] **Step 2: Run test to confirm it fails**
  Run `cargo test -p clank_app --test instanced_rendering_test`.
- [ ] **Step 3: Implement WGSL instanced dart shader & rendering refactor**
  - Create `crates/clank_app/assets/shaders/dart_instanced.wgsl` supporting instanced vertex attributes `(pos_rot: vec3f, vis: vec2u)`.
  - In `crates/clank_app/src/rendering.rs`:
    - Refactor `generate_dart_mesh_data` to consume `GpuAgentState` directly using `visual_cache` and `packed_color`.
    - Pack dart body triangles + outline vertices directly into the dynamic mesh.
    - Remove the 22,500 immediate-mode CPU gizmo line draw calls in `render_sim_gizmos_system` (remove dart outline lines, keep only selected agent reticle).
    - Eliminate redundant calls to `extract_agent_render_data` and trail vector allocations in the hot rendering loop.
- [ ] **Step 4: Run test to confirm it passes**
  Run `cargo test -p clank_app --test instanced_rendering_test`.
- [ ] **Step 5: Git commit**
  `git commit -am "perf(render): implement instanced dart rendering streaming visual_cache and eliminate CPU gizmo line flood"`

---

### Task 3: Bevy GPU Compute Driver & Workgroup Dispatch

**Files:**
- Create: `crates/clank_app/src/gpu/compute_driver.rs`
- Create: `crates/clank_app/tests/gpu_compute_dispatch_test.rs`
- Modify: `crates/clank_app/src/gpu/mod.rs`
- Modify: `crates/clank_app/src/sim.rs`
- Modify: `crates/clank_app/src/lib.rs`

- [ ] **Step 1: Write failing test for `GpuComputeDriver` dispatch**
  Create `crates/clank_app/tests/gpu_compute_dispatch_test.rs` initializing `GpuComputeDriver`, uploading test agent states and soil grid, dispatching a batch of sub-ticks via WGPU compute passes, and asserting that agent positions, energy, and soil evolve correctly on the GPU.
- [ ] **Step 2: Run test to confirm it fails**
  Run `cargo test -p clank_app --test gpu_compute_dispatch_test`.
- [ ] **Step 3: Implement `GpuComputeDriver` and wire into Bevy game loop**
  - Implement `GpuComputeDriver` in `crates/clank_app/src/gpu/compute_driver.rs`:
    - Compile compute pipelines for `soil_step.wgsl`, `morton_grid.wgsl`, `lbvh_build.wgsl`, `agent_step.wgsl`, and `birth_step.wgsl`.
    - Create storage buffers for agent states, genomes, atomics, soil cells, LBVH nodes, freelist, consolidated queue, and uniform params.
    - Create Bind Groups with strict $\le 8$ storage buffer bindings in Group 0.
    - Implement `dispatch_sub_ticks` to encode and submit compute passes with `pass.dispatch_workgroups(...)`.
  - In `crates/clank_app/src/sim.rs`:
    - In `sim_step_system`, when `sim.active_engine == ActiveEngine::Gpu`, dispatch simulation sub-ticks via `GpuComputeDriver` on the GPU silicon rather than executing CPU loops.
    - Read back telemetry and updated states for rendering and minimap caching.
- [ ] **Step 4: Run test to confirm it passes**
  Run `cargo test -p clank_app --test gpu_compute_dispatch_test`.
- [ ] **Step 5: Git commit**
  `git commit -am "feat(gpu): wire WGSL compute pipelines to WGPU dispatch_workgroups for real GPU simulation"`

---

### Task 4: End-to-End Benchmark & Verification

**Files:**
- Modify: `crates/clank_app/benches/gpu_microbench.rs`
- Run: `scratch/benchmark_engines.py`

- [ ] **Step 1: Run all workspace tests**
  `cargo test --workspace` ensuring all existing and new tests pass.
- [ ] **Step 2: Build release binary**
  `cargo build -p clank_app --release`.
- [ ] **Step 3: Benchmark live app at 16x speed with 1,000+ agents**
  Run `python3 scratch/benchmark_engines.py` to capture real live FPS and frame time ms via REST API.
  Verify framerate is $>60\text{ FPS}$ at 16x speed on GPU engine.
- [ ] **Step 4: Capture verified screenshot**
  Trigger `POST /screenshot` and inspect the visual rendering of darts and minimap.
- [ ] **Step 5: Git commit and summarize results**
  `git commit -am "bench(gpu): verify >60 FPS performance at 16x speed with instanced rendering and GPU compute"`
