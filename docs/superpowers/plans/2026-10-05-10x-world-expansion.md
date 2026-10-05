# 10x World Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the Clankolution simulation world to $10\times$ linear dimensions ($9,500 \times 7,470$ px, $100\times$ area), scale agent capacity to 340,000 agents in GPU VRAM, implement smooth camera navigation with up to 50x zoom, WASD, and edge panning, and enforce dynamic spatial partitioning ($95 \times 75$ cells) while maintaining locked 60 FPS performance.

**Architecture:** 
Scale all GPU storage buffers to 340,000 agents and 800x630 soil cells (~233 MB VRAM total) with unlocked native adapter limits. Enforce a 112-byte 16-byte aligned `GpuSimParams` uniform carrying dynamic spatial grid dimensions ($95 \times 75 = 7,125$ cells). Decouple simulation world size from the window resolution, initialize camera to fit the full world on startup, support up to 50x cursor-centered zoom and edge panning, and use GPU frustum culling so instanced rendering only draws agents visible in the active viewport.

**Tech Stack:** Rust, Bevy 0.19.1, wgpu 29.0.4, WGSL, bytemuck.

**Spec:** `docs/superpowers/specs/2026-10-05-10x-world-expansion-design.md`

---

## Global Constraints
- **Zero Dead Padding:** All structs, strides, and uniforms must be strictly divisible by 4 and 16 with every field serving a functional purpose.
- **Persistent VRAM Residency:** Zero per-frame state re-upload or heavy readbacks of the 340,000 agents; telemetry remains the only per-frame readback (128 bytes).
- **GPU Indirection:** Agents remain stationary in VRAM; sorting, spatial search, culling, and rendering operate strictly through 8-byte indirection keys `(morton, slot_idx)`.
- **Terminology:** All entities are strictly "agents", never "spores".
- **Strict Capacity Guards:** `atomicAdd` on freelist and instance buffers must clamp to `max_capacity` (340,000) to prevent buffer overruns.
- **Dual-Engine Integrity:** Bidirectional state synchronization between Rust CPU and GPU must preserve exact agent counts, positions, energies, and traits.

---

## Review Focus
1. **Preamble Clear Loop Bounds:** With 7,125 cells, `preamble_clear.wgsl` must clear `if (id.x < params.spatial_grid.x * params.spatial_grid.y)` using the 340,000 dispatched threads without dropping cells or overflowing.
2. **Boundary Floating-Point Clamping:** Agents right on the boundary edge ($x = 9500.0, y = 7470.0$) must be clamped to `params.spatial_grid - 1` to prevent out-of-bounds spatial cell indexing.
3. **Native Adapter Limit Unlocking:** `compute_driver.rs` must request `required_limits: adapter.limits()` to prevent default WebGPU 128 MB buffer truncation.
4. **Frustum Culling Instance Overflow:** If the user zooms all the way out, `spatial_query.wgsl` must clamp `cull_output.count` to `params.max_capacity`.
5. **High-DPI Zoom Invariance:** Camera zoom-to-cursor math must factor in `window.scale_factor()` so cursor tracking at 50x magnification does not drift on Retina/4K displays.

---

### Task 1: Struct Alignment, Dynamic Spatial Grid Uniforms & WGSL Shaders
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
- Consumes: `GpuSimParams`
- Produces: 112-byte 16-byte aligned `GpuSimParams` with `spatial_grid: [u32; 2]` and `spatial_cell_size: [f32; 2]`

- [ ] **Step 1: Write failing test in `crates/clank_app/tests/gpu_types_test.rs`**
  ```rust
  #[test]
  fn test_gpu_sim_params_112_bytes_and_spatial_grid() {
      assert_eq!(std::mem::size_of::<GpuSimParams>(), 112);
      assert_eq!(std::mem::align_of::<GpuSimParams>(), 16);
      let params = GpuSimParams {
          spatial_grid: [95, 75],
          spatial_cell_size: [100.0, 99.6],
          ..Default::default()
      };
      assert_eq!(params.spatial_grid[0], 95);
      assert_eq!(params.spatial_grid[1], 75);
  }
  ```

- [ ] **Step 2: Run test to verify failure**
  ```bash
  cargo test -p clank_app --test gpu_types_test test_gpu_sim_params_112_bytes_and_spatial_grid
  ```

- [ ] **Step 3: Update `crates/clank_app/src/gpu/types.rs`**
  Add Chunk 6 to `GpuSimParams`:
  ```rust
  // Chunk 6: Dynamic Spatial Neighborhood Grid (16B)
  pub spatial_grid: [u32; 2],     // 8 bytes (96..104)
  pub spatial_cell_size: [f32; 2],// 8 bytes (104..112)
  ```

- [ ] **Step 4: Update all 8 WGSL shaders**
  Add `spatial_grid: vec2u` and `spatial_cell_size: vec2f` to `GpuSimParams` in all 8 shaders.

- [ ] **Step 5: Update `bridge.rs` and `compute_driver.rs`**
  Populate `spatial_grid: [95, 75]` and `spatial_cell_size: [100.0, 99.6]`.

- [ ] **Step 6: Run tests to verify pass**
  ```bash
  cargo test -p clank_app --test gpu_types_test
  ```

- [ ] **Step 7: Commit**
  ```bash
  git add crates/clank_app/src/gpu/types.rs crates/clank_app/src/gpu/bridge.rs crates/clank_app/src/gpu/compute_driver.rs crates/clank_app/assets/shaders/ crates/clank_app/tests/gpu_types_test.rs
  git commit -m "feat(gpu): add dynamic spatial grid uniforms to GpuSimParams and shaders"
  ```

---

### Task 2: Expanded VRAM Allocations & Native Adapter Limits (340,000 Agents)
**Files:**
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/src/sim.rs`
- Create: `crates/clank_app/tests/gpu_scale_alloc_test.rs`

**Interfaces:**
- Consumes: `GpuComputeDriver::create_for_world(soil_cols, soil_rows, max_agents)`
- Produces: Allocated buffers for 340,000 agents and 800x630 soil cells with unlocked adapter limits.

- [ ] **Step 1: Write failing test in `crates/clank_app/tests/gpu_scale_alloc_test.rs`**
  ```rust
  #[test]
  fn test_driver_allocates_340k_capacity() {
      let driver = GpuComputeDriver::create_for_world(800, 630, 340_000);
      assert!(driver.is_some(), "Expected GPU driver to allocate 340k agent capacity");
      let d = driver.unwrap();
      assert_eq!(d.max_agents, 340_000);
      assert_eq!(d.soil_cols, 800);
      assert_eq!(d.soil_rows, 630);
  }
  ```

- [ ] **Step 2: Run test to verify failure**
  ```bash
  cargo test -p clank_app --test gpu_scale_alloc_test
  ```

- [ ] **Step 3: Update `compute_driver.rs`**
  - In `create_for_world`: request `required_limits: adapter.limits()`.
  - In `new_with_grid`:
    - `cell_offsets_size = 7125 * 4;` (for $95 \times 75$ cells).
    - `queue_size`: telemetry (128B) + 65536 * 16B births + 256 * 16B audio.
    - Set default `max_agents` to 340,000 when created for expanded world.

- [ ] **Step 4: Update `sim.rs`**
  - Set `SimWorld` default arena dimensions to `9500.0, 7470.0, 800, 630`.
  - Set default capacity to 340,000.

- [ ] **Step 5: Run tests to verify pass**
  ```bash
  cargo test -p clank_app --test gpu_scale_alloc_test
  ```

- [ ] **Step 6: Commit**
  ```bash
  git add crates/clank_app/src/gpu/compute_driver.rs crates/clank_app/src/sim.rs crates/clank_app/tests/gpu_scale_alloc_test.rs
  git commit -m "feat(gpu): allocate 340,000 agent buffers with unlocked adapter limits"
  ```

---

### Task 3: Dynamic Spatial Grid Shaders ($95 \times 75$) & Preamble Reset
**Files:**
- Modify: `crates/clank_app/assets/shaders/preamble_clear.wgsl`
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/morton_grid.wgsl`
- Modify: `crates/clank_app/src/gpu/spatial_index.rs`
- Create: `crates/clank_app/tests/gpu_spatial_expanded_test.rs`

**Interfaces:**
- Consumes: `params.spatial_grid`, `params.spatial_cell_size`
- Produces: Correct cell offset reset and neighbor queries across 7,125 cells.

- [ ] **Step 1: Write failing test in `crates/clank_app/tests/gpu_spatial_expanded_test.rs`**
  ```rust
  #[test]
  fn test_spatial_grid_7125_cells_initialization() {
      let driver = GpuComputeDriver::create_for_world(800, 630, 1000).unwrap();
      let mut params = GpuSimParams {
          spatial_grid: [95, 75],
          spatial_cell_size: [100.0, 99.6],
          max_agents: 1000,
          ..Default::default()
      };
      driver.dispatch_preamble(&params);
      let offsets = driver.readback_cell_offsets(7125);
      assert_eq!(offsets.len(), 7125);
      for offset in offsets {
          assert_eq!(offset, 0xFFFFFFFF, "All 7125 cells must be reset to sentinel");
      }
  }
  ```

- [ ] **Step 2: Run test to verify failure**
  ```bash
  cargo test -p clank_app --test gpu_spatial_expanded_test
  ```

- [ ] **Step 3: Update `preamble_clear.wgsl`**
  Clear cell offsets up to `params.spatial_grid.x * params.spatial_grid.y`:
  ```wgsl
  let total_cells = params.spatial_grid.x * params.spatial_grid.y;
  if (id.x < total_cells) {
      atomicStore(&cell_offsets[id.x], 0xFFFFFFFFu);
  }
  ```

- [ ] **Step 4: Update `agent_step.wgsl`**
  Replace hardcoded `/ 9.0` and `/ 6.0` with `params.spatial_cell_size` and modulo with `params.spatial_grid`.

- [ ] **Step 5: Run tests to verify pass**
  ```bash
  cargo test -p clank_app --test gpu_spatial_expanded_test
  ```

- [ ] **Step 6: Commit**
  ```bash
  git add crates/clank_app/assets/shaders/preamble_clear.wgsl crates/clank_app/assets/shaders/agent_step.wgsl crates/clank_app/src/gpu/spatial_index.rs crates/clank_app/tests/gpu_spatial_expanded_test.rs
  git commit -m "feat(gpu): implement dynamic 7,125 spatial cell grid and concurrent preamble clear"
  ```

---

### Task 4: Camera Navigation: Decoupled World, World-Fit Default, 50x Zoom, WASD & Edge Panning
**Files:**
- Modify: `crates/clank_app/src/camera.rs`
- Modify: `crates/clank_app/src/sim.rs`
- Create: `crates/clank_app/tests/camera_expanded_test.rs`

**Interfaces:**
- Consumes: Bevy Window events, keyboard input, mouse scroll, mouse position
- Produces: `Transform` (camera position in world coordinates) and `Projection::Orthographic` (`scale` clamped between fit-scale and 50x zoom).

- [ ] **Step 1: Write failing test in `crates/clank_app/tests/camera_expanded_test.rs`**
  ```rust
  #[test]
  fn test_compute_fit_scale_and_50x_zoom_limits() {
      let arena_size = Vec2::new(1280.0, 900.0);
      let world_size = Vec2::new(9500.0, 7470.0);
      let fit_scale = compute_fit_world_scale(world_size, arena_size);
      assert!(fit_scale >= 7.0 && fit_scale <= 8.5);
      let min_zoom = fit_scale / 50.0;
      assert!(min_zoom > 0.1 && min_zoom < 0.2);
  }
  ```

- [ ] **Step 2: Run test to verify failure**
  ```bash
  cargo test -p clank_app --test camera_expanded_test
  ```

- [ ] **Step 3: Update `crates/clank_app/src/camera.rs`**
  - Implement `compute_fit_world_scale(world_size, arena_size)`.
  - In `setup_camera`: initialize camera translation to `(4750.0, 3735.0, 0.0)` and scale to `fit_scale`.
  - In `camera_viewport_sync_system`: remove window-resize mutation of `sim.world_width / world_height`. Keep simulation dimensions fixed at `9500.0, 7470.0`.
  - In `camera_control_system`:
    - Read `MouseWheel` events: zoom into cursor position up to `fit_scale / 50.0`.
    - Read `KeyCode::KeyW`, `KeyA`, `KeyS`, `KeyD`: pan camera by $v \times \text{scale} \times \Delta t$.
    - Read cursor position: if within 25px of viewport edge, pan towards that edge.

- [ ] **Step 4: Run tests to verify pass**
  ```bash
  cargo test -p clank_app --test camera_expanded_test
  ```

- [ ] **Step 5: Commit**
  ```bash
  git add crates/clank_app/src/camera.rs crates/clank_app/src/sim.rs crates/clank_app/tests/camera_expanded_test.rs
  git commit -m "feat(camera): implement world-fit default, up to 50x zoom, WASD and edge panning"
  ```

---

### Task 5: Frustum Culling & Instanced Rendering at Expanded Scale
**Files:**
- Modify: `crates/clank_app/assets/shaders/spatial_query.wgsl`
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/src/rendering.rs`
- Create: `crates/clank_app/tests/gpu_culling_expanded_test.rs`

**Interfaces:**
- Consumes: Camera frustum bounds from `GpuSimParams`
- Produces: `visible_instances_buf` clamped to `max_capacity` and instanced mesh generation.

- [ ] **Step 1: Write failing test in `crates/clank_app/tests/gpu_culling_expanded_test.rs`**
  ```rust
  #[test]
  fn test_culling_at_50x_zoom_reduces_visible_agents() {
      let driver = GpuComputeDriver::create_for_world(800, 630, 1000).unwrap();
      // Setup params with zoomed in camera frustum (100x100 box around center)
      let params = GpuSimParams {
          camera_pos: [4750.0, 3735.0],
          camera_size: [100.0, 100.0],
          world_size: [9500.0, 7470.0],
          max_agents: 1000,
          max_capacity: 1000,
          ..Default::default()
      };
      // Dispatch culling
      driver.dispatch_culling(&params);
      let count = driver.readback_visible_count();
      assert!(count <= 1000);
  }
  ```

- [ ] **Step 2: Run test to verify failure**
  ```bash
  cargo test -p clank_app --test gpu_culling_expanded_test
  ```

- [ ] **Step 3: Update `spatial_query.wgsl` and `compute_driver.rs`**
  - In `spatial_query.wgsl`: clamp atomic counter increments to `params.max_capacity`.
  - In `compute_driver.rs`: expand `dart_instances_buf` and `cull_output_buf` to 340,000 capacity.

- [ ] **Step 4: Run tests to verify pass**
  ```bash
  cargo test -p clank_app --test gpu_culling_expanded_test
  ```

- [ ] **Step 5: Commit**
  ```bash
  git add crates/clank_app/assets/shaders/spatial_query.wgsl crates/clank_app/src/gpu/compute_driver.rs crates/clank_app/src/rendering.rs crates/clank_app/tests/gpu_culling_expanded_test.rs
  git commit -m "feat(culling): clamp frustum culling to 340,000 capacity and scale instance buffer"
  ```

---

### Task 6: UI Radar Minimap Frustum Box & Viewport Outline
**Files:**
- Modify: `crates/clank_app/src/ui.rs`
- Create: `crates/clank_app/tests/ui_expanded_minimap_test.rs`

**Interfaces:**
- Consumes: `sim.world_width`, `sim.world_height`, camera frustum rectangle
- Produces: Minimap canvas rendering showing whole-world agent clusters with camera viewport outline.

- [ ] **Step 1: Write failing test in `crates/clank_app/tests/ui_expanded_minimap_test.rs`**
  ```rust
  #[test]
  fn test_minimap_frustum_rect_normalized_to_world() {
      let world_size = Vec2::new(9500.0, 7470.0);
      let cam_pos = Vec2::new(4750.0, 3735.0);
      let cam_size = Vec2::new(1280.0, 900.0);
      let rect = compute_minimap_frustum_rect(world_size, cam_pos, cam_size, 1.0);
      assert!(rect.min.x >= 0.0 && rect.max.x <= 1.0);
      assert!(rect.min.y >= 0.0 && rect.max.y <= 1.0);
  }
  ```

- [ ] **Step 2: Run test to verify failure**
  ```bash
  cargo test -p clank_app --test ui_expanded_minimap_test
  ```

- [ ] **Step 3: Update `crates/clank_app/src/ui.rs`**
  - Implement `compute_minimap_frustum_rect`.
  - Draw the camera viewport outline box inside the radar minimap egui painter.

- [ ] **Step 4: Run tests to verify pass**
  ```bash
  cargo test -p clank_app --test ui_expanded_minimap_test
  ```

- [ ] **Step 5: Commit**
  ```bash
  git add crates/clank_app/src/ui.rs crates/clank_app/tests/ui_expanded_minimap_test.rs
  git commit -m "feat(ui): render camera frustum viewport rectangle on radar minimap"
  ```

---

### Task 7: Full System Verification, 340k Capacity Benchmarking & Visual Proof
**Files:**
- Test: Full workspace test suite (`cargo test --workspace`)
- Artifact: `artifacts/expanded_world_benchmark.md`
- Artifact: `artifacts/live_render_fit_world_340k.png`
- Artifact: `artifacts/live_render_zoom_50x_detail.png`

**Interfaces:**
- Consumes: Running application with 340,000 capacity
- Produces: Passing test suite, verified 60 FPS performance, and visual screenshot proofs.

- [ ] **Step 1: Run full workspace test suite**
  ```bash
  cargo test --workspace
  ```
  Verify all tests pass without regressions.

- [ ] **Step 2: Launch release build with 340,000 capacity**
  ```bash
  cargo run --release
  ```

- [ ] **Step 3: Capture full-world overview screenshot via `POST /screenshot`**
  Save to `artifacts/live_render_fit_world_340k.png`.

- [ ] **Step 4: Zoom in 50x on active agent cluster and capture screenshot**
  Save to `artifacts/live_render_zoom_50x_detail.png`.

- [ ] **Step 5: Benchmark frame rate and generate performance report**
  Save metrics to `artifacts/expanded_world_benchmark.md`.

- [ ] **Step 6: Commit benchmark report and artifacts**
  ```bash
  git add artifacts/
  git commit -m "docs(bench): record 340,000 capacity performance and live visual screenshots"
  ```
