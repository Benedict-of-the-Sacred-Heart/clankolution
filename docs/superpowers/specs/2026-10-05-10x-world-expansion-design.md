# 10x World Expansion Architecture & Design Specification

**Date:** 2026-10-05  
**Branch:** `feature/bevy-gpu`  
**Status:** In Review  

---

## 1. Overview & Objectives

This specification defines the architecture, data structures, shader modifications, and camera systems required to expand the Clankolution simulation arena by $10\times$ linear dimensions ($100\times$ arena area), scaling agent capacity to **340,000 agents** on the GPU simulation engine while maintaining 60 FPS performance and fluid navigation.

### Key Goals
1. **$10\times$ Linear Scale Arena:** Expand world dimensions from $950 \times 747$ px to **$9,500 \times 7,470$ px** ($70.965\text{ Mpx}^2$, $100\times$ total area).
2. **Biological Soil Resolution:** Scale the dual-target soil simulation grid to **$800 \times 630$ cells** ($\sim 11.875\text{ px/cell}$, $504,000$ total cells, $8.06\text{ MB}$ storage buffer) to preserve fine-grained food, taint, and scent dynamics across the vast world.
3. **Massive Agent Capacity:** Allocate GPU buffers for **340,000 agents** ($\sim 230\text{ MB}$ total VRAM footprint) with lock-free freelist management and telemetry.
4. **Decoupled Camera Navigation:**
   - **Fit Full World by Default:** On startup, camera automatically sets orthographic zoom and position to fit the entire $9,500 \times 7,470$ arena into the active viewport.
   - **Up to 50x Zoom:** Smooth mouse-wheel zooming centered on the cursor, ranging from full-world overview down to extreme 50x microscopic close-up.
   - **WASD + Screen Edge Panning:** Smooth keyboard movement and cursor edge panning when hovering near arena borders, plus selected agent focus tracking.
   - **Minimap Frustum Indicator:** UI radar minimap displays the entire $9,500 \times 7,470$ world with an outline showing the camera's current visible viewport frustum.
5. **Multi-Tier Spatial Partitioning:**
   - Dynamically dimensioned atomic spatial grid ($95 \times 75 = 7,125$ cells, $\sim 100\text{ px}$ cell size) preventing cell saturation (avg $\sim 47$ agents/cell at 340,000 capacity).
   - 32-bit Morton code spatial hashing normalized across $9,500 \times 7,470$.
   - Karras 2012 Radix LBVH hierarchy across 340,000 leaves for $O(\log N)$ picking, AoE tools, minimap cluster LOD generation, and frustum culling.
6. **Frustum-Culled Instanced Rendering:**
   - GPU compute culling pass writes instances only for agents inside the active camera frustum.
   - At high zoom levels, only visible agents ($\sim 100 - 2,000$) are instanced and drawn, keeping frame times under 16.6ms.

---

## 2. Memory Architecture & VRAM Budget (340,000 Agents)

All structures maintain 16-byte std140/std430 alignment. Modern native WGPU on Apple Silicon Metal, Vulkan, and DirectX 12 supports buffer allocations up to 1-2 GB; our total allocation is $\sim 230\text{ MB}$, well within system limits.

| Buffer Name | Element Type | Count | Element Size | Total Buffer Size | Usage |
|-------------|--------------|-------|--------------|-------------------|-------|
| `agent_states_buf` | `GpuAgentState` | 340,000 | 128 B | 43.52 MB | Storage (R/W) |
| `agent_genomes_buf` | `GpuAgentGenome` | 340,000 | 352 B | 119.68 MB | Storage (R/W) |
| `agent_atomics_buf` | `GpuAgentAtomic` | 340,000 | 16 B | 5.44 MB | Storage (R/W) |
| `freelist_buf` | `u32` (slot indices) | 340,000 | 4 B | 1.36 MB | Storage (R/W) |
| `spatial_keys_buf` | `vec2u` (next, cell) | 340,000 | 8 B | 2.72 MB | Storage (R/W) |
| `lbvh_nodes_buf` | `GpuLbvhNode` | 680,000 | 48 B | 32.64 MB | Storage (R/W) |
| `node_flags_buf` | `atomic<u32>` | 680,000 | 4 B | 2.72 MB | Storage (R/W) |
| `soil_buffer` | `GpuSoilCell` | 504,000 | 16 B | 8.06 MB | Storage (R/W) |
| `soil_bloom_buffer`| `u32` | 504,000 | 4 B | 2.02 MB | Storage (R/W) |
| `cell_offsets_buf` | `atomic<u32>` | 7,125 ($95 \times 75$) | 4 B | 0.03 MB | Storage (R/W) |
| `queue_buffer` | `ConsolidatedQueue` | 1 | 4.2 MB | 4.20 MB | Storage (R/W) |
| `dart_instances_buf`| `GpuDartInstance` | 340,000 | 32 B | 10.88 MB | Storage/Vertex |
| **Total VRAM** | | | | **~233.27 MB** | |

### Consolidated Queue Sizing
- `telemetry`: 128 bytes
- `births`: 262,144 entries $\times$ 16 bytes = 4.19 MB
- `audio`: 512 entries $\times$ 16 bytes = 8 KB

---

## 3. Spatial Partitioning & Grid Scaling

### Current vs. Expanded Grid
- **Current Arena:** $950 \times 747$ px $\implies$ $9 \times 6 = 54$ cells ($105.5 \times 124.5$ px/cell).
- **Expanded Arena:** $9,500 \times 7,470$ px $\implies$ **$95 \times 75 = 7,125$ cells** ($100.0 \times 99.6$ px/cell).

### Uniform Configuration (`GpuSimParams`)
Add dynamic spatial grid dimensions to `GpuSimParams` (divisible by 4 and 16-byte aligned):
```rust
pub struct GpuSimParams {
    ...
    pub spatial_grid: [u32; 2],     // [95, 75] spatial grid cell counts (nx, ny)
    pub spatial_cell_size: [f32; 2],// [100.0, 99.6] physical cell dimensions
    ...
}
```

### WGSL Neighbor Query in `agent_step.wgsl`
The hardcoded $9 \times 6$ modulo arithmetic is updated to dynamic dimensions:
```wgsl
let nx = params.spatial_grid.x;
let ny = params.spatial_grid.y;
let cell_w = params.spatial_cell_size.x;
let cell_h = params.spatial_cell_size.y;

let gx = i32(min(u32(max(0.0, pos.x) / cell_w), nx - 1u));
let gy = i32(min(u32(max(0.0, pos.y) / cell_h), ny - 1u));

for (var dy = -1; dy <= 1; dy++) {
    let n_gy = (gy + dy + i32(ny)) % i32(ny);
    for (var dx = -1; dx <= 1; dx++) {
        let n_gx = (gx + dx + i32(nx)) % i32(nx);
        let n_cell = u32(n_gy * i32(nx) + n_gx);
        ...
    }
}
```

### Morton Grid Normalization (`morton_grid.wgsl`)
Morton code calculation computes normalized 16-bit coordinates:
```wgsl
let x_norm = u32(clamp(pos.x / params.world_size.x, 0.0, 1.0) * 65535.0);
let y_norm = u32(clamp(pos.y / params.world_size.y, 0.0, 1.0) * 65535.0);
let code = (expand_bits(x_norm) << 1u) | expand_bits(y_norm);
```
Because coordinates are normalized by `params.world_size`, 32-bit Morton codes cover the entire $9,500 \times 7,470$ arena with full spatial fidelity.

---

## 4. Camera Controller & Navigation

### Decoupled Simulation Arena
`SimWorld` is initialized with constant world dimensions:
```rust
pub const WORLD_WIDTH: f64 = 9500.0;
pub const WORLD_HEIGHT: f64 = 7470.0;
pub const SOIL_COLS: usize = 800;
pub const SOIL_ROWS: usize = 630;
```
`camera_viewport_sync_system` no longer resizes the simulation world to the window size. Instead:
- `sim.world_width = 9500.0`
- `sim.world_height = 7470.0`
- The camera moves across this coordinate space.

### Startup World-Fit Orthographic Zoom
On startup and window resize, the default camera zoom fits the full arena within the visible arena viewport:
```rust
let (vp, arena_size) = compute_arena_viewport(window_size, scale_factor);
let fit_scale_x = WORLD_WIDTH as f32 / arena_size.x;
let fit_scale_y = WORLD_HEIGHT as f32 / arena_size.y;
let fit_scale = fit_scale_x.max(fit_scale_y); // e.g. ~7.5x
```
Camera position defaults to arena center:
```rust
transform.translation.x = (WORLD_WIDTH * 0.5) as f32;
transform.translation.y = (WORLD_HEIGHT * 0.5) as f32;
ortho.scale = fit_scale;
```

### Camera Controls & Zoom Limits
- **Min Zoom (50x zoom-in):** `fit_scale / 50.0` (or `0.1` zoom scale), allowing close inspection of individual dart morphologies.
- **Max Zoom (Full-World):** `fit_scale * 1.1`, allowing comfortable macro overview of all biomes.
- **Zoom to Cursor:** When scrolling the mouse wheel, zoom adjusts around the cursor's world-space position:
  $$\text{pos}_{\text{new}} = \text{cursor}_{\text{world}} + (\text{pos}_{\text{old}} - \text{cursor}_{\text{world}}) \times \frac{\text{scale}_{\text{new}}}{\text{scale}_{\text{old}}}$$
- **Keyboard Navigation (WASD):** Smooth pan with configurable speed (scaled by current zoom level so movement feels natural at all scales).
- **Edge Panning:** When mouse cursor is within 25px of the viewport edge, camera pans automatically in that direction.
- **Agent Tracking:** When an agent is selected in the UI, camera smoothly tracks target position:
  $$\text{pos} \leftarrow \text{pos} + (\text{agent\_pos} - \text{pos}) \times 0.1$$

---

## 5. Frustum Culling & Rendering Optimization

### GPU Culling Pass (`spatial_query.wgsl`)
1. Compute Pass reads `agent_states_buf`.
2. Clamps camera frustum bounds in world coordinates:
   $$\text{frustum\_min} = \text{cam\_pos} - \frac{\text{cam\_size}}{2} \times \text{zoom} - \text{margin}$$
   $$\text{frustum\_max} = \text{cam\_pos} + \frac{\text{cam\_size}}{2} \times \text{zoom} + \text{margin}$$
3. Toroidal wrapping is applied to camera viewports spanning arena boundaries.
4. Active, living agents inside the frustum are appended to `cull_output_buf` via atomic increment.
5. In GPU instanced rendering, Bevy draws `cull_output_buf.count` instances using `dart_instanced.wgsl`.

### UI Minimap Viewport Frustum Box
In `crates/clank_app/src/ui.rs`:
- Radar minimap renders cluster dots normalized to `[0, 9500] × [0, 7470]`.
- Draws a rectangular outline representing the current camera frustum bounds on the minimap canvas, giving the user immediate situational awareness of their field of view.

---

## 6. Verification & Test Plan

1. **Unit & Integration Tests:**
   - `test_expanded_world_memory_allocations`: Verify all buffers for 340,000 agents and 800x630 soil initialize without error.
   - `test_spatial_grid_7125_cells`: Verify cell offsets and neighbor queries across $95 \times 75$ cells.
   - `test_camera_zoom_and_fit_world`: Verify camera fits full arena on startup and clamps between fit scale and 50x zoom.
   - `test_frustum_culling_at_zoom_levels`: Verify culling count matches expected visible agents at 50x zoom vs. full world fit.
2. **Live Visual & Performance Verification:**
   - Run simulation at 340,000 capacity on native Metal GPU.
   - Measure frame rate and dispatch time: target $\ge 60\text{ FPS}$ at full fit and 50x zoom.
   - Capture live screenshots via `POST /screenshot` at fit-world scale and 50x zoomed-in view.
