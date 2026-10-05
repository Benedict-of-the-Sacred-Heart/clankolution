# GPU Bit-for-Bit Formula Parity & Strict Population Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish exact mathematical and structural parity between the GPU compute engine (`crates/clank_app`), the CPU engine (`crates/clank_core`), and the ground truth (`clankolution.html`), while strictly enforcing population capacity limits.

**Architecture:**
1. Align neural network inference in WGSL to the canonical 326-weight Elman RNN architecture with exact 15 sensory inputs, 10 recurrent units, and 6 actuator outputs.
2. Structure genome memory in Rust and WGSL to pack/unpack exact 326 `i8` weights per agent without stride skips or dummy padding offsets.
3. Align agent kinematics, antennae geometry, grazing, metabolism, combat, and senescence (age > 2100 death) to match HTML and Rust formulas.
4. Enforce strict carrying capacity (`max_capacity`) across `GpuSimParams`, `agent_step.wgsl`, `birth_step.wgsl`, `bridge.rs`, `sim.rs`, and `clank_core::seed_life_at`.
5. Add automated bit-parity tests validating Rust vs. GPU brain execution within $10^{-4}$ tolerance.

**Tech Stack:** Rust, Bevy 0.16, WebGPU / wgpu 24.0, WGSL shaders, bytemuck.

**Spec:** Bit-for-bit equivalence against `clankolution.html` (lines 570-630, 815-940, 1140-1230, 1300-1455) and `crates/clank_core/src/world.rs`.

## Global Constraints
- WGSL memory layouts must remain strictly 16-byte aligned (`std140` / `std430`).
- Storage buffers must not exceed WebGPU limit (max 8 per shader stage).
- Total genome size is fixed at `GENES = 326` bytes: 260 hidden weights (10 * 26) + 66 output weights (6 * 11).
- No agent count shall ever exceed `sim.world.max_cap` in either GPU or CPU mode.
- All workspace tests (`cargo test --workspace`) must pass cleanly.

## Review Focus
1. **Bias multiplication in brain**: In WGSL, bias weights (slot 25 for hidden, slot 10 for output) must multiply `1.0`, not `0.0`.
2. **Gene packing/unpacking continuity**: Unpacking `GpuAgentGenome` back to Rust CPU in `sync_gpu_to_rust` must restore all 326 `i8` weights into `sim.world.agents[i].genes`.
3. **Population cap bounds**: When population reaches `max_capacity`, no birth requests may be allocated by `birth_step.wgsl` or pushed into `sim.world.agents`.
4. **Antennae feeler geometry**: Tip offsets must use `reach = 19.0 + 38.0 * tr2` with `mid = pos + fwd * 0.7` and `off = perp * 0.6`.
5. **Kinematic drag and thrust**: Steering must be `o0 * (0.11 + 0.09 * tr1)`, thrust `(o1 + 1.0) * 0.5 * (0.45 + 1.1 * tr1) * 0.22`, drag `0.89`, with Barnes-Hut artificial pull completely removed.

---

### Task 1: Enforce Strict Population Capacity Limits
**Files:**
- Modify: `crates/clank_app/src/gpu/types.rs`
- Modify: `crates/clank_app/src/gpu/bridge.rs`
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/src/sim.rs`
- Modify: `crates/clank_core/src/world.rs`
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/birth_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/morton_grid.wgsl`
- Modify: `crates/clank_app/assets/shaders/lbvh_aabb.wgsl`
- Modify: `crates/clank_app/assets/shaders/lbvh_build.wgsl`
- Modify: `crates/clank_app/assets/shaders/spatial_query.wgsl`
- Modify: `crates/clank_app/assets/shaders/preamble_clear.wgsl`
- Test: `tests/population_cap_test.rs`

- [x] **Step 1: Write failing test in `tests/population_cap_test.rs` asserting population never exceeds `max_cap`**
- [x] **Step 2: Run test to confirm it fails or highlights missing cap**
- [x] **Step 3: Add `max_capacity: u32` to `GpuSimParams` and all WGSL uniform declarations (96-byte aligned struct)**
- [x] **Step 4: Update `agent_step.wgsl` to check `params.agent_count < params.max_capacity && out[5] > -0.15`**
- [x] **Step 5: Update `birth_step.wgsl` to discard births when `child_idx >= params.max_capacity` or freelist is exhausted**
- [x] **Step 6: Update `sim.rs` and `bridge.rs` to clamp agent allocations to `sim.world.max_cap`**
- [x] **Step 7: Update `clank_core::seed_life_at` to check capacity on each individual agent creation**
- [x] **Step 8: Run tests and ensure `population_cap_test.rs` passes**
- [x] **Step 9: Git commit**

---

### Task 2: Exact 326-Gene Architecture & Full Genome Readback
**Files:**
- Modify: `crates/clank_app/src/gpu/types.rs`
- Modify: `crates/clank_app/src/gpu/bridge.rs`
- Modify: `crates/clank_app/src/gpu/compute_driver.rs`
- Modify: `crates/clank_app/src/sim.rs`
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Modify: `crates/clank_app/assets/shaders/birth_step.wgsl`
- Test: `tests/genome_bridge_test.rs`

- [x] **Step 1: Write failing test in `tests/genome_bridge_test.rs` checking 326-gene roundtrip through GPU bridge**
- [x] **Step 2: Run test to confirm failure**
- [x] **Step 3: Update `bridge.rs` `pack_genes` to map exact 326 `i8` bytes into `packed_genes` without stride gaps**
- [x] **Step 4: Update `bridge.rs` `sync_gpu_to_rust` to unpack all 326 `i8` bytes from GPU back to `AgentData.genes`**
- [x] **Step 5: Update `sim.rs` to readback `agent_genomes` buffer on each sync**
- [x] **Step 6: Update `birth_step.wgsl` to mutate exact 326 bytes using crossover and triangular mutation**
- [x] **Step 7: Run test to verify `genome_bridge_test.rs` passes**
- [x] **Step 8: Git commit**

---

### Task 3: Bit-for-Bit Brain Inference & 15 Sensory Inputs in WGSL
**Files:**
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Test: `tests/brain_bit_parity_test.rs`

- [x] **Step 1: Write test in `tests/brain_bit_parity_test.rs` comparing Rust `World::brain` vs WGSL compute shader for identical inputs**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement exact 15 sensory inputs matching HTML lines 1334-1362 in `agent_step.wgsl`**
- [x] **Step 4: Implement exact 326-weight sequential brain evaluation with `ins[25] = 1.0` bias and `out_ins[10] = 1.0` bias**
- [x] **Step 5: Apply exact `SCALE_H = 0.61 / 127.0` and `SCALE_O = 0.66 / 127.0` tanh scalings**
- [x] **Step 6: Run test to verify `brain_bit_parity_test.rs` passes within $10^{-4}$ tolerance**
- [x] **Step 7: Git commit**

---

### Task 4: Kinematics, Antennae, Grazing, Combat & Lifespan Parity
**Files:**
- Modify: `crates/clank_app/assets/shaders/agent_step.wgsl`
- Test: `tests/dynamics_parity_test.rs`

- [x] **Step 1: Write test in `tests/dynamics_parity_test.rs` testing single-step kinematic, eating, and combat outcomes**
- [x] **Step 2: Run test to verify failure**
- [x] **Step 3: Update antennae probe geometry to `reach = 19.0 + 38.0 * tr2`, `mid = 0.7`, `off = 0.6`**
- [x] **Step 4: Update kinematics: steering `o0 * (0.11 + 0.09 * tr1)`, thrust `(o1 + 1.0) * 0.5 * (0.45 + 1.1 * tr1) * 0.22`, drag `0.89`, remove Barnes-Hut**
- [x] **Step 5: Update grazing, metabolism, and combat formulas to exact match HTML lines 1389-1436**
- [x] **Step 6: Add maximum lifespan check: `if (a_energy <= 0.0 || a_age > 2100u) die()`**
- [x] **Step 7: Run test to verify `dynamics_parity_test.rs` passes**
- [x] **Step 8: Git commit**

---

### Task 5: End-to-End Verification & Benchmarking
**Files:**
- Test: `cargo test --workspace`
- Artifact: `artifacts/live_render_gpu_bit_parity_verified.png`
- Artifact: `artifacts/parity_benchmark_report.md`

- [x] **Step 1: Run all workspace tests (`cargo test --workspace`) and verify 100% pass**
- [x] **Step 2: Launch release build and benchmark 5,000 agents in GPU mode**
- [x] **Step 3: Capture screenshot verifying healthy population equilibrium, kills, and food consumption**
- [x] **Step 4: Confirm population strictly stays within configured `max_cap` (never exceeds)**
- [x] **Step 5: Git commit and summarize results**
