//! Regression tests for GPU tool behaviour (vs clankolution.html `interaction()` / `die()`)
//! and for the frustum-cull -> instanced-draw indirection path.
//!
//! HTML reference (clankolution.html):
//!   food  (NOURISH):    deposit(food, x, y, 0.28, 3)                          — soil only, no creature energy
//!   toxin (BLIGHT):     deposit(taint, x, y, 0.38, 3); deposit(food, x, y, -0.14, 2) — soil only
//!   erase (EXTINGUISH): creatures with hypot < 23 -> energy = 0, die(a)
//!   die(a):             deposit(food, clamp(e*0.016+0.6, 0.3, 2), r=2); deposit(taint, 0.1, r=1)

use bevy::prelude::*;
use clank_app::api::{api_dispatch_system, ApiCommand, ApiReceiverResource, ApiToolRequest};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::{GpuAgentAtomic, GpuSimParams};
use clank_app::sim::{ActiveEngine, GpuDriverResource, SimWorld};
use clank_app::ui::UiState;
use clank_core::agent::AgentData;
use clank_core::world::World;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

const COLS: usize = 75;
const ROWS: usize = 50;
const NO_TOOL: u32 = 0xFFFF_FFFF;

fn cell(x: f32, y: f32) -> usize {
    ((y / 12.0).floor() as usize) * COLS + (x / 12.0).floor() as usize
}

fn quiet_agent(id: u32, x: f64, y: f64, energy: f64) -> AgentData {
    let mut a = AgentData::default();
    a.id = id;
    a.root = id;
    a.x = x;
    a.y = y;
    a.energy = energy; // below 58 => cannot reproduce, well above 0 => cannot starve in one tick
    a.birth = 95;
    a.tr = [0.5; 6];
    a
}

/// Builds a SimWorld with the given agents, uniform food, no taint/scent and zero renewal
/// (growth = 0) so the only soil changes in a tick come from deposits under test.
fn quiet_world(agents: Vec<AgentData>, food: f32) -> SimWorld {
    let mut sim = SimWorld::default();
    sim.world = World::new(7);
    sim.world.agents = agents;
    sim.world.sync_pos_cache();
    sim.world.growth = 0.0;
    sim.world.hostility = 0.0;
    for v in sim.world.soil.food.iter_mut() { *v = food; }
    for v in sim.world.soil.taint.iter_mut() { *v = 0.0; }
    for v in sim.world.soil.scent.iter_mut() { *v = 0.0; }
    sim
}

fn upload(driver: &GpuComputeDriver, sim: &SimWorld) -> GpuSimParams {
    let (states, genomes, atomics, soil, mut params) = clank_app::gpu::bridge::sync_rust_to_gpu(sim);
    params.max_agents = driver.max_agents;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    params
}

fn run_one_tick(driver: &GpuComputeDriver, mut params: GpuSimParams, tool_type: u32, pos: [f32; 2], radius: f32) {
    params.tool_type = tool_type;
    params.tool_pos = pos;
    params.tool_radius = radius;
    driver.update_params(&params);
    driver.dispatch_sub_ticks(1, &params);
}

fn driver() -> Option<GpuComputeDriver> {
    let d = GpuComputeDriver::create_for_world(COLS as u32, ROWS as u32, 4096);
    if d.is_none() {
        eprintln!("Skipping: no GPU adapter");
    }
    d
}

// ---------------------------------------------------------------------------------------------
// Extinguish
// ---------------------------------------------------------------------------------------------

#[test]
fn test_gpu_extinguish_returns_slot_to_freelist_and_deposits_corpse() {
    let Some(driver) = driver() else { return };
    let sim = quiet_world(
        vec![quiet_agent(1, 450.0, 300.0, 20.0), quiet_agent(2, 480.0, 300.0, 20.0)],
        0.0,
    );
    let params = upload(&driver, &sim);
    let top_before = driver.readback_telemetry().freelist_top;

    run_one_tick(&driver, params, 3, [450.0, 300.0], 23.0);

    let telem = driver.readback_telemetry();
    assert_eq!(
        telem.freelist_top,
        top_before + 1,
        "extinguished creature's slot must be pushed back onto the freelist (otherwise capacity leaks)"
    );

    let atomics = driver.readback_atomics(2);
    let states = driver.readback_agent_states(2);
    assert_ne!(atomics[0].dead_claimed, 0, "victim inside r=23 must be dead");
    assert_ne!(states[0].meta_flags & (1 << 13), 0, "victim must carry the tombstone flag");
    assert_eq!(atomics[1].dead_claimed, 0, "creature 30px away is outside HTML erase radius 23 and must survive");

    let soil = driver.readback_soil();
    let c = cell(450.0, 300.0);
    let food = soil[c].food_milli as f32 * 0.001;
    let taint = soil[c].taint_milli as f32 * 0.001;
    assert!(
        (food - 0.6).abs() < 0.02,
        "die() with energy 0 deposits food 0.6 at the corpse cell, got {food}"
    );
    assert!(taint > 0.08, "die() deposits taint 0.1 at the corpse cell, got {taint}");
}

#[test]
fn test_api_extinguish_uses_html_erase_radius() {
    let (mut app, tx) = gpu_api_app(quiet_world(vec![quiet_agent(1, 450.0, 300.0, 20.0)], 0.0));
    tx.send(ApiCommand::ApplyTool(ApiToolRequest {
        tool: "extinguish".into(),
        x: Some(450.0),
        y: Some(300.0),
        count: None,
    }))
    .unwrap();
    app.update();
    let sim = app.world().resource::<SimWorld>();
    let pt = sim.pending_tool.as_ref().expect("GPU extinguish must queue a pending tool");
    assert_eq!(pt.tool_type, 3);
    assert_eq!(pt.tool_radius, 23.0, "HTML erase radius is 23px");
}

// ---------------------------------------------------------------------------------------------
// Nourish / Blight: soil-only, applied on the GPU soil state
// ---------------------------------------------------------------------------------------------

/// Runs the same quiet world twice (with/without the tool) and returns
/// (soil_without, soil_with, energy_without, energy_with) for agent slot 0.
fn tool_ab(tool_type: u32, food: f32) -> Option<(Vec<clank_app::gpu::types::GpuSoilCell>, Vec<clank_app::gpu::types::GpuSoilCell>, i32, i32)> {
    let driver = driver()?;
    // Creature 40px from the brush centre: inside the old 45px AoE, nearly outside the r=3 deposit.
    let sim = quiet_world(vec![quiet_agent(1, 490.0, 300.0, 20.0)], food);

    let params = upload(&driver, &sim);
    run_one_tick(&driver, params, NO_TOOL, [450.0, 300.0], 45.0);
    let soil_a = driver.readback_soil();
    let e_a = driver.readback_atomics(1)[0].energy_milli;

    let params = upload(&driver, &sim);
    run_one_tick(&driver, params, tool_type, [450.0, 300.0], 45.0);
    let soil_b = driver.readback_soil();
    let e_b = driver.readback_atomics(1)[0].energy_milli;
    Some((soil_a, soil_b, e_a, e_b))
}

#[test]
fn test_gpu_nourish_deposits_food_and_does_not_feed_creatures() {
    let Some((soil_a, soil_b, e_a, e_b)) = tool_ab(1, 0.0) else { return };
    let c = cell(450.0, 300.0);
    let d_center = (soil_b[c].food_milli - soil_a[c].food_milli) as f32 * 0.001;
    let d_right = (soil_b[c + 1].food_milli - soil_a[c + 1].food_milli) as f32 * 0.001;
    assert!((d_center - 0.28).abs() < 0.005, "nourish centre deposit 0.28, got {d_center}");
    assert!((d_right - 0.28 / 1.8).abs() < 0.005, "nourish rr=1 deposit 0.28/1.8, got {d_right}");
    assert!(
        (e_b - e_a).abs() < 1000,
        "HTML nourish never touches creature energy; energy delta was {} milli",
        e_b - e_a
    );
}

#[test]
fn test_gpu_blight_deposits_taint_removes_food_and_does_not_drain_creatures() {
    let Some((soil_a, soil_b, e_a, e_b)) = tool_ab(2, 1.0) else { return };
    let c = cell(450.0, 300.0);
    let taint = soil_b[c].taint_milli as f32 * 0.001;
    let d_food = (soil_b[c].food_milli - soil_a[c].food_milli) as f32 * 0.001;
    // deposit 0.38 then one tick of decay: t * (1 - 0.006) - 0.0001
    assert!((taint - (0.38 * 0.994 - 0.0001)).abs() < 0.005, "blight centre taint, got {taint}");
    assert!((d_food + 0.14).abs() < 0.005, "blight centre food -0.14, got {d_food}");
    assert!(
        (e_b - e_a).abs() < 1000,
        "HTML blight never drains creature energy directly; energy delta was {} milli",
        e_b - e_a
    );
}

fn gpu_api_app(sim: SimWorld) -> (App, mpsc::Sender<ApiCommand>) {
    let mut app = App::new();
    let (tx, rx) = mpsc::channel();
    app.insert_resource(ApiReceiverResource(Arc::new(Mutex::new(rx))));
    app.insert_resource(UiState::default());
    let mut sim = sim;
    sim.active_engine = ActiveEngine::Gpu;
    let driver = GpuComputeDriver::create_for_world(COLS as u32, ROWS as u32, 4096);
    if let Some(ref d) = driver {
        upload(d, &sim);
    }
    app.insert_resource(GpuDriverResource { driver });
    app.insert_resource(sim);
    app.add_systems(Update, api_dispatch_system);
    (app, tx)
}

#[test]
fn test_api_gpu_nourish_does_not_overwrite_gpu_soil_with_stale_cpu_soil() {
    let (mut app, tx) = gpu_api_app(quiet_world(vec![quiet_agent(1, 100.0, 100.0, 20.0)], 1.0));
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        return;
    }
    // CPU soil is stale while the GPU engine runs; make the staleness obvious.
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        for v in sim.world.soil.food.iter_mut() { *v = 0.0; }
    }
    tx.send(ApiCommand::ApplyTool(ApiToolRequest {
        tool: "nourish".into(),
        x: Some(450.0),
        y: Some(300.0),
        count: None,
    }))
    .unwrap();
    app.update();

    let soil = app.world().resource::<GpuDriverResource>().driver.as_ref().unwrap().readback_soil();
    let edge = cell(450.0, 300.0) + 3; // inside the old 7x7 write-back window
    let food = soil[edge].food_milli as f32 * 0.001;
    assert!(
        food > 0.99,
        "GPU soil cell was overwritten from stale CPU soil (food {food}, expected the GPU value 1.0)"
    );
    let sim = app.world().resource::<SimWorld>();
    assert_eq!(sim.pending_tool.as_ref().map(|p| p.tool_type), Some(1));
}

// ---------------------------------------------------------------------------------------------
// Frustum cull -> indirect draw
// ---------------------------------------------------------------------------------------------

#[test]
fn test_frustum_cull_excludes_creatures_killed_this_tick() {
    let Some(driver) = driver() else { return };
    let sim = quiet_world(
        vec![quiet_agent(1, 450.0, 300.0, 20.0), quiet_agent(2, 460.0, 300.0, 20.0)],
        0.0,
    );
    let mut params = upload(&driver, &sim);
    // Creature 2 was killed by a predator this tick: CAS claimed, tombstone not yet written.
    let killed = GpuAgentAtomic { energy_milli: 0, mate_claim: 0, mate_energy_milli: 0, dead_claimed: 1 };
    driver.queue.write_buffer(&driver.agent_atomics_buf, 16, bytemuck::bytes_of(&killed));
    params.agent_count = 2;
    let visible = driver.dispatch_culling(&params);
    assert_eq!(visible, 1, "a claimed corpse must not be drawn");
}

#[test]
fn test_cull_output_is_valid_draw_indirect_args() {
    let Some(driver) = driver() else { return };
    let sim = quiet_world(
        vec![quiet_agent(1, 450.0, 300.0, 20.0), quiet_agent(2, 460.0, 300.0, 20.0), quiet_agent(3, 10.0, 10.0, 20.0)],
        0.0,
    );
    let mut params = upload(&driver, &sim);
    params.camera_pos = [450.0, 300.0];
    params.camera_size = [200.0, 200.0];
    let visible = driver.dispatch_culling(&params);
    let args = driver.readback_cull_args();
    assert_eq!(
        args,
        [driver.dart_template_vertex_count(), visible, 0, 0],
        "cull output must be DrawIndirectArgs {{vertex_count, instance_count, first_vertex, first_instance}}"
    );
}

#[test]
fn test_instanced_darts_drawn_at_camera_relative_position() {
    let Some(driver) = driver() else { return };
    // Camera looks at the top half of the arena, centred on a creature at sim (450, 150).
    let sim = quiet_world(vec![quiet_agent(1, 450.0, 150.0, 20.0)], 0.0);
    let mut params = upload(&driver, &sim);
    params.camera_pos = [450.0, 150.0];
    params.camera_size = [300.0, 300.0];
    let visible = driver.dispatch_culling(&params);
    assert_eq!(visible, 1);

    let size = 256u32;
    let tex = driver.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_target"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    driver.render_darts_instanced(&view, &params);

    let pixels = read_texture(&driver, &tex, size);
    let lit: Vec<(u32, u32)> = (0..size * size)
        .filter(|i| pixels[(*i * 4 + 3) as usize] > 0)
        .map(|i| (i % size, i / size))
        .collect();
    assert!(!lit.is_empty(), "no dart pixels were drawn");
    let (sx, sy) = lit.iter().fold((0u64, 0u64), |(ax, ay), (x, y)| (ax + *x as u64, ay + *y as u64));
    let (cx, cy) = (sx as f32 / lit.len() as f32, sy as f32 / lit.len() as f32);
    let mid = size as f32 / 2.0;
    assert!(
        (cx - mid).abs() < 8.0 && (cy - mid).abs() < 8.0,
        "creature at the camera centre must render at the target centre, got ({cx}, {cy})"
    );
}

fn read_texture(driver: &GpuComputeDriver, tex: &wgpu::Texture, size: u32) -> Vec<u8> {
    let bytes_per_row = size * 4; // 1024, already 256-aligned
    let buf = driver.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_readback"),
        size: (bytes_per_row * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = driver.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(size) },
        },
        wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
    );
    driver.queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    driver.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let out = slice.get_mapped_range().to_vec();
    buf.unmap();
    out
}

// ---------------------------------------------------------------------------------------------
// Driver defaults
// ---------------------------------------------------------------------------------------------

#[test]
fn test_default_driver_matches_canonical_75x50_soil_grid() {
    let Some(d) = GpuComputeDriver::create_default() else { return };
    assert_eq!((d.soil_cols, d.soil_rows), (75, 50), "900x600 arena / 12px cells = 75x50");
}
