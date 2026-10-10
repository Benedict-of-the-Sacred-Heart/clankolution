//! Shared harness for the plan verification tests: runs the real WGSL pipelines through
//! `GpuComputeDriver` on a headless device and reads results back.
//!
//! Picks go through `Gpu::pick`, which aims from the creatures' positions before the tick;
//! `Gpu::tick` refuses to dispatch one.
#![allow(dead_code)]

use bevy::app::App;
use clank_app::gpu::bridge::{sync_rust_to_gpu, unpack_gpu_agent_to_agent_data};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::*;
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};
use clank_core::agent::AgentData;

pub const DEAD: u32 = 1 << 13;
pub const NO_TOOL: u32 = 0xFFFF_FFFF;
/// GPU slots for test worlds (the app uses 65536).
pub const SLOTS: u32 = 4096;

pub struct Gpu {
    pub driver: GpuComputeDriver,
    pub params: GpuSimParams,
}

/// Uploads `sim` to a fresh driver, as the app does when the GPU engine takes over.
/// `None` when no adapter is available.
pub fn upload(sim: &SimWorld) -> Option<Gpu> {
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, SLOTS) else {
        eprintln!("Skipping test: No GPU adapter available");
        return None;
    };
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(sim);
    params.max_agents = driver.max_agents;
    params.sub_tick = 0;
    params.sub_ticks_per_frame = 1;
    params.tool_type = NO_TOOL;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    Some(Gpu { driver, params })
}

impl Gpu {
    /// One tick, as the app runs speed 1: dispatch, read the census, advance the tick.
    pub fn tick(&mut self) -> GpuTelemetry {
        assert_ne!(
            self.params.tool_type, 0,
            "a pick reads the creatures' positions before the tick; use pick"
        );
        self.params.sub_tick = 0;
        self.params.sub_ticks_per_frame = 1;
        self.driver.dispatch_sub_ticks(1, &self.params);
        let t = self.driver.readback_telemetry();
        self.params.tick += 1;
        self.params.agent_count = t.population;
        self.params.tool_type = NO_TOOL;
        t
    }

    pub fn ticks(&mut self, n: u32) -> GpuTelemetry {
        let mut t = self.driver.readback_telemetry();
        for _ in 0..n {
            t = self.tick();
        }
        t
    }

    pub fn states(&self) -> Vec<GpuAgentState> {
        self.driver
            .readback_agent_states(self.driver.max_agents as usize)
    }

    pub fn atomics(&self) -> Vec<GpuAgentAtomic> {
        self.driver
            .readback_atomics(self.driver.max_agents as usize)
    }

    /// Live creatures as `(slot, AgentData)`, energy from the atomic ground truth.
    pub fn live(&self) -> Vec<(usize, AgentData)> {
        let n = self.driver.max_agents as usize;
        let states = self.driver.readback_agent_states(n);
        let genomes = self.driver.readback_agent_genomes(n);
        let atomics = self.driver.readback_atomics(n);
        states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.meta_flags & DEAD == 0 && s.id != 0)
            .map(|(i, s)| {
                (
                    i,
                    unpack_gpu_agent_to_agent_data(s, &genomes[i], Some(&atomics[i])),
                )
            })
            .collect()
    }

    pub fn freelist(&self, count: usize) -> Vec<u32> {
        read_buffer(&self.driver, &self.driver.freelist_buf, count)
    }
}

/// Reads `count` elements of `T` from the start of a GPU buffer (blocking).
pub fn read_buffer<T: bytemuck::Pod>(
    driver: &GpuComputeDriver,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    let bytes = (count * std::mem::size_of::<T>()) as u64;
    if bytes == 0 {
        return Vec::new();
    }
    let staging = driver.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plan_test_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = driver.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    driver.queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    driver
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    out
}

/// A creature with a zero genome (all brain outputs 0: no turn, half thrust, no attack).
pub fn creature(id: u32, x: f64, y: f64, angle: f64) -> AgentData {
    AgentData {
        id,
        root: id,
        x,
        y,
        angle,
        energy: 50.0,
        ..AgentData::default()
    }
}

/// The seed-42 world with `agents` instead of its founders. `growth` is the renewal
/// slider (0 stops regrowth and random food spawns, which the engines draw differently).
pub fn world_with(agents: Vec<AgentData>, growth: f64) -> SimWorld {
    let mut sim = SimWorld::new(42);
    sim.world.next_id = agents.iter().map(|a| a.id).max().unwrap_or(0) + 1;
    sim.world.roots = agents.iter().map(|a| a.root).max().unwrap_or(0) + 1;
    sim.world.agents = agents;
    sim.world.growth = growth;
    sim.world.sync_pos_cache();
    sim
}

/// Toroidal distance on the 900×600 world.
pub fn torus_dist(ax: f64, ay: f64, bx: f64, by: f64, w: f64, h: f64) -> f64 {
    let dx = (ax - bx).abs();
    let dy = (ay - by).abs();
    let dx = dx.min(w - dx);
    let dy = dy.min(h - dy);
    (dx * dx + dy * dy).sqrt()
}

/// Smallest angle between two headings.
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d.min(std::f64::consts::TAU - d)
}

impl Gpu {
    /// One tick with a pick at the position `aim` chooses from the live creatures, which
    /// are where the pick's grid is built from. The grid pick is bounded (the cursor's 3×3
    /// cells, each clamped to the slot count), so it is safe to dispatch.
    pub fn pick(&mut self, aim: impl FnOnce(&[(usize, AgentData)]) -> [f32; 2]) -> GpuTelemetry {
        let pos = aim(&self.live());
        self.params.tool_type = 0;
        self.params.tool_pos = pos;
        self.params.tool_radius = 25.0;
        self.params.sub_tick = 0;
        self.params.sub_ticks_per_frame = 1;
        self.driver.dispatch_sub_ticks(1, &self.params);
        self.params.tool_type = NO_TOOL;
        self.params.tick += 1;
        self.driver.readback_telemetry()
    }
}

// App-level helpers: the real `ClankSimPlugin` systems, headless (no renderer, a GPU device of
// their own). Nothing here picks (tool 0).

/// The app in GPU mode on `configure`d world, after its first frame. `None` without a GPU.
pub fn gpu_app(configure: impl FnOnce(&mut SimWorld)) -> Option<App> {
    let mut app = App::new();
    app.add_plugins(bevy::MinimalPlugins)
        .add_plugins(ClankSimPlugin);
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Gpu;
        configure(&mut sim);
    }
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return None;
    }
    Some(app)
}

pub fn app_driver(app: &App) -> &GpuComputeDriver {
    app.world()
        .resource::<GpuDriverResource>()
        .driver
        .as_ref()
        .unwrap()
}

/// Pauses and runs frames until every GPU frame in flight has been applied, so the app's
/// counters describe the current tick. Leaves the app paused.
pub fn settle(app: &mut App) {
    app.world_mut().resource_mut::<SimWorld>().paused = true;
    for _ in 0..500 {
        app.update();
        if app_driver(app).pending_frame_readbacks() == 0 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    panic!("GPU frames never landed");
}

pub fn run(app: &mut App, frames: u32) {
    app.world_mut().resource_mut::<SimWorld>().paused = false;
    for _ in 0..frames {
        app.update();
    }
}

/// Every live creature in the app's GPU buffers.
pub fn gpu_creatures(driver: &GpuComputeDriver) -> Vec<AgentData> {
    let n = driver.max_agents as usize;
    let states = driver.readback_agent_states(n);
    let genomes = driver.readback_agent_genomes(n);
    let atomics = driver.readback_atomics(n);
    (0..n)
        .filter(|&i| states[i].meta_flags & DEAD == 0 && states[i].id != 0)
        .map(|i| unpack_gpu_agent_to_agent_data(&states[i], &genomes[i], Some(&atomics[i])))
        .collect()
}

pub fn distinct_roots(agents: &[AgentData]) -> usize {
    agents
        .iter()
        .map(|a| a.root)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}
