use bevy::prelude::*;
use clank_core::world::World;
use clank_core::agent::AgentData;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ActiveEngine {
    #[default]
    Rust,
    Gpu,
}

#[derive(Resource)]
pub struct SimWorld {
    pub world: World,
    pub speed: u32,
    pub paused: bool,
    pub unthrottled: bool,
    pub selected_agent_id: Option<u32>,
    pub world_width: f64,
    pub world_height: f64,
    pub active_engine: ActiveEngine,
}

impl Default for SimWorld {
    fn default() -> Self {
        let world = World::new_with_size(42, 950.0, 747.0, 80, 63);
        Self {
            world,
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            world_width: 950.0,
            world_height: 747.0,
            active_engine: ActiveEngine::Rust,
        }
    }
}

impl SimWorld {
    pub fn new(seed: u32) -> Self {
        let world = World::new_with_size(seed, 950.0, 747.0, 80, 63);
        Self {
            world,
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            world_width: 950.0,
            world_height: 747.0,
            active_engine: ActiveEngine::Rust,
        }
    }

    pub fn step(&mut self, substeps: u32) {
        if self.paused {
            return;
        }
        for _ in 0..substeps {
            self.world.evolve();
        }
    }

    pub fn force_step(&mut self, substeps: u32) {
        for _ in 0..substeps {
            self.world.evolve();
        }
    }

    pub fn reset(&mut self) {
        self.world.reset();
        self.selected_agent_id = None;
    }

    pub fn get_selected_agent(&self) -> Option<&AgentData> {
        let target_id = self.selected_agent_id?;
        self.world.agents.iter().find(|a| a.id == target_id)
    }
}

#[derive(Resource, Default)]
pub struct GpuDriverResource {
    pub driver: Option<crate::gpu::compute_driver::GpuComputeDriver>,
}

pub fn flush_gpu_to_rust(sim: &mut SimWorld, driver: &crate::gpu::compute_driver::GpuComputeDriver) {
    let telem = driver.readback_telemetry();
    let read_count = (telem.population as usize + 512).min(driver.max_agents as usize).min(sim.world.max_cap);
    let updated_states = driver.readback_agent_states(read_count);
    let updated_genomes = driver.readback_agent_genomes(read_count);
    let updated_atomics = driver.readback_atomics(read_count);
    let updated_soil = driver.readback_soil();
    let params = crate::gpu::types::GpuSimParams {
        tick: sim.world.tick,
        agent_count: telem.population,
        max_agents: driver.max_agents,
        max_capacity: sim.world.max_cap as u32,
        ..Default::default()
    };
    crate::gpu::bridge::sync_gpu_to_rust(
        &updated_states,
        &updated_genomes,
        &updated_atomics,
        &updated_soil,
        &params,
        sim,
    );
}

pub fn sim_step_system(
    mut sim: ResMut<SimWorld>,
    mut gpu_res: Option<ResMut<GpuDriverResource>>,
) {
    if sim.paused {
        return;
    }
    let steps = sim.speed.clamp(1, 32);

    match sim.active_engine {
        ActiveEngine::Rust => {
            if let Some(ref mut gpu) = gpu_res {
                if let Some(ref driver) = gpu.driver {
                    if driver.is_initialized() {
                        flush_gpu_to_rust(&mut sim, driver);
                        driver.set_initialized(false);
                    }
                }
            }
            sim.step(steps);
        }
        ActiveEngine::Gpu => {
            if let Some(ref mut gpu) = gpu_res {
                if let Some(ref d) = gpu.driver {
                    if d.soil_cols != sim.world.soil.cols as u32 || d.soil_rows != sim.world.soil.rows as u32 {
                        gpu.driver = None;
                    }
                }
                if gpu.driver.is_none() {
                    let cols = sim.world.soil.cols as u32;
                    let rows = sim.world.soil.rows as u32;
                    gpu.driver = crate::gpu::compute_driver::GpuComputeDriver::create_for_world(cols, rows, 65536);
                }
                if let Some(ref mut driver) = gpu.driver {
                    if !driver.is_initialized() {
                        let (states, genomes, atomics, soil, params) = crate::gpu::bridge::sync_rust_to_gpu(&sim);
                        driver.upload_state(&states, &genomes, &atomics, &soil, &params);
                    }

                    let params = crate::gpu::types::GpuSimParams {
                        tick: sim.world.tick,
                        agent_count: sim.world.agents.len() as u32,
                        max_agents: driver.max_agents,
                        max_capacity: sim.world.max_cap as u32,
                        hostility: (sim.world.hostility / 100.0) as f32,
                        mut_rate: (sim.world.mutation / 100.0) as f32,
                        speed: sim.speed as f32,
                        renewal: (sim.world.growth / 100.0) as f32,
                        sub_tick: 0,
                        sub_ticks_per_frame: steps,
                        tool_type: 0xFFFFFFFF,
                        tool_radius: 45.0,
                        tool_pos: [0.0, 0.0],
                        camera_pos: [(sim.world_width * 0.5) as f32, (sim.world_height * 0.5) as f32],
                        camera_size: [sim.world_width as f32, sim.world_height as f32],
                        world_size: [sim.world_width as f32, sim.world_height as f32],
                        soil_grid: [sim.world.soil.cols as u32, sim.world.soil.rows as u32],
                        eclipse: sim.world.eclipse,
                        epoch: 0,
                    };

                    driver.update_params(&params);
                    driver.dispatch_sub_ticks(steps, &params);

                    // Read back ONLY telemetry (128 bytes)
                    let telemetry = driver.readback_telemetry();
                    sim.world.tick += steps;
                    sim.world.kills += telemetry.kills;
                    sim.world.births += telemetry.birth_count;

                    if telemetry.selected_agent_idx != 0xFFFFFFFF {
                        sim.selected_agent_id = Some(telemetry.selected_agent_id);
                    }

                    // Spore replenishment if population collapses in GPU mode
                    if telemetry.population < 15 && sim.world.tick % 45 == 0 {
                        let n = (15 - telemetry.population).min(15) as usize;
                        let mut spore_positions = Vec::with_capacity(n);
                        let w = sim.world.w;
                        let h = sim.world.h;
                        for _ in 0..n {
                            let x = sim.world.prng.rand(0.0, w) as f32;
                            let y = sim.world.prng.rand(0.0, h) as f32;
                            spore_positions.push((x, y));
                        }
                        driver.seed_spores_gpu(&spore_positions);
                    }

                    return;
                }
            }
            // Fallback to CPU step if no GPU adapter available
            sim.step(steps);
        }
    }
}

pub struct ClankSimPlugin;

impl Plugin for ClankSimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimWorld>()
            .init_resource::<GpuDriverResource>()
            .add_systems(Update, sim_step_system);
    }
}
