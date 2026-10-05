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
            sim.step(steps);
        }
        ActiveEngine::Gpu => {
            if let Some(ref mut gpu) = gpu_res {
                if gpu.driver.is_none() {
                    gpu.driver = crate::gpu::compute_driver::GpuComputeDriver::create_default();
                }
                if let Some(ref mut driver) = gpu.driver {
                    let (states, genomes, atomics, soil, params) = crate::gpu::bridge::sync_rust_to_gpu(&sim);
                    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
                    driver.dispatch_sub_ticks(steps, &params);

                    let read_count = (states.len() + 512).min(driver.max_agents as usize).min(sim.world.max_cap + 256);
                    let updated_states = driver.readback_agent_states(read_count);
                    let updated_atomics = driver.readback_atomics(read_count);
                    let updated_soil = driver.readback_soil();
                    let mut updated_params = params;
                    updated_params.tick += steps;

                    crate::gpu::bridge::sync_gpu_to_rust(
                        &updated_states,
                        &genomes,
                        &updated_atomics,
                        &updated_soil,
                        &updated_params,
                        &mut sim,
                    );

                    // Spore replenishment if population collapses below 15 (matching CPU World::evolve)
                    if sim.world.agents.len() < 15 && sim.world.tick % 45 == 0 {
                        let needed = 15 - sim.world.agents.len();
                        let (w, h) = (sim.world.w, sim.world.h);
                        for _ in 0..needed {
                            let x = sim.world.prng.rand(0.0, w);
                            let y = sim.world.prng.rand(0.0, h);
                            sim.world.seed_life_at(x, y);
                        }
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
