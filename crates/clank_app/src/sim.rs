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

pub fn sim_step_system(mut sim: ResMut<SimWorld>) {
    if sim.paused {
        return;
    }
    let steps = sim.speed.clamp(1, 32);
    sim.step(steps);
}

pub struct ClankSimPlugin;

impl Plugin for ClankSimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimWorld>()
            .add_systems(Update, sim_step_system);
    }
}
