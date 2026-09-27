use bevy::prelude::*;
use clank_core::world::World;
use clank_core::agent::AgentData;

#[derive(Resource)]
pub struct SimWorld {
    pub world: World,
    pub speed: u32,
    pub paused: bool,
    pub unthrottled: bool,
    pub selected_agent_id: Option<u32>,
    pub world_width: f64,
    pub world_height: f64,
}

impl Default for SimWorld {
    fn default() -> Self {
        Self {
            world: World::new(42),
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            world_width: 900.0,
            world_height: 600.0,
        }
    }
}

impl SimWorld {
    pub fn new(seed: u32) -> Self {
        Self {
            world: World::new(seed),
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            world_width: 900.0,
            world_height: 600.0,
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
