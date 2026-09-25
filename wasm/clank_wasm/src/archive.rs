use rkyv::{Archive, Deserialize, Serialize};
use crate::agent::AgentData;
use crate::soil::GRID_SIZE;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct WorldSnapshot {
    pub tick: u32,
    pub kills: u32,
    pub births: u32,
    pub roots: u32,
    pub next_id: u32,
    pub eclipse: u32,
    pub food: Vec<f32>,
    pub taint: Vec<f32>,
    pub scent: Vec<f32>,
    pub agents: Vec<AgentData>,
}

impl WorldSnapshot {
    pub fn new(
        tick: u32,
        kills: u32,
        births: u32,
        roots: u32,
        next_id: u32,
        eclipse: u32,
        food: &[f32; GRID_SIZE],
        taint: &[f32; GRID_SIZE],
        scent: &[f32; GRID_SIZE],
        agents: &[AgentData],
    ) -> Self {
        Self {
            tick,
            kills,
            births,
            roots,
            next_id,
            eclipse,
            food: food.to_vec(),
            taint: taint.to_vec(),
            scent: scent.to_vec(),
            agents: agents.to_vec(),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 4096>(self).unwrap().into_vec()
    }
}
