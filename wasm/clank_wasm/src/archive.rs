use rkyv::{Archive, Deserialize, Serialize};
use crate::agent::AgentData;

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct WorldSnapshot {
    pub version: u32,
    pub tick: u32,
    pub kills: u32,
    pub births: u32,
    pub roots: u32,
    pub next_id: u32,
    pub eclipse: u32,
    pub width: f64,
    pub height: f64,
    pub cols: u32,
    pub rows: u32,
    pub mutation: f64,
    pub growth: f64,
    pub hostility: f64,
    pub prng_state: u32,
    pub food: Vec<f32>,
    pub taint: Vec<f32>,
    pub scent: Vec<f32>,
    pub agents: Vec<AgentData>,
}

impl WorldSnapshot {
    pub const CURRENT_VERSION: u32 = 1;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tick: u32,
        kills: u32,
        births: u32,
        roots: u32,
        next_id: u32,
        eclipse: u32,
        width: f64,
        height: f64,
        cols: u32,
        rows: u32,
        mutation: f64,
        growth: f64,
        hostility: f64,
        prng_state: u32,
        food: &[f32],
        taint: &[f32],
        scent: &[f32],
        agents: &[AgentData],
    ) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            tick,
            kills,
            births,
            roots,
            next_id,
            eclipse,
            width,
            height,
            cols,
            rows,
            mutation,
            growth,
            hostility,
            prng_state,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_snapshot_roundtrip_all_fields() {
        let mut sample_agent = AgentData::default();
        sample_agent.id = 1234;
        sample_agent.x = 450.5;
        sample_agent.y = 300.25;
        sample_agent.energy = 0.88;
        sample_agent.genes[0] = 42;
        sample_agent.genes[325] = -17;
        sample_agent.h[0] = 0.55;
        sample_agent.tr[1] = 0.92;

        let snap = WorldSnapshot::new(
            5000, 120, 340, 15, 9999, 105,
            1200.0, 800.0, 100, 60,
            0.25, 1.45, 1.10, 0x12345678,
            &[1.0, 1.5, 2.0],
            &[0.1, 0.2, 0.3],
            &[0.4, 0.5, 0.6],
            &[sample_agent],
        );

        let bytes = snap.to_bytes();
        assert!(!bytes.is_empty());

        let archived = rkyv::check_archived_root::<WorldSnapshot>(&bytes).expect("zero-copy validation must pass");
        assert_eq!(archived.version, WorldSnapshot::CURRENT_VERSION);
        assert_eq!(archived.tick, 5000);
        assert_eq!(archived.kills, 120);
        assert_eq!(archived.births, 340);
        assert_eq!(archived.roots, 15);
        assert_eq!(archived.next_id, 9999);
        assert_eq!(archived.eclipse, 105);
        assert_eq!(archived.width, 1200.0);
        assert_eq!(archived.height, 800.0);
        assert_eq!(archived.cols, 100);
        assert_eq!(archived.rows, 60);
        assert_eq!(archived.mutation, 0.25);
        assert_eq!(archived.growth, 1.45);
        assert_eq!(archived.hostility, 1.10);
        assert_eq!(archived.prng_state, 0x12345678);

        assert_eq!(archived.food.len(), 3);
        assert_eq!(archived.food[0], 1.0);
        assert_eq!(archived.food[1], 1.5);
        assert_eq!(archived.food[2], 2.0);

        assert_eq!(archived.agents.len(), 1);
        let a = &archived.agents[0];
        assert_eq!(a.id, 1234);
        assert_eq!(a.x, 450.5);
        assert_eq!(a.y, 300.25);
        assert_eq!(a.energy, 0.88);
        assert_eq!(a.genes[0], 42);
        assert_eq!(a.genes[325], -17);
        assert_eq!(a.h[0], 0.55);
        assert_eq!(a.tr[1], 0.92);

        // Deserialization roundtrip
        let restored: WorldSnapshot = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(restored.tick, 5000);
        assert_eq!(restored.food, vec![1.0, 1.5, 2.0]);
        assert_eq!(restored.agents.len(), 1);
        assert_eq!(restored.agents[0].id, 1234);
    }

    #[test]
    fn test_corrupted_snapshot_rejected() {
        let snap = WorldSnapshot::new(
            1, 0, 0, 0, 1, 0,
            900.0, 600.0, 75, 50,
            0.16, 1.0, 1.0, 12345,
            &[1.0, 2.0], &[0.0, 0.0], &[0.0, 0.0],
            &[],
        );
        let bytes = snap.to_bytes();
        assert!(rkyv::check_archived_root::<WorldSnapshot>(&bytes).is_ok());

        let len = bytes.len();
        println!("Buffer length: {}, Root size: {}", len, std::mem::size_of::<ArchivedWorldSnapshot>());

        // Truncated buffer should always fail
        assert!(rkyv::check_archived_root::<WorldSnapshot>(&bytes[..len - 1]).is_err());
        assert!(rkyv::check_archived_root::<WorldSnapshot>(&bytes[..len / 2]).is_err());
        assert!(rkyv::check_archived_root::<WorldSnapshot>(&[]).is_err());
    }
}

