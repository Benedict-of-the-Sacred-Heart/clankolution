use bytemuck::Zeroable;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::gpu::types::{
    GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuSimParams, GpuSoilCell,
};

/// The CPU engine may hold one more creature than the cap (the original seed tool spawns two
/// whenever the population is below CAP). Uploading more states than the GPU has slots must
/// keep the first `max_agents` and never write past the end of the agent buffers.
#[test]
fn test_upload_state_truncates_to_gpu_slot_capacity() {
    let Some(driver) = GpuComputeDriver::create_for_world(75, 50, 16) else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let n = 17usize;
    let mut states = vec![GpuAgentState::zeroed(); n];
    for (i, s) in states.iter_mut().enumerate() {
        s.id = i as u32 + 1;
    }
    let genomes = vec![GpuAgentGenome::zeroed(); n];
    let atomics = vec![
        GpuAgentAtomic {
            energy_milli: 30_000,
            mate_claim: 0,
            mate_energy_milli: 0,
            dead_claimed: 0
        };
        n
    ];
    let soil = vec![GpuSoilCell::zeroed(); 75 * 50];
    let mut params = GpuSimParams::zeroed();
    params.max_agents = 16;
    params.max_capacity = 16;

    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    let back = driver.readback_agent_states(16);
    assert_eq!(back.len(), 16);
    assert_eq!(back[15].id, 16, "the first max_agents states are kept");
}
