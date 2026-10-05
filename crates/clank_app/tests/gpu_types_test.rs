use clank_app::gpu::types::{
    GpuAgentState, GpuAgentGenome, GpuSimParams, GpuLbvhNode, GpuAgentAtomic, BirthEvent,
    GpuSoilCell, AudioVoice, GpuTelemetry, ConsolidatedQueue,
};

#[test]
fn test_gpu_struct_alignments_and_divisible_by_4() {
    assert_eq!(std::mem::size_of::<GpuAgentState>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentState>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuAgentState>(), 128);

    assert_eq!(std::mem::size_of::<GpuAgentGenome>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentGenome>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuAgentGenome>(), 352);

    assert_eq!(std::mem::size_of::<GpuSimParams>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSimParams>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuSimParams>(), 96);

    assert_eq!(std::mem::size_of::<GpuLbvhNode>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuLbvhNode>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuLbvhNode>(), 48);

    assert_eq!(std::mem::size_of::<GpuAgentAtomic>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentAtomic>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuAgentAtomic>(), 16);

    assert_eq!(std::mem::size_of::<BirthEvent>() % 16, 0);
    assert_eq!(std::mem::size_of::<BirthEvent>() % 4, 0);
    assert_eq!(std::mem::size_of::<BirthEvent>(), 16);

    assert_eq!(std::mem::size_of::<AudioVoice>() % 16, 0);
    assert_eq!(std::mem::size_of::<AudioVoice>() % 4, 0);
    assert_eq!(std::mem::size_of::<AudioVoice>(), 16);

    assert_eq!(std::mem::size_of::<GpuSoilCell>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSoilCell>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuSoilCell>(), 16);

    assert_eq!(std::mem::size_of::<GpuTelemetry>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuTelemetry>() % 4, 0);
    assert_eq!(std::mem::size_of::<GpuTelemetry>(), 128);

    assert_eq!(std::mem::size_of::<ConsolidatedQueue>() % 16, 0);
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>() % 4, 0);
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>(), 1_052_800);
}

#[test]
fn test_repurposed_functional_fields() {
    let mut params = GpuSimParams::default();
    params.tool_radius = 45.0;
    params.eclipse = 120;
    params.epoch = 3;
    assert_eq!(params.tool_radius, 45.0);
    assert_eq!(params.eclipse, 120);
    assert_eq!(params.epoch, 3);

    let birth = BirthEvent {
        parent_a: 1,
        parent_b: 2,
        child_slot: 3,
        birth_tick: 500,
    };
    assert_eq!(birth.birth_tick, 500);

    let soil = GpuSoilCell {
        food_milli: 100,
        taint_milli: 200,
        scent_milli: 300,
        fertility_milli: 400,
    };
    assert_eq!(soil.fertility_milli, 400);

    let telem = GpuTelemetry {
        total_births: 10,
        total_deaths: 5,
        max_generation: 12,
        extinctions: 2,
        ..Default::default()
    };
    assert_eq!(telem.total_births, 10);
    assert_eq!(telem.total_deaths, 5);
    assert_eq!(telem.max_generation, 12);
    assert_eq!(telem.extinctions, 2);
}
