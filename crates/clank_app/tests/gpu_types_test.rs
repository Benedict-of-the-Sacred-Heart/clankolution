use clank_app::gpu::types::{
    GpuAgentState, GpuAgentGenome, GpuSimParams, GpuLbvhNode, GpuAgentAtomic, BirthEvent,
    GpuSoilCell, AudioVoice, GpuTelemetry, ConsolidatedQueue,
};

#[test]
fn test_gpu_struct_alignments() {
    assert_eq!(std::mem::size_of::<GpuAgentState>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentState>(), 128);
    assert_eq!(std::mem::size_of::<GpuAgentGenome>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentGenome>(), 352);
    assert_eq!(std::mem::size_of::<GpuSimParams>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSimParams>(), 80);
    assert_eq!(std::mem::size_of::<GpuLbvhNode>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuLbvhNode>(), 48);
    assert_eq!(std::mem::size_of::<GpuAgentAtomic>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuAgentAtomic>(), 16);
    assert_eq!(std::mem::size_of::<BirthEvent>() % 16, 0);
    assert_eq!(std::mem::size_of::<BirthEvent>(), 16);
    assert_eq!(std::mem::size_of::<AudioVoice>() % 16, 0);
    assert_eq!(std::mem::size_of::<AudioVoice>(), 16);
    assert_eq!(std::mem::size_of::<GpuSoilCell>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuSoilCell>(), 16);
    assert_eq!(std::mem::size_of::<GpuTelemetry>() % 16, 0);
    assert_eq!(std::mem::size_of::<GpuTelemetry>(), 128);
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>() % 16, 0);
    assert_eq!(std::mem::size_of::<ConsolidatedQueue>(), 1_052_800);
}
