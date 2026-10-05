use clank_app::gpu::bridge::{sync_gpu_to_rust, sync_rust_to_gpu};
use clank_app::sim::SimWorld;
use clank_core::agent::GENES;

#[test]
fn test_genome_326_byte_exact_roundtrip() {
    let mut sim = SimWorld::default();
    assert!(!sim.world.agents.is_empty());

    // Set specific pattern on agent 0
    for i in 0..GENES {
        let val = ((i as i32 * 17 + 23) % 251) - 125;
        sim.world.agents[0].genes[i] = val as i8;
    }
    let expected_genes = sim.world.agents[0].genes;

    // Pack to GPU
    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    assert_eq!(genomes.len(), states.len());

    // Verify first genome packed representation
    // Reset agent 0 genes in sim to zeros to prove sync_gpu_to_rust restores them
    sim.world.agents[0].genes = [0i8; GENES];

    // Unpack back to Rust
    sync_gpu_to_rust(&states, &genomes, &atomics, &soil, &params, &mut sim);

    // Verify bit-exact restoration
    assert_eq!(
        sim.world.agents[0].genes, expected_genes,
        "All 326 genes must match bit-for-bit after roundtrip through GPU bridge"
    );
}
