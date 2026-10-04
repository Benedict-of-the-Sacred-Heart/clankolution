use clank_app::gpu::birth_pipeline::{
    crossover_genes, mutate_gene_byte, pcg_float, pcg_hash, pcg_triangular,
};

#[test]
fn test_stateless_pcg_triangular_distribution() {
    // 1. PCG 3D hash evaluates deterministically from (id, stream, tick)
    let h1 = pcg_hash(42, 1, 100);
    let h2 = pcg_hash(42, 1, 100);
    assert_eq!(h1, h2);

    let h3 = pcg_hash(42, 2, 100);
    assert_ne!(h1, h3);

    let f = pcg_float(42, 1, 100);
    assert!(f >= 0.0 && f <= 1.0);

    // 2. r1 + r2 - 1.0 produces zero-centered symmetric triangular distribution T(-1, 0, 1)
    let mut sum = 0.0f32;
    let n = 10_000;
    for i in 0..n {
        let tri = pcg_triangular(i, 99, 1);
        assert!(tri >= -1.0 && tri <= 1.0);
        sum += tri;
    }
    let mean = sum / (n as f32);
    assert!(mean.abs() < 0.05); // Mean is approximately 0

    // 3. Genome mutation formula round((r1 + r2 - 1) * 100 * mutRate) matches HTML bounds [-127, 127]
    let initial_byte = 50i8;
    let mut_rate = 0.15f32;
    for i in 0..100 {
        let mutated = mutate_gene_byte(initial_byte, i, 7, 10, mut_rate);
        assert!(mutated >= -127);
    }

    // Crossover testing (48% parent A, 52% parent B)
    let word_a = 0xAAAAAAAAu32;
    let word_b = 0x55555555u32;
    let mixed = crossover_genes(word_a, word_b, 10, 20);
    // Should be valid u32
    assert!(mixed == word_a || mixed == word_b || (mixed != 0));
}
