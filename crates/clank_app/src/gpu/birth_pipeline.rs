//! GPU Birth, Mutation & PCG PRNG Pipeline
//!
//! Provides stateless GPU-compatible PCG hashing, symmetric triangular distribution sampling,
//! and parallel genome crossover/mutation math matching `birth_step.wgsl`.

/// Stateless 3D PCG hash generating high-entropy pseudo-random numbers from (id, stream, tick).
#[inline]
pub fn pcg_hash(id: u32, stream: u32, tick: u32) -> u32 {
    let state = id
        .wrapping_mul(747796405)
        .wrapping_add(stream.wrapping_mul(2891336453))
        .wrapping_add(tick.wrapping_mul(1013904223));
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// Evaluates uniform float in [0.0, 1.0] from PCG hash.
#[inline]
pub fn pcg_float(id: u32, stream: u32, tick: u32) -> f32 {
    (pcg_hash(id, stream, tick) as f32) / 4294967295.0
}

/// Evaluates zero-centered symmetric triangular distribution $T(-1, 0, 1)$ via $r_1 + r_2 - 1.0$.
#[inline]
pub fn pcg_triangular(id: u32, stream: u32, tick: u32) -> f32 {
    let r1 = pcg_float(id, stream, tick);
    let r2 = pcg_float(id, stream.wrapping_add(100), tick);
    r1 + r2 - 1.0
}

/// Mutates a single signed gene byte matching HTML reference:
/// `round((r1 + r2 - 1.0) * 100.0 * mut_rate)` clamped to `[-127, 127]`.
#[inline]
pub fn mutate_gene_byte(val: i8, id: u32, stream: u32, tick: u32, mut_rate: f32) -> i8 {
    let delta = (pcg_triangular(id, stream, tick) * 100.0 * mut_rate).round() as i32;
    let new_val = (val as i32 + delta).clamp(-127, 127);
    new_val as i8
}

/// Crossover between two 32-bit packed gene words with 48% inheritance from parent A.
#[inline]
pub fn crossover_genes(word_a: u32, word_b: u32, id: u32, tick: u32) -> u32 {
    let p = pcg_float(id, 42, tick);
    if p < 0.48 {
        word_a
    } else {
        word_b
    }
}

/// Mutates a 32-bit packed word containing 4 signed 8-bit gene weights,
/// applying triangular mutation and masking inactive dummy genes when `expanded_cortex` is false.
pub fn mutate_word(
    word: u32,
    id: u32,
    word_idx: u32,
    tick: u32,
    mut_rate: f32,
    expanded_cortex: bool,
) -> u32 {
    let bytes = word.to_le_bytes();
    let mut mutated_bytes = [0u8; 4];
    for i in 0..4 {
        let stream = word_idx * 4 + i as u32;
        let b = mutate_gene_byte(bytes[i] as i8, id, stream, tick, mut_rate);
        mutated_bytes[i] = b as u8;
    }
    let mut result = u32::from_le_bytes(mutated_bytes);

    if !expanded_cortex {
        if word_idx < 70 && (word_idx % 7) == 6 {
            result &= 0x0000FFFF;
        }
        if word_idx >= 70 && ((word_idx - 70) % 3) == 2 {
            result &= 0x00FFFFFF;
        }
    }

    result
}

