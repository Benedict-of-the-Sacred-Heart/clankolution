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
