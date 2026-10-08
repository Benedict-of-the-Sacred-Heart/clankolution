//! Parity with the original simulation in `clankolution.html` at main@dfe8dc6
//! (the reference behavior). Every expectation below transcribes the original
//! JavaScript formula; sin/cos/atan2/tanh are the only accepted deviations.

use clank_core::agent::AgentData;
use clank_core::math::{self, Prng};
use clank_core::soil::SoilGrid;
use clank_core::world::World;

/// Original `brain()`: every term is dequantized with `/127` before summing,
/// sums are strictly sequential, then `tanh(s * .61)` / `tanh(s * .66)`.
fn original_brain(genes: &[i8], ins: &[f64; 15], h: &[f32; 10]) -> ([f32; 10], [f32; 6]) {
    let mut p = 0;
    let mut new_h = [0f32; 10];
    let mut out = [0f32; 6];
    for j in 0..10 {
        let mut s = 0.0f64;
        for i in 0..15 {
            s += (genes[p] as f64) * ins[i] / 127.0;
            p += 1;
        }
        for k in 0..10 {
            s += (genes[p] as f64) * (h[k] as f64) / 127.0;
            p += 1;
        }
        s += (genes[p] as f64) / 127.0;
        p += 1;
        new_h[j] = math::tanh(s * 0.61) as f32;
    }
    for j in 0..6 {
        let mut s = 0.0f64;
        for k in 0..10 {
            s += (genes[p] as f64) * (new_h[k] as f64) / 127.0;
            p += 1;
        }
        s += (genes[p] as f64) / 127.0;
        p += 1;
        out[j] = math::tanh(s * 0.66) as f32;
    }
    (new_h, out)
}

/// A constructed case where factored dequantization (`s * (0.61 / 127)`) and
/// grouped sums round hidden neuron 0 and two outputs to different f32 values.
const DIVERGENT_GENES: [i8; 326] = [
    122, -49, -4, 81, 2, -39, -108, 68, 126, 83, -10, 113, 99, 119, 32, -64, -7, -50, 69, 93, 37,
    71, 73, 39, -12, -108, -44, -75, 64, -101, -56, -110, 33, 109, -39, -73, -127, 46, 79, -115,
    25, 51, 111, -118, 16, 102, 22, 87, -10, -61, -60, 4, 56, -49, -103, 90, -16, 100, -90, -124,
    -80, 65, -19, 59, -18, -107, -34, 121, 121, 68, -51, -127, 23, -88, 1, -84, 83, 106, -68, 68,
    -52, 67, 116, -86, 94, 113, 11, -124, 19, 52, 40, 26, 113, 79, 120, 97, -6, -26, -67, -63, 125,
    -11, -86, -120, -37, -55, -24, -61, 80, -42, 119, 80, 21, 31, -84, -119, 124, -121, 40, 117,
    62, -37, -102, 27, -31, -127, 36, 22, -109, -45, 25, -55, 27, 118, 83, 82, 68, -29, 113, 92,
    -17, 37, 96, 89, -99, 115, -17, 81, -61, -7, -127, -81, -93, 6, 46, -125, 29, 110, 5, 4, 26,
    44, -71, 21, 120, -80, -12, 44, -33, -95, 53, 108, 65, 97, 77, 25, -92, 64, -83, -84, 16, -116,
    39, -57, 112, 122, 114, -22, 74, 33, -31, -90, -91, 58, -88, -108, -17, 12, -123, 42, -35, 20,
    55, 2, 60, -121, -57, -94, -113, -109, 32, -108, 55, 111, -126, 94, 120, -12, 15, -108, 113,
    -68, -34, -94, -16, -116, 61, -122, -100, -7, 73, 69, 45, -74, -29, -74, 71, 43, -100, 89, -12,
    16, 37, 33, -7, 92, 91, 46, 71, -42, -61, -70, 1, -100, -73, 64, -111, 109, 54, -97, -104, 112,
    31, 63, -5, 64, -91, 117, -103, 112, 99, 109, -50, -92, 81, 85, -84, 104, -119, 40, 102, -35,
    3, 18, -66, -53, 109, -59, 55, -102, 117, -18, -27, 127, 127, -37, 8, 28, 82, 108, -117, 94,
    -109, 63, 126, 43, -125, -100, 48, 69, 51, 13, 107, -85, 93, -124, 70, -13, 5, -74, -111, -8,
    -77, 54, 15, -16,
];
const DIVERGENT_INS: [f64; 15] = [
    -0.778247546086991,
    -0.9141327345627076,
    0.48982730137566155,
    0.27162694832933043,
    -0.8572969114931502,
    -0.15941413870994767,
    0.28979489993950425,
    -0.5428914676273577,
    -0.6761536564106296,
    -0.9979970136651485,
    -0.46280986616219216,
    -0.5233417473658287,
    0.8791728732396198,
    -0.9797128758414096,
    -0.9536524631302719,
];
const DIVERGENT_H: [f32; 10] = [
    0.37724122,
    0.9994359,
    0.042804725,
    -0.39179558,
    0.9609525,
    0.8966886,
    -0.9991991,
    0.12212469,
    -0.6748915,
    -0.30386615,
];

fn assert_brain_matches_original(genes: &[i8; 326], ins: &[f64; 15], h: &[f32; 10]) {
    let (want_h, want_out) = original_brain(genes, ins, h);
    let mut a = AgentData::default();
    a.genes = *genes;
    a.h = *h;
    let out = World::brain(&mut a, ins);
    for k in 0..10 {
        assert_eq!(
            a.h[k].to_bits(),
            want_h[k].to_bits(),
            "hidden {k}: {} vs original {}",
            a.h[k],
            want_h[k]
        );
    }
    for j in 0..6 {
        assert_eq!(
            out[j].to_bits(),
            want_out[j].to_bits(),
            "output {j}: {} vs original {}",
            out[j],
            want_out[j]
        );
    }
}

#[test]
fn brain_matches_original_dequantization_on_divergent_case() {
    assert_brain_matches_original(&DIVERGENT_GENES, &DIVERGENT_INS, &DIVERGENT_H);
}

#[test]
fn brain_matches_original_on_random_cases() {
    let mut rng = Prng::new(2024);
    for _ in 0..20_000 {
        let mut genes = [0i8; 326];
        for g in genes.iter_mut() {
            *g = rng.rand(-127.0, 127.0).round() as i8;
        }
        let mut ins = [0f64; 15];
        for v in ins.iter_mut() {
            *v = rng.rand(-1.0, 1.0);
        }
        let mut h = [0f32; 10];
        for v in h.iter_mut() {
            *v = rng.rand(-1.0, 1.0) as f32;
        }
        assert_brain_matches_original(&genes, &ins, &h);
    }
}

/// Original `idx()`/`sample()`/`deposit()` locate cells with `Math.floor(wrap(x, W) / cellW)`.
/// Multiplying by a precomputed reciprocal picks a different cell exactly at some boundaries
/// whenever `cellW` is not 12 (e.g. a 1000px-wide world has 84 columns of 11.9047...px).
#[test]
fn soil_cell_lookup_divides_by_cell_width_like_original() {
    let mut soil = SoilGrid::new();
    soil.resize(1000.0, 600.0, 84, 50);
    let x = 11.904761904761905_f64; // == 1000 / 84 in f64
    let y = 6.0;
    let want_col = (x / (1000.0 / 84.0)).floor() as usize;
    assert_eq!(want_col, 1);
    assert_eq!(soil.idx(x, y) % 84, want_col, "idx() column");
    assert_eq!(soil.idx_fast(x, y) % 84, want_col, "idx_fast() column");

    soil.food.fill(0.0);
    soil.deposit(0, x, y, 0.5, 1);
    assert!(
        (soil.food[want_col] - 0.5).abs() < 1e-6,
        "deposit() centre must land in column {want_col}"
    );
}

/// JavaScript `Math.round` rounds ties toward +infinity; Rust's `f64::round` rounds them away
/// from zero. Genes use `Math.round` in the original.
#[test]
fn js_round_matches_javascript_math_round() {
    assert_eq!(math::js_round(-2.5), -2.0);
    assert_eq!(math::js_round(2.5), 3.0);
    assert_eq!(math::js_round(-0.5), 0.0);
    assert_eq!(math::js_round(-1.6), -2.0);
    assert_eq!(math::js_round(0.49999999999999994), 0.0);
    assert_eq!(math::js_round(-58.0), -58.0);
}

fn agent_at(world: &mut World, x: f64, y: f64) -> usize {
    let idx = world.create_agent(x, y, None, None);
    world.agents[idx].genes = [0; 326];
    // Output neuron 3 (attack) bias: tanh(127 / 127 * .66) ≈ 0.578 > 0.25.
    world.agents[idx].genes[260 + 3 * 11 + 10] = 127;
    world.agents[idx].energy = 50.0;
    world.agents[idx].cooldown = 0;
    idx
}

/// Original: `d = best?.d || 999`, so a neighbour at distance exactly 0 counts as 999 away
/// and cannot be attacked or mated with.
#[test]
fn neighbour_at_exact_overlap_is_treated_as_999_away() {
    let mut world = World::new(7);
    world.agents.clear();
    world.sync_pos_cache();
    let first = agent_at(&mut world, 450.0, 300.0);
    agent_at(&mut world, 450.0, 300.0);
    let first_id = world.agents[first].id;
    world.sync_pos_cache();

    world.evolve();

    let a = world
        .agents
        .iter()
        .find(|a| a.id == first_id)
        .expect("agent alive");
    assert_eq!(
        a.cooldown, 0,
        "agent 0 attacked a neighbour at d == 0 (original treats it as 999)"
    );
}

/// Original extinguish: `a.energy = 0; a.dead = true; die(a)` then the array is filtered
/// immediately. `die` deposits `clamp(0 * .016 + .6, .3, 2)` = 0.6 food at radius 2 and
/// 0.1 taint at radius 1, and emits `spark(…, 5)` (15 random draws).
#[test]
fn extinguish_uses_original_die_semantics_and_compacts() {
    let mut world = World::new(11);
    world.agents.clear();
    world.sync_pos_cache();
    world.create_agent(450.0, 300.0, None, None);
    world.create_agent(100.0, 100.0, None, None);
    world.sync_pos_cache();
    world.soil.food.fill(0.0);
    world.soil.taint.fill(0.0);
    let mut expected_prng = Prng::new(world.prng.s);
    for _ in 0..15 {
        expected_prng.next_u32();
    }

    world.extinguish_at(450.0, 300.0, 23.0);

    assert_eq!(
        world.agents.len(),
        1,
        "extinguished creatures are removed immediately"
    );
    assert_eq!(world.pos_x.len(), 1);
    assert_eq!(world.agents[0].x, 100.0);
    let c = world.soil.idx(450.0, 300.0);
    let cols = world.soil.cols;
    assert!(
        (world.soil.food[c] - 0.6).abs() < 1e-6,
        "corpse food centre 0.6, got {}",
        world.soil.food[c]
    );
    assert!(
        (world.soil.food[c + 2] - (0.6 / 4.2) as f32).abs() < 1e-6,
        "corpse food radius 2"
    );
    assert!((world.soil.taint[c] - 0.1).abs() < 1e-6, "corpse taint 0.1");
    assert!(
        (world.soil.taint[c + cols] - (0.1 / 1.8) as f32).abs() < 1e-6,
        "corpse taint radius 1"
    );
    assert_eq!(
        world.prng.s, expected_prng.s,
        "die() spark consumes exactly 15 random draws"
    );
}

/// Original catastrophe button: `eclipse = 210; spark(rand(W), rand(H), '#ff6659', 30)`
/// (2 + 30 * 3 = 92 random draws).
#[test]
fn trigger_eclipse_matches_original_catastrophe() {
    let mut world = World::new(3);
    let mut expected_prng = Prng::new(world.prng.s);
    for _ in 0..92 {
        expected_prng.next_u32();
    }

    world.trigger_eclipse();

    assert_eq!(world.eclipse, 210);
    assert_eq!(
        world.prng.s, expected_prng.s,
        "eclipse spark consumes exactly 92 random draws"
    );
}

/// Original seed tool: `if (agents.length < CAP) for (i < 2) agent(...)` — two creatures
/// whenever the population is below the cap, even if that ends at CAP + 1.
#[test]
fn seed_life_spawns_two_whenever_below_cap() {
    let mut world = World::new(5);
    world.agents.clear();
    world.sync_pos_cache();
    for i in 0..(world.max_cap - 1) {
        world.create_agent((i % 90) as f64 * 10.0, (i / 90) as f64 * 10.0, None, None);
    }
    world.sync_pos_cache();

    world.seed_life_at(450.0, 300.0);

    assert_eq!(world.agents.len(), world.max_cap + 1);
}
