//! Plan Task 5: the agent pass (brain, sensing, steering, kinematics, metabolism, toroidal
//! wrap), combat, births, mods, tombstones, audio and `visual_cache` — on the real shaders,
//! one tick against the reference `World::evolve` in controlled scenarios.

#[path = "support/gpu_harness.rs"]
mod gpu_harness;

use clank_app::gpu::bridge::{pack_agent_state, sync_rust_to_gpu};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_core::agent::AgentData;
use gpu_harness::*;

/// clankolution.html brain shape: N=15 inputs, Q=10 hidden, O=6 outputs.
const HIDDEN_GENES: usize = 10 * (15 + 10 + 1);
fn output_bias(j: usize) -> usize {
    HIDDEN_GENES + j * 11 + 10
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// A creature with a random founder genome, traits and hidden state, too young to breed.
fn random_creature(rng: &mut Lcg, id: u32, x: f64, y: f64) -> AgentData {
    let mut a = creature(id, x, y, rng.range(0.0, std::f64::consts::TAU));
    for g in a.genes.iter_mut() {
        *g = rng.range(-58.0, 58.0).round() as i8;
    }
    for t in a.tr.iter_mut() {
        *t = rng.next();
    }
    for h in a.h.iter_mut() {
        *h = rng.range(-1.0, 1.0) as f32;
    }
    a.vx = rng.range(-1.0, 1.0);
    a.vy = rng.range(-1.0, 1.0);
    a.energy = rng.range(30.0, 90.0);
    a.age = 10;
    a
}

fn reference(sim: &clank_app::sim::SimWorld, id: u32) -> Option<AgentData> {
    sim.world
        .agents
        .iter()
        .find(|a| a.id == id && a.dead == 0)
        .copied()
}

/// Differences between the GPU's and the reference's view of one creature after a tick.
fn compare(g: &AgentData, r: &AgentData) -> Vec<String> {
    let mut d = Vec::new();
    let pos = torus_dist(g.x, g.y, r.x, r.y, 900.0, 600.0);
    if pos > 1e-3 {
        d.push(format!(
            "position off by {pos:.2e} ({:.4},{:.4}) vs ({:.4},{:.4})",
            g.x, g.y, r.x, r.y
        ));
    }
    if (g.vx - r.vx).abs() > 1e-4 || (g.vy - r.vy).abs() > 1e-4 {
        d.push(format!(
            "velocity ({:.5},{:.5}) vs ({:.5},{:.5})",
            g.vx, g.vy, r.vx, r.vy
        ));
    }
    if angle_diff(g.angle, r.angle) > 1e-4 {
        d.push(format!("angle {:.6} vs {:.6}", g.angle, r.angle));
    }
    if (g.energy - r.energy).abs() > 2e-3 {
        d.push(format!("energy {:.5} vs {:.5}", g.energy, r.energy));
    }
    let h =
        g.h.iter()
            .zip(&r.h)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
    if h > 1e-4 {
        d.push(format!("hidden state off by {h:.2e}"));
    }
    for (name, a, b) in [
        ("feeding", g.feeding, r.feeding),
        ("attack", g.attack, r.attack),
        ("signal", g.signal, r.signal),
    ] {
        if (a - b).abs() > 1e-4 {
            d.push(format!("{name} {a:.5} vs {b:.5}"));
        }
    }
    if g.age != r.age {
        d.push(format!("age {} vs {}", g.age, r.age));
    }
    d
}

/// A lone creature (no neighbour anywhere) takes the same tick on the GPU as in the
/// reference: brain, food sensing, steering, thrust, drag, wrap and metabolism. Includes
/// creatures crossing each seam.
#[test]
fn test_a_lone_creature_takes_the_reference_tick() {
    let mut rng = Lcg(7);
    let spots = [
        (450.0, 300.0),
        (899.9, 300.0),
        (0.05, 120.0),
        (300.0, 599.95),
        (700.0, 0.02),
        (123.4, 456.7),
        (612.0, 77.0),
        (20.0, 580.0),
    ];
    let mut failures = Vec::new();
    for (k, (x, y)) in spots.into_iter().enumerate() {
        let a = random_creature(&mut rng, 100 + k as u32, x, y);
        let mut sim = world_with(vec![a], 100.0);
        let Some(mut gpu) = upload(&sim) else {
            return;
        };
        gpu.tick();
        sim.world.evolve();
        let live = gpu.live();
        let (Some((_, g)), Some(r)) = (live.first(), reference(&sim, a.id)) else {
            failures.push(format!(
                "creature {k}: alive on GPU {} / reference {}",
                live.len(),
                reference(&sim, a.id).is_some()
            ));
            continue;
        };
        let d = compare(g, &r);
        if !d.is_empty() {
            failures.push(format!("creature {k} at ({x}, {y}): {}", d.join("; ")));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of 8 lone creatures differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The agent pass takes the slots in Morton order, not slot order: in a crowd of 300, every
/// creature still takes exactly one step per tick. A skipped creature keeps its age (10); one
/// stepped twice pays and gains energy twice, about double the reference's change.
#[test]
fn test_every_creature_in_a_crowd_takes_one_step() {
    let mut rng = Lcg(11);
    let agents: Vec<AgentData> = (0..300u32)
        .map(|k| {
            let x = 22.5 + (k % 20) as f64 * 45.0 + rng.range(-10.0, 10.0);
            let y = 20.0 + (k / 20) as f64 * 40.0 + rng.range(-10.0, 10.0);
            random_creature(&mut rng, 1 + k, x, y)
        })
        .collect();
    let mut sim = world_with(agents, 0.0);
    let before: std::collections::HashMap<u32, f64> =
        sim.world.agents.iter().map(|a| (a.id, a.energy)).collect();
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    sim.world.evolve();
    let live = gpu.live();
    assert!(
        live.len() > 250,
        "only {} survived; scenario is wrong",
        live.len()
    );
    let unstepped: Vec<String> = live
        .iter()
        .filter(|(_, a)| a.age != 11)
        .map(|(slot, a)| format!("slot {slot} (id {}): age {}", a.id, a.age))
        .collect();
    assert!(
        unstepped.is_empty(),
        "{} of {} creatures did not take exactly one step:\n{}",
        unstepped.len(),
        live.len(),
        unstepped[..unstepped.len().min(12)].join("\n")
    );
    // A few creatures differ from the reference for an accepted reason: two neighbours almost
    // equally near, and the reference has moved one of them before this creature runs (id 53
    // here: 33 at 37.00 px after its move, 37.62 before; 54 at 37.18), so the nearest differs.
    let off: Vec<String> = live
        .iter()
        .filter_map(|(_, g)| {
            let r = reference(&sim, g.id)?;
            let (dg, dr) = (g.energy - before[&g.id], r.energy - before[&g.id]);
            ((dg - dr).abs() > 0.5 * dr.abs() + 0.01)
                .then(|| format!("id {}: energy change {dg:.4} vs reference {dr:.4}", g.id))
        })
        .collect();
    assert!(
        off.len() <= 3,
        "{} creatures changed energy unlike the reference (stepped twice?):\n{}",
        off.len(),
        off[..off.len().min(12)].join("\n")
    );
}

/// An attacker next to a weak victim kills it: the victim dies, the kill is counted and
/// credited, the attacker gains the reference's energy and goes on cooldown.
#[test]
fn test_a_kill_matches_the_reference() {
    let mut attacker = creature(1, 400.0, 300.0, 0.0);
    attacker.genes[output_bias(3)] = 127; // attack = tanh(0.66) ≈ 0.58
    attacker.tr = [0.8, 0.5, 0.5, 0.5, 0.5, 0.9];
    attacker.age = 10;
    let mut victim = creature(2, 408.0, 300.0, std::f64::consts::PI);
    victim.energy = 0.4;
    victim.age = 10;
    victim.tr[3] = 0.0;
    let mut sim = world_with(vec![attacker, victim], 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let t = gpu.tick();
    let t2 = gpu.tick();
    sim.world.evolve();
    sim.world.evolve();
    let r = reference(&sim, 1).expect("reference attacker alive");
    assert!(
        reference(&sim, 2).is_none(),
        "reference victim survived; scenario is wrong"
    );
    let live = gpu.live();
    assert!(
        live.iter().all(|(_, a)| a.id != 2),
        "victim still alive on the GPU"
    );
    let (_, g) = live
        .iter()
        .find(|(_, a)| a.id == 1)
        .expect("attacker alive on GPU");
    assert_eq!(t.kills + t2.kills, 1, "kills counted");
    assert_eq!(g.kills, r.kills, "kill credited to the attacker");
    assert!(
        (g.energy - r.energy).abs() < 2e-3,
        "attacker energy {:.4} vs reference {:.4}",
        g.energy,
        r.energy
    );
    assert_eq!(g.cooldown, r.cooldown, "attacker cooldown");
}

/// A creature ready to breed (energy, age > 65, no birth cooldown) and alone gives birth:
/// child energy 24, gen + 1, same lineage, within ±9px of the parent; the parent pays 24
/// and starts its 95-tick birth countdown.
#[test]
fn test_an_asexual_birth_matches_the_reference() {
    let mut parent = creature(1, 400.0, 300.0, 0.0);
    parent.energy = 100.0;
    parent.age = 70;
    parent.gen = 3;
    let mut sim = world_with(vec![parent], 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    sim.world.evolve();
    let r_parent = reference(&sim, 1).unwrap();
    let r_child = sim
        .world
        .agents
        .iter()
        .find(|a| a.id != 1)
        .expect("reference child");
    let live = gpu.live();
    let (_, g_parent) = live.iter().find(|(_, a)| a.id == 1).expect("GPU parent");
    let Some((_, g_child)) = live.iter().find(|(_, a)| a.id != 1) else {
        panic!(
            "no child on the GPU (reference child gen {} energy {:.1})",
            r_child.gen, r_child.energy
        );
    };
    assert!(
        (g_child.energy - r_child.energy).abs() < 1e-3,
        "child energy {:.3} vs {:.3}",
        g_child.energy,
        r_child.energy
    );
    assert_eq!(g_child.gen, r_child.gen, "child generation");
    assert_eq!(g_child.root, r_child.root, "child lineage");
    assert!(
        (g_child.x - g_parent.x).abs() <= 9.0 + 1e-3
            && (g_child.y - g_parent.y).abs() <= 9.0 + 1e-3,
        "child placed away from parent"
    );
    assert!(
        (g_parent.energy - r_parent.energy).abs() < 2e-3,
        "parent energy {:.4} vs {:.4}",
        g_parent.energy,
        r_parent.energy
    );
    assert_eq!(g_parent.birth, r_parent.birth, "parent birth countdown");
}

/// A dead slot is left alone by the agent pass (tombstone guard): its state does not change.
#[test]
fn test_dead_slots_are_not_modified() {
    let a = creature(1, 400.0, 300.0, 0.0);
    let mut b = creature(2, 100.0, 100.0, 0.0);
    b.dead = 1;
    b.energy = 0.0;
    let sim = world_with(vec![a, b], 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let before = gpu.states()[1];
    gpu.ticks(3);
    let after = gpu.states()[1];
    assert_eq!(before, after, "the dead slot changed");
}

/// A kill in view emits a kill audio voice at the attacker.
#[test]
fn test_a_kill_emits_an_audio_voice() {
    let mut attacker = creature(1, 400.0, 300.0, 0.0);
    attacker.genes[output_bias(3)] = 127;
    attacker.tr = [0.8, 0.5, 0.5, 0.5, 0.5, 0.9];
    let mut victim = creature(2, 408.0, 300.0, std::f64::consts::PI);
    victim.energy = 0.4;
    victim.tr[3] = 0.0;
    let sim = world_with(vec![attacker, victim], 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let t = gpu.tick();
    let voices = gpu
        .driver
        .readback_audio_voices(t.audio_voice_count.min(256) as usize);
    let kill = voices.iter().find(|v| v.event_type == 1);
    assert!(kill.is_some(), "no kill voice among {voices:?}");
    let v = kill.unwrap();
    assert!(
        (v.pos[0] - 400.0).abs() < 2.0 && (v.pos[1] - 300.0).abs() < 2.0,
        "voice at {:?}",
        v.pos
    );
}

/// The GPU packs `visual_cache` (radius, signal, energy, attack flag, birth countdown) as the
/// Rust bridge does for the same creature state.
#[test]
fn test_visual_cache_is_packed_like_the_bridge() {
    let sim = clank_app::sim::SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(120);
    let states = gpu.states();
    let mut wrong = Vec::new();
    for (slot, a) in gpu.live() {
        let want = pack_agent_state(&a, [900.0, 600.0]).visual_cache;
        let got = states[slot].visual_cache;
        if got != want {
            wrong.push(format!("slot {slot}: {got:#010x} vs bridge {want:#010x}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} creatures packed differently:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(8)].join("\n")
    );
}

/// A creature with a zero genome except the attack bias (attack = tanh(0.66) ≈ 0.58), long
/// reach (14 + 10 × 0.8 = 22px) and a full kill bonus.
fn attacker(id: u32, x: f64, y: f64) -> AgentData {
    let mut a = creature(id, x, y, 0.0);
    a.genes[output_bias(3)] = 127;
    a.tr = [0.8, 0.5, 0.5, 0.5, 0.5, 0.9];
    a.age = 10;
    a
}

/// A zero-genome creature ready to breed: energy 100, age 70, no birth countdown.
fn ready_parent(id: u32, x: f64, y: f64) -> AgentData {
    let mut a = creature(id, x, y, 0.0);
    a.energy = 100.0;
    a.age = 70;
    a
}

fn ids(agents: &[AgentData]) -> std::collections::BTreeSet<u32> {
    agents.iter().map(|a| a.id).collect()
}

/// A victim that survives the hit keeps the damage even when its own pass runs after the
/// attacker's: the victim's energy update must not overwrite the attacker's subtraction.
/// 64 attacker/victim pairs 60px apart, with the victims in the slots after all the
/// attackers and, in a second world, before them.
#[test]
fn test_damage_is_kept_when_the_victim_runs_after_the_attacker() {
    for victims_first in [false, true] {
        let mut attackers = Vec::new();
        let mut victims = Vec::new();
        for k in 0..64u32 {
            let (x, y) = (30.0 + (k % 15) as f64 * 60.0, 30.0 + (k / 15) as f64 * 60.0);
            attackers.push(attacker(1 + k, x, y));
            let mut v = creature(100 + k, x + 8.0, y, std::f64::consts::PI);
            v.energy = 90.0;
            v.age = 10;
            victims.push(v);
        }
        let agents = if victims_first {
            [victims.clone(), attackers].concat()
        } else {
            [attackers, victims.clone()].concat()
        };
        let mut sim = world_with(agents, 0.0);
        let Some(mut gpu) = upload(&sim) else {
            return;
        };
        gpu.tick();
        sim.world.evolve();
        let live = gpu.live();
        let mut wrong = Vec::new();
        for v in &victims {
            let r = reference(&sim, v.id).expect("reference victim survives");
            assert!(
                r.energy < 89.0,
                "reference victim was not hit; scenario is wrong"
            );
            match live.iter().find(|(_, a)| a.id == v.id) {
                Some((slot, g)) if (g.energy - r.energy).abs() > 2e-3 => wrong.push(format!(
                    "victim {} in slot {slot}: energy {:.3} vs reference {:.3}",
                    v.id, g.energy, r.energy
                )),
                None => wrong.push(format!("victim {} died on the GPU", v.id)),
                _ => {}
            }
        }
        let order = if victims_first {
            "victims first"
        } else {
            "attackers first"
        };
        assert!(
            wrong.is_empty(),
            "{order}: {} of 64 victims differ:\n{}",
            wrong.len(),
            wrong[..wrong.len().min(8)].join("\n")
        );
    }
}

/// At the population cap a ready parent cannot give birth and keeps its energy and its
/// readiness. With 10 ready parents and a cap of 10 nobody breeds; with a cap of 15 exactly 5
/// breed, and exactly those 5 pay.
#[test]
fn test_parents_pay_only_for_births_that_get_a_slot() {
    for cap in [10usize, 15] {
        let parents: Vec<AgentData> = (0..10)
            .map(|k| ready_parent(1 + k, 60.0 + k as f64 * 80.0, 300.0))
            .collect();
        let mut sim = world_with(parents.clone(), 0.0);
        sim.world.max_cap = cap;
        let Some(mut gpu) = upload(&sim) else {
            return;
        };
        gpu.tick();
        sim.world.evolve();
        let live = gpu.live();
        let births = live.iter().filter(|(_, a)| a.id > 10).count();
        let ref_births = sim.world.agents.iter().filter(|a| a.id > 10).count();
        assert_eq!(
            ref_births,
            cap - 10,
            "reference births at cap {cap}; scenario is wrong"
        );
        let unpaid = sim
            .world
            .agents
            .iter()
            .find(|a| a.id <= 10 && a.birth == 0)
            .map(|a| a.energy);
        let paid = sim
            .world
            .agents
            .iter()
            .find(|a| a.id <= 10 && a.birth != 0)
            .map(|a| a.energy);
        let mut wrong = Vec::new();
        for (_, g) in live.iter().filter(|(_, a)| a.id <= 10) {
            let want = if g.birth != 0 { paid } else { unpaid };
            match want {
                Some(e) if (g.energy - e).abs() < 2e-3 => {}
                _ => wrong.push(format!("parent {} (birth countdown {}): energy {:.3}, reference paid {paid:?} unpaid {unpaid:?}", g.id, g.birth, g.energy)),
            }
        }
        let charged = live
            .iter()
            .filter(|(_, a)| a.id <= 10 && a.energy < 80.0)
            .count();
        assert_eq!(
            (births, charged),
            (cap - 10, cap - 10),
            "cap {cap}: {births} births, {charged} parents charged (reference {ref_births})"
        );
        assert!(wrong.is_empty(), "cap {cap}:\n{}", wrong.join("\n"));
    }
}

/// 100 ready parents in one tick give 100 children, as in the reference: the birth pass
/// handles the whole queue, not one 64-thread workgroup.
#[test]
fn test_more_than_64_births_in_one_sub_tick_all_land() {
    let parents: Vec<AgentData> = (0..100)
        .map(|k| {
            ready_parent(
                1 + k,
                30.0 + (k % 15) as f64 * 60.0,
                30.0 + (k / 15) as f64 * 80.0,
            )
        })
        .collect();
    let mut sim = world_with(parents, 0.0);
    sim.world.max_cap = 4000;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    sim.world.evolve();
    let ref_children = sim.world.agents.iter().filter(|a| a.id > 100).count();
    let children = gpu.live().iter().filter(|(_, a)| a.id > 100).count();
    assert_eq!(ref_children, 100, "reference children; scenario is wrong");
    assert_eq!(children, 100, "children on the GPU");
}

/// 32 sub-ticks in one frame never put more live creatures than the cap: 150 ready parents
/// could make 150 children, but the cap is 200.
#[test]
fn test_the_cap_holds_across_sub_ticks() {
    let parents: Vec<AgentData> = (0..150)
        .map(|k| {
            ready_parent(
                1 + k,
                30.0 + (k % 15) as f64 * 60.0,
                30.0 + (k / 15) as f64 * 60.0,
            )
        })
        .collect();
    let mut sim = world_with(parents, 0.0);
    sim.world.max_cap = 200;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let mut counts = Vec::new();
    for frame in 0..3 {
        gpu.params.sub_tick = 0;
        gpu.params.sub_ticks_per_frame = 32;
        gpu.driver.dispatch_sub_ticks(32, &gpu.params);
        gpu.params.tick += 32;
        let live = gpu.live().len();
        gpu.params.agent_count = live as u32;
        counts.push((frame, live));
    }
    assert!(
        counts[0].1 > 150,
        "nobody was born; the test proves nothing: {counts:?}"
    );
    assert!(
        counts.iter().all(|&(_, n)| n <= 200),
        "live creatures per frame exceed the cap of 200: {counts:?}"
    );
}

/// Output genes that keep the mating lattice still and peaceful: steering and thrust
/// weights 0 (both partners move the same way), attack off, breeding wish on.
const CONTROLLED_OUTPUTS: [usize; 4] = [0, 1, 3, 5];
fn controlled(gene: usize) -> bool {
    CONTROLLED_OUTPUTS
        .iter()
        .any(|&j| (HIDDEN_GENES + j * 11..HIDDEN_GENES + j * 11 + 11).contains(&gene))
}

/// 450 pairs 10px apart (30px between pairs): a ready parent with every other gene +50 and
/// a partner that is not ready to breed (birth countdown 50) but able to mate, every other
/// gene −50, of another lineage. Each parent gives birth this tick: sexual with the
/// reference's 15% chance, asexual otherwise.
fn mating_lattice() -> SimWorld {
    let mut agents = Vec::new();
    let mut id = 1;
    for row in 0..15 {
        for col in 0..30 {
            let (x, y) = (15.0 + col as f64 * 30.0, 15.0 + row as f64 * 40.0);
            for (sign, dy) in [(50i8, 0.0), (-50, 10.0)] {
                let mut a = ready_parent(id, x, y + dy);
                for g in 0..a.genes.len() {
                    a.genes[g] = if controlled(g) { 0 } else { sign };
                }
                a.genes[output_bias(3)] = -127;
                a.genes[output_bias(5)] = 127;
                if sign < 0 {
                    a.birth = 50;
                }
                agents.push(a);
                id += 1;
            }
        }
    }
    let mut sim = world_with(agents, 0.0);
    sim.world.max_cap = 4000;
    sim
}
use clank_app::sim::SimWorld;

/// The children of one lattice tick: (sexual?, from-other-parent and mutated flags of each
/// uncontrolled gene). A gene from the +50 parent stays positive after a mutation of at
/// most ±16, one from the −50 parent negative; the birthing parent is the one whose
/// lineage the child carries (odd roots are the +50 parents).
fn lattice_children(children: &[AgentData]) -> Vec<(bool, Vec<(bool, bool)>)> {
    children
        .iter()
        .map(|c| {
            let own_sign = if c.root % 2 == 1 { 1 } else { -1 };
            let genes: Vec<(bool, bool)> = (0..c.genes.len())
                .filter(|&g| !controlled(g))
                .map(|g| {
                    (
                        c.genes[g].signum() as i32 != own_sign,
                        c.genes[g].abs() != 50,
                    )
                })
                .collect();
            (genes.iter().any(|(other, _)| *other), genes)
        })
        .collect()
}

struct LatticeStats {
    births: usize,
    sexual: usize,
    from_other: f64,
    mutated_own: f64,
    mutated_other: f64,
}

fn lattice_stats(children: &[AgentData]) -> LatticeStats {
    let kids = lattice_children(children);
    let genes: Vec<(bool, bool)> = kids
        .iter()
        .filter(|(s, _)| *s)
        .flat_map(|(_, g)| g.clone())
        .collect();
    let rate = |from_other: bool| {
        let group: Vec<_> = genes.iter().filter(|(o, _)| *o == from_other).collect();
        group.iter().filter(|(_, m)| *m).count() as f64 / group.len().max(1) as f64
    };
    LatticeStats {
        births: kids.len(),
        sexual: kids.iter().filter(|(s, _)| *s).count(),
        from_other: genes.iter().filter(|(o, _)| *o).count() as f64 / genes.len().max(1) as f64,
        mutated_own: rate(false),
        mutated_other: rate(true),
    }
}

fn run_lattice() -> Option<(LatticeStats, LatticeStats)> {
    let mut sim = mating_lattice();
    let parents = ids(&sim.world.agents);
    let mut gpu = upload(&sim)?;
    gpu.tick();
    sim.world.evolve();
    let gpu_children: Vec<AgentData> = gpu
        .live()
        .into_iter()
        .map(|(_, a)| a)
        .filter(|a| !parents.contains(&a.id))
        .collect();
    let ref_children: Vec<AgentData> = sim
        .world
        .agents
        .iter()
        .copied()
        .filter(|a| !parents.contains(&a.id))
        .collect();
    Some((lattice_stats(&gpu_children), lattice_stats(&ref_children)))
}

/// In sexual births, which parent a gene comes from is independent of whether it mutates:
/// genes from either parent mutate at the same rate (the reference draws the two separately),
/// and the other parent supplies its reference share of genes.
#[test]
fn test_crossover_and_mutation_draws_are_uncorrelated() {
    let Some((g, r)) = run_lattice() else {
        return;
    };
    assert!(
        r.sexual > 20,
        "reference had only {} sexual births; scenario is wrong",
        r.sexual
    );
    assert!(
        g.sexual > 20,
        "only {} sexual births on the GPU ({} births); nothing to measure",
        g.sexual,
        g.births
    );
    let report = format!(
        "GPU: {:.3} of genes from the other parent, mutated {:.3} (own) / {:.3} (other); reference {:.3}, {:.3} / {:.3}",
        g.from_other, g.mutated_own, g.mutated_other, r.from_other, r.mutated_own, r.mutated_other
    );
    assert!(
        (g.mutated_own - g.mutated_other).abs() < 0.02,
        "mutation depends on the parent: {report}"
    );
    assert!(
        (g.from_other - r.from_other).abs() < 0.03,
        "crossover share differs: {report}"
    );
    assert!(
        (g.mutated_own - r.mutated_own).abs() < 0.02,
        "mutation rate differs: {report}"
    );
}

/// The share of births that are sexual matches the reference: every lattice parent gives
/// birth, sexually with its partner 15% of the time.
#[test]
fn test_sexual_births_happen_at_the_reference_rate() {
    let Some((g, r)) = run_lattice() else {
        return;
    };
    assert_eq!(r.births, 450, "reference births; scenario is wrong");
    let (gf, rf) = (
        g.sexual as f64 / g.births.max(1) as f64,
        r.sexual as f64 / r.births as f64,
    );
    assert_eq!(g.births, 450, "GPU births (sexual {})", g.sexual);
    assert!(
        (gf - rf).abs() < 0.07,
        "sexual share {gf:.3} ({} of {}) vs reference {rf:.3} ({} of {})",
        g.sexual,
        g.births,
        r.sexual,
        r.births
    );
}

/// New agents get founder genomes like the reference `create_agent`: genes rounded from
/// uniform [−58, 58], traits uniform [0, 1], energy uniform [34, 52], generation 0.
#[test]
fn test_new_agents_get_reference_founder_genomes() {
    let sim = world_with(Vec::new(), 0.0);
    let Some(gpu) = upload(&sim) else {
        return;
    };
    let spots: Vec<(f32, f32)> = (0..200)
        .map(|k| (20.0 + (k % 20) as f32 * 44.0, 20.0 + (k / 20) as f32 * 58.0))
        .collect();
    gpu.driver.replenish_agents_gpu(&spots);
    let agents: Vec<AgentData> = gpu.live().into_iter().map(|(_, a)| a).collect();
    assert_eq!(agents.len(), 200, "agents seeded");
    let genes: Vec<i32> = agents
        .iter()
        .flat_map(|a| a.genes.iter().map(|&g| g as i32))
        .collect();
    let (gmin, gmax) = (*genes.iter().min().unwrap(), *genes.iter().max().unwrap());
    let gmean = genes.iter().sum::<i32>() as f64 / genes.len() as f64;
    let at_edge = genes.iter().filter(|g| g.abs() >= 55).count() as f64 / genes.len() as f64;
    let energies: Vec<f64> = agents.iter().map(|a| a.energy).collect();
    let (emin, emax) = energies
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &e| (lo.min(e), hi.max(e)));
    let mut wrong = Vec::new();
    // Uniform [−58, 58] rounded: about 6/116 of genes at |g| ≥ 55, mean near 0.
    if gmin < -58
        || gmax > 58
        || gmin > -56
        || gmax < 56
        || gmean.abs() > 1.0
        || !(0.035..0.07).contains(&at_edge)
    {
        wrong.push(format!(
            "genes span [{gmin}, {gmax}], mean {gmean:.2}, {at_edge:.3} at |g| ≥ 55"
        ));
    }
    // The reference's create_agent draws each founder trait uniformly from its own range.
    let trait_ranges = [
        (0.25, 0.8),
        (0.3, 0.8),
        (0.3, 0.8),
        (0.25, 0.75),
        (0.2, 0.8),
        (0.2, 0.8),
    ];
    for (c, &(lo, hi)) in trait_ranges.iter().enumerate() {
        let values: Vec<f64> = agents.iter().map(|a| a.tr[c]).collect();
        let (tmin, tmax) = values
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &t| (l.min(t), h.max(t)));
        let tmean = values.iter().sum::<f64>() / values.len() as f64;
        let slack = (hi - lo) * 0.04;
        if tmin < lo
            || tmax > hi
            || tmin > lo + slack
            || tmax < hi - slack
            || (tmean - (lo + hi) / 2.0).abs() > (hi - lo) * 0.06
        {
            wrong.push(format!(
                "trait {c} spans [{tmin:.3}, {tmax:.3}], mean {tmean:.3}; reference [{lo}, {hi}]"
            ));
        }
    }
    if emin < 34.0 || emax > 52.0 || emin > 36.0 || emax < 50.0 {
        wrong.push(format!("energy spans [{emin:.2}, {emax:.2}]"));
    }
    if agents.iter().any(|a| a.gen != 0 || a.age != 0) {
        wrong.push("a new agent is not generation 0 / age 0".into());
    }
    assert!(
        wrong.is_empty(),
        "new agents are not reference founders:\n{}",
        wrong.join("\n")
    );
}

/// Deposits never push a soil cell outside [0, 3], as the reference's `deposit` clamps:
/// 60 creatures dying of old age in one cell (food and taint), and 60 signalling creatures
/// sitting together for 40 ticks (scent). The dying cell matches the reference exactly.
#[test]
fn test_deposits_keep_cells_within_0_and_3() {
    let pile = |id0: u32, configure: &dyn Fn(&mut AgentData)| -> Vec<AgentData> {
        (0..60)
            .map(|k| {
                let mut a = creature(
                    id0 + k,
                    450.0 + (k % 8) as f64 * 0.5,
                    300.0 + (k / 8) as f64 * 0.5,
                    0.0,
                );
                a.genes[output_bias(1)] = -127; // barely any thrust
                configure(&mut a);
                a
            })
            .collect()
    };
    let dying = pile(1, &|a| a.age = 2101);
    let signalling = pile(1, &|a| a.genes[output_bias(4)] = 127);
    for (name, agents, ticks) in [("dying", dying, 1), ("signalling", signalling, 40)] {
        let mut sim = world_with(agents, 0.0);
        let Some(mut gpu) = upload(&sim) else {
            return;
        };
        gpu.ticks(ticks);
        for _ in 0..ticks {
            sim.world.evolve();
        }
        let soil: Vec<(f32, f32, f32)> = gpu
            .driver
            .readback_soil()
            .iter()
            .map(clank_app::gpu::soil_pipeline::GpuSoilPipeline::to_physics)
            .collect();
        let out: Vec<String> = soil
            .iter()
            .enumerate()
            .filter(|(_, (f, t, s))| [f, t, s].iter().any(|v| !(0.0..=3.0 + 1e-6).contains(*v)))
            .map(|(k, c)| format!("cell ({}, {}): {c:?}", k % 75, k / 75))
            .collect();
        let rs = &sim.world.soil;
        let hottest = (0..soil.len())
            .max_by(|&a, &b| {
                (rs.food[a] + rs.taint[a] + rs.scent[a])
                    .total_cmp(&(rs.food[b] + rs.taint[b] + rs.scent[b]))
            })
            .unwrap();
        let (rf, rt, rsc) = (rs.food[hottest], rs.taint[hottest], rs.scent[hottest]);
        let field_max = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
        let ref_max = field_max(&rs.food)
            .max(field_max(&rs.taint))
            .max(field_max(&rs.scent));
        assert!(
            ref_max >= 2.99,
            "{name}: reference never reached the clamp (highest cell {ref_max}); scenario is wrong"
        );
        assert!(
            out.is_empty(),
            "{name}: cells outside [0, 3]:\n{}",
            out[..out.len().min(8)].join("\n")
        );
        if name == "dying" {
            let (f, t, s) = soil[hottest];
            let gpu_hottest = (0..soil.len())
                .max_by(|&a, &b| (soil[a].0 + soil[a].1).total_cmp(&(soil[b].0 + soil[b].1)))
                .unwrap();
            assert!(
                (f - rf).abs() < 1e-3 && (t - rt).abs() < 1e-3 && (s - rsc).abs() < 1e-3,
                "dying cell ({}, {}): GPU ({f:.4}, {t:.4}, {s:.4}) vs reference ({rf:.4}, {rt:.4}, {rsc:.4}); {} of 60 alive on the GPU, {} in the reference; GPU's fullest cell ({}, {}) is {:?}",
                hottest % 75,
                hottest / 75,
                gpu.live().len(),
                sim.world.agents.len(),
                gpu_hottest % 75,
                gpu_hottest / 75,
                soil[gpu_hottest]
            );
        }
    }
}

/// A killed creature's corpse food lands where the reference's does. The victim starts 1px
/// short of a cell edge moving 3px across it, in the slot after its attacker: the reference
/// victim dies at the top of its own turn, before moving or eating. Every soil cell matches
/// the reference but one: the GPU victim runs at the same time as its attacker, so it may move
/// into the next cell and eat there (at most one meal, under 0.03 food) before the blow lands.
/// That one meal is a known difference of the parallel pass.
#[test]
fn test_corpse_food_lands_where_the_victim_died() {
    let a = attacker(1, 400.0, 300.0);
    let mut victim = creature(2, 407.0, 300.0, 0.0);
    victim.energy = 0.4;
    victim.age = 10;
    victim.vx = 3.0;
    let mut sim = world_with(vec![a, victim], 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(2);
    sim.world.evolve();
    sim.world.evolve();
    assert!(
        reference(&sim, 2).is_none(),
        "reference victim survived; scenario is wrong"
    );
    assert!(
        gpu.live().iter().all(|(_, a)| a.id != 2),
        "victim still alive on the GPU"
    );
    let soil: Vec<(f32, f32, f32)> = gpu
        .driver
        .readback_soil()
        .iter()
        .map(clank_app::gpu::soil_pipeline::GpuSoilPipeline::to_physics)
        .collect();
    let rs = &sim.world.soil;
    // The cell the victim moves into on the GPU (x ≈ 409.7 of 12px cells, row 25).
    let meal_cell = 25 * 75 + 34;
    let food_ok = |k: usize| {
        let eaten = rs.food[k] - soil[k].0;
        if k == meal_cell {
            (-2e-3..0.03).contains(&eaten)
        } else {
            eaten.abs() <= 2e-3
        }
    };
    let wrong: Vec<String> = (0..soil.len())
        .filter(|&k| !food_ok(k) || (soil[k].1 - rs.taint[k]).abs() > 2e-3)
        .map(|k| {
            format!(
                "cell ({}, {}): GPU food {:.3} taint {:.3} vs reference {:.3} {:.3}",
                k % 75,
                k / 75,
                soil[k].0,
                soil[k].1,
                rs.food[k],
                rs.taint[k]
            )
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{} cells differ:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(12)].join("\n")
    );
}

/// The census runs after the birth pass: population, highest generation and living lineages
/// describe the creatures alive at the end of the tick — this tick's newborns included, this
/// tick's dead excluded. 40 ready parents of 13 full-width lineages give birth; 5 starving
/// creatures of their own lineages and generation 50 die.
#[test]
fn test_the_census_counts_newborns_and_not_the_dead() {
    let mut agents: Vec<AgentData> = (0..40)
        .map(|k| {
            let mut a = ready_parent(
                1 + k,
                30.0 + (k % 10) as f64 * 85.0,
                40.0 + (k / 10) as f64 * 70.0,
            );
            a.root = 5000 + (k % 13) * 300;
            a.gen = k % 5;
            a
        })
        .collect();
    for k in 0..5u32 {
        let mut a = creature(100 + k, 60.0 + k as f64 * 160.0, 500.0, 0.0);
        a.energy = 0.001;
        a.age = 10;
        a.root = 9000 + k;
        a.gen = 50;
        agents.push(a);
    }
    let mut sim = world_with(agents, 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let t = gpu.tick();
    sim.world.evolve();
    let census = |live: &[AgentData]| {
        (
            live.len() as u32,
            live.iter().map(|a| a.gen).max().unwrap_or(0),
            distinct_roots(live) as u32,
        )
    };
    let reference = census(&sim.world.agents);
    assert!(
        reference.0 > 40 && reference.1 < 50,
        "reference had no births or kept the starving: {reference:?}; scenario is wrong"
    );
    let gpu_live: Vec<AgentData> = gpu.live().into_iter().map(|(_, a)| a).collect();
    assert_eq!(
        census(&gpu_live),
        reference,
        "GPU creatures (count, max generation, lineages) differ from the reference"
    );
    assert_eq!(
        (t.population, t.max_generation, t.living_lineages),
        reference,
        "GPU census (population, max_generation, living_lineages) does not describe its creatures"
    );
}

/// The lineage census holds as many distinct lineages as there can be living creatures: 6000
/// founders, each its own lineage, on 8192 slots (the app has 65536). A table of fixed size
/// overflows here — it stops counting at its size, and every creature past that probes the
/// whole table, which made a speed-1 tick of 16000 founders cost 47 ms against 17.5 ms.
#[test]
fn test_the_census_counts_every_lineage_in_a_large_world() {
    let agents: Vec<AgentData> = (0..6000u32)
        .map(|k| {
            creature(
                1 + k,
                4.5 + (k % 100) as f64 * 9.0,
                5.0 + (k / 100) as f64 * 10.0,
                0.0,
            )
        })
        .collect();
    let mut sim = world_with(agents, 0.0);
    sim.world.max_cap = 6000;
    let (cols, rows) = (sim.world.soil.cols as u32, sim.world.soil.rows as u32);
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 8192) else {
        return;
    };
    let (states, genomes, atomics, soil, mut params) = sync_rust_to_gpu(&sim);
    params.max_agents = driver.max_agents;
    params.tool_type = NO_TOOL;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    let mut gpu = Gpu { driver, params };
    let t = gpu.tick();
    let live: Vec<AgentData> = gpu.live().into_iter().map(|(_, a)| a).collect();
    let lineages = distinct_roots(&live) as u32;
    assert!(
        lineages > 4096,
        "only {lineages} lineages survived; scenario is wrong"
    );
    assert_eq!(
        (t.population, t.living_lineages),
        (live.len() as u32, lineages),
        "GPU census (population, living_lineages) does not describe its creatures"
    );
}
