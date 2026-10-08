//! Plan Task 2: tombstone freelist, pass-decoupled push/pop, CAS death ownership — on the
//! real shaders.

#[path = "support/gpu_harness.rs"]
mod gpu_harness;

use clank_app::sim::SimWorld;
use gpu_harness::*;
use std::collections::BTreeSet;

/// Gene index of the attack output's bias (10 hidden neurons × 26 genes, then 11 per output).
const OUTPUT_BIAS_ATTACK: usize = 10 * (15 + 10 + 1) + 3 * 11 + 10;

/// Slot accounting holds across a long run: every slot under the cap is either alive or on
/// the freelist exactly once, and the freelist holds only dead slots.
#[test]
fn test_every_slot_is_alive_or_free_exactly_once() {
    let sim = SimWorld::new(42);
    let cap = sim.world.max_cap;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    for checkpoint in 0..6 {
        let t = gpu.ticks(100);
        let states = gpu.states();
        let top = t.freelist_top as usize;
        let free = gpu.freelist(top.min(gpu.driver.max_agents as usize));
        let live: BTreeSet<usize> = (0..cap)
            .filter(|&i| states[i].meta_flags & DEAD == 0 && states[i].id != 0)
            .collect();
        let free_set: BTreeSet<u32> = free.iter().copied().collect();
        assert_eq!(
            free_set.len(),
            free.len(),
            "checkpoint {checkpoint}: a slot is on the freelist twice"
        );
        let live_on_freelist: Vec<_> = free
            .iter()
            .filter(|&&s| live.contains(&(s as usize)))
            .collect();
        assert!(
            live_on_freelist.is_empty(),
            "checkpoint {checkpoint}: live slots on the freelist: {live_on_freelist:?}"
        );
        assert_eq!(
            live.len() + top,
            cap,
            "checkpoint {checkpoint}: {} live + {top} free != cap {cap}",
            live.len()
        );
    }
}

/// Each death is counted once: over a long window, the creatures that disappear equal the
/// increase in `total_deaths` (a victim may vanish one tick after its kill is counted, so
/// the comparison is cumulative, allowing the final tick's kills to be still pending).
#[test]
fn test_each_death_is_counted_exactly_once() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(300); // a full, fighting population
    let start = gpu.driver.readback_telemetry().total_deaths;
    let (mut died, mut last_kills) = (0u32, 0u32);
    for _ in 0..800 {
        let before: BTreeSet<u32> = gpu.live().iter().map(|(_, a)| a.id).collect();
        let t = gpu.tick();
        let after: BTreeSet<u32> = gpu.live().iter().map(|(_, a)| a.id).collect();
        died += before.difference(&after).count() as u32;
        last_kills = t.kills;
    }
    let counted = gpu.driver.readback_telemetry().total_deaths - start;
    assert!(died > 20, "only {died} deaths in 800 ticks");
    assert!(
        counted >= died && counted <= died + last_kills,
        "{died} creatures died but total_deaths rose by {counted}"
    );
}

/// Slots freed by deaths are reused by later births instead of growing the slot range.
#[test]
fn test_dead_slots_are_recycled_by_births() {
    let sim = SimWorld::new(42);
    let cap = sim.world.max_cap;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let first: Vec<u32> = gpu.states()[..cap].iter().map(|s| s.id).collect();
    gpu.ticks(600);
    let states = gpu.states();
    let reused = (0..cap)
        .filter(|&i| first[i] != 0 && states[i].id != first[i] && states[i].meta_flags & DEAD == 0)
        .count();
    let beyond_cap = states[cap..]
        .iter()
        .filter(|s| s.meta_flags & DEAD == 0 && s.id != 0)
        .count();
    assert!(
        reused > 0,
        "no founder slot was reused by a newborn in 600 ticks"
    );
    assert_eq!(beyond_cap, 0, "live creatures in slots at or above the cap");
}

/// A creature killed or starved is tombstoned in its own slot within a tick of dying:
/// dead bit set and `visual_cache` cleared, so renderers and spatial passes skip it.
#[test]
fn test_dead_creatures_are_tombstoned() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(300);
    let mut stale = Vec::new();
    for _ in 0..100 {
        gpu.tick();
        let states = gpu.states();
        let atomics = gpu.atomics();
        for (i, (s, a)) in states.iter().zip(&atomics).enumerate() {
            let claimed = a.dead_claimed != 0;
            let dead = s.meta_flags & DEAD != 0;
            if claimed && !dead {
                stale.push(i);
            }
            if dead && s.visual_cache != 0 {
                stale.push(i);
            }
        }
        // One tick of lag is allowed: a victim learns of its death on its next pass.
        gpu.tick();
        let states = gpu.states();
        let atomics = gpu.atomics();
        stale.retain(|&i| {
            let dead = states[i].meta_flags & DEAD != 0;
            (atomics[i].dead_claimed != 0 && !dead) || (dead && states[i].visual_cache != 0)
        });
        assert!(
            stale.is_empty(),
            "slots claimed dead but still drawn after a tick: {stale:?}"
        );
    }
}

/// A slot freed this sub-tick is not reused until the next one: 48 parents each give birth
/// in the pass where they die of old age, with plenty of older free slots. Pushing deaths
/// straight onto the stack hands the dying parents' slots to the newborns, so a newborn
/// can overwrite a parent another newborn is still copying. Each lineage must get exactly
/// one child.
#[test]
fn test_a_slot_freed_this_sub_tick_is_not_reused_until_the_next() {
    let parents: Vec<_> = (0..48u32)
        .map(|k| {
            let (x, y) = (30.0 + (k % 15) as f64 * 60.0, 30.0 + (k / 15) as f64 * 60.0);
            let mut a = creature(1 + k, x, y, 0.0);
            a.energy = 100.0;
            a.age = 2100;
            a
        })
        .collect();
    let mut sim = world_with(parents, 0.0);
    sim.world.max_cap = 1000;
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    let live = gpu.live();
    assert!(
        live.iter().all(|(_, a)| a.id > 48),
        "a parent outlived age 2100"
    );
    let mut roots: Vec<u32> = live.iter().map(|(_, a)| a.root).collect();
    roots.sort_unstable();
    assert_eq!(
        roots,
        (1..=48).collect::<Vec<u32>>(),
        "children's lineages (each parent had one child)"
    );
}

/// Changing the cap rebuilds the freelist without losing slots. A creature killed by a
/// thread that runs after its own pass is claimed and freed at once, but sets its own dead
/// bit only on its next pass; a cap change in between must not drop its slot. 64 weak
/// victims in the first slots, each beside an attacker in the slots after them; the cap
/// changes right after the kills.
#[test]
fn test_changing_the_cap_loses_no_slots() {
    let mut victims = Vec::new();
    let mut attackers = Vec::new();
    for k in 0..64u32 {
        let (x, y) = (30.0 + (k % 15) as f64 * 60.0, 30.0 + (k / 15) as f64 * 60.0);
        let mut v = creature(1 + k, x + 8.0, y, std::f64::consts::PI);
        v.energy = 1.0;
        v.age = 10;
        victims.push(v);
        let mut a = creature(100 + k, x, y, 0.0);
        a.genes[OUTPUT_BIAS_ATTACK] = 127;
        a.tr = [0.8, 0.5, 0.5, 0.5, 0.5, 0.9];
        a.age = 10;
        attackers.push(a);
    }
    let sim = world_with([victims, attackers].concat(), 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let kills = gpu.tick().kills;
    gpu.params.max_capacity += 1;
    gpu.tick();
    let t = gpu.tick();
    let cap = gpu.params.max_capacity as usize;
    let states = gpu.states();
    let live = (0..cap)
        .filter(|&i| states[i].meta_flags & DEAD == 0 && states[i].id != 0)
        .count();
    assert!(kills > 20, "only {kills} kills; scenario is wrong");
    assert_eq!(
        live + t.freelist_top as usize,
        cap,
        "{live} live + {} free != cap {cap} after {kills} kills",
        t.freelist_top
    );
}
