//! Plan Task 4: Morton codes, the 8-byte (morton, slot) radix sort with dead slots sorted to
//! the tail, the Karras LBVH over sorted keys, picking, and the AoE tools — on the real
//! shaders. Picking is only dispatched once the tree is proven valid (see the harness).

#[path = "support/gpu_harness.rs"]
mod gpu_harness;

use clank_app::gpu::spatial_index::compute_morton_32_with_size;
use clank_app::sim::SimWorld;
use gpu_harness::*;

const WORLD: [f32; 2] = [900.0, 600.0];

/// Every live creature carries the Morton code of the position it was indexed at (its
/// position at the start of the tick, before it moved).
#[test]
fn test_live_creatures_carry_the_morton_code_of_their_position() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(20);
    let before = gpu.states();
    gpu.tick();
    let after = gpu.states();
    let wrong: Vec<String> = (0..before.len())
        .filter(|&i| before[i].meta_flags & DEAD == 0 && after[i].meta_flags & DEAD == 0)
        .filter(|&i| before[i].id == after[i].id && before[i].id != 0)
        .filter_map(|i| {
            let p = [before[i].pos_vel[0], before[i].pos_vel[1]];
            let want = compute_morton_32_with_size(p, WORLD);
            (after[i].morton_code != want)
                .then(|| format!("slot {i}: {:#x} vs {want:#x}", after[i].morton_code))
        })
        .collect();
    assert!(wrong.is_empty(), "wrong Morton codes: {wrong:?}");
}

/// The 8-byte indirection sort: after indexing, `spatial_keys` holds `(morton, slot)` pairs in
/// ascending key order, every live creature exactly once, dead slots with the `0xFFFFFFFF`
/// sentinel at the tail (plan §9 "Dead Agent Partitioning").
#[test]
fn test_spatial_keys_are_sorted_morton_slot_pairs_with_dead_slots_last() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(20);
    let before = gpu.states();
    gpu.tick();
    let n = gpu.driver.max_agents as usize;
    let keys: Vec<[u32; 2]> = read_buffer(&gpu.driver, &gpu.driver.spatial_keys_buf, n);
    let live: Vec<usize> = (0..n)
        .filter(|&i| before[i].meta_flags & DEAD == 0 && before[i].id != 0)
        .collect();
    let unsorted = keys.windows(2).filter(|w| w[0] > w[1]).count();
    assert_eq!(
        unsorted, 0,
        "{unsorted} adjacent key pairs are out of order"
    );
    for (rank, &slot) in live.iter().enumerate() {
        let _ = rank;
        let p = [before[slot].pos_vel[0], before[slot].pos_vel[1]];
        let want = [compute_morton_32_with_size(p, WORLD), slot as u32];
        assert!(
            keys[..live.len()].contains(&want),
            "live slot {slot} has no key {want:?}"
        );
    }
    assert!(
        keys[live.len()..].iter().all(|k| k[0] == u32::MAX),
        "dead slots are not sentinel keys at the tail"
    );
}

/// The sort covers `params.max_agents` slots even when the driver has more (the driver
/// dispatches for all of its slots): with params for the cap's 340 slots on a 65,536-slot
/// driver (the app's size), the first 340 keys are still sorted, every live creature once.
#[test]
fn test_the_sort_holds_when_params_cover_fewer_slots_than_the_driver() {
    use clank_app::gpu::compute_driver::GpuComputeDriver;
    let sim = SimWorld::new(42);
    let (cols, rows) = (sim.world.soil.cols as u32, sim.world.soil.rows as u32);
    let Some(driver) = GpuComputeDriver::create_for_world(cols, rows, 65536) else {
        return;
    };
    let (states, genomes, atomics, soil, mut params) =
        clank_app::gpu::bridge::sync_rust_to_gpu(&sim);
    let n = sim.world.max_cap;
    params.max_agents = n as u32;
    params.sub_ticks_per_frame = 1;
    params.tool_type = NO_TOOL;
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    let mut gpu = Gpu { driver, params };
    gpu.ticks(20);
    let before = gpu.states();
    gpu.tick();
    let keys: Vec<[u32; 2]> = read_buffer(&gpu.driver, &gpu.driver.spatial_keys_buf, n);
    let live: Vec<u32> = (0..n)
        .filter(|&i| before[i].meta_flags & DEAD == 0 && before[i].id != 0)
        .map(|i| i as u32)
        .collect();
    let unsorted = keys.windows(2).filter(|w| w[0] > w[1]).count();
    let mut listed: Vec<u32> = keys[..live.len()].iter().map(|k| k[1]).collect();
    listed.sort_unstable();
    assert_eq!(
        unsorted, 0,
        "{unsorted} adjacent key pairs are out of order"
    );
    assert_eq!(listed, live, "the live keys are not the live creatures");
}

/// The grid cell table built from the sorted keys holds every live creature exactly once,
/// under the 100px cell of its position, with no per-cell cap: a cell holding 300 creatures
/// lists all 300. `spatial_keys` is [sorted keys][54 (start, count) cells][cell_slots].
#[test]
fn test_the_grid_table_holds_every_live_creature_once() {
    let crowd: Vec<_> = (0..300u32)
        .map(|k| {
            creature(
                1 + k,
                410.0 + (k % 20) as f64 * 4.0,
                310.0 + (k / 20) as f64 * 4.0,
                0.0,
            )
        })
        .collect();
    let mut agents = crowd;
    agents.extend((0..60u32).map(|k| creature(400 + k, 15.0 * k as f64, 7.0 * k as f64, 0.0)));
    let sim = world_with(agents, 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    // Rebuilt every sub-tick: nothing from earlier ticks may carry over.
    gpu.ticks(3);
    let before = gpu.states();
    gpu.tick();
    let n = gpu.driver.max_agents as usize;
    let words: Vec<[u32; 2]> = read_buffer(&gpu.driver, &gpu.driver.spatial_keys_buf, 2 * n + 54);
    let cell_of = |slot: usize| {
        let (x, y) = (before[slot].pos_vel[0], before[slot].pos_vel[1]);
        let gx = ((x.max(0.0) / 100.0) as usize).min(8);
        let gy = ((y.max(0.0) / 100.0) as usize).min(5);
        gy * 9 + gx
    };
    let mut seen = vec![0u32; n];
    let mut misplaced = Vec::new();
    for c in 0..54 {
        let [start, count] = words[n + c];
        if start == u32::MAX {
            continue;
        }
        for k in 0..count as usize {
            let slot = words[n + 54 + start as usize + k][0] as usize;
            seen[slot] += 1;
            if cell_of(slot) != c {
                misplaced.push(format!(
                    "slot {slot} listed in cell {c}, lives in {}",
                    cell_of(slot)
                ));
            }
        }
    }
    let live: Vec<usize> = (0..n)
        .filter(|&i| before[i].meta_flags & DEAD == 0 && before[i].id != 0)
        .collect();
    let wrong_count = live.iter().filter(|&&i| seen[i] != 1).count();
    let listed_dead = (0..n)
        .filter(|&i| seen[i] > 0 && !live.contains(&i))
        .count();
    let crowded = words[n + cell_of(0)][1];
    assert!(misplaced.is_empty(), "{misplaced:?}");
    assert_eq!(
        wrong_count, 0,
        "{wrong_count} live creatures not listed exactly once"
    );
    assert_eq!(listed_dead, 0, "{listed_dead} dead slots listed");
    assert_eq!(
        crowded, 300,
        "the crowded cell lists {crowded} of 300 creatures"
    );
}

/// The wrap `p - w * floor(p / w)` can land exactly on the far edge, and a creature exactly on
/// the far corner has the all-ones Morton code. It is still live: it must not sort among the
/// dead slots and drop out of the live count and the grid table.
#[test]
fn test_a_creature_on_the_far_corner_stays_in_the_index() {
    let agents = vec![
        creature(1, 100.0, 100.0, 0.0),
        creature(2, WORLD[0] as f64, WORLD[1] as f64, 0.0),
        creature(3, 300.0, 200.0, 0.0),
    ];
    let sim = world_with(agents, 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let telemetry = gpu.tick();
    let n = gpu.driver.max_agents as usize;
    let words: Vec<[u32; 2]> = read_buffer(&gpu.driver, &gpu.driver.spatial_keys_buf, 2 * n + 54);
    let listed: Vec<u32> = (0..54)
        .filter(|&c| words[n + c][0] != u32::MAX)
        .flat_map(|c| {
            let [start, count] = words[n + c];
            (0..count).map(move |k| (start + k) as usize)
        })
        .map(|at| words[n + 54 + at][0])
        .collect();
    assert_eq!(telemetry.live_count, 3, "live count");
    assert!(
        listed.contains(&1),
        "the corner creature (slot 1) is missing from the grid table: {listed:?}"
    );
}

/// The Karras LBVH is a tree over exactly the live creatures: from the root every node is
/// reached once and every live creature is one leaf.
#[test]
fn test_lbvh_is_a_valid_tree_over_the_live_creatures() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let mut bad = Vec::new();
    for tick in 0..30 {
        gpu.tick();
        let problems = gpu.lbvh_problems();
        if !problems.is_empty() {
            bad.push(format!("tick {tick}: {}", problems.join("; ")));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of 30 trees are invalid:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// Leaves appear in ascending Morton order (Karras construction over sorted keys).
#[test]
fn test_lbvh_leaves_are_in_morton_order() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(5);
    let states = gpu.states();
    let n = gpu.live().len();
    let nodes = gpu.lbvh_nodes(2 * n - 1);
    let codes: Vec<u32> = nodes[n - 1..]
        .iter()
        .map(|leaf| {
            states
                .get(leaf.leaf_idx as usize)
                .map(|s| s.morton_code)
                .unwrap_or(u32::MAX)
        })
        .collect();
    let unsorted = codes.windows(2).filter(|w| w[0] > w[1]).count();
    assert_eq!(
        unsorted,
        0,
        "{unsorted} of {} leaf pairs out of Morton order",
        n - 1
    );
}

/// Every node's box contains its children's boxes and each leaf's box its creature.
#[test]
fn test_lbvh_boxes_contain_their_children() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.ticks(5);
    let problems = gpu.lbvh_problems();
    assert!(
        problems.is_empty(),
        "not a tree, boxes undefined: {problems:?}"
    );
    let n = gpu.live().len();
    let nodes = gpu.lbvh_nodes(2 * n - 1);
    let contains = |o: &clank_app::gpu::types::GpuLbvhNode,
                    i: &clank_app::gpu::types::GpuLbvhNode| {
        o.aabb_min[0] <= i.aabb_min[0]
            && o.aabb_min[1] <= i.aabb_min[1]
            && o.aabb_max[0] >= i.aabb_max[0]
            && o.aabb_max[1] >= i.aabb_max[1]
    };
    let bad = nodes
        .iter()
        .filter(|nd| nd.leaf_idx == u32::MAX)
        .filter(|nd| {
            !contains(nd, &nodes[nd.left_child as usize])
                || !contains(nd, &nodes[nd.right_child as usize])
        })
        .count();
    assert_eq!(bad, 0, "{bad} internal boxes do not contain their children");
}

/// With a single creature the tree is that one leaf (degenerate N <= 1 guard).
#[test]
fn test_lbvh_of_one_creature_is_its_leaf() {
    let sim = world_with(vec![creature(1, 300.0, 200.0, 0.0)], 0.0);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    gpu.tick();
    let root = gpu.lbvh_nodes(1)[0];
    assert_eq!(
        root.leaf_idx, 0,
        "root of a one-creature tree is not its leaf"
    );
}

/// Picking returns the live creature under the cursor: aimed at a creature with no other
/// creature within the pick radius, that creature's slot and id come back.
#[test]
fn test_picking_returns_the_creature_under_the_cursor() {
    let sim = SimWorld::new(42);
    let Some(mut gpu) = upload(&sim) else {
        return;
    };
    let target = std::cell::Cell::new((u32::MAX, 0u32));
    let result = gpu.pick_if_tree_is_valid(|live| {
        let isolated = live
            .iter()
            .find(|(_, a)| {
                live.iter().all(|(_, b)| {
                    b.id == a.id || torus_dist(a.x, a.y, b.x, b.y, 900.0, 600.0) > 60.0
                })
            })
            .expect("an isolated creature");
        target.set((isolated.0 as u32, isolated.1.id));
        [isolated.1.x as f32, isolated.1.y as f32]
    });
    let t = result.unwrap_or_else(|why| panic!("{why}"));
    let (slot, id) = target.get();
    assert_eq!(
        (t.selected_agent_idx, t.selected_agent_id),
        (slot, id),
        "picked the wrong creature"
    );
}

/// The Extinguish tool kills the creatures the reference `extinguish_at` kills: within 23px
/// of the cursor by plain (non-toroidal) distance, as clankolution.html's erase tool does.
#[test]
fn test_extinguish_tool_kills_what_the_reference_kills() {
    for (x, y) in [(452.0, 298.0), (4.0, 300.0), (896.0, 2.0)] {
        let mut sim = SimWorld::new(42);
        // Creatures on both sides of the seam around the cursor.
        let mut id = 1000;
        for (dx, dy) in [
            (-15.0, 0.0),
            (10.0, 5.0),
            (0.0, 20.0),
            (-22.0, -3.0),
            (30.0, 0.0),
        ] {
            let mut c = creature(id, (x + dx + 900.0) % 900.0, (y + dy + 600.0) % 600.0, 0.0);
            c.root = 1;
            sim.world.agents.push(c);
            id += 1;
        }
        sim.world.next_id = id;
        sim.world.sync_pos_cache();
        let Some(mut gpu) = upload(&sim) else {
            return;
        };
        gpu.params.tool_type = 3;
        gpu.params.tool_pos = [x as f32, y as f32];
        gpu.params.tool_radius = 23.0;
        let before: std::collections::BTreeSet<u32> =
            gpu.live().iter().map(|(_, a)| a.id).collect();
        gpu.tick();
        let after: std::collections::BTreeSet<u32> = gpu.live().iter().map(|(_, a)| a.id).collect();
        let gpu_killed: Vec<u32> = before.difference(&after).copied().collect();
        let alive_before: Vec<u32> = sim.world.agents.iter().map(|a| a.id).collect();
        sim.world.extinguish_at(x, y, 23.0);
        let alive: std::collections::BTreeSet<u32> =
            sim.world.agents.iter().map(|a| a.id).collect();
        let ref_killed: Vec<u32> = alive_before
            .into_iter()
            .filter(|i| !alive.contains(i))
            .collect();
        // Other deaths that tick (starvation) are not the tool's; compare the tool's targets.
        let gpu_tool: Vec<u32> = gpu_killed.iter().copied().filter(|i| *i >= 1000).collect();
        assert_eq!(
            gpu_tool, ref_killed,
            "cursor ({x}, {y}): GPU killed {gpu_tool:?}, reference {ref_killed:?}"
        );
    }
}

/// The Seed Life tool adds two creatures within ±11px of the cursor, as the reference
/// `seed_life_at` does (while under the cap).
#[test]
fn test_seed_tool_adds_two_creatures_at_the_cursor() {
    let sim = SimWorld::new(42);
    let Some(gpu) = upload(&sim) else {
        return;
    };
    let before = gpu.live().len();
    let (x, y) = (450.0f32, 300.0f32);
    gpu.driver.seed_agents_gpu(&[(x, y)]);
    let live = gpu.live();
    let new: Vec<_> = live
        .iter()
        .filter(|(_, a)| {
            a.age == 0
                && a.gen == 0
                && (a.x as f32 - x).abs() <= 11.0
                && (a.y as f32 - y).abs() <= 11.0
        })
        .collect();
    assert_eq!(
        live.len() - before,
        2,
        "seeding added {} creatures",
        live.len() - before
    );
    assert_eq!(
        new.len(),
        2,
        "seeded creatures not within 11px of the cursor"
    );
}

/// Senses copied into hidden neurons 0..5 by `observer`'s genome: (input index, name).
const SENSES: [(usize, &str); 5] = [
    (7, "sin bearing"),
    (8, "cos bearing"),
    (9, "closeness"),
    (14, "neighbour energy"),
    (10, "density"),
];

/// Creature 1 wired so each of its first five hidden neurons copies one sense: weight 127
/// from that input and nothing else, so after one tick `h[j] = tanh(0.61 × input)` and the
/// hidden state reads back what it sensed. Every other output is 0 (no attack, no turn).
fn observer(x: f64, y: f64) -> clank_core::agent::AgentData {
    let mut a = creature(1, x, y, 0.3);
    for (j, (input, _)) in SENSES.iter().enumerate() {
        a.genes[j * 26 + input] = 127;
    }
    a.age = 10;
    a
}

fn sensed(a: &clank_core::agent::AgentData) -> [f64; 5] {
    std::array::from_fn(|j| (a.h[j] as f64).atanh() / clank_core::world::HIDDEN_GAIN)
}

/// One tick of `observer` among `others` on both engines; returns what each engine's
/// observer sensed, or `None` without a GPU. The observer is first in the reference's update
/// order, so every other creature is still where it started when it looks around, as on the
/// GPU. `dead_slots` dead creatures sit between the observer and the others on the GPU only,
/// putting the others in high slots.
fn senses_after_one_tick(
    observer: clank_core::agent::AgentData,
    others: &[clank_core::agent::AgentData],
    dead_slots: u32,
) -> Option<Result<(), String>> {
    let mut reference = world_with([&[observer][..], others].concat(), 0.0);
    let mut gpu_world = vec![observer];
    for k in 0..dead_slots {
        let mut d = creature(9000 + k, 880.0, 580.0, 0.0);
        d.dead = 1;
        d.energy = 0.0;
        gpu_world.push(d);
    }
    gpu_world.extend_from_slice(others);
    let mut gpu = upload(&world_with(gpu_world, 0.0))?;
    reference.world.build_grid(); // evolve rebuilds it too; built here for the near() probe
    let near = reference.world.near(0);
    let picked = near.best_idx.map(|i| reference.world.agents[i].id);
    gpu.tick();
    reference.world.evolve();
    let r = reference
        .world
        .agents
        .iter()
        .find(|a| a.id == 1)
        .expect("reference observer");
    let live = gpu.live();
    let Some((_, g)) = live.iter().find(|(_, a)| a.id == 1) else {
        return Some(Err("observer died on the GPU".into()));
    };
    let (gs, rs) = (sensed(g), sensed(r));
    let wrong: Vec<String> = SENSES
        .iter()
        .enumerate()
        .filter(|(j, _)| (gs[*j] - rs[*j]).abs() > 1e-4)
        .map(|(j, (_, name))| format!("{name} {:.4} vs {:.4}", gs[j], rs[j]))
        .collect();
    Some(if wrong.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} (reference nearest: creature {picked:?} at {:.1}px, density {})",
            wrong.join("; "),
            near.best_d,
            near.density
        ))
    })
}

/// The agent pass finds the nearest neighbour the reference `near()` finds: the 3×3 cells
/// around the creature first, then a scan of every live creature when nothing is within
/// 100px — whatever slot it is in, however crowded its cell.
#[test]
fn test_nearest_neighbour_matches_near() {
    let at = |id: u32, x: f64, y: f64, energy: f64| {
        let mut c = creature(id, x, y, 0.0);
        c.energy = energy;
        c.age = 10;
        c
    };
    // 150 creatures in the observer's cell, 40px away, plus one 30px away.
    let ring: Vec<_> = (0..150)
        .map(|k| {
            let t = k as f64 / 150.0 * std::f64::consts::TAU;
            at(10 + k, 450.0 + 40.0 * t.cos(), 350.0 + 40.0 * t.sin(), 50.0)
        })
        .collect();
    let closest = at(500, 480.0, 350.0, 90.0);
    // 196 creatures in the top-left of the cell [400, 500) × [300, 400), all at least 31px from
    // an observer at (480, 380), and the nearest (21px) in the cell's bottom-right corner: its
    // Morton code is the cell's highest, so it comes last in the cell's list of creatures.
    let crowd: Vec<_> = (0..196)
        .map(|k| {
            at(
                600 + k,
                402.0 + (k % 14) as f64 * 4.0,
                302.0 + (k / 14) as f64 * 4.0,
                50.0,
            )
        })
        .collect();
    let corner = at(900, 495.0, 395.0, 90.0);
    let scenarios: Vec<(&str, clank_core::agent::AgentData, Vec<_>, u32)> = vec![
        (
            "in the next cell",
            observer(395.0, 300.0),
            vec![at(2, 455.0, 320.0, 80.0)],
            0,
        ),
        (
            "beyond 100px",
            observer(450.0, 300.0),
            vec![at(2, 630.0, 340.0, 80.0)],
            0,
        ),
        (
            "beyond 100px across the seam",
            observer(30.0, 300.0),
            vec![at(2, 780.0, 330.0, 80.0)],
            0,
        ),
        (
            "beyond 100px in a high slot",
            observer(450.0, 300.0),
            vec![at(2, 630.0, 340.0, 80.0)],
            40,
        ),
        (
            "nearer one outside the 3×3 cells than the one inside",
            observer(310.0, 210.0),
            vec![at(2, 440.0, 330.0, 20.0), at(3, 310.0, 95.0, 80.0)],
            0,
        ),
        (
            "151 in the cell, nearest first",
            observer(450.0, 350.0),
            [vec![closest], ring.clone()].concat(),
            0,
        ),
        (
            "151 in the cell, nearest last",
            observer(450.0, 350.0),
            [ring, vec![closest]].concat(),
            0,
        ),
        (
            "196 in the cell, nearest with the cell's highest Morton code",
            observer(480.0, 380.0),
            [crowd, vec![corner]].concat(),
            0,
        ),
    ];
    let count = scenarios.len();
    let mut failures = Vec::new();
    for (name, obs, others, dead_slots) in scenarios {
        match senses_after_one_tick(obs, &others, dead_slots) {
            None => return,
            Some(Err(why)) => failures.push(format!("{name}: {why}")),
            Some(Ok(())) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {count} scenarios sense a different neighbour:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Density counts the other creatures strictly within 100px (the reference's `d² < 10000`),
/// across the seam too; one exactly 100px away is not counted.
#[test]
fn test_density_counts_neighbours_within_100px() {
    let at = |id: u32, x: f64, y: f64| {
        let mut c = creature(id, x, y, 0.0);
        c.age = 10;
        c
    };
    let scenarios = [
        (
            "middle of the world",
            observer(450.0, 350.0),
            vec![
                at(2, 500.0, 350.0),
                at(3, 450.0, 250.5),
                at(4, 350.0, 350.0),
                at(5, 450.0, 450.5),
                at(6, 556.0, 456.0),
            ],
        ),
        (
            "across the seam",
            observer(40.0, 350.0),
            vec![at(2, 870.0, 350.0), at(3, 40.0, 449.0), at(4, 140.0, 350.0)],
        ),
    ];
    let mut failures = Vec::new();
    for (name, obs, others) in scenarios {
        match senses_after_one_tick(obs, &others, 0) {
            None => return,
            Some(Err(why)) => failures.push(format!("{name}: {why}")),
            Some(Ok(())) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of 2 scenarios count density differently:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
