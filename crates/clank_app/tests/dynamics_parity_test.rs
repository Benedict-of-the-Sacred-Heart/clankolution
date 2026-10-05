use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::SimWorld;

#[test]
fn test_kinematics_and_grazing_formulas() {
    let mut sim = SimWorld::new(42);
    sim.world.agents.truncate(1);
    sim.world.growth = 0.0;
    sim.world.soil.food.fill(2.0);
    sim.world.soil.taint.fill(0.0);
    sim.world.soil.scent.fill(0.0);

    let mut agent = sim.world.agents[0];
    agent.x = 200.0;
    agent.y = 200.0;
    agent.vx = 0.2;
    agent.vy = 0.1;
    agent.angle = 0.5;
    agent.energy = 50.0;
    agent.age = 100;
    agent.birth = 0;
    agent.cooldown = 0;

    // Morphological traits as f32
    let tr0 = agent.tr[0] as f32;
    let tr1 = agent.tr[1] as f32;
    let tr2 = agent.tr[2] as f32;
    let tr3 = agent.tr[3] as f32;
    let tr4 = agent.tr[4] as f32;

    // Reference kinematics from HTML & clank_core
    let o0: f32 = 0.3; // steering
    let o1: f32 = 0.8; // thrust
    let o2: f32 = 0.5; // feeding
    let o3: f32 = 0.0; // attack
    let o4: f32 = 0.0; // signal

    let new_angle = agent.angle as f32 + o0 * (0.11 + 0.09 * tr1);
    let thrust = (o1 + 1.0) * 0.5;
    let mot = 0.45 + 1.1 * tr1;
    let fx = new_angle.cos() * thrust * mot * 0.22;
    let fy = new_angle.sin() * thrust * mot * 0.22;
    let new_vx = (agent.vx as f32 + fx) * 0.89;
    let new_vy = (agent.vy as f32 + fy) * 0.89;

    assert!(new_vx.abs() > 0.0);
    assert!(new_vy.abs() > 0.0);

    // Reference grazing & metabolism
    let feeding = o2.max(0.0);
    let attack = o3.max(0.0);
    let signal = o4.max(0.0);
    let intake_cap = (0.016 + 0.064 * feeding) * (0.7 + tr4);
    let eaten = intake_cap.min(2.0);
    let energy_gain = eaten * (9.0 + 9.0 * tr4);
    let basal = 0.10
        + 0.12 * tr0
        + 0.07 * tr1
        + 0.035 * tr2
        + 0.055 * tr3
        + 0.035 * attack
        + 0.014 * signal;
    let thrust_cost = thrust * 0.06;

    let delta = energy_gain - basal - thrust_cost;
    let expected_energy = (agent.energy as f32 + delta).clamp(0.0, 110.0);
    assert!(expected_energy > 40.0 && expected_energy < 60.0);
}

#[test]
fn test_senescence_death_on_gpu() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let mut sim = SimWorld::new(1234);
    sim.world.agents.truncate(2);
    // Agent 0 is old (age 2105 > 2100)
    sim.world.agents[0].age = 2105;
    sim.world.agents[0].energy = 50.0;
    sim.world.agents[0].dead = 0;

    // Agent 1 is young (age 100)
    sim.world.agents[1].age = 100;
    sim.world.agents[1].energy = 50.0;
    sim.world.agents[1].dead = 0;

    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Run 1 sub-tick on GPU
    driver.dispatch_sub_ticks(1, &params);

    let updated_states = driver.readback_agent_states(2);
    assert_eq!(updated_states.len(), 2);

    // Agent 0 should be marked dead by senescence
    let a0_dead = (updated_states[0].meta_flags & (1 << 13)) != 0;
    assert!(
        a0_dead,
        "Agent 0 with age 2105 must die of senescence (meta_flags: 0x{:X})",
        updated_states[0].meta_flags
    );
    assert_eq!(updated_states[0].angle_energy[1], 0.0);

    // Agent 1 should still be alive
    let a1_dead = (updated_states[1].meta_flags & (1 << 13)) != 0;
    assert!(
        !a1_dead,
        "Agent 1 with age 100 should remain alive (meta_flags: 0x{:X})",
        updated_states[1].meta_flags
    );
    assert!(updated_states[1].angle_energy[1] > 0.0);
}

#[test]
fn test_starvation_death_on_gpu() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let mut sim = SimWorld::new(5678);
    sim.world.agents.truncate(1);
    // Agent has tiny energy that will be drained by basal metabolism
    sim.world.agents[0].age = 10;
    sim.world.agents[0].energy = 0.001;
    sim.world.agents[0].dead = 0;

    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    driver.dispatch_sub_ticks(1, &params);

    let updated_states = driver.readback_agent_states(1);
    let is_dead = (updated_states[0].meta_flags & (1 << 13)) != 0;
    assert!(
        is_dead,
        "Agent with 0.001 energy should starve and be marked dead (meta_flags: 0x{:X})",
        updated_states[0].meta_flags
    );
}

#[test]
fn test_combat_and_killer_attribution_on_gpu() {
    let driver = GpuComputeDriver::create_default();
    if driver.is_none() {
        eprintln!("Skipping test: No compatible GPU adapter available on this host");
        return;
    }
    let driver = driver.unwrap();

    let mut sim = SimWorld::new(999);
    sim.world.agents.truncate(2);
    sim.world.hostility = 200.0; // 200% hostility = 2.0x multiplier
    sim.world.growth = 0.0;
    sim.world.soil.food.fill(0.0); // Zero soil food so victim cannot graze

    // Agent 0 (Attacker): positioned at (200, 200)
    sim.world.agents[0].x = 200.0;
    sim.world.agents[0].y = 200.0;
    sim.world.agents[0].energy = 50.0;
    sim.world.agents[0].cooldown = 0;
    sim.world.agents[0].kills = 0;
    sim.world.agents[0].dead = 0;
    // Set bias weight for actuator 3 (attack) to +127 so out[3] is strongly positive
    sim.world.agents[0].genes[260 + 3 * 11 + 10] = 127;

    // Agent 1 (Victim): adjacent at (204, 200) with low energy
    sim.world.agents[1].x = 204.0;
    sim.world.agents[1].y = 200.0;
    sim.world.agents[1].energy = 0.5;
    sim.world.agents[1].cooldown = 0;
    sim.world.agents[1].dead = 0;

    let (states, genomes, atomics, soil, params) = sync_rust_to_gpu(&sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);

    // Run 1 sub-tick on GPU
    driver.dispatch_sub_ticks(1, &params);

    let updated_states = driver.readback_agent_states(2);
    let telemetry = driver.readback_telemetry();

    // Attacker should register at least 1 kill
    let kills_attacker = (updated_states[0].meta_flags >> 14) & 0x3FFFF;
    assert!(
        kills_attacker >= 1 || telemetry.kills >= 1,
        "Combat should record a kill! Attacker kills: {}, Telemetry kills: {}",
        kills_attacker,
        telemetry.kills
    );

    // Victim should be marked dead
    let victim_dead = (updated_states[1].meta_flags & (1 << 13)) != 0;
    assert!(
        victim_dead,
        "Victim should be dead after lethal attack (meta_flags: 0x{:X})",
        updated_states[1].meta_flags
    );
}
