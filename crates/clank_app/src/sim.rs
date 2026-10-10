use bevy::prelude::*;
use clank_core::agent::AgentData;
use clank_core::world::World;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ActiveEngine {
    #[default]
    Rust,
    Gpu,
}

#[derive(Resource)]
pub struct SimWorld {
    pub world: World,
    pub speed: u32,
    pub paused: bool,
    pub unthrottled: bool,
    pub selected_agent_id: Option<u32>,
    pub selected_agent_slot: Option<u32>,
    pub selected_agent_cache: Option<AgentData>,
    pub pending_tool: Option<PendingTool>,
    pub world_width: f64,
    pub world_height: f64,
    pub active_engine: ActiveEngine,
    pub gpu_population: u32,
    /// The GPU census's highest living generation and distinct living lineages.
    pub gpu_max_generation: u32,
    pub gpu_living_lineages: u32,
    /// Set when the CPU world was replaced (NEW WORLD, API reset, load); the GPU copy is
    /// stale and must be re-uploaded rather than flushed back.
    pub gpu_resync_pending: bool,
    /// `world.births` at the last GPU upload; GPU telemetry counts births since then.
    pub gpu_births_base: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PendingTool {
    pub tool_type: u32,
    pub tool_pos: [f32; 2],
    pub tool_radius: f32,
}

impl Default for SimWorld {
    fn default() -> Self {
        Self::new(42)
    }
}

impl SimWorld {
    pub fn new(seed: u32) -> Self {
        let world = World::new(seed);
        Self {
            world,
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            selected_agent_slot: None,
            selected_agent_cache: None,
            pending_tool: None,
            world_width: 900.0,
            world_height: 600.0,
            active_engine: ActiveEngine::Rust,
            gpu_population: 0,
            gpu_max_generation: 0,
            gpu_living_lineages: 0,
            gpu_resync_pending: false,
            gpu_births_base: 0,
        }
    }

    pub fn active_population(&self) -> usize {
        if self.active_engine == ActiveEngine::Gpu {
            self.gpu_population as usize
        } else {
            self.world.agents.iter().filter(|a| a.dead == 0).count()
        }
    }

    /// The highest generation among living creatures, from the engine that owns them.
    pub fn max_generation(&self) -> u32 {
        if self.active_engine == ActiveEngine::Gpu {
            self.gpu_max_generation
        } else {
            self.cpu_max_generation()
        }
    }

    /// The number of distinct lineages among living creatures, from the engine that owns them.
    pub fn living_lineages(&self) -> usize {
        if self.active_engine == ActiveEngine::Gpu {
            self.gpu_living_lineages as usize
        } else {
            self.cpu_living_lineages()
        }
    }

    fn cpu_max_generation(&self) -> u32 {
        self.world
            .agents
            .iter()
            .filter(|a| a.dead == 0)
            .map(|a| a.gen)
            .max()
            .unwrap_or(0)
    }

    fn cpu_living_lineages(&self) -> usize {
        self.world
            .agents
            .iter()
            .filter(|a| a.dead == 0)
            .map(|a| a.root)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    pub fn step(&mut self, substeps: u32) {
        if self.paused {
            return;
        }
        for _ in 0..substeps {
            self.world.evolve();
        }
    }

    pub fn force_step(&mut self, substeps: u32) {
        for _ in 0..substeps {
            self.world.evolve();
        }
    }

    pub fn reset(&mut self) {
        self.world.reset();
        self.selected_agent_id = None;
        self.selected_agent_slot = None;
        self.selected_agent_cache = None;
        self.pending_tool = None;
        self.gpu_resync_pending = true;
    }

    pub fn get_selected_agent(&self) -> Option<&AgentData> {
        if self.active_engine == ActiveEngine::Gpu {
            self.selected_agent_cache.as_ref()
        } else {
            let target_id = self.selected_agent_id?;
            self.world.agents.iter().find(|a| a.id == target_id)
        }
    }
}

#[derive(Resource, Default)]
pub struct GpuDriverResource {
    pub driver: Option<crate::gpu::compute_driver::GpuComputeDriver>,
}

pub fn flush_gpu_to_rust(
    sim: &mut SimWorld,
    driver: &crate::gpu::compute_driver::GpuComputeDriver,
) {
    let telem = driver.readback_telemetry();
    let read_count = sim.world.max_cap.min(driver.max_agents as usize);
    let updated_states = driver.readback_agent_states(read_count);
    let updated_genomes = driver.readback_agent_genomes(read_count);
    let updated_atomics = driver.readback_atomics(read_count);
    let updated_soil = driver.readback_soil();
    let params = crate::gpu::types::GpuSimParams {
        tick: sim.world.tick,
        agent_count: telem.population,
        max_agents: driver.max_agents,
        max_capacity: sim.world.max_cap as u32,
        ..Default::default()
    };
    crate::gpu::bridge::sync_gpu_to_rust(
        &updated_states,
        &updated_genomes,
        &updated_atomics,
        &updated_soil,
        &params,
        sim,
    );
    sim.world.roots = sim.world.roots.max(
        driver
            .next_agent_root
            .load(std::sync::atomic::Ordering::Relaxed),
    );
    // Ids the GPU issued stay spent, also those of creatures already gone.
    sim.world.next_id = sim.world.next_id.max(telem.apex_agent_id);
}

/// Slots the app's GPU driver allocates (simulation and presentation capacity).
pub const GPU_MAX_AGENTS: u32 = 65536;

/// Keeps a GPU driver sized to the current soil grid in both engine modes: the GPU engine
/// simulates with it and both engines are presented from its buffers. In the app it is built
/// on Bevy's render device so the renderer reads those buffers in place; without a renderer
/// (headless) it gets a device of its own.
pub fn ensure_gpu_driver(
    sim: Res<SimWorld>,
    mut gpu: ResMut<GpuDriverResource>,
    render_device: Option<Res<bevy::render::renderer::RenderDevice>>,
    render_queue: Option<Res<bevy::render::renderer::RenderQueue>>,
) {
    let cols = sim.world.soil.cols as u32;
    let rows = sim.world.soil.rows as u32;
    let render_available = render_device.is_some() && render_queue.is_some();
    if let Some(ref d) = gpu.driver {
        let grid_matches = d.soil_cols == cols && d.soil_rows == rows;
        // A driver made before the renderer existed moves onto its device, unless the GPU
        // engine already holds live state on the old device.
        let can_move = render_available && !d.on_render_device && !d.is_initialized();
        if grid_matches && !can_move {
            return;
        }
    }
    gpu.driver = match (render_device, render_queue) {
        (Some(device), Some(queue)) => Some(
            crate::gpu::compute_driver::GpuComputeDriver::from_render_device(
                &device,
                &queue,
                cols,
                rows,
                GPU_MAX_AGENTS,
            ),
        ),
        _ => crate::gpu::compute_driver::GpuComputeDriver::create_for_world(
            cols,
            rows,
            GPU_MAX_AGENTS,
        ),
    };
}

/// Uploads the CPU world to the GPU, which then owns the simulation.
pub fn upload_world_to_gpu(
    sim: &mut SimWorld,
    driver: &crate::gpu::compute_driver::GpuComputeDriver,
) {
    let (states, genomes, atomics, soil, params) = crate::gpu::bridge::sync_rust_to_gpu(sim);
    driver.upload_state(&states, &genomes, &atomics, &soil, &params);
    // Lineages that died out still count: GPU founders continue after the world's `roots`.
    driver
        .next_agent_root
        .fetch_max(sim.world.roots, std::sync::atomic::Ordering::Relaxed);
    driver.raise_next_agent_id(sim.world.next_id);
    // Present the uploaded world as it was, before the first GPU tick: its soil colormap and
    // its creatures' trails (a NEW WORLD has none; an engine switch keeps them).
    let mut rgba = vec![0u8; sim.world.soil.grid_size * 4];
    crate::rendering::generate_soil_rgba(&sim.world.soil, &mut rgba);
    driver.write_soil_display(&rgba);
    driver.write_trails(&crate::gpu::bridge::agent_trails(sim));
    sim.gpu_population = sim.world.agents.iter().filter(|a| a.dead == 0).count() as u32;
    // Until the first GPU frame lands, the census is the uploaded world's.
    sim.gpu_max_generation = sim.cpu_max_generation();
    sim.gpu_living_lineages = sim.cpu_living_lineages() as u32;
    sim.gpu_births_base = sim.world.births;
}

pub fn sim_step_system(
    mut sim: ResMut<SimWorld>,
    mut gpu_res: Option<ResMut<GpuDriverResource>>,
    mut audio_queue: Option<ResMut<crate::audio::AudioVoiceQueue>>,
    camera_query: Query<(&Camera, &Transform, &Projection), With<Camera2d>>,
) {
    // A replaced CPU world invalidates the GPU copy. Never flush it back; in GPU mode upload
    // the new world at once so the renderer (which reads GPU buffers) shows it even while paused.
    if sim.gpu_resync_pending {
        sim.gpu_resync_pending = false;
        if let Some(driver) = gpu_res.as_ref().and_then(|g| g.driver.as_ref()) {
            driver.set_initialized(false);
            let grid_matches = driver.soil_cols == sim.world.soil.cols as u32
                && driver.soil_rows == sim.world.soil.rows as u32;
            if sim.active_engine == ActiveEngine::Gpu && grid_matches {
                upload_world_to_gpu(&mut sim, driver);
            }
        }
    }
    // Finished GPU frames land here, also while paused, so nothing stays in flight.
    if let Some(driver) = gpu_res.as_ref().and_then(|g| g.driver.as_ref()) {
        if driver.is_initialized() {
            for frame in driver.poll_frame_readbacks() {
                apply_frame_readback(&mut sim, driver, &frame, audio_queue.as_deref_mut());
            }
        }
    }
    // An engine switch hands the world over at once, also while paused, so the panel and
    // picking use the engine that now owns it.
    if let Some(driver) = gpu_res.as_ref().and_then(|g| g.driver.as_ref()) {
        match (sim.active_engine, driver.is_initialized()) {
            (ActiveEngine::Rust, true) => {
                flush_gpu_to_rust(&mut sim, driver);
                for frame in driver.poll_frame_readbacks() {
                    apply_frame_readback(&mut sim, driver, &frame, None);
                }
                driver.set_initialized(false);
            }
            (ActiveEngine::Gpu, false) => upload_world_to_gpu(&mut sim, driver),
            _ => {}
        }
    }
    if sim.paused {
        return;
    }
    let steps = sim.speed.clamp(1, 32);

    match sim.active_engine {
        ActiveEngine::Rust => {
            sim.step(steps);
            sim.gpu_population = sim.world.agents.iter().filter(|a| a.dead == 0).count() as u32;
        }
        ActiveEngine::Gpu => {
            if let Some(ref mut gpu) = gpu_res {
                if let Some(ref mut driver) = gpu.driver {
                    let current_count = if sim.gpu_population > 0 {
                        sim.gpu_population
                    } else {
                        sim.world.agents.len() as u32
                    };

                    let (cam_pos, cam_size) = if let Ok((camera, transform, proj)) =
                        camera_query.single()
                    {
                        let pos = transform.translation.truncate();
                        let sim_cam =
                            crate::rendering::bevy_to_sim_coord(pos, sim.world_height as f32);
                        let mut size = camera
                            .logical_viewport_size()
                            .unwrap_or(Vec2::new(sim.world_width as f32, sim.world_height as f32));
                        if let Projection::Orthographic(ref ortho) = *proj {
                            size *= ortho.scale;
                        }
                        ([sim_cam.x, sim_cam.y], [size.x, size.y])
                    } else {
                        (
                            [
                                (sim.world_width * 0.5) as f32,
                                (sim.world_height * 0.5) as f32,
                            ],
                            [sim.world_width as f32, sim.world_height as f32],
                        )
                    };

                    let (tool_type, tool_pos, tool_radius) =
                        if let Some(pt) = sim.pending_tool.take() {
                            (pt.tool_type, pt.tool_pos, pt.tool_radius)
                        } else {
                            (0xFFFFFFFF, [0.0, 0.0], 45.0)
                        };

                    let params = crate::gpu::types::GpuSimParams {
                        tick: sim.world.tick,
                        agent_count: current_count,
                        // Only the slots that can hold a creature, not all of the driver's.
                        max_agents: driver.active_slots(sim.world.max_cap as u32),
                        max_capacity: sim.world.max_cap as u32,
                        hostility: (sim.world.hostility / 100.0) as f32,
                        mut_rate: (sim.world.mutation / 100.0) as f32,
                        speed: sim.speed as f32,
                        renewal: (sim.world.growth / 100.0) as f32,
                        sub_tick: 0,
                        sub_ticks_per_frame: steps,
                        tool_type,
                        tool_radius,
                        tool_pos,
                        camera_pos: cam_pos,
                        camera_size: cam_size,
                        world_size: [sim.world_width as f32, sim.world_height as f32],
                        soil_grid: [sim.world.soil.cols as u32, sim.world.soil.rows as u32],
                        eclipse: sim.world.eclipse,
                        epoch: 0,
                    };

                    driver.update_params(&params);
                    driver.dispatch_sub_ticks(steps, &params);

                    // Results arrive through the async ring on a later frame.
                    driver.request_frame_readback(sim.selected_agent_slot, tool_type == 0);
                    sim.world.tick += steps;
                    // The reference counts the eclipse down once per tick.
                    sim.world.eclipse = sim.world.eclipse.saturating_sub(steps);
                    if sim.selected_agent_slot.is_none() {
                        sim.selected_agent_cache = None;
                    }

                    // Agent replenishment if GPU population collapses (matching HTML line 1464 & Rust world.rs line 984)
                    if sim.gpu_population < 15 && sim.world.tick % 45 < steps {
                        let n = 15 - sim.gpu_population;
                        let mut positions = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            let x = (sim.world.prng.next_f64() * sim.world_width) as f32;
                            let y = (sim.world.prng.next_f64() * sim.world_height) as f32;
                            positions.push((x, y));
                        }
                        let new_slots = driver.replenish_agents_gpu(&positions);
                        // Seeding waited for the GPU, so every frame requested so far has
                        // landed; apply them now or their pre-seed counts would arrive later.
                        for frame in driver.poll_frame_readbacks() {
                            apply_frame_readback(
                                &mut sim,
                                driver,
                                &frame,
                                audio_queue.as_deref_mut(),
                            );
                        }
                        sim.gpu_population += new_slots.len() as u32;
                        let cur_tick = sim.world.tick;
                        if sim.world.events.len() < 1024 {
                            sim.world.events.push(clank_core::SimEvent {
                                tick: cur_tick,
                                event_type: 1,
                                p1: 0,
                                p2: 0,
                            });
                        }
                    }

                    return;
                }
            }
            // Fallback to CPU step if no GPU adapter available
            sim.step(steps);
        }
    }
}

/// Applies one finished GPU frame: counters, audio, picking and the selected creature.
fn apply_frame_readback(
    sim: &mut SimWorld,
    driver: &crate::gpu::compute_driver::GpuComputeDriver,
    frame: &crate::gpu::frame_readback::FrameReadback,
    audio_queue: Option<&mut crate::audio::AudioVoiceQueue>,
) {
    let telemetry = &frame.telemetry;
    sim.world.kills += telemetry.kills;
    sim.world.births = sim.gpu_births_base + telemetry.total_births;
    sim.gpu_population = telemetry.population;
    sim.gpu_max_generation = telemetry.max_generation;
    sim.gpu_living_lineages = telemetry.living_lineages;

    if telemetry.apex_agent_id > 0 {
        driver.next_agent_id.fetch_max(
            telemetry.apex_agent_id,
            std::sync::atomic::Ordering::Relaxed,
        );
        sim.world.next_id = sim.world.next_id.max(telemetry.apex_agent_id);
    }

    // The GPU's chronicle entries join the queue the UI drains, as the CPU engine's do.
    for e in &frame.chronicle {
        if sim.world.events.len() < 1024 {
            sim.world.events.push(clank_core::SimEvent {
                tick: e.tick,
                event_type: e.event_type,
                p1: e.p1,
                p2: e.p2,
            });
        }
    }

    for v in &frame.audio {
        let (color_idx, count) = match v.event_type {
            1 => (8, 2),  // Kill spark
            2 => (9, 9),  // Birth spark
            3 => (10, 9), // Lineage spark
            _ => (8, 2),
        };
        if sim.world.spark_events.len() < 512 {
            sim.world.spark_events.push(clank_core::SparkEvent {
                x: v.pos[0],
                y: v.pos[1],
                color_idx,
                count,
            });
        }
    }
    if let Some(aq) = audio_queue {
        if !frame.audio.is_empty() {
            aq.ingest_voices(&frame.audio);
        }
    }

    // Specimen Picking & Identity Guard
    if frame.picking {
        if telemetry.selected_agent_idx != 0xFFFFFFFF {
            if sim.selected_agent_id != Some(telemetry.selected_agent_id) {
                sim.selected_agent_cache = None;
            }
            sim.selected_agent_id = Some(telemetry.selected_agent_id);
            sim.selected_agent_slot = Some(telemetry.selected_agent_idx);
        } else {
            sim.selected_agent_id = None;
            sim.selected_agent_slot = None;
            sim.selected_agent_cache = None;
        }
    }

    // Only a snapshot of the creature that is selected now; frames requested before a new
    // pick describe the previous selection.
    let Some(sel) = frame.selected else {
        return;
    };
    let (Some(slot), Some(expected_id)) = (sim.selected_agent_slot, sim.selected_agent_id) else {
        return;
    };
    if sel.slot != slot {
        return;
    }
    // A specimen that died is deselected, as the reference does (`selected = null`); its slot
    // may already hold a newborn, which must not take its place on the panel.
    let is_dead = (sel.state.meta_flags & (1 << 13)) != 0;
    if sel.state.id != expected_id || is_dead {
        sim.selected_agent_id = None;
        sim.selected_agent_slot = None;
        sim.selected_agent_cache = None;
        return;
    }
    let mut data = crate::gpu::bridge::unpack_gpu_agent_to_agent_data(
        &sel.state,
        &sel.genome,
        Some(&sel.atomic),
    );
    if sel.trail.owner_id == sel.state.id {
        let n = (sel.trail.count as usize).min(crate::gpu::types::TRAIL_LEN);
        for (i, p) in sel.trail.pts[..n].iter().enumerate() {
            data.trail_x[i] = p[0] as f64;
            data.trail_y[i] = p[1] as f64;
        }
        data.trail_count = n as u32;
    }
    sim.selected_agent_cache = Some(data);
}

pub struct ClankSimPlugin;

impl Plugin for ClankSimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimWorld>()
            .init_resource::<GpuDriverResource>()
            .add_systems(Update, (ensure_gpu_driver, sim_step_system).chain());
    }
}
