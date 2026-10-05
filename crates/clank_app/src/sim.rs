use bevy::prelude::*;
use clank_core::world::World;
use clank_core::agent::AgentData;

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
}

#[derive(Clone, Copy, Debug)]
pub struct PendingTool {
    pub tool_type: u32,
    pub tool_pos: [f32; 2],
    pub tool_radius: f32,
}

impl Default for SimWorld {
    fn default() -> Self {
        let world = World::new_with_size(42, 950.0, 747.0, 80, 63);
        Self {
            world,
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            selected_agent_slot: None,
            selected_agent_cache: None,
            pending_tool: None,
            world_width: 950.0,
            world_height: 747.0,
            active_engine: ActiveEngine::Rust,
            gpu_population: 0,
        }
    }
}

impl SimWorld {
    pub fn new(seed: u32) -> Self {
        let world = World::new_with_size(seed, 950.0, 747.0, 80, 63);
        Self {
            world,
            speed: 1,
            paused: false,
            unthrottled: false,
            selected_agent_id: None,
            selected_agent_slot: None,
            selected_agent_cache: None,
            pending_tool: None,
            world_width: 950.0,
            world_height: 747.0,
            active_engine: ActiveEngine::Rust,
            gpu_population: 0,
        }
    }

    pub fn active_population(&self) -> usize {
        if self.active_engine == ActiveEngine::Gpu {
            self.gpu_population as usize
        } else {
            self.world.agents.iter().filter(|a| a.dead == 0).count()
        }
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

pub fn flush_gpu_to_rust(sim: &mut SimWorld, driver: &crate::gpu::compute_driver::GpuComputeDriver) {
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
}

pub fn sim_step_system(
    mut sim: ResMut<SimWorld>,
    mut gpu_res: Option<ResMut<GpuDriverResource>>,
    mut audio_queue: Option<ResMut<crate::audio::AudioVoiceQueue>>,
    ui_state: Option<Res<crate::ui::UiState>>,
    camera_query: Query<(&Camera, &Transform), With<Camera2d>>,
) {
    if sim.paused {
        return;
    }
    let steps = sim.speed.clamp(1, 32);

    match sim.active_engine {
        ActiveEngine::Rust => {
            if let Some(ref mut gpu) = gpu_res {
                if let Some(ref driver) = gpu.driver {
                    if driver.is_initialized() {
                        flush_gpu_to_rust(&mut sim, driver);
                        driver.set_initialized(false);
                    }
                }
            }
            sim.step(steps);
            sim.gpu_population = sim.world.agents.iter().filter(|a| a.dead == 0).count() as u32;
        }
        ActiveEngine::Gpu => {
            if let Some(ref mut gpu) = gpu_res {
                if let Some(ref d) = gpu.driver {
                    if d.soil_cols != sim.world.soil.cols as u32 || d.soil_rows != sim.world.soil.rows as u32 {
                        gpu.driver = None;
                    }
                }
                if gpu.driver.is_none() {
                    let cols = sim.world.soil.cols as u32;
                    let rows = sim.world.soil.rows as u32;
                    gpu.driver = crate::gpu::compute_driver::GpuComputeDriver::create_for_world(cols, rows, 65536);
                }
                if let Some(ref mut driver) = gpu.driver {
                    if !driver.is_initialized() {
                        let (states, genomes, atomics, soil, params) = crate::gpu::bridge::sync_rust_to_gpu(&sim);
                        driver.upload_state(&states, &genomes, &atomics, &soil, &params);
                        sim.gpu_population = sim.world.agents.iter().filter(|a| a.dead == 0).count() as u32;
                    }

                    let current_count = if sim.gpu_population > 0 {
                        sim.gpu_population
                    } else {
                        sim.world.agents.len() as u32
                    };

                    let (cam_pos, cam_size) = if let Ok((camera, transform)) = camera_query.single() {
                        let pos = transform.translation.truncate();
                        let sim_cam = crate::rendering::bevy_to_sim_coord(pos, sim.world_height as f32);
                        let size = camera.logical_viewport_size().unwrap_or(Vec2::new(sim.world_width as f32, sim.world_height as f32));
                        ([sim_cam.x, sim_cam.y], [size.x, size.y])
                    } else {
                        ([(sim.world_width * 0.5) as f32, (sim.world_height * 0.5) as f32], [sim.world_width as f32, sim.world_height as f32])
                    };

                    let (tool_type, tool_pos, tool_radius) = if let Some(pt) = sim.pending_tool.take() {
                        (pt.tool_type, pt.tool_pos, pt.tool_radius)
                    } else {
                        (0xFFFFFFFF, [0.0, 0.0], 45.0)
                    };

                    let (bh, cortex, sex) = if let Some(ref ui) = ui_state {
                        (ui.barnes_hut, ui.expanded_cortex, ui.sexual_selection)
                    } else {
                        (false, false, false)
                    };

                    let params = crate::gpu::types::GpuSimParams {
                        tick: sim.world.tick,
                        agent_count: current_count,
                        max_agents: driver.max_agents,
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
                    driver.dispatch_sub_ticks_with_mods(steps, &params, bh, cortex, sex);

                    // Read back ONLY telemetry (128 bytes)
                    let telemetry = driver.readback_telemetry();
                    sim.world.tick += steps;
                    sim.world.kills += telemetry.kills;
                    sim.world.births += telemetry.total_births.max(telemetry.birth_count);
                    sim.gpu_population = telemetry.population;

                    if telemetry.apex_agent_id > 0 {
                        driver.next_agent_id.fetch_max(telemetry.apex_agent_id, std::sync::atomic::Ordering::Relaxed);
                        sim.world.next_id = sim.world.next_id.max(telemetry.apex_agent_id);
                    }

                    // Ingest audio voices into AudioVoiceQueue
                    if let Some(ref mut aq) = audio_queue {
                        let v_count = (telemetry.audio_voice_count as usize).min(256);
                        if v_count > 0 {
                            let voices = driver.readback_audio_voices(v_count);
                            aq.ingest_voices(&voices);
                        }
                    }

                    // Specimen Picking & Identity Guard
                    if telemetry.selected_agent_idx != 0xFFFFFFFF {
                        sim.selected_agent_id = Some(telemetry.selected_agent_id);
                        sim.selected_agent_slot = Some(telemetry.selected_agent_idx);
                    }

                    if let (Some(slot), Some(expected_id)) = (sim.selected_agent_slot, sim.selected_agent_id) {
                        if let Some((state, genome, atomic)) = driver.readback_agent_slot(slot) {
                            let is_dead = (state.meta_flags & (1 << 13)) != 0;
                            let mut data = crate::gpu::bridge::unpack_gpu_agent_to_agent_data(&state, &genome, Some(&atomic));
                            if state.id != expected_id || is_dead {
                                data.dead = 1;
                            }
                            sim.selected_agent_cache = Some(data);
                        } else {
                            sim.selected_agent_cache = None;
                        }
                    } else {
                        sim.selected_agent_cache = None;
                    }

                    // Automatic agent replenishment if population collapses in GPU mode
                    if telemetry.population < 15 && sim.world.tick % 45 == 0 {
                        let n = (15 - telemetry.population).min(15) as usize;
                        let mut agent_positions = Vec::with_capacity(n);
                        let w = sim.world.w;
                        let h = sim.world.h;
                        for _ in 0..n {
                            let x = sim.world.prng.rand(0.0, w) as f32;
                            let y = sim.world.prng.rand(0.0, h) as f32;
                            agent_positions.push((x, y));
                        }
                        driver.seed_agents_gpu(&agent_positions);
                    }

                    return;
                }
            }
            // Fallback to CPU step if no GPU adapter available
            sim.step(steps);
        }
    }
}

pub struct ClankSimPlugin;

impl Plugin for ClankSimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimWorld>()
            .init_resource::<GpuDriverResource>()
            .add_systems(Update, sim_step_system);
    }
}
