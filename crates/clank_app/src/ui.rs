use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use egui::{Color32, CornerRadius, LayerId, Pos2, RichText, Sense, Stroke, Ui, UiBuilder, Vec2};
use crate::sim::SimWorld;
use crate::persistence::{save_clank_file, load_clank_file, export_json_file, import_json_file};
use crate::theme::{self, COLOR_CYAN, COLOR_GOLD, COLOR_INK, COLOR_MUTED, COLOR_PANEL_LINE, COLOR_RED, COLOR_SIDE_BG, COLOR_STAT_BG, COLOR_STAT_BORDER, COLOR_TOP_BG, COLOR_BTN_BG, COLOR_BTN_BORDER};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTool {
    Observe,
    Nourish,
    Blight,
    SeedLife,
    Extinguish,
    Eclipse,
}

#[derive(Debug, Clone)]
pub struct HistoryPoint {
    pub population: usize,
    pub food: f32,
    pub kills: usize,
}

#[derive(Resource)]
pub struct UiState {
    pub speed: f32,
    pub active_tool: ActiveTool,
    pub history: Vec<HistoryPoint>,
    pub chronicle: Vec<String>,
    pub show_stats: bool,
    pub show_controls: bool,
    pub file_path: String,
    pub status_message: Option<String>,
    pub theme_initialized: bool,
    pub history_timer: f32,
    pub scroll_offset: Option<f32>,
    pub barnes_hut: bool,
    pub expanded_cortex: bool,
    pub sexual_selection: bool,
    pub show_mod_drawer: bool,
}

impl Default for UiState {
    fn default() -> Self {
        let mut chronicle = Vec::new();
        chronicle.push("00000  The first hunger begins.".to_string());
        Self {
            speed: 1.0,
            active_tool: ActiveTool::Observe,
            history: Vec::new(),
            chronicle,
            show_stats: true,
            show_controls: true,
            file_path: "clankolution_save.clank".to_string(),
            status_message: None,
            theme_initialized: false,
            history_timer: 0.0,
            scroll_offset: None,
            barnes_hut: false,
            expanded_cortex: false,
            sexual_selection: false,
            show_mod_drawer: true,
        }
    }
}

impl UiState {
    pub fn record_history(&mut self, population: usize, food: f32, kills: usize) {
        if self.history.len() >= 300 {
            self.history.remove(0);
        }
        self.history.push(HistoryPoint {
            population,
            food,
            kills,
        });
    }

    pub fn add_chronicle(&mut self, text: String) {
        self.chronicle.insert(0, text);
        if self.chronicle.len() > 50 {
            self.chronicle.truncate(50);
        }
    }
}

pub const WHAT_SAVES_NOTE: &str = ".clank saves an instant binary snapshot in microseconds via rkyv. .json saves a portable human-readable format.";
pub const DEFAULT_SAVE_HINT: &str = "Export a snapshot to continue generations later. Import it here on this or another device.";

pub fn compute_export_clank_filename(tick: u32) -> String {
    format!("clankolution-cycle-{}.clank", tick)
}

pub fn compute_export_json_filename(tick: u32) -> String {
    format!("clankolution-cycle-{}.json", tick)
}

pub fn compute_export_clank_status(tick: u32, bytes: usize) -> String {
    format!("Cycle {} saved to binary snapshot (.clank, {:.1} KB).", tick, bytes as f32 / 1024.0)
}

pub fn compute_export_json_status(tick: u32) -> String {
    format!("Cycle {} exported. Keep the JSON file to restore this world.", tick)
}

pub fn compute_import_clank_status(tick: u32, bytes: usize) -> String {
    format!("Cycle {} restored from .clank binary snapshot ({:.1} KB).", tick, bytes as f32 / 1024.0)
}

pub fn compute_import_json_status(tick: u32) -> String {
    format!("Cycle {} restored. The simulation continues here.", tick)
}

pub fn toggle_pause(sim: &mut SimWorld) {
    sim.paused = !sim.paused;
}

pub fn set_simulation_speed(sim: &mut SimWorld, state: &mut UiState, speed: f32) {
    let clamped = speed.clamp(1.0, 32.0);
    state.speed = clamped;
    sim.speed = clamped.round() as u32;
}

pub fn trigger_spore_catastrophe(sim: &mut SimWorld) {
    sim.world.eclipse = 210;
}

pub fn compute_cycle_subtitle(tick: u32, eclipse: u32, active_count: usize) -> &'static str {
    if tick == 0 {
        "THE FIRST HUNGER"
    } else if eclipse > 0 {
        "THE ECLIPSE"
    } else if active_count > 200 {
        "THE SWARM"
    } else if active_count < 25 {
        "THE REMNANT"
    } else {
        "THE HUNGER"
    }
}

pub fn compute_tool_hint(active_tool: ActiveTool) -> &'static str {
    match active_tool {
        ActiveTool::Observe => "Click a creature to inspect its lineage. Choose a tool, then paint on the world.",
        ActiveTool::Nourish => "Drag on the world to grow food.",
        ActiveTool::Blight => "Drag on the world to spread blight.",
        ActiveTool::SeedLife => "Drag on the world to seed life.",
        ActiveTool::Extinguish => "Drag on the world to extinguish creatures.",
        ActiveTool::Eclipse => "Click a creature to inspect its lineage. Choose a tool, then paint on the world.",
    }
}

#[derive(Resource, Default, Clone, Debug)]
pub struct MinimapCache {
    pub clusters: Vec<crate::gpu::lbvh::MinimapCluster>,
}

pub fn update_minimap_cache_from_sim(sim: &SimWorld, cache: &mut MinimapCache) {
    let living_states: Vec<crate::gpu::types::GpuAgentState> = sim.world.agents.iter()
        .filter(|a| a.dead == 0)
        .map(|a| crate::gpu::types::GpuAgentState {
            pos_vel: [a.x as f32, a.y as f32, 0.0, 0.0],
            angle_energy: [0.0; 4],
            traits: [a.tr[0] as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            hidden: [0.0; 10],
            id: a.id,
            meta_flags: a.root & 0x0F,
            age_gen: 0,
            morton_code: crate::gpu::spatial_index::compute_morton_32([a.x as f32, a.y as f32]),
            packed_color: 0,
            visual_cache: 1,
        })
        .collect();

    if !living_states.is_empty() {
        let tree = crate::gpu::lbvh::LbvhTree::build(&living_states);
        cache.clusters = tree.extract_minimap_clusters(4);
    } else {
        cache.clusters.clear();
    }
}

pub fn update_minimap_cache_system(
    sim: Option<Res<SimWorld>>,
    gpu_res: Option<Res<crate::sim::GpuDriverResource>>,
    mut cache: ResMut<MinimapCache>,
) {
    if let Some(sim) = sim {
        if sim.active_engine == crate::sim::ActiveEngine::Gpu {
            if let Some(ref gpu) = gpu_res {
                if let Some(ref driver) = gpu.driver {
                    if driver.is_initialized() {
                        cache.clusters = driver.extract_gpu_minimap_clusters(4);
                        return;
                    }
                }
            }
        }
        update_minimap_cache_from_sim(&sim, &mut cache);
    }
}

pub fn clank_ui_system(
    mut contexts: EguiContexts,
    mut sim: ResMut<SimWorld>,
    mut state: ResMut<UiState>,
    time: Res<Time>,
    minimap_cache: Option<Res<MinimapCache>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    if !state.theme_initialized {
        theme::setup_clank_theme(ctx);
        state.theme_initialized = true;
        return;
    }

    // Periodic history recorder
    state.history_timer += time.delta_secs();
    if state.history_timer >= 0.25 {
        state.history_timer = 0.0;
        let pop = sim.active_population();
        let food_sum: f32 = sim.world.soil.food.iter().sum::<f32>() / (sim.world.soil.cols * sim.world.soil.rows) as f32;
        let kills = sim.world.kills as usize;
        state.record_history(pop, food_sum, kills);
    }

    // Drain simulation chronicle events from clank_core
    let sim_events: Vec<_> = sim.world.events.drain(..).collect();
    for ev in sim_events {
        let msg = match ev.event_type {
            1 => "Spores emerge from the sediment.".to_string(),
            2 => format!("Generation {} opens in lineage {}.", ev.p1, ev.p2),
            3 => format!("Lineage {} has taken {} lives.", ev.p1, ev.p2),
            _ => continue,
        };
        state.add_chronicle(format!("{:05}  {}", ev.tick, msg));
    }

    let mut root_ui = Ui::new(
        ctx.clone(),
        "root_ui".into(),
        UiBuilder::new()
            .layer_id(LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    // 1. Top Panel (Exact Brand Header: height 53px, background #0b1719, bottom line #1a3033)
    egui::Panel::top("top_header")
        .exact_size(53.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(COLOR_TOP_BG))
        .show(&mut root_ui, |ui| {
            let r = ui.max_rect();
            ui.painter().line_segment(
                [Pos2::new(r.left(), r.bottom()), Pos2::new(r.right(), r.bottom())],
                Stroke::new(1.0, COLOR_PANEL_LINE),
            );
            ui.add_space(6.0);
            ui.horizontal_centered(|ui| {
                ui.add_space(18.0);
                // Brand
                ui.horizontal(|ui| {
                    ui.label(RichText::new("C L A N K ").size(13.5).monospace().strong().color(COLOR_INK));
                    ui.label(RichText::new("O ").size(13.5).monospace().strong().color(COLOR_RED));
                    ui.label(RichText::new("L U T I O N").size(13.5).monospace().strong().color(COLOR_INK));
                    ui.add_space(14.0);
                    ui.label(RichText::new("A N   E X P E R I M E N T   I N   I N H E R I T E D   A P P E T I T E").size(9.5).monospace().color(COLOR_MUTED));
                });

                // Top Controls on Right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(18.0);
                    let btn_new = egui::Button::new(RichText::new("NEW WORLD").size(10.5).monospace().color(Color32::from_rgb(180, 203, 198)))
                        .fill(COLOR_BTN_BG)
                        .stroke(Stroke::new(1.0, COLOR_BTN_BORDER))
                        .corner_radius(CornerRadius::same(3));
                    if ui.add_sized([95.0, 28.0], btn_new).clicked() {
                        sim.reset();
                        state.add_chronicle(format!("{:05}  The first hunger begins.", sim.world.tick));
                    }

                    ui.add_space(7.0);

                    // Speed slider pill container
                    ui.allocate_ui_with_layout(
                        Vec2::new(145.0, 28.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            egui::Frame::NONE
                                .fill(COLOR_BTN_BG)
                                .stroke(Stroke::new(1.0, COLOR_BTN_BORDER))
                                .corner_radius(CornerRadius::same(3))
                                .inner_margin(egui::Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new("SPEED").size(9.5).monospace().color(COLOR_MUTED));
                                        let mut speed_val = state.speed;
                                        ui.spacing_mut().slider_width = 46.0;
                                        if ui.add(egui::Slider::new(&mut speed_val, 1.0..=32.0).show_value(false)).changed() {
                                             set_simulation_speed(&mut sim, &mut state, speed_val);
                                        }
                                        ui.label(RichText::new(format!("{}×", sim.speed)).size(10.5).monospace().strong().color(COLOR_GOLD));
                                    });
                                });
                        },
                    );

                    ui.add_space(7.0);

                    let pause_label = if sim.paused { "RESUME" } else { "PAUSE" };
                    let btn_pause = egui::Button::new(RichText::new(pause_label).size(10.5).monospace().color(Color32::from_rgb(180, 203, 198)))
                        .fill(COLOR_BTN_BG)
                        .stroke(Stroke::new(1.0, COLOR_BTN_BORDER))
                        .corner_radius(CornerRadius::same(3));
                    if ui.add_sized([70.0, 28.0], btn_pause).clicked() {
                        toggle_pause(&mut sim);
                    }

                    ui.add_space(7.0);

                    let (label, bg_col, stroke_col, text_col) = match sim.active_engine {
                        crate::sim::ActiveEngine::Rust => ("ENGINE: RUST", Color32::from_rgb(20, 53, 43), Color32::from_rgb(56, 239, 125), Color32::from_rgb(56, 239, 125)),
                        crate::sim::ActiveEngine::Gpu => ("ENGINE: GPU", Color32::from_rgb(18, 48, 65), Color32::from_rgb(0, 220, 255), Color32::from_rgb(0, 220, 255)),
                    };
                    let btn_engine = egui::Button::new(RichText::new(label).size(10.0).monospace().strong().color(text_col))
                        .fill(bg_col)
                        .stroke(Stroke::new(1.0, stroke_col))
                        .corner_radius(CornerRadius::same(3));
                    if ui.add_sized([106.0, 28.0], btn_engine).clicked() {
                        match sim.active_engine {
                            crate::sim::ActiveEngine::Rust => {
                                sim.active_engine = crate::sim::ActiveEngine::Gpu;
                                state.add_chronicle(format!("{:05}  Switched to GPU compute simulation engine.", sim.world.tick));
                            }
                            crate::sim::ActiveEngine::Gpu => {
                                sim.active_engine = crate::sim::ActiveEngine::Rust;
                                state.add_chronicle(format!("{:05}  Switched to Rust reference simulation engine.", sim.world.tick));
                            }
                        }
                    }
                });
            });
        });

    // 2. Right Sidebar Panel (Exact 330px width, #0c191b background, left line #1a3033)
    egui::Panel::right("right_sidebar")
        .exact_size(330.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(COLOR_SIDE_BG))
        .show(&mut root_ui, |ui| {
            let r = ui.max_rect();
            ui.painter().line_segment(
                [Pos2::new(r.left(), r.top()), Pos2::new(r.left(), r.bottom())],
                Stroke::new(1.0, COLOR_PANEL_LINE),
            );
            ui.add_space(4.0);
            let mut scroll_area = egui::ScrollArea::vertical().auto_shrink([false, false]);
            if let Some(offset) = state.scroll_offset {
                scroll_area = scroll_area.vertical_scroll_offset(offset);
            }
            scroll_area.show(ui, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add_space(8.0);
                    ui.vertical(|ui| {
                        ui.set_max_width(290.0);
                        // Eyebrow & Hero Title
                        ui.label(RichText::new("FIELD NOTES / 001").size(9.5).monospace().strong().color(COLOR_CYAN));
                        ui.add_space(3.0);
                        ui.label(
                            RichText::new("Let them become\nsomething else.")
                                .family(egui::FontFamily::Name("Georgia".into()))
                                .size(26.0)
                                .strong()
                                .color(COLOR_INK)
                                .line_height(Some(30.0))
                        );
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new("Each creature inherits a tiny quantized recurrent brain and a body. Food blooms. Blood feeds the soil. Nothing is told how to behave.")
                                .size(11.0)
                                .monospace()
                                .color(Color32::from_rgb(160, 184, 178))
                                .line_height(Some(16.0))
                        );
                        ui.add_space(10.0);

                        // 2x2 Stat Grid
                        let active_count = sim.active_population();
                        let max_gen = sim.world.agents.iter().map(|a| a.gen).max().unwrap_or(0);
                        let living_roots = sim.world.agents.iter()
                            .filter(|a| a.dead == 0)
                            .map(|a| a.root)
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        let kills_count = sim.world.kills;

                        egui::Grid::new("stat_grid").num_columns(2).spacing([8.0, 8.0]).show(ui, |ui| {
                            render_stat_box(ui, "POPULATION", &format!("{}", active_count), &format!(" / {}", sim.world.max_cap), Some("capacity"));
                            render_stat_box(ui, "GENERATION", &format!("{}", max_gen), " / oldest", Some("living"));
                            ui.end_row();
                            render_stat_box(ui, "PREDATIONS", &format!("{}", kills_count), " / total", None);
                            render_stat_box(ui, "LINEAGES", &format!("{}", living_roots), " / living", Some("roots"));
                            ui.end_row();
                        });

                        ui.add_space(14.0);
                        ui.separator();

                        // INTERVENE Section
                        ui.label(RichText::new("INTERVENE").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_tool_matrix(ui, &mut state, &mut sim);
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new("Drag to paint. Blight is a fading toxin: it destroys food and drains creatures crossing it. An eclipse starves the surface, then it regrows.")
                                .size(10.5)
                                .monospace()
                                .color(COLOR_MUTED)
                                .line_height(Some(15.0))
                        );

                        ui.add_space(14.0);
                        ui.separator();

                        // KEEP YOUR WORLD Section
                        ui.label(RichText::new("KEEP YOUR WORLD").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_persistence_section(ui, &mut state, &mut sim);

                        ui.add_space(14.0);
                        ui.separator();

                        // SELECTIVE PRESSURE Section
                        ui.label(RichText::new("SELECTIVE PRESSURE").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_pressure_sliders(ui, &mut sim);

                        ui.add_space(14.0);
                        ui.separator();

                        // THE RECORD Section (3-Series Live History Chart)
                        ui.label(RichText::new("THE RECORD").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_history_chart(ui, &state, sim.world.max_cap);
                        ui.add_space(4.0);
                        render_chart_legend(ui);

                        ui.add_space(14.0);
                        ui.separator();

                        // SPECIMEN Section
                        ui.label(RichText::new("SPECIMEN").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_specimen_box(ui, &sim);

                        ui.add_space(14.0);
                        ui.separator();

                        // RADAR MINIMAP (LBVH CLUSTERING)
                        ui.label(RichText::new("RADAR MINIMAP").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        let cached_clusters = minimap_cache.as_ref().map(|c| &c.clusters[..]).unwrap_or(&[]);
                        render_radar_minimap(ui, cached_clusters);

                        ui.add_space(14.0);
                        ui.separator();

                        // EXPERIMENTAL MUTATIONS (GPU SIM MODS)
                        ui.label(RichText::new("EXPERIMENTAL MUTATIONS").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_mod_drawer(ui, &mut state);

                        ui.add_space(14.0);
                        ui.separator();


                        // CHRONICLE Section
                        ui.label(RichText::new("CHRONICLE").size(10.5).monospace().strong().color(COLOR_GOLD));
                        ui.add_space(4.0);
                        render_chronicle_log(ui, &state);

                        ui.add_space(14.0);
                        ui.label(RichText::new("One file. No assets. No API. All decisions happen on your machine.").size(10.0).monospace().color(Color32::from_rgb(105, 128, 122)));
                        ui.add_space(14.0);
                    });
                });
            });
        });

    // 3. Central Transparent Panel with HUD Overlays
    egui::CentralPanel::no_frame()
        .show(&mut root_ui, |ui| {
            ui.add_space(14.0);
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                let active_count = sim.active_population();
                let cycle_sub = compute_cycle_subtitle(sim.world.tick, sim.world.eclipse, active_count);
                let cycle_text = format!("CYCLE {:05} / {}", sim.world.tick, cycle_sub);
                ui.label(RichText::new(cycle_text).size(11.0).monospace().color(Color32::from_rgb(160, 195, 188)));
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    let hint_text = compute_tool_hint(state.active_tool);
                    ui.label(RichText::new(hint_text).size(11.0).monospace().color(Color32::from_rgb(160, 195, 188)));
                });
            });
        });
}

fn render_stat_box(ui: &mut egui::Ui, label: &str, value: &str, sub: &str, sub2: Option<&str>) {
    egui::Frame::NONE
        .fill(COLOR_STAT_BG)
        .stroke(Stroke::new(1.0, COLOR_STAT_BORDER))
        .corner_radius(CornerRadius::same(2))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(142.0);
            ui.set_min_height(60.0);
            ui.vertical(|ui| {
                ui.label(RichText::new(label).size(9.0).monospace().color(COLOR_MUTED));
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(value).size(22.0).monospace().strong().color(COLOR_INK));
                    ui.label(RichText::new(sub).size(10.0).monospace().color(COLOR_MUTED));
                });
                if let Some(s2) = sub2 {
                    ui.label(RichText::new(s2).size(9.5).monospace().color(COLOR_MUTED));
                }
            });
        });
}

fn render_tool_matrix(ui: &mut egui::Ui, state: &mut UiState, sim: &mut SimWorld) {
    let tools = [
        ("OBSERVE", ActiveTool::Observe),
        ("NOURISH", ActiveTool::Nourish),
        ("BLIGHT", ActiveTool::Blight),
        ("SEED LIFE", ActiveTool::SeedLife),
        ("EXTINGUISH", ActiveTool::Extinguish),
        ("ECLIPSE", ActiveTool::Eclipse),
    ];

    egui::Grid::new("tool_grid").num_columns(3).spacing([6.0, 6.0]).show(ui, |ui| {
        for (i, (name, tool)) in tools.into_iter().enumerate() {
            let is_active = state.active_tool == tool;
            let btn_fill = if is_active { Color32::from_rgb(23, 52, 54) } else { COLOR_BTN_BG };
            let btn_stroke = if is_active { Stroke::new(1.0, COLOR_CYAN) } else { Stroke::new(1.0, COLOR_BTN_BORDER) };
            let text_color = if is_active { Color32::from_rgb(224, 245, 240) } else { Color32::from_rgb(166, 196, 191) };

            let btn = egui::Button::new(RichText::new(name).size(10.0).monospace().color(text_color))
                .fill(btn_fill)
                .stroke(btn_stroke)
                .corner_radius(CornerRadius::same(3));

            if ui.add_sized([92.0, 30.0], btn).clicked() {
                if tool == ActiveTool::Eclipse {
                    trigger_spore_catastrophe(sim);
                    state.add_chronicle(format!("{:05}  An eclipse consumes the harvest.", sim.world.tick));
                } else {
                    state.active_tool = tool;
                }
            }

            if (i + 1) % 3 == 0 {
                ui.end_row();
            }
        }
    });
}

fn render_persistence_section(ui: &mut egui::Ui, state: &mut UiState, sim: &mut SimWorld) {
    egui::Grid::new("persist_grid").num_columns(3).spacing([6.0, 6.0]).show(ui, |ui| {
        let btn_clank = egui::Button::new(RichText::new("EXPORT (.CLANK)").size(9.0).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([92.0, 30.0], btn_clank).clicked() {
            let filename = compute_export_clank_filename(sim.world.tick);
            match save_clank_file(sim, &filename) {
                Ok(bytes) => {
                    state.status_message = Some(compute_export_clank_status(sim.world.tick, bytes));
                    state.file_path = filename;
                }
                Err(e) => state.status_message = Some(format!("Export error: {}", e)),
            }
        }

        let btn_json = egui::Button::new(RichText::new("EXPORT (.JSON)").size(9.0).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([92.0, 30.0], btn_json).clicked() {
            let filename = compute_export_json_filename(sim.world.tick);
            match export_json_file(sim, &filename) {
                Ok(_bytes) => {
                    state.status_message = Some(compute_export_json_status(sim.world.tick));
                    state.file_path = filename;
                }
                Err(e) => state.status_message = Some(format!("JSON error: {}", e)),
            }
        }

        let btn_import = egui::Button::new(RichText::new("IMPORT").size(10.0).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([92.0, 30.0], btn_import).clicked() {
            let picked = rfd::FileDialog::new()
                .add_filter("World Snapshot (.clank, .json)", &["clank", "json"])
                .pick_file();
            if let Some(path) = picked {
                let path_str = path.to_string_lossy().to_string();
                let is_json = path_str.ends_with(".json");
                let result = if is_json {
                    import_json_file(sim, &path_str)
                } else {
                    load_clank_file(sim, &path_str)
                };
                match result {
                    Ok(()) => {
                        sim.selected_agent_id = None;
                        let (event_msg, status_msg) = if is_json {
                            ("A world returns from its record.", compute_import_json_status(sim.world.tick))
                        } else {
                            let byte_len = std::fs::metadata(&path).map(|m| m.len() as usize).unwrap_or(0);
                            ("A world returns from its .clank record.", compute_import_clank_status(sim.world.tick, byte_len))
                        };
                        state.add_chronicle(format!("{:05}  {}", sim.world.tick, event_msg));
                        state.status_message = Some(status_msg);
                        state.file_path = path_str;
                    }
                    Err(e) => state.status_message = Some(format!("Could not load this world: {}", e)),
                }
            }
        }
        ui.end_row();

        let btn_saves = egui::Button::new(RichText::new("WHAT SAVES?").size(9.0).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([96.0, 30.0], btn_saves).clicked() {
            state.status_message = Some(WHAT_SAVES_NOTE.to_string());
        }
        ui.end_row();
    });

    ui.add_space(4.0);
    let hint_text = state.status_message.as_deref().unwrap_or(DEFAULT_SAVE_HINT);
    ui.label(RichText::new(hint_text).size(10.5).monospace().color(COLOR_MUTED));
}


fn render_slider_row(ui: &mut egui::Ui, label: &str, val_str: &str, width: f32) {
    ui.allocate_ui_with_layout(
        Vec2::new(width, 16.0),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.label(RichText::new(label).size(11.0).color(Color32::from_rgb(196, 208, 202)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(val_str).size(11.0).color(COLOR_CYAN));
            });
        },
    );
}

fn render_pressure_sliders(ui: &mut egui::Ui, sim: &mut SimWorld) {
    let slider_w = 290.0;
    ui.spacing_mut().slider_width = slider_w;

    let mut mutation = sim.world.mutation as f32;
    render_slider_row(ui, "Mutation", &format!("{:.2}", mutation / 100.0), slider_w);
    if ui.add(egui::Slider::new(&mut mutation, 0.0..=50.0).show_value(false)).changed() {
        sim.world.mutation = mutation as f64;
    }

    let mut growth = sim.world.growth as f32;
    render_slider_row(ui, "Food renewal", &format!("{:.2}×", growth / 100.0), slider_w);
    if ui.add(egui::Slider::new(&mut growth, 0.0..=200.0).show_value(false)).changed() {
        sim.world.growth = growth as f64;
    }

    let mut hostility = sim.world.hostility as f32;
    render_slider_row(ui, "Hostility of contact", &format!("{:.2}×", hostility / 100.0), slider_w);
    if ui.add(egui::Slider::new(&mut hostility, 0.0..=200.0).show_value(false)).changed() {
        sim.world.hostility = hostility as f64;
    }

    let mut cap = sim.world.max_cap;
    render_slider_row(ui, "Creature capacity", &format!("{}", cap), slider_w);
    if ui.add(egui::Slider::new(&mut cap, 50..=10000).show_value(false)).changed() {
        sim.world.set_max_capacity(cap as u32);
    }
}

fn render_mod_drawer(ui: &mut egui::Ui, state: &mut UiState) {
    ui.horizontal(|ui| {
        let label = if state.barnes_hut { "[BARNES-HUT: ON]" } else { "[BARNES-HUT: OFF]" };
        let col = if state.barnes_hut { COLOR_CYAN } else { COLOR_MUTED };
        if ui.button(RichText::new(label).size(9.5).monospace().color(col)).clicked() {
            state.barnes_hut = !state.barnes_hut;
        }

        let label_ctx = if state.expanded_cortex { "[EXPANDED CORTEX: ON]" } else { "[EXPANDED CORTEX: OFF]" };
        let col_ctx = if state.expanded_cortex { COLOR_CYAN } else { COLOR_MUTED };
        if ui.button(RichText::new(label_ctx).size(9.5).monospace().color(col_ctx)).clicked() {
            state.expanded_cortex = !state.expanded_cortex;
        }
    });
    ui.add_space(3.0);
    ui.horizontal(|ui| {
        let label_sex = if state.sexual_selection { "[SEXUAL SELECTION: ON]" } else { "[SEXUAL SELECTION: OFF]" };
        let col_sex = if state.sexual_selection { COLOR_CYAN } else { COLOR_MUTED };
        if ui.button(RichText::new(label_sex).size(9.5).monospace().color(col_sex)).clicked() {
            state.sexual_selection = !state.sexual_selection;
        }
    });
}

fn render_history_chart(ui: &mut egui::Ui, state: &UiState, max_cap: usize) {
    let (response, painter) = ui.allocate_painter(Vec2::new(290.0, 82.0), Sense::hover());
    let rect = response.rect;

    // Background and border
    painter.rect_filled(rect, CornerRadius::same(0), Color32::from_rgb(10, 21, 23));
    painter.rect_stroke(rect, CornerRadius::same(0), Stroke::new(1.0, Color32::from_rgb(37, 58, 59)), egui::StrokeKind::Outside);

    // Grid lines
    for dy in [20.0, 40.0, 60.0] {
        let y = rect.top() + dy;
        painter.line_segment([Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)], Stroke::new(1.0, Color32::from_rgb(25, 42, 43)));
    }

    if state.history.len() < 2 {
        return;
    }

    let count = state.history.len();
    let x_step = (rect.width() - 2.0) / (count - 1) as f32;

    // Series 1: Population (Life: #8fe3cf)
    let color_life = Color32::from_rgb(143, 227, 207);
    let cap_f = (max_cap as f32).max(220.0);
    let mut life_pts = Vec::with_capacity(count);
    for (i, p) in state.history.iter().enumerate() {
        let x = rect.left() + 1.0 + i as f32 * x_step;
        let y = rect.bottom() - 3.0 - (p.population as f32 / cap_f).clamp(0.0, 1.0) * (rect.height() - 7.0);
        life_pts.push(Pos2::new(x, y));
    }
    painter.line(life_pts, Stroke::new(1.5, color_life));

    // Series 2: Food (Food: #d9ad72)
    let color_food = Color32::from_rgb(217, 173, 114);
    let mut food_pts = Vec::with_capacity(count);
    for (i, p) in state.history.iter().enumerate() {
        let x = rect.left() + 1.0 + i as f32 * x_step;
        let y = rect.bottom() - 3.0 - (p.food / 1.7).clamp(0.0, 1.0) * (rect.height() - 7.0);
        food_pts.push(Pos2::new(x, y));
    }
    painter.line(food_pts, Stroke::new(1.5, color_food));

    // Series 3: Violence (Predations: #ed7869)
    let color_pred = Color32::from_rgb(237, 120, 105);
    let mut pred_pts = Vec::with_capacity(count);
    for i in 0..count {
        let x = rect.left() + 1.0 + i as f32 * x_step;
        let k0 = state.history[i].kills;
        let k_prev = state.history[i.saturating_sub(10)].kills;
        let delta = (k0.saturating_sub(k_prev) as f32 / 30.0).clamp(0.0, 1.0);
        let y = rect.bottom() - 3.0 - delta * (rect.height() - 7.0);
        pred_pts.push(Pos2::new(x, y));
    }
    painter.line(pred_pts, Stroke::new(1.5, color_pred));
}

fn render_chart_legend(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("■").color(Color32::from_rgb(143, 227, 207)));
        ui.label(RichText::new("life").size(10.0).color(COLOR_MUTED));
        ui.add_space(8.0);
        ui.label(RichText::new("■").color(Color32::from_rgb(217, 173, 114)));
        ui.label(RichText::new("food").size(10.0).color(COLOR_MUTED));
        ui.add_space(8.0);
        ui.label(RichText::new("■").color(Color32::from_rgb(237, 120, 105)));
        ui.label(RichText::new("violence").size(10.0).color(COLOR_MUTED));
    });
}

fn render_specimen_box(ui: &mut egui::Ui, sim: &SimWorld) {
    egui::Frame::NONE
        .fill(Color32::from_rgb(16, 30, 31))
        .stroke(Stroke::new(1.0, Color32::from_rgb(45, 65, 64)))
        .corner_radius(CornerRadius::same(2))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(290.0);
            if let Some(agent) = sim.get_selected_agent() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("SPECIMEN {}", agent.id)).size(13.5).strong().monospace().color(Color32::from_rgb(240, 230, 217)));
                    ui.label(RichText::new(format!("  ·  lineage {}  ·  generation {}", agent.root, agent.gen)).size(11.0).monospace().color(Color32::from_rgb(173, 191, 186)));
                });
                ui.add_space(2.0);
                ui.label(
                    RichText::new(format!("Energy {:.1}  ·  age {}  ·  kills {}", agent.energy, agent.age, agent.kills))
                        .size(11.0)
                        .monospace()
                        .color(Color32::from_rgb(173, 191, 186))
                );
                ui.add_space(2.0);
                let labels = ["bulk", "speed", "sight", "armor", "foraging", "carnivory"];
                let traits_str = labels
                    .iter()
                    .zip(agent.tr.iter())
                    .map(|(name, val)| format!("{} {}%", name, (val * 100.0).round() as i32))
                    .collect::<Vec<_>>()
                    .join("  ·  ");
                ui.label(RichText::new(traits_str).size(11.0).monospace().color(Color32::from_rgb(173, 191, 186)));
                ui.add_space(2.0);
                let h_str = agent
                    .h
                    .iter()
                    .take(5)
                    .map(|v| format!("{:.2}", v))
                    .collect::<Vec<_>>()
                    .join(" / ");
                ui.label(
                    RichText::new(format!("Recurrent state {}", h_str))
                        .size(11.0)
                        .monospace()
                        .color(Color32::from_rgb(173, 191, 186))
                );
            } else {
                ui.label(RichText::new("Select a creature in the arena.\nIts recurrent state, ancestry, and traits will appear here.").size(11.0).color(Color32::from_rgb(173, 191, 186)));
            }
        });
}

fn render_chronicle_log(ui: &mut egui::Ui, state: &UiState) {
    ui.allocate_ui_with_layout(
        Vec2::new(290.0, 112.0),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            for (i, entry) in state.chronicle.iter().take(7).enumerate() {
                let color = if i == 0 {
                    Color32::from_rgb(234, 215, 184)
                } else {
                    Color32::from_rgb(167, 184, 175)
                };
                ui.label(
                    RichText::new(entry)
                        .size(11.0)
                        .monospace()
                        .color(color)
                        .line_height(Some(16.0)),
                );
            }
        },
    );
}

/// Maps a hierarchical LBVH MinimapCluster into 2D canvas coordinates for the radar minimap.
pub fn compute_minimap_cluster_disc(
    cluster: &crate::gpu::lbvh::MinimapCluster,
    map_width: f32,
    map_height: f32,
) -> ([f32; 2], f32, u32) {
    let cx = (cluster.center[0] / 900.0).clamp(0.0, 1.0) * map_width;
    let cy = (cluster.center[1] / 600.0).clamp(0.0, 1.0) * map_height;
    let scale = map_width / 900.0;
    let r = (cluster.radius * scale).max(2.0);
    ([cx, cy], r, cluster.dominant_lineage)
}

fn render_radar_minimap(ui: &mut egui::Ui, clusters: &[crate::gpu::lbvh::MinimapCluster]) {
    let (response, painter) = ui.allocate_painter(Vec2::new(290.0, 100.0), Sense::hover());
    let rect = response.rect;

    // Background and border
    painter.rect_filled(rect, CornerRadius::same(0), Color32::from_rgb(10, 21, 23));
    painter.rect_stroke(rect, CornerRadius::same(0), Stroke::new(1.0, Color32::from_rgb(37, 58, 59)), egui::StrokeKind::Outside);

    // Crosshairs
    let mid_x = rect.left() + rect.width() * 0.5;
    let mid_y = rect.top() + rect.height() * 0.5;
    painter.line_segment([Pos2::new(mid_x, rect.top()), Pos2::new(mid_x, rect.bottom())], Stroke::new(0.5, Color32::from_rgba_unmultiplied(0, 220, 255, 30)));
    painter.line_segment([Pos2::new(rect.left(), mid_y), Pos2::new(rect.right(), mid_y)], Stroke::new(0.5, Color32::from_rgba_unmultiplied(0, 220, 255, 30)));

    for cluster in clusters {
        let (disc_pos, radius, lineage) = compute_minimap_cluster_disc(cluster, rect.width(), rect.height());
        let screen_pos = Pos2::new(rect.left() + disc_pos[0], rect.top() + disc_pos[1]);
        let pal_color = crate::theme::PALETTE[(lineage as usize) % crate::theme::PALETTE.len()];
        let col = Color32::from_rgba_unmultiplied(pal_color.r(), pal_color.g(), pal_color.b(), 180);
        painter.circle_filled(screen_pos, radius.min(12.0), col);
    }
}

pub struct ClankUiPlugin;

impl Plugin for ClankUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiState>()
            .init_resource::<MinimapCache>()
            .add_plugins(EguiPlugin::default())
            .add_systems(Update, update_minimap_cache_system)
            .add_systems(EguiPrimaryContextPass, clank_ui_system);
    }
}
