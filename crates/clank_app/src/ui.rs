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
}

impl Default for UiState {
    fn default() -> Self {
        let mut chronicle = Vec::new();
        chronicle.push("World seeded. Primordial creatures awakened.".to_string());
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
        if self.chronicle.len() >= 50 {
            self.chronicle.remove(0);
        }
        self.chronicle.push(text);
    }
}

pub fn toggle_pause(sim: &mut SimWorld) {
    sim.paused = !sim.paused;
}

pub fn set_simulation_speed(sim: &mut SimWorld, state: &mut UiState, speed: f32) {
    let clamped = speed.clamp(1.0, 10.0);
    state.speed = clamped;
    sim.speed = clamped.round() as u32;
}

pub fn trigger_spore_catastrophe(sim: &mut SimWorld) {
    sim.world.eclipse = 210;
}

pub fn clank_ui_system(
    mut contexts: EguiContexts,
    mut sim: ResMut<SimWorld>,
    mut state: ResMut<UiState>,
    time: Res<Time>,
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
        let pop = sim.world.agents.iter().filter(|a| a.dead == 0).count();
        let food_sum: f32 = sim.world.soil.food.iter().sum::<f32>() / (sim.world.soil.cols * sim.world.soil.rows) as f32;
        let kills = sim.world.kills as usize;
        state.record_history(pop, food_sum, kills);
    }

    let mut root_ui = Ui::new(
        ctx.clone(),
        "root_ui".into(),
        UiBuilder::new()
            .layer_id(LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    // 1. Top Panel (Exact Brand Header: height 53px, background #0b1719, line #1a3033)
    egui::Panel::top("top_header")
        .exact_size(53.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(COLOR_TOP_BG).stroke(Stroke::new(1.0, COLOR_PANEL_LINE)))
        .show(&mut root_ui, |ui| {
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
                        state.add_chronicle("Cycle 00000: New primordial world seeded.".to_string());
                    }

                    ui.add_space(6.0);

                    // Speed slider pill container
                    ui.allocate_ui_with_layout(
                        Vec2::new(140.0, 28.0),
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
                                        ui.spacing_mut().slider_width = 44.0;
                                        if ui.add(egui::Slider::new(&mut speed_val, 1.0..=10.0).show_value(false)).changed() {
                                            set_simulation_speed(&mut sim, &mut state, speed_val);
                                        }
                                        ui.label(RichText::new(format!("{}×", sim.speed)).size(10.5).monospace().strong().color(COLOR_GOLD));
                                    });
                                });
                        },
                    );

                    ui.add_space(6.0);

                    let pause_label = if sim.paused { "RESUME" } else { "PAUSE" };
                    let btn_pause = egui::Button::new(RichText::new(pause_label).size(10.5).monospace().color(Color32::from_rgb(180, 203, 198)))
                        .fill(COLOR_BTN_BG)
                        .stroke(Stroke::new(1.0, COLOR_BTN_BORDER))
                        .corner_radius(CornerRadius::same(3));
                    if ui.add_sized([70.0, 28.0], btn_pause).clicked() {
                        toggle_pause(&mut sim);
                    }
                });
            });
        });

    // 2. Right Sidebar Panel (Exact 330px width, #0c191b background)
    egui::Panel::right("right_sidebar")
        .exact_size(330.0)
        .resizable(false)
        .frame(egui::Frame::NONE.fill(COLOR_SIDE_BG).stroke(Stroke::new(1.0, COLOR_PANEL_LINE)))
        .show(&mut root_ui, |ui| {
            ui.add_space(4.0);
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
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
                        let active_count = sim.world.agents.iter().filter(|a| a.dead == 0).count();
                        let max_gen = sim.world.agents.iter().map(|a| a.gen).max().unwrap_or(0);
                        let root_count = sim.world.roots;
                        let kills_count = sim.world.kills;

                        egui::Grid::new("stat_grid").num_columns(2).spacing([8.0, 8.0]).show(ui, |ui| {
                            render_stat_box(ui, "POPULATION", &format!("{}", active_count), &format!(" / {}", sim.world.max_cap), Some("capacity"));
                            render_stat_box(ui, "GENERATION", &format!("{}", max_gen), " / oldest", Some("living"));
                            ui.end_row();
                            render_stat_box(ui, "PREDATIONS", &format!("{}", kills_count), " / total", None);
                            render_stat_box(ui, "LINEAGES", &format!("{}", root_count), " / living", Some("roots"));
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
                let cycle_sub = if sim.world.eclipse > 0 {
                    format!("THE HUNGER ECLIPSE ({})", sim.world.eclipse)
                } else {
                    "THE FIRST HUNGER".to_string()
                };
                let cycle_text = format!("CYCLE {:05} / {}", sim.world.tick, cycle_sub);
                ui.label(RichText::new(cycle_text).size(11.0).monospace().color(Color32::from_rgb(160, 195, 188)));
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    ui.label(RichText::new("Click a creature to inspect its lineage. Choose a tool, then paint on the world.").size(11.0).monospace().color(Color32::from_rgb(160, 195, 188)));
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

            if ui.add_sized([96.0, 30.0], btn).clicked() {
                if tool == ActiveTool::Eclipse {
                    trigger_spore_catastrophe(sim);
                    state.add_chronicle(format!("Cycle {:05}: Spore Catastrophe triggered! Sunlight obscured.", sim.world.tick));
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
        let btn_clank = egui::Button::new(RichText::new("EXPORT\n(.CLANK)").size(9.5).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([96.0, 38.0], btn_clank).clicked() {
            match save_clank_file(sim, &state.file_path) {
                Ok(bytes) => {
                    let msg = format!("Exported {} bytes to {}", bytes, state.file_path);
                    state.add_chronicle(msg.clone());
                    state.status_message = Some(msg);
                }
                Err(e) => state.status_message = Some(format!("Export error: {}", e)),
            }
        }

        let json_path = state.file_path.replace(".clank", ".json");
        let btn_json = egui::Button::new(RichText::new("EXPORT\n(.JSON)").size(9.5).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([96.0, 38.0], btn_json).clicked() {
            match export_json_file(sim, &json_path) {
                Ok(bytes) => {
                    let msg = format!("Exported JSON ({} bytes) to {}", bytes, json_path);
                    state.add_chronicle(msg.clone());
                    state.status_message = Some(msg);
                }
                Err(e) => state.status_message = Some(format!("JSON error: {}", e)),
            }
        }

        let btn_import = egui::Button::new(RichText::new("IMPORT").size(10.0).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([96.0, 38.0], btn_import).clicked() {
            let file_is_json = state.file_path.ends_with(".json");
            let result = if file_is_json {
                import_json_file(sim, &state.file_path)
            } else {
                load_clank_file(sim, &state.file_path)
            };
            match result {
                Ok(()) => {
                    sim.selected_agent_id = None;
                    let msg = format!("Imported {} successfully", state.file_path);
                    state.add_chronicle(msg.clone());
                    state.status_message = Some(msg);
                }
                Err(e) => state.status_message = Some(format!("Import error: {}", e)),
            }
        }
        ui.end_row();

        let btn_saves = egui::Button::new(RichText::new("WHAT SAVES?").size(10.0).monospace().color(Color32::from_rgb(166, 196, 191)))
            .fill(COLOR_BTN_BG).stroke(Stroke::new(1.0, COLOR_BTN_BORDER)).corner_radius(CornerRadius::same(3));
        if ui.add_sized([106.0, 28.0], btn_saves).clicked() {
            state.status_message = Some("Zero-copy .clank saves all creatures, soil grids, genes, brains & PRNG state.".to_string());
        }
        ui.end_row();
    });

    ui.add_space(4.0);
    ui.label(RichText::new("Export a snapshot to continue generations later. Import it here on this or another machine.").size(10.5).monospace().color(COLOR_MUTED));

    if let Some(ref msg) = state.status_message {
        ui.add_space(2.0);
        ui.label(RichText::new(msg).size(10.5).monospace().color(COLOR_GOLD));
    }
}


fn render_pressure_sliders(ui: &mut egui::Ui, sim: &mut SimWorld) {
    let mut mutation = sim.world.mutation as f32;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Mutation").size(11.0).color(Color32::from_rgb(196, 208, 202)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(format!("{:.2}", mutation)).size(11.0).color(COLOR_CYAN));
        });
    });
    if ui.add(egui::Slider::new(&mut mutation, 0.0..=0.5).show_value(false)).changed() {
        sim.world.mutation = mutation as f64;
    }

    let mut growth = sim.world.growth as f32;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Food renewal").size(11.0).color(Color32::from_rgb(196, 208, 202)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(format!("{:.2}×", growth)).size(11.0).color(COLOR_CYAN));
        });
    });
    if ui.add(egui::Slider::new(&mut growth, 0.0..=2.0).show_value(false)).changed() {
        sim.world.growth = growth as f64;
    }

    let mut hostility = sim.world.hostility as f32;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Hostility of contact").size(11.0).color(Color32::from_rgb(196, 208, 202)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(format!("{:.2}×", hostility)).size(11.0).color(COLOR_CYAN));
        });
    });
    if ui.add(egui::Slider::new(&mut hostility, 0.0..=2.0).show_value(false)).changed() {
        sim.world.hostility = hostility as f64;
    }

    let mut cap = sim.world.max_cap;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Creature capacity").size(11.0).color(Color32::from_rgb(196, 208, 202)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(format!("{}", cap)).size(11.0).color(COLOR_CYAN));
        });
    });
    if ui.add(egui::Slider::new(&mut cap, 50..=1000).show_value(false)).changed() {
        sim.world.max_cap = cap;
    }
}

fn render_history_chart(ui: &mut egui::Ui, state: &UiState, max_cap: usize) {
    let (response, painter) = ui.allocate_painter(Vec2::new(300.0, 82.0), Sense::hover());
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
    let x_step = rect.width() / (count - 1) as f32;

    // Series 1: Population (Life: #8fe3cf)
    let color_life = Color32::from_rgb(143, 227, 207);
    let cap_f = (max_cap as f32).max(220.0);
    for i in 0..count - 1 {
        let p0 = &state.history[i];
        let p1 = &state.history[i + 1];
        let y0 = rect.bottom() - 3.0 - (p0.population as f32 / cap_f).clamp(0.0, 1.0) * (rect.height() - 7.0);
        let y1 = rect.bottom() - 3.0 - (p1.population as f32 / cap_f).clamp(0.0, 1.0) * (rect.height() - 7.0);
        painter.line_segment([Pos2::new(rect.left() + i as f32 * x_step, y0), Pos2::new(rect.left() + (i + 1) as f32 * x_step, y1)], Stroke::new(1.5, color_life));
    }

    // Series 2: Food (Food: #d9ad72)
    let color_food = Color32::from_rgb(217, 173, 114);
    for i in 0..count - 1 {
        let p0 = &state.history[i];
        let p1 = &state.history[i + 1];
        let y0 = rect.bottom() - 3.0 - (p0.food / 1.7).clamp(0.0, 1.0) * (rect.height() - 7.0);
        let y1 = rect.bottom() - 3.0 - (p1.food / 1.7).clamp(0.0, 1.0) * (rect.height() - 7.0);
        painter.line_segment([Pos2::new(rect.left() + i as f32 * x_step, y0), Pos2::new(rect.left() + (i + 1) as f32 * x_step, y1)], Stroke::new(1.5, color_food));
    }

    // Series 3: Violence (Predations: #ed7869)
    let color_pred = Color32::from_rgb(237, 120, 105);
    for i in 0..count - 1 {
        let k0 = state.history[i].kills;
        let k_prev = state.history[i.saturating_sub(10)].kills;
        let delta0 = (k0.saturating_sub(k_prev) as f32 / 30.0).clamp(0.0, 1.0);

        let k1 = state.history[i + 1].kills;
        let k_prev1 = state.history[(i + 1).saturating_sub(10)].kills;
        let delta1 = (k1.saturating_sub(k_prev1) as f32 / 30.0).clamp(0.0, 1.0);

        let y0 = rect.bottom() - 3.0 - delta0 * (rect.height() - 7.0);
        let y1 = rect.bottom() - 3.0 - delta1 * (rect.height() - 7.0);
        painter.line_segment([Pos2::new(rect.left() + i as f32 * x_step, y0), Pos2::new(rect.left() + (i + 1) as f32 * x_step, y1)], Stroke::new(1.5, color_pred));
    }
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
            ui.set_width(300.0);
            if let Some(agent) = sim.get_selected_agent() {
                ui.label(RichText::new(format!("Creature #{} (Gen {})", agent.id, agent.gen)).size(13.0).strong().color(Color32::from_rgb(240, 230, 217)));
                ui.add_space(4.0);
                ui.label(RichText::new(format!("Energy: {:.1}  |  Age: {} ticks  |  Kills: {}", agent.energy, agent.age, agent.kills)).size(10.5).color(Color32::from_rgb(173, 191, 186)));
                ui.label(RichText::new(format!("Lineage Root: {}", agent.root)).size(10.5).color(COLOR_GOLD));
                ui.add_space(4.0);

                // Trait display
                let trait_names = ["Bulk", "Speed", "Sight", "Armor", "Forage", "Carn"];
                for (name, val) in trait_names.iter().zip(agent.tr.iter()) {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(*name).size(10.0).color(COLOR_MUTED));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{:.2}", val)).size(10.0).color(COLOR_CYAN));
                        });
                    });
                }

                ui.add_space(6.0);
                ui.label(RichText::new("Recurrent Brain Activations (h[0..9]):").size(10.0).color(COLOR_GOLD));
                for (i, val) in agent.h.iter().enumerate() {
                    let norm = ((*val as f32 + 1.0) * 0.5).clamp(0.0, 1.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("h[{}]", i)).size(9.5).color(COLOR_MUTED));
                        ui.add(egui::ProgressBar::new(norm).desired_width(180.0));
                    });
                }
            } else {
                ui.label(RichText::new("Select a creature in the arena.\nIts recurrent state, ancestry, and traits will appear here.").size(11.0).color(Color32::from_rgb(173, 191, 186)));
            }
        });
}

fn render_chronicle_log(ui: &mut egui::Ui, state: &UiState) {
    egui::Frame::NONE
        .fill(Color32::from_rgb(10, 21, 23))
        .stroke(Stroke::new(1.0, Color32::from_rgb(37, 58, 59)))
        .inner_margin(egui::Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(300.0);
            ui.set_height(100.0);
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for (i, entry) in state.chronicle.iter().rev().take(6).enumerate() {
                    let color = if i == 0 { Color32::from_rgb(234, 215, 184) } else { Color32::from_rgb(167, 184, 175) };
                    ui.label(RichText::new(entry).size(10.5).color(color));
                }
            });
        });
}

pub struct ClankUiPlugin;

impl Plugin for ClankUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiState>()
            .add_plugins(EguiPlugin::default())
            .add_systems(EguiPrimaryContextPass, clank_ui_system);
    }
}
