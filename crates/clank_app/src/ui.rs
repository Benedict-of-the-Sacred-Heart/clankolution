use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use crate::sim::SimWorld;
use crate::persistence::{save_clank_file, load_clank_file, export_json_file, import_json_file};

#[derive(Resource)]
pub struct UiState {
    pub speed: f32,
    pub show_stats: bool,
    pub show_controls: bool,
    pub file_path: String,
    pub status_message: Option<String>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            speed: 1.0,
            show_stats: true,
            show_controls: true,
            file_path: "clankolution_save.clank".to_string(),
            status_message: None,
        }
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
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::Window::new("CLANKOLUTION 2.0")
        .default_pos([16.0, 16.0])
        .default_width(320.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("CLANKOLUTION 2.0");
            ui.label(egui::RichText::new("Native Bevy High-Performance Core").italics().weak());
            ui.separator();

            // Simulation Controls
            ui.heading("Controls");
            ui.horizontal(|ui| {
                let play_pause_label = if sim.paused { "▶ Play" } else { "⏸ Pause" };
                if ui.button(play_pause_label).clicked() {
                    toggle_pause(&mut sim);
                }
                if ui.button("⏭ Step").clicked() {
                    sim.force_step(1);
                }
                if ui.button("↺ Reset").clicked() {
                    sim.reset();
                    state.status_message = Some("World reset".into());
                }
            });

            let mut speed_val = state.speed;
            ui.add(egui::Slider::new(&mut speed_val, 1.0..=10.0).text("Speed Multiplier"));
            if (speed_val - state.speed).abs() > 0.01 {
                set_simulation_speed(&mut sim, &mut state, speed_val);
            }

            ui.separator();

            // Ecosystem Stats
            let active_count = sim.world.agents.iter().filter(|a| a.dead == 0).count();
            ui.heading("Ecosystem Stats");
            ui.label(format!("Tick: {}", sim.world.tick));
            ui.label(format!("Active Creatures: {} / {}", active_count, sim.world.max_cap));
            ui.label(format!("Total Births: {}", sim.world.births));
            ui.label(format!("Total Kills: {}", sim.world.kills));
            ui.label(format!("Lineage Roots: {}", sim.world.roots));
            if sim.world.eclipse > 0 {
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), format!("Eclipse Active: {} ticks remaining", sim.world.eclipse));
            }

            ui.separator();

            // Selective Pressures
            ui.heading("Selective Pressures");
            let mut mutation = sim.world.mutation as f32;
            if ui.add(egui::Slider::new(&mut mutation, 0.0..=1.0).text("Mutation Rate")).changed() {
                sim.world.mutation = mutation as f64;
            }

            let mut growth = sim.world.growth as f32;
            if ui.add(egui::Slider::new(&mut growth, 0.0..=3.0).text("Soil Growth")).changed() {
                sim.world.growth = growth as f64;
            }

            let mut hostility = sim.world.hostility as f32;
            if ui.add(egui::Slider::new(&mut hostility, 0.0..=1.0).text("Hostility")).changed() {
                sim.world.hostility = hostility as f64;
            }

            let mut cap = sim.world.max_cap;
            if ui.add(egui::Slider::new(&mut cap, 50..=1000).text("Max Population")).changed() {
                sim.world.max_cap = cap;
            }

            if ui.button("⚡ Trigger Spore Catastrophe").clicked() {
                trigger_spore_catastrophe(&mut sim);
                state.status_message = Some("Spore catastrophe triggered!".into());
            }

            ui.separator();

            // Persistence
            ui.heading("Persistence");
            ui.text_edit_singleline(&mut state.file_path);
            ui.horizontal(|ui| {
                if ui.button("💾 Save .clank").clicked() {
                    match save_clank_file(&sim, &state.file_path) {
                        Ok(bytes) => state.status_message = Some(format!("Saved {} bytes to {}", bytes, state.file_path)),
                        Err(e) => state.status_message = Some(format!("Save error: {}", e)),
                    }
                }
                if ui.button("📂 Load .clank").clicked() {
                    match load_clank_file(&mut sim, &state.file_path) {
                        Ok(()) => {
                            sim.selected_agent_id = None;
                            state.status_message = Some(format!("Loaded {} successfully", state.file_path));
                        }
                        Err(e) => state.status_message = Some(format!("Load error: {}", e)),
                    }
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Export JSON").clicked() {
                    let json_path = state.file_path.replace(".clank", ".json");
                    match export_json_file(&sim, &json_path) {
                        Ok(bytes) => state.status_message = Some(format!("Exported {} bytes to {}", bytes, json_path)),
                        Err(e) => state.status_message = Some(format!("JSON error: {}", e)),
                    }
                }
                if ui.button("Import JSON").clicked() {
                    let json_path = state.file_path.replace(".clank", ".json");
                    match import_json_file(&mut sim, &json_path) {
                        Ok(()) => {
                            sim.selected_agent_id = None;
                            state.status_message = Some(format!("Imported {} successfully", json_path));
                        }
                        Err(e) => state.status_message = Some(format!("Import error: {}", e)),
                    }
                }
            });

            if let Some(ref msg) = state.status_message {
                ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(250, 204, 21)));
            }

            ui.separator();

            // Creature Inspector
            if let Some(agent) = sim.get_selected_agent() {
                ui.heading(format!("Creature #{} (Gen {})", agent.id, agent.gen));
                ui.label(format!("Energy: {:.1}", agent.energy));
                ui.label(format!("Age: {} ticks", agent.age));
                ui.label(format!("Lineage Root: {}", agent.root));
                ui.label(format!("Kills: {}", agent.kills));
                ui.label(format!("Traits: Bulk={:.2} Spd={:.2} Sight={:.2}", agent.tr[0], agent.tr[1], agent.tr[2]));
                ui.label(format!("Armor={:.2} Forage={:.2} Carn={:.2}", agent.tr[3], agent.tr[4], agent.tr[5]));
                ui.label(format!("Attack={:.2} Signal={:.2}", agent.attack, agent.signal));

                ui.collapsing("Neural Hidden State (h[0..10])", |ui| {
                    for (i, val) in agent.h.iter().enumerate() {
                        ui.label(format!("h[{}]: {:.3}", i, val));
                    }
                });

                if ui.button("Deselect").clicked() {
                    sim.selected_agent_id = None;
                }
            } else {
                ui.label(egui::RichText::new("Click a creature to inspect").weak().italics());
            }
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
