use bevy_egui::egui::{self, Color32, CornerRadius, Stroke, Visuals};

pub const COLOR_BG: Color32 = Color32::from_rgb(7, 16, 18);
pub const COLOR_TOP_BG: Color32 = Color32::from_rgb(11, 23, 25);
pub const COLOR_SIDE_BG: Color32 = Color32::from_rgb(12, 25, 27);
pub const COLOR_STAT_BG: Color32 = Color32::from_rgb(16, 31, 33);
pub const COLOR_PANEL_LINE: Color32 = Color32::from_rgb(41, 64, 65);
pub const COLOR_STAT_BORDER: Color32 = Color32::from_rgb(43, 66, 67);
pub const COLOR_BTN_BG: Color32 = Color32::from_rgb(20, 35, 38);
pub const COLOR_BTN_BORDER: Color32 = Color32::from_rgb(57, 80, 84);
pub const COLOR_BTN_HOVER: Color32 = Color32::from_rgb(43, 72, 73);

pub const COLOR_INK: Color32 = Color32::from_rgb(231, 228, 220);
pub const COLOR_MUTED: Color32 = Color32::from_rgb(143, 156, 155);
pub const COLOR_CYAN: Color32 = Color32::from_rgb(124, 228, 213);
pub const COLOR_GOLD: Color32 = Color32::from_rgb(230, 188, 120);
pub const COLOR_RED: Color32 = Color32::from_rgb(255, 117, 94);

pub const PALETTE: [Color32; 8] = [
    Color32::from_rgb(145, 229, 210), // #91e5d2
    Color32::from_rgb(230, 184, 120), // #e6b878
    Color32::from_rgb(238, 120, 107), // #ee786b
    Color32::from_rgb(190, 151, 215), // #be97d7
    Color32::from_rgb(127, 184, 223), // #7fb8df
    Color32::from_rgb(212, 223, 139), // #d4df8b
    Color32::from_rgb(228, 163, 187), // #e4a3bb
    Color32::from_rgb(137, 197, 161), // #89c5a1
];

pub const PALETTE_40: [Color32; 8] = [
    Color32::from_rgba_premultiplied(145, 229, 210, 64),
    Color32::from_rgba_premultiplied(230, 184, 120, 64),
    Color32::from_rgba_premultiplied(238, 120, 107, 64),
    Color32::from_rgba_premultiplied(190, 151, 215, 64),
    Color32::from_rgba_premultiplied(127, 184, 223, 64),
    Color32::from_rgba_premultiplied(212, 223, 139, 64),
    Color32::from_rgba_premultiplied(228, 163, 187, 64),
    Color32::from_rgba_premultiplied(137, 197, 161, 64),
];

pub fn setup_clank_theme(ctx: &egui::Context) {
    let mut visuals = Visuals::dark();
    visuals.override_text_color = Some(COLOR_INK);
    visuals.panel_fill = COLOR_SIDE_BG;
    visuals.window_fill = COLOR_SIDE_BG;
    visuals.widgets.noninteractive.bg_fill = COLOR_STAT_BG;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, COLOR_PANEL_LINE);
    visuals.widgets.inactive.bg_fill = COLOR_BTN_BG;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, COLOR_BTN_BORDER);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(4);
    visuals.widgets.hovered.bg_fill = COLOR_BTN_HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, COLOR_CYAN);
    visuals.widgets.active.bg_fill = COLOR_BTN_HOVER;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, COLOR_CYAN);
    ctx.set_visuals(visuals);
}
