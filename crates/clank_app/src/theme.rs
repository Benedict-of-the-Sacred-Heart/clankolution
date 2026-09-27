use std::sync::Arc;
use bevy_egui::egui::{self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, Stroke, Visuals};

pub const COLOR_BG: Color32 = Color32::from_rgb(7, 16, 18);
pub const COLOR_TOP_BG: Color32 = Color32::from_rgb(11, 23, 25);
pub const COLOR_SIDE_BG: Color32 = Color32::from_rgb(12, 25, 27);
pub const COLOR_STAT_BG: Color32 = Color32::from_rgb(12, 25, 27);
pub const COLOR_PANEL_LINE: Color32 = Color32::from_rgb(26, 48, 51);
pub const COLOR_STAT_BORDER: Color32 = Color32::from_rgb(26, 48, 51);
pub const COLOR_BTN_BG: Color32 = Color32::from_rgb(14, 31, 34);
pub const COLOR_BTN_BORDER: Color32 = Color32::from_rgb(29, 56, 59);
pub const COLOR_BTN_HOVER: Color32 = Color32::from_rgb(23, 52, 54);

pub const COLOR_INK: Color32 = Color32::from_rgb(238, 229, 213);
pub const COLOR_MUTED: Color32 = Color32::from_rgb(109, 143, 138);
pub const COLOR_CYAN: Color32 = Color32::from_rgb(91, 196, 176);
pub const COLOR_GOLD: Color32 = Color32::from_rgb(204, 168, 90);
pub const COLOR_RED: Color32 = Color32::from_rgb(255, 84, 78);

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
    visuals.widgets.inactive.corner_radius = CornerRadius::same(3);
    visuals.widgets.hovered.bg_fill = COLOR_BTN_HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, COLOR_CYAN);
    visuals.widgets.active.bg_fill = COLOR_BTN_HOVER;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, COLOR_CYAN);
    ctx.set_visuals(visuals);

    let mut fonts = FontDefinitions::default();
    // 1. Try Georgia
    for path in &[
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/Library/Fonts/Georgia.ttf",
        "/usr/share/fonts/truetype/georgia/georgia.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(
                "Georgia".to_owned(),
                Arc::new(FontData::from_owned(bytes)),
            );
            fonts.families.entry(FontFamily::Name("Georgia".into())).or_default().push("Georgia".to_owned());
            break;
        }
    }

    // Fallback for Georgia family to ensure it never panics if missing
    if let Some(prop) = fonts.families.get(&FontFamily::Proportional).cloned() {
        fonts.families.entry(FontFamily::Name("Georgia".into())).or_default().extend(prop);
    }

    // 2. Try Monaco / Menlo for crisp monospace
    for path in &[
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(
                "CustomMono".to_owned(),
                Arc::new(FontData::from_owned(bytes)),
            );
            fonts.families.entry(FontFamily::Monospace).or_default().insert(0, "CustomMono".to_owned());
            break;
        }
    }

    ctx.set_fonts(fonts);
}
