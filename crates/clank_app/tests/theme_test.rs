use clank_app::theme::*;

#[test]
fn test_palette_and_colors() {
    assert_eq!(PALETTE.len(), 8);
    assert_eq!(PALETTE_40.len(), 8);
    // Verify Cyan accent (#5bc4b0 -> 91, 196, 176)
    assert_eq!(COLOR_CYAN.r(), 91);
    assert_eq!(COLOR_CYAN.g(), 196);
    assert_eq!(COLOR_CYAN.b(), 176);
    // Verify Gold accent (#cca85a -> 204, 168, 90)
    assert_eq!(COLOR_GOLD.r(), 204);
    assert_eq!(COLOR_GOLD.g(), 168);
    assert_eq!(COLOR_GOLD.b(), 90);
    // Verify Red accent (#ff544e -> 255, 84, 78)
    assert_eq!(COLOR_RED.r(), 255);
    assert_eq!(COLOR_RED.g(), 84);
    assert_eq!(COLOR_RED.b(), 78);
}

#[test]
fn test_setup_clank_theme() {
    let ctx = bevy_egui::egui::Context::default();
    setup_clank_theme(&ctx);
    let _ = ctx.run_ui(bevy_egui::egui::RawInput::default(), |_| {});
    let has_georgia = ctx.fonts(|f| f.families().contains(&bevy_egui::egui::FontFamily::Name("Georgia".into())));
    assert!(has_georgia);
}

#[test]
fn test_font_definitions() {
    let mut fonts = bevy_egui::egui::FontDefinitions::default();
    let data = vec![0u8; 10];
    let font_data = bevy_egui::egui::FontData::from_owned(data);
    fonts.font_data.insert("Georgia".to_owned(), std::sync::Arc::new(font_data));
    fonts.families.entry(bevy_egui::egui::FontFamily::Name("Georgia".into())).or_default().push("Georgia".to_owned());
    let _text = bevy_egui::egui::RichText::new("test").family(bevy_egui::egui::FontFamily::Name("Georgia".into()));
}
