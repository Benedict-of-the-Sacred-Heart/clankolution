use clank_app::theme::*;

#[test]
fn test_palette_and_colors() {
    assert_eq!(PALETTE.len(), 8);
    assert_eq!(PALETTE_40.len(), 8);
    // Verify Cyan accent
    assert_eq!(COLOR_CYAN.r(), 124);
    assert_eq!(COLOR_CYAN.g(), 228);
    assert_eq!(COLOR_CYAN.b(), 213);
    // Verify Gold accent
    assert_eq!(COLOR_GOLD.r(), 230);
    assert_eq!(COLOR_GOLD.g(), 188);
    assert_eq!(COLOR_GOLD.b(), 120);
    // Verify Red accent
    assert_eq!(COLOR_RED.r(), 255);
    assert_eq!(COLOR_RED.g(), 117);
    assert_eq!(COLOR_RED.b(), 94);
}
