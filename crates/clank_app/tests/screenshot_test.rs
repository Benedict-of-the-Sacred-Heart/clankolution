use clank_app::api::*;
use std::sync::mpsc::channel;

#[test]
fn test_screenshot_command_structure() {
    let (tx, _rx) = channel();
    let cmd = ApiCommand::TakeScreenshot {
        path: "scratch/test.png".to_string(),
        response_tx: tx,
    };
    match cmd {
        ApiCommand::TakeScreenshot { path, .. } => assert_eq!(path, "scratch/test.png"),
        _ => panic!("wrong command variant"),
    }
}

#[test]
fn test_save_screenshot_image_buffer() {
    let (tx, rx) = channel();
    let dynamic_img = image::DynamicImage::new_rgb8(100, 100);
    let path = "target/test_save_screenshot.png".to_string();
    let res = save_screenshot_to_disk(&dynamic_img, &path, tx);
    assert!(res.is_ok());
    let recv_res = rx.recv().expect("recv result");
    assert!(recv_res.is_ok());
    let saved = recv_res.unwrap();
    assert_eq!(saved.width, 100);
    assert_eq!(saved.height, 100);
    assert!(std::path::Path::new(&saved.path).exists());
    let _ = std::fs::remove_file(&saved.path);
}
