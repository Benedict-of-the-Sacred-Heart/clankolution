//! Headless GPU work (tests, tools, benchmarks) runs on a card that drives no display when the
//! machine has one: long compute on the display's card freezes the whole screen (a Mac Pro has
//! two FirePro D500s, and only one drives the monitor).

use clank_app::gpu::compute_driver::GpuComputeDriver;

#[cfg(target_os = "macos")]
#[test]
fn test_standalone_device_avoids_the_display_card() {
    use objc2_metal::MTLDevice;
    let any_headless = objc2_metal::MTLCopyAllDevices()
        .iter()
        .any(|d| d.isHeadless());
    let Some((device, _queue)) = GpuComputeDriver::request_standalone_device() else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    // SAFETY: only reads a property of the raw Metal device.
    let headless = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
        .map(|d| d.raw_device().isHeadless())
        .expect("a Metal device");
    assert_eq!(
        headless, any_headless,
        "the standalone device is on a card that drives a display while a headless card exists"
    );
}
