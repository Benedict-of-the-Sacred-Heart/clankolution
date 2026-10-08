use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue, WgpuWrapper};
use clank_app::gpu::compute_driver::GpuComputeDriver;
use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};
use std::sync::Arc;

/// Binds `buffer` through Bevy's device. wgpu rejects a resource that belongs to another
/// device, so this only succeeds when the driver was built on Bevy's device.
fn bind_with(render_device: &RenderDevice, buffer: &wgpu::Buffer) -> Option<wgpu::Error> {
    let device = render_device.wgpu_device();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let _ = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    bevy::tasks::block_on(scope.pop())
}

/// Zero-copy presentation needs Bevy's renderer to read the simulation buffers in place,
/// so the app must build the compute driver on Bevy's `RenderDevice`.
#[test]
fn test_app_builds_gpu_driver_on_bevy_render_device() {
    let Some((device, queue)) = GpuComputeDriver::request_standalone_device() else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let render_device = RenderDevice::from(device);
    let render_queue = RenderQueue(Arc::new(WgpuWrapper::new(queue)));

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(ClankSimPlugin)
        .insert_resource(render_device.clone())
        .insert_resource(render_queue);
    app.world_mut().resource_mut::<SimWorld>().active_engine = ActiveEngine::Gpu;
    app.update();

    let gpu = app.world().resource::<GpuDriverResource>();
    let driver = gpu.driver.as_ref().expect("driver created");
    assert!(
        bind_with(&render_device, &driver.agent_states_buf).is_none(),
        "driver buffers must live on Bevy's render device"
    );
}

/// Rust mode needs the driver too: both engines are presented by the same GPU renderer.
#[test]
fn test_driver_exists_in_rust_mode() {
    let Some((device, queue)) = GpuComputeDriver::request_standalone_device() else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(ClankSimPlugin)
        .insert_resource(RenderDevice::from(device))
        .insert_resource(RenderQueue(Arc::new(WgpuWrapper::new(queue))));
    app.update();
    let sim = app.world().resource::<SimWorld>();
    assert_eq!(sim.active_engine, ActiveEngine::Rust);
    assert!(app.world().resource::<GpuDriverResource>().driver.is_some());
}

/// If the sim ran a frame before Bevy's renderer existed, the standalone driver must move onto
/// the render device once it appears (the GPU engine holds no live state yet).
#[test]
fn test_driver_moves_onto_render_device_when_it_appears() {
    let Some((device, queue)) = GpuComputeDriver::request_standalone_device() else {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    };
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ClankSimPlugin);
    app.update();
    assert!(
        !app.world()
            .resource::<GpuDriverResource>()
            .driver
            .as_ref()
            .unwrap()
            .on_render_device
    );

    let render_device = RenderDevice::from(device);
    app.insert_resource(render_device.clone())
        .insert_resource(RenderQueue(Arc::new(WgpuWrapper::new(queue))));
    app.update();
    let gpu = app.world().resource::<GpuDriverResource>();
    let driver = gpu.driver.as_ref().unwrap();
    assert!(driver.on_render_device);
    assert!(bind_with(&render_device, &driver.agent_states_buf).is_none());
}
