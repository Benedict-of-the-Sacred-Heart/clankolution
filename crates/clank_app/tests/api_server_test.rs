use clank_app::api::*;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[test]
fn test_http_api_endpoints() {
    let port = 9336;
    let (_server, rx, shared) = create_test_api_server(port);
    {
        let mut data = shared.write().unwrap();
        data.metrics.update(120.0, 8.33, 100, 50, 2.0);
        data.state = Some(ApiStateResponse {
            tick: 100,
            paused: false,
            speed: 1,
            population: 50,
            max_capacity: 400,
            generation: 1,
            kills: 0,
            births: 50,
            roots: 10,
            eclipse: 0,
            active_tool: "none".to_string(),
            mutation: 0.15,
            growth: 1.0,
            hostility: 1.0,
            selected_agent: None,
            active_engine: "rust".to_string(),
        });
    }

    // 1. Request GET /metrics
    {
        let mut stream =
            TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /metrics");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains("\"fps\""));
        assert!(response.contains("120.0"));
    }

    // 2. Request GET /state
    {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /state");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .write_all(b"GET /state HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains("\"population\":50") || response.contains("\"population\": 50"));
    }

    // 3. Request POST /settings
    {
        let payload = r#"{"speed":4.0,"paused":true}"#;
        let mut stream =
            TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /settings");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let request = format!(
            "POST /settings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream.write_all(request.as_bytes()).expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        assert!(response.contains("HTTP/1.1 200 OK"));

        let cmd = rx
            .recv_timeout(Duration::from_secs(1))
            .expect("receive command");
        match cmd {
            ApiCommand::UpdateSettings(s) => {
                assert_eq!(s.speed, Some(4.0));
                assert_eq!(s.paused, Some(true));
            }
            _ => panic!("unexpected command variant"),
        }
    }
}

#[test]
fn test_concurrent_http_requests() {
    let port = 9337;
    let (_server, _rx, shared) = create_test_api_server(port);
    shared
        .write()
        .unwrap()
        .metrics
        .update(100.0, 10.0, 1, 10, 1.0);

    // Stream 1 connects but does not send data (hanging connection)
    let _slow_stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect slow");

    // Stream 2 connects and immediately queries /metrics
    let start = std::time::Instant::now();
    let mut stream2 = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect stream 2");
    stream2
        .set_read_timeout(Some(Duration::from_millis(500)))
        .unwrap();
    stream2
        .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .expect("write");
    let mut response = String::new();
    let res = stream2.read_to_string(&mut response);
    let elapsed = start.elapsed();

    assert!(res.is_ok(), "Stream 2 timed out or failed: {:?}", res);
    assert!(response.contains("HTTP/1.1 200 OK"));
    assert!(
        elapsed < Duration::from_millis(500),
        "Concurrent request blocked for {:?}",
        elapsed
    );
}

#[test]
fn test_settings_active_engine_switching() {
    let port = 9338;
    let (_server, rx, _shared) = create_test_api_server(port);

    let payload = r#"{"active_engine":"gpu"}"#;
    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /settings");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let request = format!(
        "POST /settings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream.write_all(request.as_bytes()).expect("write");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read");
    assert!(response.contains("HTTP/1.1 200 OK"));

    let cmd = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("receive command");
    match cmd {
        ApiCommand::UpdateSettings(s) => {
            assert_eq!(s.active_engine.as_deref(), Some("gpu"));
        }
        _ => panic!("unexpected command variant"),
    }
}

/// `/state`'s generation is the panel's: the highest living generation, from the engine that
/// owns the world (the GPU's census in GPU mode, where the CPU copy is not kept current).
#[test]
fn test_state_generation_comes_from_the_engine_that_owns_the_world() {
    use bevy::prelude::*;
    use clank_app::sim::{ActiveEngine, SimWorld};
    use std::sync::{Arc, RwLock};

    let shared = Arc::new(RwLock::new(SharedApiData::default()));
    let mut app = App::new();
    app.insert_resource(SimWorld::new(42))
        .insert_resource(SharedApiResource(shared.clone()))
        .add_systems(Update, api_state_sync_system);
    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Gpu;
        sim.gpu_max_generation = 7;
    }
    app.update();
    let generation = shared.read().unwrap().state.as_ref().unwrap().generation;
    assert_eq!(generation, 7);

    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.active_engine = ActiveEngine::Rust;
        sim.world.agents[0].gen = 12;
        sim.world.agents[1].gen = 30;
        sim.world.agents[1].dead = 1;
    }
    app.update();
    let generation = shared.read().unwrap().state.as_ref().unwrap().generation;
    assert_eq!(generation, 12, "dead creatures do not count");
}

#[test]
fn test_gpu_settings_selected_agent_beyond_500_and_invalid_id() {
    use bevy::prelude::*;
    use clank_app::sim::{ActiveEngine, ClankSimPlugin, GpuDriverResource, SimWorld};
    use std::sync::{mpsc, Arc, Mutex};

    let mut app = App::new();
    let (tx, rx) = mpsc::channel::<ApiCommand>();
    app.add_plugins(MinimalPlugins)
        .add_plugins(ClankSimPlugin)
        .insert_resource(ApiSenderResource(tx.clone()))
        .insert_resource(ApiReceiverResource(Arc::new(Mutex::new(rx))))
        .add_systems(Update, api_dispatch_system);

    {
        let mut sim = app.world_mut().resource_mut::<SimWorld>();
        sim.paused = true;
        sim.world.set_max_capacity(700);
        sim.world.agents.clear();
        for idx in 0..601u32 {
            let mut a = clank_core::agent::AgentData::default();
            a.id = 1000 + idx;
            a.dead = if idx == 600 { 0 } else { 1 };
            a.energy = 100.0;
            sim.world.agents.push(a);
        }
        sim.active_engine = ActiveEngine::Gpu;
    }
    app.update();
    if app.world().resource::<GpuDriverResource>().driver.is_none() {
        eprintln!("Skipping test: No GPU adapter available");
        return;
    }

    // Select creature at slot 600 (id = 1600), which is beyond the old 500-slot cap.
    tx.send(ApiCommand::UpdateSettings(ApiSettingsRequest {
        selected_agent: Some(1600),
        ..Default::default()
    }))
    .unwrap();
    app.update();

    {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(sim.selected_agent_id, Some(1600));
        assert_eq!(
            sim.selected_agent_slot,
            Some(600),
            "must resolve actual buffer slot beyond 500 rather than falling back to agent_id"
        );
    }

    // Selecting a non-existent creature ID must clear selection rather than setting slot = id.
    tx.send(ApiCommand::UpdateSettings(ApiSettingsRequest {
        selected_agent: Some(999_999),
        ..Default::default()
    }))
    .unwrap();
    app.update();

    {
        let sim = app.world().resource::<SimWorld>();
        assert_eq!(sim.selected_agent_id, None);
        assert_eq!(sim.selected_agent_slot, None);
    }
}
