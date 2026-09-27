use std::net::TcpStream;
use std::io::{Read, Write};
use std::time::Duration;
use clank_app::api::*;

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
        });
    }

    // 1. Request GET /metrics
    {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /metrics");
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        stream.write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains("\"fps\""));
        assert!(response.contains("120.0"));
    }

    // 2. Request GET /state
    {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /state");
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        stream.write_all(b"GET /state HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        assert!(response.contains("HTTP/1.1 200 OK"));
        assert!(response.contains("\"population\":50") || response.contains("\"population\": 50"));
    }

    // 3. Request POST /settings
    {
        let payload = r#"{"speed":4.0,"paused":true}"#;
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect /settings");
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let request = format!(
            "POST /settings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream.write_all(request.as_bytes()).expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        assert!(response.contains("HTTP/1.1 200 OK"));

        let cmd = rx.recv_timeout(Duration::from_secs(1)).expect("receive command");
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
    shared.write().unwrap().metrics.update(100.0, 10.0, 1, 10, 1.0);

    // Stream 1 connects but does not send data (hanging connection)
    let _slow_stream = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect slow");

    // Stream 2 connects and immediately queries /metrics
    let start = std::time::Instant::now();
    let mut stream2 = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("connect stream 2");
    stream2.set_read_timeout(Some(Duration::from_millis(500))).unwrap();
    stream2.write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").expect("write");
    let mut response = String::new();
    let res = stream2.read_to_string(&mut response);
    let elapsed = start.elapsed();

    assert!(res.is_ok(), "Stream 2 timed out or failed: {:?}", res);
    assert!(response.contains("HTTP/1.1 200 OK"));
    assert!(elapsed < Duration::from_millis(500), "Concurrent request blocked for {:?}", elapsed);
}
