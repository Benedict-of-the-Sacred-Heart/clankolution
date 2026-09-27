use serde::{Deserialize, Serialize};
use std::sync::mpsc::{Receiver, Sender};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApiSettingsRequest {
    pub speed: Option<f32>,
    pub paused: Option<bool>,
    pub mutation: Option<f64>,
    pub growth: Option<f64>,
    pub hostility: Option<f64>,
    pub max_cap: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiToolRequest {
    pub tool: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResetRequest {
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiStateResponse {
    pub tick: u32,
    pub paused: bool,
    pub speed: u32,
    pub population: usize,
    pub max_capacity: usize,
    pub generation: u32,
    pub kills: u32,
    pub births: u32,
    pub roots: u32,
    pub eclipse: u32,
    pub active_tool: String,
    pub mutation: f64,
    pub growth: f64,
    pub hostility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMetricsResponse {
    pub fps: f64,
    pub frame_time_ms: f64,
    pub tick: u32,
    pub population: usize,
    pub uptime_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LiveMetrics {
    pub fps: f64,
    pub frame_time_ms: f64,
    pub tick: u32,
    pub population: usize,
    pub uptime_secs: f64,
}

impl LiveMetrics {
    pub fn update(&mut self, fps: f64, frame_time_ms: f64, tick: u32, population: usize, uptime_secs: f64) {
        self.fps = fps;
        self.frame_time_ms = frame_time_ms;
        self.tick = tick;
        self.population = population;
        self.uptime_secs = uptime_secs;
    }
}

#[derive(Debug, Clone, Default)]
pub struct SharedApiData {
    pub metrics: LiveMetrics,
    pub state: Option<ApiStateResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotResult {
    pub path: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub enum ApiCommand {
    TakeScreenshot {
        path: String,
        response_tx: Sender<Result<ScreenshotResult, String>>,
    },
    UpdateSettings(ApiSettingsRequest),
    ApplyTool(ApiToolRequest),
    Reset {
        seed: u64,
    },
    PersistSave {
        path: String,
        response_tx: Sender<Result<usize, String>>,
    },
    PersistLoad {
        path: String,
        response_tx: Sender<Result<(), String>>,
    },
}

pub type ApiCommandSender = Sender<ApiCommand>;
pub type ApiCommandReceiver = Receiver<ApiCommand>;

pub struct ApiServerHandle {
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    port: u16,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for ApiServerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, std::sync::atomic::Ordering::SeqCst);
        // Connect to unblock listener
        let _ = std::net::TcpStream::connect(format!("127.0.0.1:{}", self.port));
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

pub fn create_test_api_server(
    port: u16,
) -> (
    ApiServerHandle,
    Receiver<ApiCommand>,
    std::sync::Arc<std::sync::RwLock<SharedApiData>>,
) {
    let (tx, rx) = std::sync::mpsc::channel();
    let shared = std::sync::Arc::new(std::sync::RwLock::new(SharedApiData::default()));
    let server = start_api_server(port, tx, shared.clone()).expect("start test api server");
    (server, rx, shared)
}

pub fn start_api_server(
    port: u16,
    command_tx: Sender<ApiCommand>,
    shared_data: std::sync::Arc<std::sync::RwLock<SharedApiData>>,
) -> std::io::Result<ApiServerHandle> {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = shutdown.clone();

    let thread = std::thread::spawn(move || {
        while !shutdown_flag.load(Ordering::SeqCst) {
            let (mut stream, _) = match listener.accept() {
                Ok(conn) => conn,
                Err(_) => break,
            };

            if shutdown_flag.load(Ordering::SeqCst) {
                break;
            }

            let _ = stream.set_read_timeout(Some(Duration::from_secs(6)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(6)));

            let mut buffer = Vec::with_capacity(4096);
            let mut temp = [0u8; 1024];
            let mut header_end = None;

            loop {
                match stream.read(&mut temp) {
                    Ok(0) => break,
                    Ok(n) => {
                        buffer.extend_from_slice(&temp[..n]);
                        if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                            header_end = Some(pos);
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            let header_end_idx = match header_end {
                Some(idx) => idx,
                None => continue,
            };

            let header_bytes = &buffer[..header_end_idx];
            let header_str = String::from_utf8_lossy(header_bytes);
            let mut lines = header_str.lines();
            let req_line = match lines.next() {
                Some(line) => line,
                None => continue,
            };

            let parts: Vec<&str> = req_line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            let method = parts[0];
            let path = parts[1];

            let mut content_length = 0usize;
            for line in lines {
                let lower = line.to_lowercase();
                if lower.starts_with("content-length:") {
                    if let Some(val_str) = line.split(':').nth(1) {
                        content_length = val_str.trim().parse::<usize>().unwrap_or(0);
                    }
                }
            }

            let mut body_bytes = buffer[header_end_idx + 4..].to_vec();
            while body_bytes.len() < content_length {
                match stream.read(&mut temp) {
                    Ok(0) => break,
                    Ok(n) => {
                        body_bytes.extend_from_slice(&temp[..n]);
                    }
                    Err(_) => break,
                }
            }

            let (status, resp_body) = match (method, path) {
                ("GET", "/metrics") => {
                    let metrics = {
                        let lock = shared_data.read().unwrap();
                        lock.metrics.clone()
                    };
                    let resp = ApiMetricsResponse {
                        fps: metrics.fps,
                        frame_time_ms: metrics.frame_time_ms,
                        tick: metrics.tick,
                        population: metrics.population,
                        uptime_secs: metrics.uptime_secs,
                    };
                    ("200 OK", serde_json::to_string(&resp).unwrap_or_default())
                }
                ("GET", "/state") => {
                    let maybe_state = {
                        let lock = shared_data.read().unwrap();
                        lock.state.clone()
                    };
                    if let Some(state) = maybe_state {
                        ("200 OK", serde_json::to_string(&state).unwrap_or_default())
                    } else {
                        ("200 OK", "{}".to_string())
                    }
                }
                ("POST", "/settings") => {
                    match serde_json::from_slice::<ApiSettingsRequest>(&body_bytes) {
                        Ok(req) => {
                            let _ = command_tx.send(ApiCommand::UpdateSettings(req));
                            ("200 OK", r#"{"status":"ok"}"#.to_string())
                        }
                        Err(e) => (
                            "400 Bad Request",
                            format!(r#"{{"status":"error","message":"{}"}}"#, e),
                        ),
                    }
                }
                ("POST", "/tool") => {
                    match serde_json::from_slice::<ApiToolRequest>(&body_bytes) {
                        Ok(req) => {
                            let _ = command_tx.send(ApiCommand::ApplyTool(req));
                            ("200 OK", r#"{"status":"ok"}"#.to_string())
                        }
                        Err(e) => (
                            "400 Bad Request",
                            format!(r#"{{"status":"error","message":"{}"}}"#, e),
                        ),
                    }
                }
                ("POST", "/reset") => {
                    let req: Result<ApiResetRequest, _> = serde_json::from_slice(&body_bytes);
                    let seed = req.ok().and_then(|r| r.seed).unwrap_or_else(|| {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    });
                    let _ = command_tx.send(ApiCommand::Reset { seed });
                    ("200 OK", format!(r#"{{"status":"ok","seed":{}}}"#, seed))
                }
                ("POST", "/screenshot") => {
                    #[derive(Deserialize)]
                    struct ScreenshotReq {
                        path: Option<String>,
                    }
                    let req: Result<ScreenshotReq, _> = serde_json::from_slice(&body_bytes);
                    let path = req
                        .ok()
                        .and_then(|r| r.path)
                        .unwrap_or_else(|| "scratch/screenshot.png".to_string());

                    let (resp_tx, resp_rx) = std::sync::mpsc::channel();
                    let _ = command_tx.send(ApiCommand::TakeScreenshot {
                        path: path.clone(),
                        response_tx: resp_tx,
                    });

                    match resp_rx.recv_timeout(Duration::from_secs(5)) {
                        Ok(Ok(res)) => ("200 OK", serde_json::to_string(&res).unwrap_or_default()),
                        Ok(Err(err)) => (
                            "500 Internal Server Error",
                            format!(r#"{{"status":"error","message":"{}"}}"#, err),
                        ),
                        Err(_) => (
                            "504 Gateway Timeout",
                            r#"{"status":"error","message":"Screenshot capture timed out"}"#.to_string(),
                        ),
                    }
                }
                ("POST", "/persist") => {
                    #[derive(Deserialize)]
                    struct PersistReq {
                        action: String,
                        path: Option<String>,
                    }
                    match serde_json::from_slice::<PersistReq>(&body_bytes) {
                        Ok(req) => {
                            let path = req.path.unwrap_or_else(|| "snapshot.clank".to_string());
                            if req.action == "save" {
                                let (resp_tx, resp_rx) = std::sync::mpsc::channel();
                                let _ = command_tx.send(ApiCommand::PersistSave {
                                    path,
                                    response_tx: resp_tx,
                                });
                                match resp_rx.recv_timeout(Duration::from_secs(5)) {
                                    Ok(Ok(bytes)) => (
                                        "200 OK",
                                        format!(r#"{{"status":"ok","bytes":{}}}"#, bytes),
                                    ),
                                    Ok(Err(err)) => (
                                        "500 Internal Server Error",
                                        format!(r#"{{"status":"error","message":"{}"}}"#, err),
                                    ),
                                    Err(_) => (
                                        "504 Gateway Timeout",
                                        r#"{"status":"error","message":"Timeout"}"#.to_string(),
                                    ),
                                }
                            } else if req.action == "load" {
                                let (resp_tx, resp_rx) = std::sync::mpsc::channel();
                                let _ = command_tx.send(ApiCommand::PersistLoad {
                                    path,
                                    response_tx: resp_tx,
                                });
                                match resp_rx.recv_timeout(Duration::from_secs(5)) {
                                    Ok(Ok(())) => ("200 OK", r#"{"status":"ok"}"#.to_string()),
                                    Ok(Err(err)) => (
                                        "500 Internal Server Error",
                                        format!(r#"{{"status":"error","message":"{}"}}"#, err),
                                    ),
                                    Err(_) => (
                                        "504 Gateway Timeout",
                                        r#"{"status":"error","message":"Timeout"}"#.to_string(),
                                    ),
                                }
                            } else {
                                (
                                    "400 Bad Request",
                                    r#"{"status":"error","message":"invalid action"}"#.to_string(),
                                )
                            }
                        }
                        Err(e) => (
                            "400 Bad Request",
                            format!(r#"{{"status":"error","message":"{}"}}"#, e),
                        ),
                    }
                }
                _ => (
                    "404 Not Found",
                    r#"{"status":"error","message":"Not Found"}"#.to_string(),
                ),
            };

            let response = format!(
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status,
                resp_body.len(),
                resp_body
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    Ok(ApiServerHandle {
        shutdown,
        port,
        thread: Some(thread),
    })
}
