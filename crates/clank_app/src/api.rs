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
    pub scroll_offset: Option<f32>,
    pub selected_agent: Option<u32>,
    pub active_engine: Option<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiToolRequest {
    pub tool: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
    #[serde(default)]
    pub count: Option<usize>,
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
    #[serde(default)]
    pub selected_agent: Option<u32>,
    #[serde(default = "default_active_engine")]
    pub active_engine: String,
}

fn default_active_engine() -> String {
    "rust".to_string()
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
    pub fn update(&mut self, fps: f64, frame_time: f64, tick: u32, population: usize, uptime_secs: f64) {
        self.fps = fps;
        self.frame_time_ms = if frame_time <= 1.0 {
            frame_time * 1000.0
        } else {
            frame_time
        };
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
    #[serde(default = "default_status_ok")]
    pub status: String,
    pub path: String,
    pub width: u32,
    pub height: u32,
}

fn default_status_ok() -> String {
    "ok".to_string()
}

pub fn save_screenshot_to_disk(
    dyn_img: &image::DynamicImage,
    path_str: &str,
    response_tx: Sender<Result<ScreenshotResult, String>>,
) -> Result<ScreenshotResult, String> {
    let path = std::path::Path::new(path_str);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let img = dyn_img.to_rgb8();
    let width = img.width();
    let height = img.height();
    match img.save(path) {
        Ok(_) => {
            let full_path = std::fs::canonicalize(path)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path_str.to_string());
            let result = ScreenshotResult {
                status: "ok".to_string(),
                path: full_path,
                width,
                height,
            };
            let _ = response_tx.send(Ok(result.clone()));
            Ok(result)
        }
        Err(e) => {
            let err_msg = format!("Failed to save screenshot: {}", e);
            let _ = response_tx.send(Err(err_msg.clone()));
            Err(err_msg)
        }
    }
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
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = shutdown.clone();

    let thread = std::thread::spawn(move || {
        while !shutdown_flag.load(Ordering::SeqCst) {
            let (stream, _) = match listener.accept() {
                Ok(conn) => conn,
                Err(_) => break,
            };

            if shutdown_flag.load(Ordering::SeqCst) {
                break;
            }

            let command_tx = command_tx.clone();
            let shared_data = shared_data.clone();
            std::thread::spawn(move || {
                handle_http_connection(stream, command_tx, shared_data);
            });
        }
    });

    Ok(ApiServerHandle {
        shutdown,
        port,
        thread: Some(thread),
    })
}

fn handle_http_connection(
    mut stream: std::net::TcpStream,
    command_tx: Sender<ApiCommand>,
    shared_data: std::sync::Arc<std::sync::RwLock<SharedApiData>>,
) {
    use std::io::{Read, Write};
    use std::time::Duration;

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
        None => return,
    };

    let header_bytes = &buffer[..header_end_idx];
    let header_str = String::from_utf8_lossy(header_bytes);
    let mut lines = header_str.lines();
    let req_line = match lines.next() {
        Some(line) => line,
        None => return,
    };

    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
        return;
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

use bevy::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use crate::sim::SimWorld;
use crate::ui::{UiState, ActiveTool, trigger_spore_catastrophe};
use crate::persistence::{save_clank_file, load_clank_file};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Resource)]
pub struct ApiReceiverResource(pub Arc<Mutex<Receiver<ApiCommand>>>);

#[derive(Resource)]
pub struct ApiSenderResource(pub Sender<ApiCommand>);

#[derive(Resource)]
pub struct SharedApiResource(pub Arc<RwLock<SharedApiData>>);

#[derive(Resource)]
pub struct ApiServerResource(pub Option<ApiServerHandle>);

pub struct ClankApiPlugin {
    pub port: u16,
}

impl Default for ClankApiPlugin {
    fn default() -> Self {
        let port = std::env::var("CLANK_API_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(9335);
        Self { port }
    }
}

impl Plugin for ClankApiPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<FrameTimeDiagnosticsPlugin>() {
            app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        }

        let (tx, rx) = std::sync::mpsc::channel();
        let shared = Arc::new(RwLock::new(SharedApiData::default()));

        let server_handle = match start_api_server(self.port, tx.clone(), shared.clone()) {
            Ok(server) => {
                bevy::log::info!("Clankolution HTTP API listening on 127.0.0.1:{}", self.port);
                Some(server)
            }
            Err(e) => {
                bevy::log::warn!("Could not bind Clankolution HTTP API on port {}: {}", self.port, e);
                None
            }
        };

        app.insert_resource(ApiSenderResource(tx))
            .insert_resource(ApiReceiverResource(Arc::new(Mutex::new(rx))))
            .insert_resource(SharedApiResource(shared))
            .insert_resource(ApiServerResource(server_handle))
            .add_systems(
                Update,
                (api_dispatch_system, api_metrics_sync_system, api_state_sync_system),
            );
    }
}

pub fn api_dispatch_system(
    mut commands: Commands,
    rx_res: Option<Res<ApiReceiverResource>>,
    mut sim_res: Option<ResMut<SimWorld>>,
    mut ui_res: Option<ResMut<UiState>>,
    mut gpu_res: Option<ResMut<crate::sim::GpuDriverResource>>,
) {
    let Some(rx_res) = rx_res else { return; };
    let rx = match rx_res.0.lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };

    while let Ok(cmd) = rx.try_recv() {
        match cmd {
            ApiCommand::TakeScreenshot { path, response_tx } => {
                let resp_tx = response_tx.clone();
                let save_path = path.clone();
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(move |captured: On<ScreenshotCaptured>| {
                        match captured.image.clone().try_into_dynamic() {
                            Ok(dyn_img) => {
                                let _ = save_screenshot_to_disk(&dyn_img, &save_path, resp_tx.clone());
                            }
                            Err(e) => {
                                let _ = resp_tx.send(Err(format!("dynamic image conversion error: {}", e)));
                            }
                        }
                    });
            }
            ApiCommand::UpdateSettings(req) => {
                if let Some(ref mut sim) = sim_res {
                    if let Some(speed) = req.speed {
                        sim.speed = (speed.max(1.0).min(64.0)) as u32;
                    }
                    if let Some(paused) = req.paused {
                        sim.paused = paused;
                    }
                    if let Some(mut_rate) = req.mutation {
                        sim.world.mutation = if mut_rate <= 1.0 {
                            (mut_rate * 100.0).clamp(0.0, 50.0)
                        } else {
                            mut_rate.clamp(0.0, 50.0)
                        };
                    }
                    if let Some(growth) = req.growth {
                        sim.world.growth = if growth <= 2.0 {
                            (growth * 100.0).clamp(0.0, 200.0)
                        } else {
                            growth.clamp(0.0, 200.0)
                        };
                    }
                    if let Some(hostility) = req.hostility {
                        sim.world.hostility = if hostility <= 2.0 {
                            (hostility * 100.0).clamp(0.0, 200.0)
                        } else {
                            hostility.clamp(0.0, 200.0)
                        };
                    }
                    if let Some(cap) = req.max_cap {
                        sim.world.set_max_capacity(cap.clamp(50, 10_000) as u32);
                    }
                    if let Some(agent_id) = req.selected_agent {
                        if agent_id == 0 {
                            sim.selected_agent_id = sim.world.agents.iter().find(|a| a.dead == 0).map(|a| a.id);
                        } else {
                            sim.selected_agent_id = Some(agent_id);
                        }
                    }
                    if let Some(ref engine) = req.active_engine {
                        match engine.to_lowercase().as_str() {
                            "gpu" => sim.active_engine = crate::sim::ActiveEngine::Gpu,
                            "rust" | "cpu" => sim.active_engine = crate::sim::ActiveEngine::Rust,
                            _ => {}
                        }
                    }
                }

                if let Some(ref mut ui) = ui_res {
                    if let Some(speed) = req.speed {
                        ui.speed = speed.max(1.0).min(64.0);
                    }
                    if req.scroll_offset.is_some() {
                        ui.scroll_offset = req.scroll_offset;
                    }
                }
            }
            ApiCommand::ApplyTool(req) => {
                let tool_lower = req.tool.to_lowercase();
                if let Some(ref mut ui) = ui_res {
                    match tool_lower.as_str() {
                        "observe" | "inspect" | "select" => ui.active_tool = ActiveTool::Observe,
                        "nourish" => ui.active_tool = ActiveTool::Nourish,
                        "blight" => ui.active_tool = ActiveTool::Blight,
                        "seed" | "seedlife" => ui.active_tool = ActiveTool::SeedLife,
                        "extinguish" | "kill" => ui.active_tool = ActiveTool::Extinguish,
                        "eclipse" => ui.active_tool = ActiveTool::Eclipse,
                        _ => {}
                    }
                }
                if let Some(ref mut sim) = sim_res {
                    let x = req.x.unwrap_or(sim.world_width / 2.0);
                    let y = req.y.unwrap_or(sim.world_height / 2.0);
                    match tool_lower.as_str() {
                        "observe" | "inspect" | "select" => {
                            if let (Some(x), Some(y)) = (req.x, req.y) {
                                sim.selected_agent_id = crate::rendering::find_agent_at_position(&sim, Vec2::new(x as f32, y as f32), 25.0);
                            }
                        }
                        "nourish" => {
                            let repeat = req.count.unwrap_or(1).clamp(1, 10_000);
                            let (w, h) = (sim.world.w, sim.world.h);
                            for _ in 0..repeat {
                                let (nx, ny) = if repeat > 1 {
                                    (
                                        sim.world.prng.rand(20.0, w - 20.0),
                                        sim.world.prng.rand(20.0, h - 20.0),
                                    )
                                } else {
                                    (x, y)
                                };
                                sim.world.nourish_at(nx, ny);
                            }
                        }
                        "blight" => {
                            sim.world.blight_at(x, y);
                        }
                        "seed" | "seedlife" => {
                            let repeat = req.count.unwrap_or(1).clamp(1, 10_000);
                            let (w, h) = (sim.world.w, sim.world.h);
                            for _ in 0..repeat {
                                let (sx, sy) = if repeat > 1 {
                                    (
                                        sim.world.prng.rand(20.0, w - 20.0),
                                        sim.world.prng.rand(20.0, h - 20.0),
                                    )
                                } else {
                                    (x, y)
                                };
                                sim.world.seed_life_at(sx, sy);
                            }
                        }
                        "extinguish" | "kill" => {
                            sim.world.extinguish_at(x, y, 23.0);
                        }
                        "eclipse" => {
                            trigger_spore_catastrophe(sim);
                            if let Some(ref mut ui) = ui_res {
                                ui.add_chronicle(format!("{:05}  An eclipse consumes the harvest.", sim.world.tick));
                            }
                        }
                        _ => {}
                    }
                }
            }
            ApiCommand::Reset { seed } => {
                if let Some(ref mut sim) = sim_res {
                    if let Some(ref mut gpu) = gpu_res {
                        if let Some(ref d) = gpu.driver {
                            d.set_initialized(false);
                        }
                    }
                    sim.world = clank_core::world::World::new(seed as u32);
                    sim.selected_agent_id = None;
                    if let Some(ref mut ui) = ui_res {
                        ui.add_chronicle(format!("{:05}  The first hunger begins.", sim.world.tick));
                    }
                }
            }
            ApiCommand::PersistSave { path, response_tx } => {
                if let Some(ref mut sim) = sim_res {
                    if sim.active_engine == crate::sim::ActiveEngine::Gpu {
                        if let Some(ref gpu) = gpu_res {
                            if let Some(ref driver) = gpu.driver {
                                if driver.is_initialized() {
                                    crate::sim::flush_gpu_to_rust(sim, driver);
                                }
                            }
                        }
                    }
                    let res = save_clank_file(sim, &path).map_err(|e| e.to_string());
                    let _ = response_tx.send(res);
                } else {
                    let _ = response_tx.send(Err("Simulation not initialized".to_string()));
                }
            }
            ApiCommand::PersistLoad { path, response_tx } => {
                if let Some(ref mut sim) = sim_res {
                    let res = load_clank_file(sim, &path).map_err(|e| e.to_string());
                    if res.is_ok() {
                        if let Some(ref mut gpu) = gpu_res {
                            if let Some(ref d) = gpu.driver {
                                d.set_initialized(false);
                            }
                        }
                        if let Some(ref mut ui) = ui_res {
                            ui.add_chronicle(format!("{:05}  A world returns from its .clank record.", sim.world.tick));
                        }
                    }
                    let _ = response_tx.send(res);
                } else {
                    let _ = response_tx.send(Err("Simulation not initialized".to_string()));
                }
            }
        }
    }
}

pub fn api_state_sync_system(
    sim_res: Option<Res<SimWorld>>,
    ui_res: Option<Res<UiState>>,
    shared: Option<Res<SharedApiResource>>,
) {
    let (Some(sim), Some(shared)) = (sim_res, shared) else { return; };
    let ui_tool = ui_res
        .as_ref()
        .map(|u| format!("{:?}", u.active_tool))
        .unwrap_or_else(|| "Observe".to_string());

    let max_gen = sim.world.agents.iter().map(|a| a.gen).max().unwrap_or(0);
    let state = ApiStateResponse {
        tick: sim.world.tick,
        paused: sim.paused,
        speed: sim.speed,
        population: sim.world.agents.iter().filter(|a| a.dead == 0).count(),
        max_capacity: sim.world.max_cap,
        generation: max_gen as u32,
        kills: sim.world.kills,
        births: sim.world.births,
        roots: sim.world.roots,
        eclipse: sim.world.eclipse,
        active_tool: ui_tool,
        mutation: sim.world.mutation / 100.0,
        growth: sim.world.growth / 100.0,
        hostility: sim.world.hostility / 100.0,
        selected_agent: sim.selected_agent_id,
        active_engine: match sim.active_engine {
            crate::sim::ActiveEngine::Rust => "rust".to_string(),
            crate::sim::ActiveEngine::Gpu => "gpu".to_string(),
        },
    };
    if let Ok(mut lock) = shared.0.write() {
        lock.state = Some(state);
    };
}

pub fn api_metrics_sync_system(
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    sim_res: Option<Res<SimWorld>>,
    shared: Option<Res<SharedApiResource>>,
) {
    let Some(shared) = shared else { return; };
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed().or_else(|| d.average()).or_else(|| d.value()))
        .unwrap_or(0.0);
    let frame_time_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed().or_else(|| d.average()).or_else(|| d.value()))
        .unwrap_or(0.0);
    let (tick, pop) = if let Some(ref sim) = sim_res {
        (sim.world.tick, sim.world.agents.iter().filter(|a| a.dead == 0).count())
    } else {
        (0, 0)
    };
    if let Ok(mut lock) = shared.0.write() {
        lock.metrics.update(fps, frame_time_ms, tick, pop, time.elapsed_secs_f64());
    };
}
