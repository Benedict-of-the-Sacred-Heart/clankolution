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
