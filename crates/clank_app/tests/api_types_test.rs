use clank_app::api::*;

#[test]
fn test_api_types_json_roundtrip() {
    let settings = ApiSettingsRequest {
        speed: Some(4.0),
        paused: Some(true),
        mutation: Some(0.25),
        growth: Some(1.5),
        hostility: Some(1.2),
        max_cap: Some(500),
        scroll_offset: Some(100.0),
        selected_agent: Some(42),
        active_engine: Some("gpu".to_string()),
    };
    let json = serde_json::to_string(&settings).expect("serialize settings");
    let deserialized: ApiSettingsRequest = serde_json::from_str(&json).expect("deserialize settings");
    assert_eq!(deserialized.speed, Some(4.0));
    assert_eq!(deserialized.mutation, Some(0.25));
    assert_eq!(deserialized.selected_agent, Some(42));
    assert_eq!(deserialized.active_engine, Some("gpu".to_string()));


    let tool = ApiToolRequest {
        tool: "nourish".to_string(),
        x: Some(120.0),
        y: Some(340.0),
    };
    let tool_json = serde_json::to_string(&tool).expect("serialize tool");
    let deserialized_tool: ApiToolRequest = serde_json::from_str(&tool_json).expect("deserialize tool");
    assert_eq!(deserialized_tool.tool, "nourish");
    assert_eq!(deserialized_tool.x, Some(120.0));

    let state = ApiStateResponse {
        tick: 100,
        paused: false,
        speed: 2,
        population: 50,
        max_capacity: 400,
        generation: 5,
        kills: 10,
        births: 40,
        roots: 8,
        eclipse: 0,
        active_tool: "none".to_string(),
        mutation: 0.15,
        growth: 1.0,
        hostility: 1.0,
        selected_agent: Some(42),
        active_engine: "rust".to_string(),
    };
    let state_json = serde_json::to_string(&state).expect("serialize state");
    let deserialized_state: ApiStateResponse = serde_json::from_str(&state_json).expect("deserialize state");
    assert_eq!(deserialized_state.population, 50);
    assert_eq!(deserialized_state.active_engine, "rust");

    let metrics = ApiMetricsResponse {
        fps: 60.0,
        frame_time_ms: 16.6,
        tick: 100,
        population: 50,
        uptime_secs: 1.5,
    };
    let metrics_json = serde_json::to_string(&metrics).expect("serialize metrics");
    let deserialized_metrics: ApiMetricsResponse = serde_json::from_str(&metrics_json).expect("deserialize metrics");
    assert_eq!(deserialized_metrics.fps, 60.0);
}
