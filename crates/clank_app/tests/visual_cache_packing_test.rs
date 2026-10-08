use clank_app::gpu::bridge::sync_rust_to_gpu;
use clank_app::rendering::unpack_visual_cache_detailed;
use clank_app::sim::SimWorld;

/// Rust-mode creatures are drawn from `sync_rust_to_gpu`'s `visual_cache`, so it must carry what
/// clankolution.html draws (line 58) exactly as `agent_step.wgsl` packs it for the GPU engine:
/// the red border only when `a.attack>.5`, antennae of length `r*(1.1+a.signal)`, and the birth
/// halo `r+3+(95-a.birth)*.12` faded by `a.birth/120`, which needs the full birth countdown.
#[test]
fn test_rust_visual_cache_matches_html_and_gpu_packing() {
    let mut sim = SimWorld::new(42);
    {
        let a = &mut sim.world.agents[0];
        a.attack = 0.4;
        a.signal = 0.3;
        a.birth = 50;
    }
    {
        let a = &mut sim.world.agents[1];
        a.attack = 0.6;
        a.signal = 0.0;
        a.birth = 0;
    }
    let (states, ..) = sync_rust_to_gpu(&sim);

    let v0 = unpack_visual_cache_detailed(
        states[0].visual_cache,
        clank_app::gpu::bridge::lineage_packed_color(states[0].root),
    )
    .unwrap();
    assert!(
        !v0.is_attacking,
        "attack 0.4 is not drawn as attacking (html: a.attack>.5)"
    );
    assert_eq!(
        v0.birth_val, 50,
        "the halo needs the birth countdown, not a flag"
    );
    assert_eq!(
        v0.signal,
        (0.3f32 * 255.0) as u32 as f32 / 255.0,
        "bits 8..15 carry signal"
    );

    let v1 = unpack_visual_cache_detailed(
        states[1].visual_cache,
        clank_app::gpu::bridge::lineage_packed_color(states[1].root),
    )
    .unwrap();
    assert!(v1.is_attacking, "attack 0.6 is drawn as attacking");
    assert_eq!(v1.birth_val, 0);
    assert_eq!(v1.signal, 0.0, "attack does not leak into the signal bits");
}
