//! Plan Task 1: "unit tests for GPU struct memory layouts and 16-byte WGSL alignment".
//!
//! The Rust structs are only half of a layout; the GPU reads the struct declared in each
//! shader. Every shader redeclares the shared structs by hand, so each copy is parsed with
//! naga (the compiler wgpu uses) and compared field by field with the Rust struct: every
//! 4-byte scalar's offset and kind (f32 / i32 / u32), and the total size.

#[path = "support/gpu_harness.rs"]
mod gpu_harness;

use clank_app::gpu::creature_render::CreatureRenderParams;
use clank_app::gpu::soil_pipeline::SoilParams;
use clank_app::gpu::types::*;
use clank_app::sim::SimWorld;
use gpu_harness::{distinct_roots, upload};
use std::collections::BTreeMap;
use std::mem::{align_of, offset_of, size_of};
use wgpu::naga;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    F32,
    I32,
    U32,
}

/// Byte offset -> scalar kind, for every 4-byte scalar in a struct.
type Layout = BTreeMap<u32, Kind>;
/// Byte offset -> name of the field that starts there.
type Names = BTreeMap<u32, &'static str>;

fn push(layout: &mut Layout, base: u32, offset: usize, count: usize, kind: Kind) {
    for i in 0..count {
        layout.insert(base + (offset + 4 * i) as u32, kind);
    }
}

fn nest(
    layout: &mut Layout,
    base: u32,
    offset: usize,
    count: usize,
    stride: usize,
    inner: &Layout,
) {
    for i in 0..count {
        for (o, k) in inner {
            layout.insert(base + (offset + i * stride) as u32 + o, *k);
        }
    }
}

/// `fields!(Type; name kind count, ...)`: the Rust layout from `offset_of!`.
macro_rules! fields {
    ($t:ty; $($f:ident $k:ident $n:expr),* $(,)?) => {{
        let mut l = Layout::new();
        let mut names = Names::new();
        $(
            push(&mut l, 0, offset_of!($t, $f), $n, Kind::$k);
            names.insert(offset_of!($t, $f) as u32, stringify!($f));
        )*
        (l, size_of::<$t>() as u32, names)
    }};
}

fn rust_layouts() -> BTreeMap<&'static str, (Layout, u32, Names)> {
    let mut m = BTreeMap::new();
    m.insert(
        "GpuSimParams",
        fields!(GpuSimParams;
            tick U32 1, agent_count U32 1, max_agents U32 1, max_capacity U32 1,
            hostility F32 1, mut_rate F32 1, speed F32 1, renewal F32 1,
            sub_tick U32 1, sub_ticks_per_frame U32 1, tool_type U32 1, tool_radius F32 1,
            tool_pos F32 2, camera_pos F32 2, camera_size F32 2, world_size F32 2,
            soil_grid U32 2, eclipse U32 1, epoch U32 1),
    );
    m.insert(
        "GpuAgentState",
        fields!(GpuAgentState;
            pos_vel F32 4, angle_energy F32 4, traits F32 8, hidden F32 10,
            id U32 1, meta_flags U32 1, age_gen U32 1, morton_code U32 1,
            root U32 1, visual_cache U32 1),
    );
    m.insert(
        "GpuAgentGenome",
        fields!(GpuAgentGenome; packed_genes U32 88),
    );
    m.insert(
        "GpuAgentAtomic",
        fields!(GpuAgentAtomic;
            energy_milli I32 1, mate_claim U32 1, mate_energy_milli U32 1, dead_claimed U32 1),
    );
    m.insert(
        "GpuSoilCell",
        fields!(GpuSoilCell;
            food_micro I32 1, taint_micro I32 1, scent_micro I32 1, fertility_micro I32 1),
    );
    m.insert(
        "BirthEvent",
        fields!(BirthEvent; parent_a U32 1, parent_b U32 1, child_slot U32 1, birth_tick U32 1),
    );
    m.insert(
        "AudioVoice",
        fields!(AudioVoice; pos F32 2, event_type U32 1, volume F32 1),
    );
    let telemetry = fields!(GpuTelemetry;
        population U32 1, kills U32 1, starvations U32 1, apex_record_milli U32 1,
        food_grazed_milli U32 1, sub_ticks_elapsed U32 1, apex_agent_id U32 1,
        freelist_top U32 1, birth_count U32 1, audio_voice_count U32 1,
        selected_agent_idx U32 1, selected_agent_id U32 1, total_births U32 1,
        total_deaths U32 1, max_generation U32 1, living_lineages U32 1,
        live_count U32 1, pending_free_count U32 1, available_slots U32 1, _reserved1 U32 13);
    m.insert("GpuTelemetry", telemetry.clone());
    {
        let mut l = Layout::new();
        nest(
            &mut l,
            0,
            offset_of!(ConsolidatedQueue, telemetry),
            1,
            0,
            &telemetry.0,
        );
        let birth = &m["BirthEvent"].0;
        nest(
            &mut l,
            0,
            offset_of!(ConsolidatedQueue, births),
            65536,
            16,
            birth,
        );
        let voice = &m["AudioVoice"].0;
        nest(
            &mut l,
            0,
            offset_of!(ConsolidatedQueue, audio),
            256,
            16,
            voice,
        );
        let names = Names::from([
            (offset_of!(ConsolidatedQueue, telemetry) as u32, "telemetry"),
            (offset_of!(ConsolidatedQueue, births) as u32, "births"),
            (offset_of!(ConsolidatedQueue, audio) as u32, "audio"),
        ]);
        m.insert(
            "ConsolidatedQueue",
            (l, size_of::<ConsolidatedQueue>() as u32, names),
        );
    }
    m.insert(
        "GpuLbvhNode",
        fields!(GpuLbvhNode;
            aabb_min F32 2, aabb_max F32 2, center_of_mass F32 2, count U32 1,
            dominant_lineage U32 1, left_child U32 1, right_child U32 1, parent U32 1,
            leaf_idx U32 1),
    );
    m.insert(
        "GpuAgentTrail",
        fields!(GpuAgentTrail; pts F32 18, count U32 1, owner_id U32 1, _pad U32 4),
    );
    m.insert(
        "GpuDartInstance",
        fields!(GpuDartInstance; pos_angle F32 3, pad0 F32 1, vis_data U32 2, pad1 U32 2),
    );
    m.insert(
        "CreatureRenderParams",
        fields!(CreatureRenderParams;
            clip_from_world F32 16, camera_pos F32 2, camera_size F32 2, world_size F32 2,
            slot_count U32 1, _pad U32 1),
    );
    m.insert(
        "SoilParams",
        fields!(SoilParams; renewal F32 1, width U32 1, height U32 1, decay_rate F32 1),
    );
    m
}

/// The Rust struct each shader struct name stands for.
fn rust_name(wgsl: &str) -> Option<&'static str> {
    Some(match wgsl {
        "GpuSimParams" => "GpuSimParams",
        "GpuAgentState" => "GpuAgentState",
        "GpuAgentGenome" => "GpuAgentGenome",
        "GpuAgentAtomic" | "AgentAtomicView" => "GpuAgentAtomic",
        "SoilCell" => "GpuSoilCell",
        "BirthEvent" => "BirthEvent",
        "AudioVoice" => "AudioVoice",
        "GpuTelemetry" => "GpuTelemetry",
        "ConsolidatedQueue" => "ConsolidatedQueue",
        "GpuLbvhNode" => "GpuLbvhNode",
        "GpuAgentTrail" => "GpuAgentTrail",
        "DartInstance" => "GpuDartInstance",
        "RenderParams" => "CreatureRenderParams",
        "SoilParams" => "SoilParams",
        _ => return None,
    })
}

fn kind_of(s: naga::Scalar) -> Kind {
    match (s.kind, s.width) {
        (naga::ScalarKind::Float, 4) => Kind::F32,
        (naga::ScalarKind::Sint, 4) => Kind::I32,
        (naga::ScalarKind::Uint, 4) => Kind::U32,
        other => panic!("unexpected scalar {other:?}"),
    }
}

/// The shader's layout of a type, flattened to 4-byte scalars at byte offsets.
fn flatten(module: &naga::Module, ty: naga::Handle<naga::Type>, base: u32, out: &mut Layout) {
    match &module.types[ty].inner {
        naga::TypeInner::Scalar(s) | naga::TypeInner::Atomic(s) => {
            out.insert(base, kind_of(*s));
        }
        naga::TypeInner::Vector { size, scalar } => {
            for i in 0..*size as u32 {
                out.insert(base + 4 * i, kind_of(*scalar));
            }
        }
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => {
            let column_stride = if *rows as u32 == 2 { 8 } else { 16 };
            for c in 0..*columns as u32 {
                for r in 0..*rows as u32 {
                    out.insert(base + c * column_stride + 4 * r, kind_of(*scalar));
                }
            }
        }
        naga::TypeInner::Array {
            base: elem,
            size,
            stride,
        } => {
            let naga::ArraySize::Constant(n) = size else {
                panic!("runtime-sized array inside a shared struct");
            };
            for i in 0..n.get() {
                flatten(module, *elem, base + i * stride, out);
            }
        }
        naga::TypeInner::Struct { members, .. } => {
            for m in members {
                flatten(module, m.ty, base + m.offset, out);
            }
        }
        other => panic!("unexpected type in a shared struct: {other:?}"),
    }
}

fn describe(layout: &Layout, offset: u32) -> String {
    layout
        .get(&offset)
        .map(|k| format!("{k:?}"))
        .unwrap_or_else(|| "padding".into())
}

/// Every shader's copy of every shared struct has the Rust layout: same scalar at every
/// offset and the same size.
#[test]
fn test_every_shader_struct_matches_its_rust_layout() {
    let rust = rust_layouts();
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/shaders");
    let mut shaders: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "wgsl"))
        .collect();
    shaders.sort();

    let mut problems = Vec::new();
    let mut checked = 0;
    for path in &shaders {
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let source = std::fs::read_to_string(path).unwrap();
        let module = match naga::front::wgsl::parse_str(&source) {
            Ok(m) => m,
            Err(e) => {
                problems.push(format!(
                    "{file}: does not parse: {}",
                    e.emit_to_string(&source)
                ));
                continue;
            }
        };
        for (handle, ty) in module.types.iter() {
            let (Some(name), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) else {
                continue;
            };
            let Some(rust_struct) = rust_name(name) else {
                continue;
            };
            let (want, want_size, want_names) = &rust[rust_struct];
            let mut got = Layout::new();
            flatten(&module, handle, 0, &mut got);
            checked += 1;

            if span != want_size {
                problems.push(format!(
                    "{file}: {name} is {span} bytes, Rust {rust_struct} is {want_size}"
                ));
            }
            let offsets: std::collections::BTreeSet<u32> =
                want.keys().chain(got.keys()).copied().collect();
            let wrong: Vec<String> = offsets
                .into_iter()
                .filter(|o| want.get(o) != got.get(o))
                .take(6)
                .map(|o| {
                    format!(
                        "byte {o}: shader {} vs Rust {}",
                        describe(&got, o),
                        describe(want, o)
                    )
                })
                .collect();
            if !wrong.is_empty() {
                problems.push(format!(
                    "{file}: {name} differs from Rust {rust_struct}: {}",
                    wrong.join("; ")
                ));
            }
            // Same-kind fields in swapped order keep every offset's kind, so also require
            // a field starting where a Rust field starts to carry the Rust field's name.
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                for m in members {
                    let (Some(got_name), Some(want_name)) =
                        (m.name.as_deref(), want_names.get(&m.offset))
                    else {
                        continue;
                    };
                    // Rust marks unused padding with a leading underscore.
                    if got_name != want_name.trim_start_matches('_') {
                        problems.push(format!(
                            "{file}: {name}.{got_name} at byte {} is Rust {rust_struct}.{want_name}",
                            m.offset
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 30, "only {checked} shader structs were checked");
    assert!(
        problems.is_empty(),
        "{} of {checked} shader struct copies differ from Rust:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// The Rust side of the plan's claim: every shared struct is 16-byte aligned with the sizes
/// the plan fixes (GpuSimParams is 96 bytes per the 2026-10-06 corrections).
#[test]
fn test_rust_structs_are_16_byte_aligned_with_planned_sizes() {
    let sizes = [
        (
            "GpuAgentState",
            size_of::<GpuAgentState>(),
            align_of::<GpuAgentState>(),
            128,
        ),
        (
            "GpuAgentGenome",
            size_of::<GpuAgentGenome>(),
            align_of::<GpuAgentGenome>(),
            352,
        ),
        (
            "GpuAgentAtomic",
            size_of::<GpuAgentAtomic>(),
            align_of::<GpuAgentAtomic>(),
            16,
        ),
        (
            "GpuSimParams",
            size_of::<GpuSimParams>(),
            align_of::<GpuSimParams>(),
            96,
        ),
        (
            "GpuLbvhNode",
            size_of::<GpuLbvhNode>(),
            align_of::<GpuLbvhNode>(),
            48,
        ),
        (
            "GpuSoilCell",
            size_of::<GpuSoilCell>(),
            align_of::<GpuSoilCell>(),
            16,
        ),
        (
            "BirthEvent",
            size_of::<BirthEvent>(),
            align_of::<BirthEvent>(),
            16,
        ),
        (
            "AudioVoice",
            size_of::<AudioVoice>(),
            align_of::<AudioVoice>(),
            16,
        ),
        (
            "GpuTelemetry",
            size_of::<GpuTelemetry>(),
            align_of::<GpuTelemetry>(),
            128,
        ),
        (
            "ConsolidatedQueue",
            size_of::<ConsolidatedQueue>(),
            align_of::<ConsolidatedQueue>(),
            1_052_800,
        ),
    ];
    for (name, size, align, want) in sizes {
        assert_eq!(size, want, "{name} size");
        assert_eq!(align, 16, "{name} alignment");
    }
}

/// Rust → GPU → Rust keeps every creature's id, lineage, generation and position, and the
/// 72 founding lineages stay 72 (the lineage root crosses the bridge at full width).
#[test]
fn test_72_lineages_survive_the_engine_round_trip() {
    let mut original = SimWorld::new(42);
    for (k, a) in original.world.agents.iter_mut().enumerate() {
        a.gen = k as u32 % 7;
    }
    assert_eq!(
        distinct_roots(&original.world.agents),
        72,
        "seed-42 founders; scenario is wrong"
    );
    let Some(gpu) = upload(&original) else {
        return;
    };
    let mut back = SimWorld::new(7);
    clank_app::sim::flush_gpu_to_rust(&mut back, &gpu.driver);
    let mut wrong = Vec::new();
    for a in &original.world.agents {
        match back.world.agents.iter().find(|b| b.id == a.id) {
            None => wrong.push(format!("creature {} lost", a.id)),
            Some(b) if b.root != a.root || b.gen != a.gen || (b.x - a.x).abs() > 1e-3 || (b.y - a.y).abs() > 1e-3 => wrong.push(format!(
                "creature {}: root {} gen {} at ({:.3}, {:.3}) came back as root {} gen {} at ({:.3}, {:.3})",
                a.id, a.root, a.gen, a.x, a.y, b.root, b.gen, b.x, b.y
            )),
            _ => {}
        }
    }
    let roots = distinct_roots(&back.world.agents);
    assert!(
        wrong.is_empty() && roots == 72,
        "{roots} lineages came back; {} creatures differ:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(8)].join("\n")
    );
}
