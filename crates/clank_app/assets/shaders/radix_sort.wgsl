// crates/clank_app/assets/shaders/radix_sort.wgsl
// Stable LSD radix sort of the (morton, slot) keys, and the 9x6 grid cell table.
//
// spatial_keys layout: [0, N) sorted keys (live first, dead 0xFFFFFFFF keys last),
// [N, N + 54) cell table (start, count) into cell_slots, empty cells (0xFFFFFFFF, 0xFFFFFFFF),
// [N + 54, 2N + 54) cell_slots (slot in .x). N = params.max_agents.
//
// Sort: 4 passes of 8 bits (PASS 0..3), two dispatches each. radix_count writes each
// 256-key workgroup's digit counts to hist[digit * num_wg + wg]. radix_scatter then works out
// its own offsets from hist (all workgroups' totals per digit, scanned over digits, plus the
// earlier workgroups' counts of each digit) and scatters stably: within a workgroup a key ranks
// after the earlier keys with the same digit. Keys start in slot order, so equal Morton codes
// stay in slot order. Passes 0 and 2 write the scratch buffer, 1 and 3 write back, so the
// result ends in spatial_keys. No workgroup waits on another (that can hang on Metal), and
// each dispatch costs the CPU about 0.1 ms to record in a debug build, so there are only 8.
//
// Grid: pass 0 also builds it from the unsorted keys. radix_count counts live creatures per
// cell and in live_count (zeroed by the preamble); radix_scatter scans the 54 cell counts,
// writes the cell table and places each live slot in cell_slots. Pass 3 clears the cell
// counters for the next sub-tick.
//
// The driver dispatches for params.max_agents; workgroups past it return at once. Every
// workgroup variable is written before it is read, so the pipelines skip zero-initialising
// workgroup memory (that cost about 120 us per dispatch on the D500).
struct GpuSimParams {
    tick: u32,
    agent_count: u32,
    max_agents: u32,
    max_capacity: u32,

    hostility: f32,
    mut_rate: f32,
    speed: f32,
    renewal: f32,

    sub_tick: u32,
    sub_ticks_per_frame: u32,
    tool_type: u32,
    tool_radius: f32,

    tool_pos: vec2f,
    camera_pos: vec2f,

    camera_size: vec2f,
    world_size: vec2f,

    soil_grid: vec2u,
    eclipse: u32,
    epoch: u32,
}

struct GpuAgentState {
    pos_vel: vec4f,
    angle_energy: vec4f,
    traits: array<vec4f, 2>,
    hidden: array<vec4f, 2>,
    hidden_tail: vec2f,
    id: u32,
    meta_flags: u32,
    age_gen: u32,
    morton_code: u32,
    root: u32,
    visual_cache: u32,
}

override PASS: u32 = 0u;

const SENTINEL: u32 = 0xFFFFFFFFu;
const WG: u32 = 256u;
const CELLS: u32 = 54u;
const LIVE_COUNT_WORD: u32 = 16u; // GpuTelemetry.live_count

@group(0) @binding(0) var<storage, read> agent_states: array<GpuAgentState>;
@group(0) @binding(1) var<storage, read_write> spatial_keys: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> sort_scratch: array<vec2u>;
// [hist: 256 * num_wg][cell counts: 54][cell fill: 54]
@group(0) @binding(3) var<storage, read_write> sort_hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> queue_words: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: GpuSimParams;

var<workgroup> wg_counts: array<atomic<u32>, 256>;
// Per digit, a 256-bit mask of the threads holding a key with that digit (8 words per digit).
var<workgroup> wg_masks: array<atomic<u32>, 2048>;
var<workgroup> wg_offsets: array<u32, 256>;
var<workgroup> wg_scan: array<u32, 256>;
var<workgroup> wg_group: array<u32, 16>;
var<workgroup> wg_cell: array<atomic<u32>, 54>;
var<workgroup> wg_cell_base: array<u32, 54>;

fn num_wg() -> u32 { return (params.max_agents + WG - 1u) / WG; }
fn cell_counts_base() -> u32 { return WG * num_wg(); }
fn cell_fill_base() -> u32 { return cell_counts_base() + CELLS; }

fn src_key(i: u32) -> vec2u {
    if ((PASS & 1u) == 0u) { return spatial_keys[i]; }
    return sort_scratch[i];
}

fn digit_of(key: u32) -> u32 { return (key >> (PASS * 8u)) & 0xFFu; }

fn cell_of(slot: u32) -> u32 {
    let pos = agent_states[slot].pos_vel.xy;
    let gx = min(u32(max(0.0, pos.x) / (params.world_size.x / 9.0)), 8u);
    let gy = min(u32(max(0.0, pos.y) / (params.world_size.y / 6.0)), 5u);
    return gy * 9u + gx;
}

// Exclusive scan of v over the workgroup: 16 threads each sum 16 values, one thread scans
// those 16 sums, and each thread adds at most 15 values.
fn exclusive_scan(t: u32, v: u32) -> u32 {
    workgroupBarrier();
    wg_scan[t] = v;
    workgroupBarrier();
    if (t < 16u) {
        var sum = 0u;
        for (var k = 0u; k < 16u; k += 1u) {
            sum += wg_scan[t * 16u + k];
        }
        wg_group[t] = sum;
    }
    workgroupBarrier();
    if (t == 0u) {
        var running = 0u;
        for (var g = 0u; g < 16u; g += 1u) {
            let sum = wg_group[g];
            wg_group[g] = running;
            running += sum;
        }
    }
    workgroupBarrier();
    let group = t / 16u;
    var result = wg_group[group];
    for (var k = group * 16u; k < t; k += 1u) {
        result += wg_scan[k];
    }
    return result;
}

@compute @workgroup_size(256)
fn radix_count(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    if (wg.x >= num_wg()) { return; }
    atomicStore(&wg_counts[lid.x], 0u);
    workgroupBarrier();
    let i = wg.x * WG + lid.x;
    if (i < params.max_agents) {
        let key = src_key(i);
        atomicAdd(&wg_counts[digit_of(key.x)], 1u);
        if (PASS == 0u && key.x != SENTINEL) {
            atomicAdd(&sort_hist[cell_counts_base() + cell_of(key.y)], 1u);
            atomicAdd(&queue_words[LIVE_COUNT_WORD], 1u);
        }
    }
    workgroupBarrier();
    atomicStore(&sort_hist[lid.x * num_wg() + wg.x], atomicLoad(&wg_counts[lid.x]));
}

@compute @workgroup_size(256)
fn radix_scatter(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    if (wg.x >= num_wg()) { return; }
    let t = lid.x;
    let n = params.max_agents;

    // Thread t handles digit t: all workgroups' keys with digit t, and the earlier ones'.
    var total = 0u;
    var before = 0u;
    for (var w = 0u; w < num_wg(); w += 1u) {
        let c = atomicLoad(&sort_hist[t * num_wg() + w]);
        total += c;
        before += select(0u, c, w < wg.x);
    }
    wg_offsets[t] = exclusive_scan(t, total) + before;

    // A key ranks after this workgroup's earlier keys with the same digit: count the set bits
    // below thread t in its digit's mask (a loop over all earlier threads cost about 60 us).
    for (var w = 0u; w < 8u; w += 1u) {
        atomicStore(&wg_masks[t * 8u + w], 0u);
    }
    let i = wg.x * WG + t;
    var key = vec2u(SENTINEL, SENTINEL);
    var d = 0u;
    if (i < n) {
        key = src_key(i);
        d = digit_of(key.x);
    }
    workgroupBarrier();
    if (i < n) {
        atomicOr(&wg_masks[d * 8u + t / 32u], 1u << (t % 32u));
    }
    workgroupBarrier();
    if (i < n) {
        var rank = countOneBits(atomicLoad(&wg_masks[d * 8u + t / 32u]) & ((1u << (t % 32u)) - 1u));
        for (var w = 0u; w < t / 32u; w += 1u) {
            rank += countOneBits(atomicLoad(&wg_masks[d * 8u + w]));
        }
        let dst = wg_offsets[d] + rank;
        if ((PASS & 1u) == 0u) {
            sort_scratch[dst] = key;
        } else {
            spatial_keys[dst] = key;
        }
    }

    if (PASS == 0u) {
        // Grid: every workgroup scans the 54 cell counts; workgroup 0 writes the table. Places
        // in a cell are taken per workgroup (one global atomic per cell, not per creature).
        var count = 0u;
        if (t < CELLS) {
            count = atomicLoad(&sort_hist[cell_counts_base() + t]);
            atomicStore(&wg_cell[t], 0u);
        }
        let start = exclusive_scan(t, count);
        wg_offsets[t] = start;
        if (wg.x == 0u && t < CELLS) {
            if (count == 0u) {
                spatial_keys[n + t] = vec2u(SENTINEL, SENTINEL);
            } else {
                spatial_keys[n + t] = vec2u(start, count);
            }
        }
        let live = i < n && key.x != SENTINEL;
        var c = 0u;
        var local = 0u;
        if (live) {
            c = cell_of(key.y);
            local = atomicAdd(&wg_cell[c], 1u);
        }
        workgroupBarrier();
        if (t < CELLS) {
            let k = atomicLoad(&wg_cell[t]);
            if (k > 0u) {
                wg_cell_base[t] = atomicAdd(&sort_hist[cell_fill_base() + t], k);
            }
        }
        workgroupBarrier();
        if (live) {
            let at = wg_offsets[c] + wg_cell_base[c] + local;
            spatial_keys[n + CELLS + at] = vec2u(key.y, 0u);
        }
    }
    if (PASS == 3u && wg.x == 0u && t < CELLS) {
        atomicStore(&sort_hist[cell_counts_base() + t], 0u);
        atomicStore(&sort_hist[cell_fill_base() + t], 0u);
    }
}
