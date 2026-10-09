// crates/clank_app/assets/shaders/census.wgsl
// The living census, recounted on a frame's final sub-tick after the birth pass so it includes
// that sub-tick's newborns and excludes its dead: population, highest generation and the number
// of distinct full-width lineage roots. `census_clear_main` (one thread per table entry) clears
// the counters and the root table; `census_main` then visits every active slot.

@group(0) @binding(0) var<storage, read_write> telemetry_words: array<atomic<u32>, 32>;
// GpuAgentState as 32 words per slot, GpuAgentAtomic as 4.
@group(0) @binding(1) var<storage, read> agent_words: array<u32>;
@group(0) @binding(2) var<storage, read> atomic_words: array<u32>;
// Open-addressed set of `root + 1` (0 = empty): a power of two of at least twice the slots, so
// it never fills.
@group(0) @binding(3) var<storage, read_write> root_table: array<atomic<u32>>;

const POPULATION_WORD: u32 = 0u;       // GpuTelemetry.population
const MAX_GENERATION_WORD: u32 = 14u;  // GpuTelemetry.max_generation
const LIVING_LINEAGES_WORD: u32 = 15u; // GpuTelemetry.living_lineages

const STATE_WORDS: u32 = 32u;
const META_FLAGS_WORD: u32 = 27u;
const AGE_GEN_WORD: u32 = 28u;
const ROOT_WORD: u32 = 30u;
const DEAD_FLAG: u32 = 1u << 13u;
const DEAD_CLAIMED_WORD: u32 = 3u;

@compute @workgroup_size(64)
fn census_clear_main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < arrayLength(&root_table)) {
        atomicStore(&root_table[id.x], 0u);
    }
    if (id.x == 0u) {
        atomicStore(&telemetry_words[POPULATION_WORD], 0u);
        atomicStore(&telemetry_words[MAX_GENERATION_WORD], 0u);
        atomicStore(&telemetry_words[LIVING_LINEAGES_WORD], 0u);
    }
}

@compute @workgroup_size(64)
fn census_main(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    if (slot >= arrayLength(&agent_words) / STATE_WORDS) { return; }
    let base = slot * STATE_WORDS;
    // A creature killed this sub-tick is claimed dead before its own flag is set.
    if ((agent_words[base + META_FLAGS_WORD] & DEAD_FLAG) != 0u
        || atomic_words[slot * 4u + DEAD_CLAIMED_WORD] != 0u) {
        return;
    }

    atomicAdd(&telemetry_words[POPULATION_WORD], 1u);
    atomicMax(&telemetry_words[MAX_GENERATION_WORD], agent_words[base + AGE_GEN_WORD] >> 16u);

    // Insert root + 1 by linear probing; the thread that fills an empty entry counts the
    // lineage. The probe count is bounded as well, so the loop always ends.
    let table_size = arrayLength(&root_table);
    let key = agent_words[base + ROOT_WORD] + 1u;
    var h = (key * 2654435761u) >> (32u - countTrailingZeros(table_size));
    for (var probe = 0u; probe < 2u * table_size; probe++) {
        let r = atomicCompareExchangeWeak(&root_table[h], 0u, key);
        if (r.exchanged) {
            atomicAdd(&telemetry_words[LIVING_LINEAGES_WORD], 1u);
            break;
        }
        if (r.old_value == key) {
            break;
        }
        if (r.old_value != 0u) {
            h = (h + 1u) & (table_size - 1u);
        }
        // old_value == 0 without an exchange is a spurious failure: retry the same entry.
    }
}
