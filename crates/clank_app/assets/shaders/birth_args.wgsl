// crates/clank_app/assets/shaders/birth_args.wgsl
// Sets the birth pass's indirect dispatch to one workgroup per queued birth. Runs after the
// agent pass, in its own dispatch: the birth pass cannot bind the buffer it is dispatched from.

@group(0) @binding(0) var<storage, read> telemetry_words: array<u32, 32>;
@group(0) @binding(1) var<storage, read_write> birth_args: array<u32, 4>;

const BIRTH_COUNT_WORD: u32 = 8u; // GpuTelemetry.birth_count

@compute @workgroup_size(1)
fn birth_args_main() {
    // Births never exceed the free slots; 65535 is the per-dimension dispatch limit.
    birth_args[0] = min(telemetry_words[BIRTH_COUNT_WORD], 65535u);
    birth_args[1] = 1u;
    birth_args[2] = 1u;
}
