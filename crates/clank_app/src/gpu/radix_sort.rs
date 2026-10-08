//! The spatial sort: a stable 4-pass LSD radix sort of the `(morton, slot)` keys that
//! `morton_encode` writes, and the 9×6 grid cell table and `cell_slots`, in 8 dispatches
//! (`assets/shaders/radix_sort.wgsl`, which documents the `spatial_keys` layout).

/// Grid cells in the cell table (9 × 6).
pub const GRID_CELLS: u64 = 54;
/// Keys per sort workgroup (one per thread).
const WG: u32 = 256;

/// Bytes `spatial_keys` needs for `max_agents` slots: sorted keys, cell table, `cell_slots`.
pub fn spatial_keys_size(max_agents: u32) -> u64 {
    (2 * max_agents as u64 + GRID_CELLS) * 8
}

pub struct SpatialSort {
    bind_group: wgpu::BindGroup,
    count: [wgpu::ComputePipeline; 4],
    scatter: [wgpu::ComputePipeline; 4],
    num_wg: u32,
    // Kept alive for the bind group.
    _scratch: wgpu::Buffer,
    _hist: wgpu::Buffer,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SpatialSort {
    pub fn new(
        device: &wgpu::Device,
        agent_states: &wgpu::Buffer,
        spatial_keys: &wgpu::Buffer,
        queue_buffer: &wgpu::Buffer,
        sim_params: &wgpu::Buffer,
        max_agents: u32,
    ) -> Self {
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_scratch_buf"),
            size: max_agents as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let num_wg = max_agents.div_ceil(WG);
        // [hist: 256 * num_wg][cell counts: 54][cell fill: 54]; zeroed at creation, and the
        // cell counters are cleared again by the last pass of every sort.
        let hist = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sort_hist_buf"),
            size: (WG as u64 * num_wg as u64 + 2 * GRID_CELLS) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix_sort_bgl"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radix_sort_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: sim_params.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radix_sort_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix_sort_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/radix_sort.wgsl").into(),
            ),
        });
        let pipeline = |entry: &str, pass: u32| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("PASS", pass as f64)],
                    // Every workgroup variable is written before it is read (see the shader).
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        Self {
            count: std::array::from_fn(|p| pipeline("radix_count", p as u32)),
            scatter: std::array::from_fn(|p| pipeline("radix_scatter", p as u32)),
            bind_group,
            num_wg,
            _scratch: scratch,
            _hist: hist,
        }
    }

    /// Records the sort into `cpass`: it sorts the keys, counts `live_count`, and builds the
    /// cell table and `cell_slots`. `max_agents` is this sub-tick's `params.max_agents`. Each
    /// dispatch is its own WebGPU synchronization scope, so every step sees the previous
    /// one's writes.
    pub fn encode(&self, cpass: &mut wgpu::ComputePass<'_>, max_agents: u32) {
        let num_wg = max_agents.div_ceil(WG).min(self.num_wg);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        for p in 0..4 {
            cpass.set_pipeline(&self.count[p]);
            cpass.dispatch_workgroups(num_wg, 1, 1);
            cpass.set_pipeline(&self.scatter[p]);
            cpass.dispatch_workgroups(num_wg, 1, 1);
        }
    }
}
