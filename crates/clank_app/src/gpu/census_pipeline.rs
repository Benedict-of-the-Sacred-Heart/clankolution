//! The living census (`census.wgsl`): population, highest generation and distinct living
//! lineages, recounted after the birth pass on a frame's final sub-tick.

pub struct CensusPass {
    clear_pipeline: wgpu::ComputePipeline,
    count_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _root_table: wgpu::Buffer,
    clear_workgroups: u32,
    soil_workgroups: u32,
}

impl CensusPass {
    pub fn new(
        device: &wgpu::Device,
        queue_buffer: &wgpu::Buffer,
        agent_states: &wgpu::Buffer,
        agent_atomics: &wgpu::Buffer,
        soil_buf: &wgpu::Buffer,
        soil_cells: u32,
        max_agents: u32,
    ) -> Self {
        // Every living creature may head its own lineage, so the root set gets at least twice
        // as many entries as there are slots: it never fills and its probe chains stay short.
        let table_len = (2 * max_agents).next_power_of_two().max(64);
        let root_table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("census_root_table_buf"),
            size: table_len as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("census_bgl"),
            entries: &[
                entry(0, false),
                entry(1, true),
                entry(2, true),
                entry(3, false),
                entry(4, true),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("census_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: queue_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(128),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: agent_atomics.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: root_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: soil_buf.as_entire_binding(),
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("census_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/census.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("census_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            clear_pipeline: pipeline("census_clear_pipeline", "census_clear_main"),
            count_pipeline: pipeline("census_pipeline", "census_main"),
            bind_group,
            _root_table: root_table,
            clear_workgroups: table_len / 64,
            soil_workgroups: (soil_cells + 63) / 64,
        }
    }

    /// Clears the census and recounts it over `workgroups` × 64 slots (and all soil cells).
    pub fn encode(&self, cpass: &mut wgpu::ComputePass<'_>, workgroups: u32) {
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.set_pipeline(&self.clear_pipeline);
        cpass.dispatch_workgroups(self.clear_workgroups, 1, 1);
        cpass.set_pipeline(&self.count_pipeline);
        cpass.dispatch_workgroups(workgroups.max(self.soil_workgroups), 1, 1);
    }
}
