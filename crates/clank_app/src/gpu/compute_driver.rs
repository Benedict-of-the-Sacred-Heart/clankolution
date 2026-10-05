//! GPU Compute Simulation Driver
//!
//! Owns the WGPU compute pipelines and VRAM storage buffers. Dispatches multi-tick
//! simulation passes directly onto GPU silicon via `dispatch_workgroups`.

use std::sync::Arc;
use crate::gpu::types::{
    GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuSimParams, GpuSoilCell, GpuTelemetry,
};

pub struct GpuComputeDriver {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,

    // Storage Buffers
    pub agent_states_buf: wgpu::Buffer,
    pub agent_genomes_buf: wgpu::Buffer,
    pub agent_atomics_buf: wgpu::Buffer,
    pub soil_buf: wgpu::Buffer,
    pub bloom_table_buf: wgpu::Buffer,
    pub spatial_keys_buf: wgpu::Buffer,
    pub lbvh_nodes_buf: wgpu::Buffer,
    pub node_flags_buf: wgpu::Buffer,
    pub cell_offsets_buf: wgpu::Buffer,
    pub freelist_buf: wgpu::Buffer,
    pub queue_buffer: wgpu::Buffer,

    // Uniform Buffers
    pub sim_params_buf: wgpu::Buffer,
    pub soil_params_buf: wgpu::Buffer,

    // Staging Buffers for readbacks
    pub staging_telemetry: wgpu::Buffer,
    pub staging_agent_states: wgpu::Buffer,
    pub staging_genomes: wgpu::Buffer,
    pub staging_soil: wgpu::Buffer,
    pub staging_atomics: wgpu::Buffer,

    // Compute Pipelines
    pub soil_pipeline: wgpu::ComputePipeline,
    pub morton_clear_pipeline: wgpu::ComputePipeline,
    pub morton_encode_pipeline: wgpu::ComputePipeline,
    pub morton_offsets_pipeline: wgpu::ComputePipeline,
    pub lbvh_pipeline: wgpu::ComputePipeline,
    pub agent_pipeline: wgpu::ComputePipeline,
    pub birth_pipeline: wgpu::ComputePipeline,

    // Bind Groups
    pub soil_bind_group: wgpu::BindGroup,
    pub morton_bind_group: wgpu::BindGroup,
    pub lbvh_bind_group: wgpu::BindGroup,
    pub agent_group0: wgpu::BindGroup,
    pub agent_group1: wgpu::BindGroup,
    pub birth_group0: wgpu::BindGroup,
    pub birth_group1: wgpu::BindGroup,

    pub max_agents: u32,
    pub soil_cols: u32,
    pub soil_rows: u32,
}

impl GpuComputeDriver {
    pub fn create_default() -> Option<Self> {
        Self::create_for_world(80, 63, 65536)
    }

    pub fn create_for_world(soil_cols: u32, soil_rows: u32, max_agents: u32) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok()?;

        let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("clank_gpu_compute_device"),
            ..Default::default()
        })).ok()?;

        Some(Self::new_with_grid(Arc::new(device), Arc::new(queue), max_agents, soil_cols, soil_rows))
    }

    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, max_agents: u32) -> Self {
        Self::new_with_grid(device, queue, max_agents, 80, 63)
    }

    pub fn new_with_grid(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        max_agents: u32,
        soil_cols: u32,
        soil_rows: u32,
    ) -> Self {
        use wgpu::BufferUsages;

        let total_soil_cells = (soil_cols * soil_rows) as u64;
        let agent_states_size = (max_agents as u64) * 128;
        let agent_genomes_size = (max_agents as u64) * 352;
        let agent_atomics_size = (max_agents as u64) * 16;
        let soil_size = total_soil_cells * 16;
        let bloom_size = total_soil_cells * 4;
        let spatial_keys_size = (max_agents as u64) * 8;
        let lbvh_nodes_size = (max_agents as u64) * 2 * 48;
        let node_flags_size = (max_agents as u64) * 4;
        let cell_offsets_size = 54 * 8;
        let freelist_size = (max_agents as u64) * 4;
        let queue_size = 1052800; // ConsolidatedQueue: 128B telemetry + 65536*16B births + 256*16B audio

        let agent_states_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_states_buf"),
            size: agent_states_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let agent_genomes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_genomes_buf"),
            size: agent_genomes_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let agent_atomics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agent_atomics_buf"),
            size: agent_atomics_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let soil_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil_buf"),
            size: soil_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bloom_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom_table_buf"),
            size: bloom_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bloom_table = vec![0.0f32; total_soil_cells as usize];
        for y in 0..soil_rows {
            let y_f = y as f32;
            for x in 0..soil_cols {
                let x_f = x as f32;
                let i = (y * soil_cols + x) as usize;
                bloom_table[i] = 0.0008 + 0.0022 * (0.5 + 0.5 * (x_f * 0.13 + (y_f * 0.19).sin()).sin() * (y_f * 0.11).cos());
            }
        }
        queue.write_buffer(&bloom_table_buf, 0, bytemuck::cast_slice(&bloom_table));

        let spatial_keys_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spatial_keys_buf"),
            size: spatial_keys_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lbvh_nodes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbvh_nodes_buf"),
            size: lbvh_nodes_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let node_flags_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node_flags_buf"),
            size: node_flags_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_offsets_buf"),
            size: cell_offsets_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let freelist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("freelist_buf"),
            size: freelist_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("queue_buffer"),
            size: queue_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_params_buf"),
            size: std::mem::size_of::<GpuSimParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let soil_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil_params_buf"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_telemetry = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_telemetry"),
            size: 128,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_agent_states = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_agent_states"),
            size: agent_states_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_genomes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_genomes"),
            size: agent_genomes_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_soil = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_soil"),
            size: soil_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_atomics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_atomics"),
            size: agent_atomics_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Soil Textures
        let soil_data_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("soil_data_tex"),
            size: wgpu::Extent3d { width: soil_cols, height: soil_rows, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let soil_data_view = soil_data_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let soil_display_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("soil_display_tex"),
            size: wgpu::Extent3d { width: soil_cols, height: soil_rows, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let soil_display_view = soil_display_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let soil_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("soil_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Compile Shader Modules
        let soil_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/soil_step.wgsl").into()),
        });
        let morton_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morton_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/morton_grid.wgsl").into()),
        });
        let lbvh_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbvh_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/lbvh_build.wgsl").into()),
        });
        let agent_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agent_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/agent_step.wgsl").into()),
        });
        let birth_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("birth_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/birth_step.wgsl").into()),
        });

        // 1. Soil Pipeline & Bind Group
        let soil_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let soil_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil_bg"),
            layout: &soil_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: soil_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&soil_data_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&soil_display_view) },
                wgpu::BindGroupEntry { binding: 3, resource: bloom_table_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: soil_params_buf.as_entire_binding() },
            ],
        });
        let soil_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil_pl"),
            bind_group_layouts: &[Some(&soil_bgl)],
            immediate_size: 0,
        });
        let soil_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil_pipeline"),
            layout: Some(&soil_pl),
            module: &soil_sm,
            entry_point: Some("soil_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 2. Morton Grid Pipelines & Bind Group
        let morton_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("morton_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let morton_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton_bg"),
            layout: &morton_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_states_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spatial_keys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sim_params_buf.as_entire_binding() },
            ],
        });
        let morton_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("morton_pl"),
            bind_group_layouts: &[Some(&morton_bgl)],
            immediate_size: 0,
        });
        let morton_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morton_clear"),
            layout: Some(&morton_pl),
            module: &morton_sm,
            entry_point: Some("clear_cell_offsets"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let morton_encode_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morton_encode"),
            layout: Some(&morton_pl),
            module: &morton_sm,
            entry_point: Some("morton_encode"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let morton_offsets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morton_offsets"),
            layout: Some(&morton_pl),
            module: &morton_sm,
            entry_point: Some("populate_cell_offsets"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 3. LBVH Pipeline & Bind Group
        let lbvh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbvh_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let lbvh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbvh_bg"),
            layout: &lbvh_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: spatial_keys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lbvh_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: node_flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: agent_states_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sim_params_buf.as_entire_binding() },
            ],
        });
        let lbvh_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbvh_pl"),
            bind_group_layouts: &[Some(&lbvh_bgl)],
            immediate_size: 0,
        });
        let lbvh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lbvh_pipeline"),
            layout: Some(&lbvh_pl),
            module: &lbvh_sm,
            entry_point: Some("build_lbvh_hierarchy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 4. Agent Pipeline & Bind Groups (Group 0: 8 storage buffers; Group 1: uniform + texture + sampler)
        let agent_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let agent_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_bg0"),
            layout: &agent_bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_states_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: agent_genomes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: agent_atomics_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: soil_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: spatial_keys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cell_offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: freelist_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: queue_buffer.as_entire_binding() },
            ],
        });

        let agent_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_bgl1"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let agent_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_bg1"),
            layout: &agent_bgl1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sim_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&soil_data_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&soil_sampler) },
            ],
        });

        let agent_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_pl"),
            bind_group_layouts: &[Some(&agent_bgl0), Some(&agent_bgl1)],
            immediate_size: 0,
        });
        let agent_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("agent_pipeline"),
            layout: Some(&agent_pl),
            module: &agent_sm,
            entry_point: Some("agent_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // 5. Birth Pipeline & Bind Groups
        let birth_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("birth_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let birth_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("birth_bg0"),
            layout: &birth_bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_states_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: agent_genomes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: agent_atomics_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: queue_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: freelist_buf.as_entire_binding() },
            ],
        });

        let birth_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("birth_bgl1"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let birth_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("birth_bg1"),
            layout: &birth_bgl1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sim_params_buf.as_entire_binding() },
            ],
        });

        let birth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("birth_pl"),
            bind_group_layouts: &[Some(&birth_bgl0), Some(&birth_bgl1)],
            immediate_size: 0,
        });
        let birth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("birth_pipeline"),
            layout: Some(&birth_pl),
            module: &birth_sm,
            entry_point: Some("birth_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            agent_states_buf,
            agent_genomes_buf,
            agent_atomics_buf,
            soil_buf,
            bloom_table_buf,
            spatial_keys_buf,
            lbvh_nodes_buf,
            node_flags_buf,
            cell_offsets_buf,
            freelist_buf,
            queue_buffer,
            sim_params_buf,
            soil_params_buf,
            staging_telemetry,
            staging_agent_states,
            staging_genomes,
            staging_soil,
            staging_atomics,
            soil_pipeline,
            morton_clear_pipeline,
            morton_encode_pipeline,
            morton_offsets_pipeline,
            lbvh_pipeline,
            agent_pipeline,
            birth_pipeline,
            soil_bind_group,
            morton_bind_group,
            lbvh_bind_group,
            agent_group0,
            agent_group1,
            birth_group0,
            birth_group1,
            max_agents,
            soil_cols,
            soil_rows,
        }
    }

    pub fn upload_state(
        &self,
        states: &[GpuAgentState],
        genomes: &[GpuAgentGenome],
        atomics: &[GpuAgentAtomic],
        soil: &[GpuSoilCell],
        params: &GpuSimParams,
    ) {
        if !states.is_empty() {
            self.queue.write_buffer(&self.agent_states_buf, 0, bytemuck::cast_slice(states));
        }
        let clear_tail = (states.len() + 512).min(self.max_agents as usize);
        if clear_tail > states.len() {
            let empty_count = clear_tail - states.len();
            let tombstone = GpuAgentState {
                pos_vel: [0.0; 4],
                angle_energy: [0.0; 4],
                traits: [0.0; 8],
                hidden: [0.0; 10],
                id: 0,
                meta_flags: 1 << 13,
                age_gen: 0,
                morton_code: 0,
                packed_color: 0,
                visual_cache: 0,
            };
            let empty_vec = vec![tombstone; empty_count];
            self.queue.write_buffer(&self.agent_states_buf, (states.len() * 128) as u64, bytemuck::cast_slice(&empty_vec));
        }
        if !genomes.is_empty() {
            self.queue.write_buffer(&self.agent_genomes_buf, 0, bytemuck::cast_slice(genomes));
        }
        if !atomics.is_empty() {
            self.queue.write_buffer(&self.agent_atomics_buf, 0, bytemuck::cast_slice(atomics));
        }
        if !soil.is_empty() {
            self.queue.write_buffer(&self.soil_buf, 0, bytemuck::cast_slice(soil));
        }
        self.queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(params));

        // Initialize tombstone freelist strictly up to params.max_capacity (lowest slots on top of stack)
        let cap = (params.max_capacity as usize).min(self.max_agents as usize);
        let mut free_slots: Vec<u32> = Vec::with_capacity(cap);
        if cap > states.len() {
            for i in (states.len()..cap).rev() {
                free_slots.push(i as u32);
            }
        }
        for (i, s) in states.iter().take(cap).enumerate().rev() {
            if (s.meta_flags & (1 << 13)) != 0 {
                free_slots.push(i as u32);
            }
        }
        let freelist_top = free_slots.len() as u32;
        if !free_slots.is_empty() {
            self.queue.write_buffer(&self.freelist_buf, 0, bytemuck::cast_slice(&free_slots));
        }
        self.queue.write_buffer(&self.queue_buffer, 28, bytemuck::bytes_of(&freelist_top));

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SoilParams {
            renewal: f32,
            width: u32,
            height: u32,
            pad: u32,
        }
        let sp = SoilParams { renewal: params.renewal, width: self.soil_cols, height: self.soil_rows, pad: 0 };
        self.queue.write_buffer(&self.soil_params_buf, 0, bytemuck::bytes_of(&sp));
    }

    pub fn dispatch_sub_ticks(&self, sub_ticks: u32, params: &GpuSimParams) {
        let mut cur_params = *params;

        for step in 0..sub_ticks {
            cur_params.sub_tick = step;
            cur_params.tick = params.tick + 1 + step;
            self.queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(&cur_params));

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_sim_sub_tick_encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_sim_pass"),
                    timestamp_writes: None,
                });

                // 1. Soil decay and diffusion
                cpass.set_pipeline(&self.soil_pipeline);
                cpass.set_bind_group(0, &self.soil_bind_group, &[]);
                cpass.dispatch_workgroups((self.soil_cols + 7) / 8, (self.soil_rows + 7) / 8, 1);

                // 2. Morton grid spatial hashing
                let active_slots = cur_params.max_agents.max(cur_params.agent_count);
                let agent_workgroups = (active_slots + 63) / 64;
                if agent_workgroups > 0 {
                    cpass.set_pipeline(&self.morton_clear_pipeline);
                    cpass.set_bind_group(0, &self.morton_bind_group, &[]);
                    cpass.dispatch_workgroups(1, 1, 1);

                    cpass.set_pipeline(&self.morton_encode_pipeline);
                    cpass.set_bind_group(0, &self.morton_bind_group, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    cpass.set_pipeline(&self.morton_offsets_pipeline);
                    cpass.set_bind_group(0, &self.morton_bind_group, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    // 3. LBVH construction
                    if cur_params.agent_count >= 2 {
                        cpass.set_pipeline(&self.lbvh_pipeline);
                        cpass.set_bind_group(0, &self.lbvh_bind_group, &[]);
                        cpass.dispatch_workgroups((cur_params.agent_count - 1 + 63) / 64, 1, 1);
                    }

                    // 4. Agent forward pass, kinematics & combat
                    cpass.set_pipeline(&self.agent_pipeline);
                    cpass.set_bind_group(0, &self.agent_group0, &[]);
                    cpass.set_bind_group(1, &self.agent_group1, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);

                    // 5. Decoupled birth step (parallel dispatch for up to 64 births per sub-tick)
                    cpass.set_pipeline(&self.birth_pipeline);
                    cpass.set_bind_group(0, &self.birth_group0, &[]);
                    cpass.set_bind_group(1, &self.birth_group1, &[]);
                    cpass.dispatch_workgroups(64, 1, 1);
                }
            }

            self.queue.submit([encoder.finish()]);
        }
    }

    pub fn readback_telemetry(&self) -> GpuTelemetry {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_telemetry_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.queue_buffer, 0, &self.staging_telemetry, 0, 128);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_telemetry.slice(0..128);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let telemetry = *bytemuck::from_bytes::<GpuTelemetry>(&data[0..128]);
        drop(data);
        self.staging_telemetry.unmap();

        telemetry
    }

    pub fn readback_agent_states(&self, count: usize) -> Vec<GpuAgentState> {
        let byte_len = (count * 128) as u64;
        if byte_len == 0 {
            return Vec::new();
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_states_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.agent_states_buf, 0, &self.staging_agent_states, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_agent_states.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let states = bytemuck::cast_slice::<u8, GpuAgentState>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_agent_states.unmap();

        states
    }

    pub fn readback_agent_genomes(&self, count: usize) -> Vec<GpuAgentGenome> {
        let byte_len = (count * 352) as u64;
        if byte_len == 0 {
            return Vec::new();
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_genomes_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.agent_genomes_buf, 0, &self.staging_genomes, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_genomes.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let genomes = bytemuck::cast_slice::<u8, GpuAgentGenome>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_genomes.unmap();

        genomes
    }

    pub fn readback_soil(&self) -> Vec<GpuSoilCell> {
        let total_cells = (self.soil_cols * self.soil_rows) as usize;
        let byte_len = (total_cells * 16) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_soil_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.soil_buf, 0, &self.staging_soil, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_soil.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let soil = bytemuck::cast_slice::<u8, GpuSoilCell>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_soil.unmap();

        soil
    }

    pub fn readback_atomics(&self, count: usize) -> Vec<GpuAgentAtomic> {
        let byte_len = (count * 16) as u64;
        if byte_len == 0 {
            return Vec::new();
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_atomics_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.agent_atomics_buf, 0, &self.staging_atomics, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_atomics.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let atomics = bytemuck::cast_slice::<u8, GpuAgentAtomic>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_atomics.unmap();

        atomics
    }
}
