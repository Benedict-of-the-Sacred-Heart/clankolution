//! GPU Compute Simulation Driver
//!
//! Owns the WGPU compute pipelines and VRAM storage buffers. Dispatches multi-tick
//! simulation passes directly onto GPU silicon via `dispatch_workgroups`.

use std::sync::Arc;
use crate::gpu::types::{
    AudioVoice, GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuLbvhNode, GpuSimParams, GpuSoilCell, GpuTelemetry,
};

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DartVertex {
    pub position: [f32; 2],
    pub edge_flag: f32,
    pub _pad: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
    pub view_proj: [f32; 16],
    pub world_size: [f32; 2],
    pub _pad: [f32; 2],
}

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
    pub visible_instances_buf: wgpu::Buffer,
    pub cull_output_buf: wgpu::Buffer,
    pub dart_instances_buf: wgpu::Buffer,
    pub dart_template_buf: wgpu::Buffer,
    pub soil_display_tex: wgpu::Texture,

    // Uniform Buffers
    pub sim_params_buf: wgpu::Buffer,
    pub soil_params_buf: wgpu::Buffer,
    pub view_uniform_buf: wgpu::Buffer,

    // Staging Buffers for readbacks
    pub staging_telemetry: wgpu::Buffer,
    pub staging_agent_states: wgpu::Buffer,
    pub staging_genomes: wgpu::Buffer,
    pub staging_soil: wgpu::Buffer,
    pub staging_soil_display: wgpu::Buffer,
    pub staging_atomics: wgpu::Buffer,
    pub staging_cull_buf: wgpu::Buffer,
    pub staging_visible_instances: wgpu::Buffer,
    pub staging_dart_instances: wgpu::Buffer,
    pub staging_freelist: wgpu::Buffer,
    pub staging_audio: wgpu::Buffer,
    pub staging_lbvh: wgpu::Buffer,

    // Pipelines
    pub soil_pipeline: wgpu::ComputePipeline,
    pub preamble_pipeline: wgpu::ComputePipeline,
    pub morton_clear_pipeline: wgpu::ComputePipeline,
    pub morton_encode_pipeline: wgpu::ComputePipeline,
    pub morton_offsets_pipeline: wgpu::ComputePipeline,
    pub lbvh_pipeline: wgpu::ComputePipeline,
    pub lbvh_aabb_pipeline: wgpu::ComputePipeline,
    pub agent_pipeline: wgpu::ComputePipeline,
    pub agent_pipelines: [wgpu::ComputePipeline; 8],
    pub birth_pipeline: wgpu::ComputePipeline,
    pub birth_pipelines: [wgpu::ComputePipeline; 2],
    pub spatial_query_pipeline: wgpu::ComputePipeline,
    pub cull_clear_pipeline: wgpu::ComputePipeline,
    pub cull_pipeline: wgpu::ComputePipeline,
    pub dart_render_pipeline: wgpu::RenderPipeline,

    // Bind Groups
    pub soil_bind_group: wgpu::BindGroup,
    pub preamble_bg0: wgpu::BindGroup,
    pub preamble_bg1: wgpu::BindGroup,
    pub morton_bind_group: wgpu::BindGroup,
    pub lbvh_bind_group: wgpu::BindGroup,
    pub agent_group0: wgpu::BindGroup,
    pub agent_group1: wgpu::BindGroup,
    pub birth_group0: wgpu::BindGroup,
    pub birth_group1: wgpu::BindGroup,
    pub spatial_query_bind_group: wgpu::BindGroup,
    pub dart_render_bind_group: wgpu::BindGroup,

    pub max_agents: u32,
    pub soil_cols: u32,
    pub soil_rows: u32,
    pub initialized: std::sync::atomic::AtomicBool,
    pub next_agent_id: std::sync::atomic::AtomicU32,
    pub next_agent_root: std::sync::atomic::AtomicU32,
}

impl GpuComputeDriver {
    pub fn create_default() -> Option<Self> {
        Self::create_for_world(75, 50, 65536)
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
        Self::new_with_grid(device, queue, max_agents, 75, 50)
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
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

        let unpadded_bytes_per_row = soil_cols * 8;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
        let staging_soil_display_size = (padded_bytes_per_row * soil_rows) as u64;

        let staging_soil_display = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_soil_display"),
            size: staging_soil_display_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_atomics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_atomics"),
            size: agent_atomics_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_freelist = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_freelist"),
            size: freelist_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_audio = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_audio"),
            size: 4096, // 256 AudioVoice * 16 bytes
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_lbvh = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_lbvh"),
            size: (64 * std::mem::size_of::<crate::gpu::types::GpuLbvhNode>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_instances_size = (max_agents as u64) * 4;
        let visible_instances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible_instances_buf"),
            size: visible_instances_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let cull_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull_output_buf"),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        let staging_cull_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_cull_buf"),
            size: 16,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_visible_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_visible_instances"),
            size: visible_instances_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dart_template: [DartVertex; 12] = [
            // Body fill (2 triangles):
            DartVertex { position: [0.0, 1.3], edge_flag: 0.0, _pad: 0.0 },
            DartVertex { position: [-1.0, -1.0], edge_flag: 0.0, _pad: 0.0 },
            DartVertex { position: [0.0, -0.4], edge_flag: 0.0, _pad: 0.0 },

            DartVertex { position: [0.0, 1.3], edge_flag: 0.0, _pad: 0.0 },
            DartVertex { position: [0.0, -0.4], edge_flag: 0.0, _pad: 0.0 },
            DartVertex { position: [1.0, -1.0], edge_flag: 0.0, _pad: 0.0 },

            // Outline border (2 triangles):
            DartVertex { position: [0.0, 1.4], edge_flag: 1.0, _pad: 0.0 },
            DartVertex { position: [-1.1, -1.1], edge_flag: 1.0, _pad: 0.0 },
            DartVertex { position: [0.0, -0.5], edge_flag: 1.0, _pad: 0.0 },

            DartVertex { position: [0.0, 1.4], edge_flag: 1.0, _pad: 0.0 },
            DartVertex { position: [0.0, -0.5], edge_flag: 1.0, _pad: 0.0 },
            DartVertex { position: [1.1, -1.1], edge_flag: 1.0, _pad: 0.0 },
        ];

        let dart_template_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dart_template_buf"),
            size: 12 * 16,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dart_template_buf, 0, bytemuck::cast_slice(&dart_template));

        let dart_instances_size = (max_agents as u64) * 32;
        let dart_instances_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dart_instances_buf"),
            size: dart_instances_size,
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_dart_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_dart_instances"),
            size: dart_instances_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view_uniform_buf"),
            size: 80,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
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
                wgpu::BindGroupEntry { binding: 5, resource: sim_params_buf.as_entire_binding() },
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

        let lbvh_aabb_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbvh_aabb_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/lbvh_aabb.wgsl").into()),
        });
        let lbvh_aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lbvh_aabb_pipeline"),
            layout: Some(&lbvh_pl),
            module: &lbvh_aabb_sm,
            entry_point: Some("fit_lbvh_aabbs"),
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

        let agent_pipelines: [wgpu::ComputePipeline; 8] = core::array::from_fn(|i| {
            let bh = (i & 1) != 0;
            let cortex = (i & 2) != 0;
            let sex = (i & 4) != 0;
            let label = format!("agent_pipeline_mod_{}", i);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&agent_pl),
                module: &agent_sm,
                entry_point: Some("agent_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ENABLE_BARNES_HUT", if bh { 1.0 } else { 0.0 }),
                        ("ENABLE_EXPANDED_CORTEX", if cortex { 1.0 } else { 0.0 }),
                        ("ENABLE_SEXUAL_SELECTION", if sex { 1.0 } else { 0.0 }),
                    ],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
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

        let birth_pipelines: [wgpu::ComputePipeline; 2] = core::array::from_fn(|i| {
            let cortex = i == 1;
            let label = format!("birth_pipeline_mod_{}", i);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&birth_pl),
                module: &birth_sm,
                entry_point: Some("birth_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("ENABLE_EXPANDED_CORTEX", if cortex { 1.0 } else { 0.0 }),
                    ],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        });

        let spatial_query_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spatial_query_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../../assets/shaders/spatial_query.wgsl"))),
        });

        let spatial_query_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spatial_query_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let spatial_query_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_query_bg"),
            layout: &spatial_query_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_states_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lbvh_nodes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: queue_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sim_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: visible_instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cull_output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dart_instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: agent_atomics_buf.as_entire_binding() },
            ],
        });

        let spatial_query_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spatial_query_pl"),
            bind_group_layouts: &[Some(&spatial_query_bgl)],
            immediate_size: 0,
        });

        let spatial_query_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spatial_query_pipeline"),
            layout: Some(&spatial_query_pl),
            module: &spatial_query_sm,
            entry_point: Some("spatial_query_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cull_clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull_clear_pipeline"),
            layout: Some(&spatial_query_pl),
            module: &spatial_query_sm,
            entry_point: Some("frustum_cull_clear"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull_pipeline"),
            layout: Some(&spatial_query_pl),
            module: &spatial_query_sm,
            entry_point: Some("frustum_cull_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let preamble_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preamble_sm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../assets/shaders/preamble_clear.wgsl").into()),
        });

        let preamble_bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preamble_bgl0"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let preamble_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preamble_bgl1"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let preamble_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preamble_bg0"),
            layout: &preamble_bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: agent_atomics_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: queue_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_offsets_buf.as_entire_binding() },
            ],
        });
        let preamble_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preamble_bg1"),
            layout: &preamble_bgl1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sim_params_buf.as_entire_binding() },
            ],
        });
        let preamble_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preamble_pl"),
            bind_group_layouts: &[Some(&preamble_bgl0), Some(&preamble_bgl1)],
            immediate_size: 0,
        });
        let preamble_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preamble_pipeline"),
            layout: Some(&preamble_pl),
            module: &preamble_sm,
            entry_point: Some("preamble_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dart_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dart_instanced_sm"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("../../assets/shaders/dart_instanced.wgsl"))),
        });

        let dart_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dart_render_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let dart_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dart_render_bg"),
            layout: &dart_render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_uniform_buf.as_entire_binding() },
            ],
        });

        let dart_render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dart_render_pl"),
            bind_group_layouts: &[Some(&dart_render_bgl)],
            immediate_size: 0,
        });

        let dart_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dart_render_pipeline"),
            layout: Some(&dart_render_pl),
            vertex: wgpu::VertexState {
                module: &dart_sm,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 8, shader_location: 1 },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 2 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32x2, offset: 16, shader_location: 3 },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dart_sm,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
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
            visible_instances_buf,
            cull_output_buf,
            dart_instances_buf,
            dart_template_buf,
            soil_display_tex,
            sim_params_buf,
            soil_params_buf,
            view_uniform_buf,
            staging_telemetry,
            staging_agent_states,
            staging_genomes,
            staging_soil,
            staging_soil_display,
            staging_atomics,
            staging_cull_buf,
            staging_visible_instances,
            staging_dart_instances,
            staging_freelist,
            staging_audio,
            staging_lbvh,
            soil_pipeline,
            preamble_pipeline,
            morton_clear_pipeline,
            morton_encode_pipeline,
            morton_offsets_pipeline,
            lbvh_pipeline,
            lbvh_aabb_pipeline,
            agent_pipeline,
            agent_pipelines,
            birth_pipeline,
            birth_pipelines,
            spatial_query_pipeline,
            cull_clear_pipeline,
            cull_pipeline,
            dart_render_pipeline,
            soil_bind_group,
            preamble_bg0,
            preamble_bg1,
            morton_bind_group,
            lbvh_bind_group,
            agent_group0,
            agent_group1,
            birth_group0,
            birth_group1,
            spatial_query_bind_group,
            dart_render_bind_group,
            max_agents,
            soil_cols,
            soil_rows,
            initialized: std::sync::atomic::AtomicBool::new(false),
            next_agent_id: std::sync::atomic::AtomicU32::new(10000),
            next_agent_root: std::sync::atomic::AtomicU32::new(0),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_initialized(&self, val: bool) {
        self.initialized.store(val, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn update_params(&self, params: &GpuSimParams) {
        self.queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(params));
        let sp = crate::gpu::soil_pipeline::SoilParams {
            renewal: params.renewal,
            width: self.soil_cols,
            height: self.soil_rows,
            decay_rate: 0.006,
        };
        self.queue.write_buffer(&self.soil_params_buf, 0, bytemuck::bytes_of(&sp));
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
        let total_agents = self.max_agents as usize;
        if total_agents > states.len() {
            let chunk_size = 2048;
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
            let dead_atomic = GpuAgentAtomic {
                energy_milli: 0,
                mate_claim: 0,
                mate_energy_milli: 0,
                dead_claimed: 1,
            };
            let tombstone_chunk = vec![tombstone; chunk_size];
            let atomic_chunk = vec![dead_atomic; chunk_size];
            let mut curr = states.len();
            while curr < total_agents {
                let this_chunk = (total_agents - curr).min(chunk_size);
                self.queue.write_buffer(&self.agent_states_buf, (curr * 128) as u64, bytemuck::cast_slice(&tombstone_chunk[..this_chunk]));
                self.queue.write_buffer(&self.agent_atomics_buf, (curr * 16) as u64, bytemuck::cast_slice(&atomic_chunk[..this_chunk]));
                curr += this_chunk;
            }
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

        // Initialize unique creature ID counter in queue_buffer offset 24 (apex_agent_id)
        let max_incoming_id = states.iter().map(|s| s.id).max().unwrap_or(0);
        let next_id = (max_incoming_id + 1).max(self.next_agent_id.load(std::sync::atomic::Ordering::Relaxed)).max(1000);
        self.next_agent_id.store(next_id, std::sync::atomic::Ordering::Relaxed);
        self.queue.write_buffer(&self.queue_buffer, 24, bytemuck::bytes_of(&next_id));

        let sp = crate::gpu::soil_pipeline::SoilParams {
            renewal: params.renewal,
            width: self.soil_cols,
            height: self.soil_rows,
            decay_rate: 0.006,
        };
        self.queue.write_buffer(&self.soil_params_buf, 0, bytemuck::bytes_of(&sp));
        self.initialized.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn dispatch_sub_ticks(&self, sub_ticks: u32, params: &GpuSimParams) {
        self.dispatch_sub_ticks_with_mods(sub_ticks, params, false, false, false);
    }

    pub fn dispatch_sub_ticks_with_mods(
        &self,
        sub_ticks: u32,
        params: &GpuSimParams,
        barnes_hut: bool,
        expanded_cortex: bool,
        sexual_selection: bool,
    ) {
        let mut cur_params = *params;
        let mod_idx = (barnes_hut as usize) | ((expanded_cortex as usize) << 1) | ((sexual_selection as usize) << 2);
        let birth_mod_idx = expanded_cortex as usize;

        for step in 0..sub_ticks {
            cur_params.sub_tick = step;
            cur_params.tick = params.tick + step;
            self.queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(&cur_params));

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_sim_sub_tick_encoder"),
            });

            // 1. Soil decay and renewal
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_soil_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.soil_pipeline);
                cpass.set_bind_group(0, &self.soil_bind_group, &[]);
                cpass.dispatch_workgroups((self.soil_cols + 7) / 8, (self.soil_rows + 7) / 8, 1);
            }

            let active_slots = cur_params.max_agents.max(cur_params.agent_count);
            let agent_workgroups = (active_slots + 63) / 64;

            if agent_workgroups > 0 {
                // 2. Preamble clear pass (resets atomics, sub-tick telemetry counters, and cell_offsets sentinels)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_preamble_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.preamble_pipeline);
                    cpass.set_bind_group(0, &self.preamble_bg0, &[]);
                    cpass.set_bind_group(1, &self.preamble_bg1, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);
                }

                // 2b. Morton Grid spatial hashing (strictly after preamble barrier)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_morton_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.morton_encode_pipeline);
                    cpass.set_bind_group(0, &self.morton_bind_group, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);
                }

                // 3. LBVH Phase 1: Hierarchy Topology
                if cur_params.agent_count >= 2 {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_lbvh_phase1_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.lbvh_pipeline);
                    cpass.set_bind_group(0, &self.lbvh_bind_group, &[]);
                    cpass.dispatch_workgroups((cur_params.agent_count - 1 + 63) / 64, 1, 1);
                }

                // 4. LBVH Phase 2: Bottom-Up AABB fitting across explicit compute pass barrier
                if cur_params.agent_count >= 1 {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_lbvh_phase2_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.lbvh_aabb_pipeline);
                    cpass.set_bind_group(0, &self.lbvh_bind_group, &[]);
                    cpass.dispatch_workgroups((cur_params.agent_count + 63) / 64, 1, 1);
                }

                // 5. Unified Spatial Query (Mouse Picking & AoE tools) on sub_tick 0
                if cur_params.tool_type != 0xFFFFFFFF && step == 0 {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_spatial_query_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.spatial_query_pipeline);
                    cpass.set_bind_group(0, &self.spatial_query_bind_group, &[]);
                    if cur_params.tool_type == 0 {
                        cpass.dispatch_workgroups(1, 1, 1);
                    } else {
                        cpass.dispatch_workgroups(agent_workgroups, 1, 1);
                    }
                }

                // 6. Agent forward pass, kinematics & combat (using specialized mod pipeline)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_agent_step_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.agent_pipelines[mod_idx]);
                    cpass.set_bind_group(0, &self.agent_group0, &[]);
                    cpass.set_bind_group(1, &self.agent_group1, &[]);
                    cpass.dispatch_workgroups(agent_workgroups, 1, 1);
                }

                // 7. Decoupled birth step across compute pass barrier (using specialized birth pipeline)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("gpu_birth_step_pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.birth_pipelines[birth_mod_idx]);
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

    pub fn copy_soil_display_rgba(&self, out: &mut [u8]) {
        let expected_len = (self.soil_cols * self.soil_rows * 4) as usize;
        assert!(out.len() >= expected_len, "out buffer too small");

        let unpadded_bytes_per_row = self.soil_cols * 8;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
        let total_bytes = (padded_bytes_per_row * self.soil_rows) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_soil_display_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.soil_display_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_soil_display,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.soil_rows),
                },
            },
            wgpu::Extent3d {
                width: self.soil_cols,
                height: self.soil_rows,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);

        let slice = self.staging_soil_display.slice(0..total_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        for y in 0..self.soil_rows {
            let row_offset = (y * padded_bytes_per_row) as usize;
            for x in 0..self.soil_cols {
                let px_offset = row_offset + (x * 8) as usize;
                let r_bits = u16::from_le_bytes([data[px_offset], data[px_offset + 1]]);
                let g_bits = u16::from_le_bytes([data[px_offset + 2], data[px_offset + 3]]);
                let b_bits = u16::from_le_bytes([data[px_offset + 4], data[px_offset + 5]]);
                let a_bits = u16::from_le_bytes([data[px_offset + 6], data[px_offset + 7]]);

                let r = half::f16::from_bits(r_bits).to_f32();
                let g = half::f16::from_bits(g_bits).to_f32();
                let b = half::f16::from_bits(b_bits).to_f32();
                let a = half::f16::from_bits(a_bits).to_f32();

                let out_idx = ((y * self.soil_cols + x) * 4) as usize;
                out[out_idx] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
                out[out_idx + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
                out[out_idx + 2] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
                out[out_idx + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        drop(data);
        self.staging_soil_display.unmap();
    }

    pub fn readback_soil_display_rgba(&self) -> Vec<u8> {
        let mut out = vec![0u8; (self.soil_cols * self.soil_rows * 4) as usize];
        self.copy_soil_display_rgba(&mut out);
        out
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

    pub fn dispatch_culling(&self, params: &GpuSimParams) -> u32 {
        self.queue.write_buffer(&self.sim_params_buf, 0, bytemuck::bytes_of(params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cull_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull_pass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.spatial_query_bind_group, &[]);

            // Clear cull counter to 0
            cpass.set_pipeline(&self.cull_clear_pipeline);
            cpass.dispatch_workgroups(1, 1, 1);

            // Frustum cull active agents
            let active_slots = params.max_agents.max(params.agent_count);
            let workgroups = (active_slots + 63) / 64;
            if workgroups > 0 {
                cpass.set_pipeline(&self.cull_pipeline);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        encoder.copy_buffer_to_buffer(&self.cull_output_buf, 0, &self.staging_cull_buf, 0, 16);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_cull_buf.slice(0..16);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let visible_count = {
            let data = slice.get_mapped_range();
            let count = u32::from_le_bytes(data[4..8].try_into().unwrap());
            drop(data);
            count
        };
        self.staging_cull_buf.unmap();

        visible_count.min(params.max_capacity).min(params.max_agents)
    }

    pub fn readback_visible_instances(&self, count: usize) -> Vec<u32> {
        let count = count.min(self.max_agents as usize);
        if count == 0 {
            return Vec::new();
        }
        let byte_len = (count * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_visible_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.visible_instances_buf, 0, &self.staging_visible_instances, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_visible_instances.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let result = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        self.staging_visible_instances.unmap();
        result
    }

    pub fn readback_dart_instances(&self, count: usize) -> Vec<crate::gpu::types::GpuDartInstance> {
        if count == 0 {
            return Vec::new();
        }
        let byte_len = (count * 32) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_dart_instances_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.dart_instances_buf, 0, &self.staging_dart_instances, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_dart_instances.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let result = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, crate::gpu::types::GpuDartInstance>(&data[0..byte_len as usize]).to_vec()
        };
        self.staging_dart_instances.unmap();
        result
    }

    pub fn readback_cull_args(&self) -> [u32; 4] {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_cull_args_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.cull_output_buf, 0, &self.staging_cull_buf, 0, 16);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_cull_buf.slice(0..16);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let args = {
            let data = slice.get_mapped_range();
            let mut arr = [0u32; 4];
            arr[0] = u32::from_le_bytes(data[0..4].try_into().unwrap());
            arr[1] = u32::from_le_bytes(data[4..8].try_into().unwrap());
            arr[2] = u32::from_le_bytes(data[8..12].try_into().unwrap());
            arr[3] = u32::from_le_bytes(data[12..16].try_into().unwrap());
            drop(data);
            arr
        };
        self.staging_cull_buf.unmap();
        args
    }

    pub fn render_darts_instanced(
        &self,
        target_view: &wgpu::TextureView,
        params: &GpuSimParams,
    ) {
        // Update view uniform with orthographic projection matching dart_instanced.wgsl inverted Y
        let cam_y = params.world_size[1] - params.camera_pos[1];
        let left = params.camera_pos[0] - params.camera_size[0] * 0.5;
        let right = params.camera_pos[0] + params.camera_size[0] * 0.5;
        let bottom = cam_y - params.camera_size[1] * 0.5;
        let top = cam_y + params.camera_size[1] * 0.5;

        let sx = 2.0 / (right - left);
        let sy = 2.0 / (top - bottom);
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);

        let mut view_proj = [0.0f32; 16];
        view_proj[0] = sx;
        view_proj[5] = sy;
        view_proj[10] = 0.0;
        view_proj[12] = tx;
        view_proj[13] = ty;
        view_proj[14] = 0.5;
        view_proj[15] = 1.0;

        let vu = ViewUniform {
            view_proj,
            world_size: params.world_size,
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.view_uniform_buf, 0, bytemuck::bytes_of(&vu));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dart_render_encoder"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dart_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.dart_render_pipeline);
            rpass.set_bind_group(0, &self.dart_render_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.dart_template_buf.slice(..));
            rpass.set_vertex_buffer(1, self.dart_instances_buf.slice(..));
            rpass.draw_indirect(&self.cull_output_buf, 0);
        }

        self.queue.submit([encoder.finish()]);
    }

    pub fn dart_template_vertex_count(&self) -> u32 {
        12
    }

    pub fn seed_spores_gpu(&self, agents: &[(f32, f32)]) -> Vec<u32> {
        self.seed_agents_gpu(agents)
    }

    pub fn seed_agents_gpu(&self, agents: &[(f32, f32)]) -> Vec<u32> {
        if agents.is_empty() {
            return Vec::new();
        }

        // 1. Read back current telemetry to check available freelist slots
        let telem = self.readback_telemetry();
        let cur_freelist_top = telem.freelist_top.min(self.max_agents);
        if cur_freelist_top == 0 {
            return Vec::new();
        }

        let count = agents.len().min(cur_freelist_top as usize);
        if count == 0 {
            return Vec::new();
        }

        // 2. Read back top `count` slots from freelist_buf
        let start_slot = cur_freelist_top - count as u32;
        let start_byte = (start_slot as u64) * 4;
        let byte_len = (count as u64) * 4;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_freelist_slots_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.freelist_buf, start_byte, &self.staging_freelist, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_freelist.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let slots: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_freelist.unmap();

        let world_w = (self.soil_cols as f32) * 12.0;
        let world_h = (self.soil_rows as f32) * 12.0;

        let mut states = Vec::with_capacity(slots.len());
        let mut genomes = Vec::with_capacity(slots.len());
        let mut atomics = Vec::with_capacity(slots.len());

        // 3. For each slot, initialize agent data
        for (i, &_slot) in slots.iter().enumerate() {
            let (x, y) = agents[i];
            let id = self.next_agent_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let root = (self.next_agent_root.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 16) as u32;
            let energy = 43.0f32;
            let phi = (1.0 + 5.0f32.sqrt()) * 0.5;
            let angle = ((i as f32) * phi) % (2.0 * std::f32::consts::PI);

            let pal_color = crate::theme::PALETTE[(root as usize) % crate::theme::PALETTE.len()];
            let packed_color = (pal_color.r() as u32)
                | ((pal_color.g() as u32) << 8)
                | ((pal_color.b() as u32) << 16)
                | (0xFF << 24);

            let r_u8 = 128u32; // tr[0] = 0.5
            let e_u8 = ((energy / 100.0).clamp(0.0, 1.0) * 255.0) as u32;
            let visual_cache = r_u8 | (e_u8 << 16);

            let morton = crate::gpu::spatial_index::compute_morton_32_with_size(
                [x, y],
                [world_w, world_h],
            );

            states.push(GpuAgentState {
                pos_vel: [x, y, 0.0, 0.0],
                angle_energy: [angle, energy, 0.0, 0.0],
                traits: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
                hidden: [0.0; 10],
                id,
                meta_flags: root, // living: bit 13 is 0
                age_gen: 0,
                morton_code: morton,
                packed_color,
                visual_cache,
            });

            // Spore genome with pseudo-random weights
            let mut packed_genes = [0u32; 88];
            for g in 0..88 {
                let w0 = (((i * 73 + g * 31) % 116) as i32 - 58) as i8 as u8;
                let w1 = (((i * 97 + g * 47) % 116) as i32 - 58) as i8 as u8;
                let w2 = (((i * 113 + g * 59) % 116) as i32 - 58) as i8 as u8;
                let w3 = (((i * 127 + g * 71) % 116) as i32 - 58) as i8 as u8;
                packed_genes[g] = (w0 as u32) | ((w1 as u32) << 8) | ((w2 as u32) << 16) | ((w3 as u32) << 24);
            }
            genomes.push(GpuAgentGenome { packed_genes });

            atomics.push(GpuAgentAtomic {
                energy_milli: (energy * 1000.0) as i32,
                mate_claim: 0,
                mate_energy_milli: 0,
                dead_claimed: 0,
            });
        }

        // Group into contiguous slot runs to minimize write_buffer overhead
        let mut run_start = 0;
        while run_start < slots.len() {
            let mut run_end = run_start + 1;
            while run_end < slots.len() && slots[run_end] == slots[run_end - 1] + 1 {
                run_end += 1;
            }

            let first_slot = slots[run_start];
            let run_states: Vec<GpuAgentState> = (run_start..run_end).map(|idx| states[idx]).collect();
            let run_genomes: Vec<GpuAgentGenome> = (run_start..run_end).map(|idx| genomes[idx]).collect();
            let run_atomics: Vec<GpuAgentAtomic> = (run_start..run_end).map(|idx| atomics[idx]).collect();

            self.queue.write_buffer(&self.agent_states_buf, (first_slot as u64) * 128, bytemuck::cast_slice(&run_states));
            self.queue.write_buffer(&self.agent_genomes_buf, (first_slot as u64) * 352, bytemuck::cast_slice(&run_genomes));
            self.queue.write_buffer(&self.agent_atomics_buf, (first_slot as u64) * 16, bytemuck::cast_slice(&run_atomics));

            run_start = run_end;
        }

        // 4. Update freelist_top and apex_agent_id in queue_buffer
        let new_freelist_top = start_slot;
        self.queue.write_buffer(&self.queue_buffer, 28, bytemuck::bytes_of(&new_freelist_top));
        let next_id = self.next_agent_id.load(std::sync::atomic::Ordering::Relaxed);
        self.queue.write_buffer(&self.queue_buffer, 24, bytemuck::bytes_of(&next_id));

        slots
    }

    pub fn readback_audio_voices(&self, count: usize) -> Vec<AudioVoice> {
        let count = count.min(256);
        if count == 0 {
            return Vec::new();
        }
        let byte_len = (count * 16) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_audio_voices_encoder"),
        });
        // Offset 1,048,704 in queue_buffer
        encoder.copy_buffer_to_buffer(&self.queue_buffer, 1_048_704, &self.staging_audio, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_audio.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let voices = bytemuck::cast_slice::<u8, AudioVoice>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_audio.unmap();

        voices
    }

    pub fn readback_lbvh_nodes(&self, count: usize) -> Vec<GpuLbvhNode> {
        let count = count.min(64);
        if count == 0 {
            return Vec::new();
        }
        let node_size = std::mem::size_of::<GpuLbvhNode>() as u64;
        let byte_len = (count as u64) * node_size;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_lbvh_nodes_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.lbvh_nodes_buf, 0, &self.staging_lbvh, 0, byte_len);
        self.queue.submit([encoder.finish()]);

        let slice = self.staging_lbvh.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let nodes = bytemuck::cast_slice::<u8, GpuLbvhNode>(&data[0..byte_len as usize]).to_vec();
        drop(data);
        self.staging_lbvh.unmap();

        nodes
    }

    pub fn extract_gpu_minimap_clusters(&self, target_depth: usize) -> Vec<crate::gpu::lbvh::MinimapCluster> {
        let nodes = self.readback_lbvh_nodes(63);
        if nodes.is_empty() {
            return Vec::new();
        }
        let tree = crate::gpu::lbvh::LbvhTree { nodes };
        tree.extract_minimap_clusters(target_depth)
    }

    pub fn readback_agent_slot(&self, slot: u32) -> Option<(GpuAgentState, GpuAgentGenome, GpuAgentAtomic)> {
        if slot >= self.max_agents {
            return None;
        }
        let state_offset = (slot as u64) * 128;
        let genome_offset = (slot as u64) * 352;
        let atomic_offset = (slot as u64) * 16;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_agent_slot_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.agent_states_buf, state_offset, &self.staging_agent_states, 0, 128);
        encoder.copy_buffer_to_buffer(&self.agent_genomes_buf, genome_offset, &self.staging_genomes, 0, 352);
        encoder.copy_buffer_to_buffer(&self.agent_atomics_buf, atomic_offset, &self.staging_atomics, 0, 16);
        self.queue.submit([encoder.finish()]);

        // Map state
        let s_slice = self.staging_agent_states.slice(0..128);
        let (s_tx, s_rx) = std::sync::mpsc::channel();
        s_slice.map_async(wgpu::MapMode::Read, move |res| { let _ = s_tx.send(res); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        s_rx.recv().unwrap().unwrap();
        let state = *bytemuck::from_bytes::<GpuAgentState>(&s_slice.get_mapped_range()[0..128]);
        self.staging_agent_states.unmap();

        // Map genome
        let g_slice = self.staging_genomes.slice(0..352);
        let (g_tx, g_rx) = std::sync::mpsc::channel();
        g_slice.map_async(wgpu::MapMode::Read, move |res| { let _ = g_tx.send(res); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        g_rx.recv().unwrap().unwrap();
        let genome = *bytemuck::from_bytes::<GpuAgentGenome>(&g_slice.get_mapped_range()[0..352]);
        self.staging_genomes.unmap();

        // Map atomic
        let a_slice = self.staging_atomics.slice(0..16);
        let (a_tx, a_rx) = std::sync::mpsc::channel();
        a_slice.map_async(wgpu::MapMode::Read, move |res| { let _ = a_tx.send(res); });
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        a_rx.recv().unwrap().unwrap();
        let atomic = *bytemuck::from_bytes::<GpuAgentAtomic>(&a_slice.get_mapped_range()[0..16]);
        self.staging_atomics.unmap();

        Some((state, genome, atomic))
    }

    pub fn sync_soil_cells_gpu(&self, soil: &clank_core::soil::SoilGrid, cx: i32, cy: i32, radius: i32) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let x = (cx + dx).rem_euclid(self.soil_cols as i32) as u32;
                    let y = (cy + dy).rem_euclid(self.soil_rows as i32) as u32;
                    let idx = (y * self.soil_cols + x) as usize;
                    let food = soil.food.get(idx).copied().unwrap_or(0.0);
                    let taint = soil.taint.get(idx).copied().unwrap_or(0.0);
                    let scent = soil.scent.get(idx).copied().unwrap_or(0.0);
                    let cell = crate::gpu::soil_pipeline::GpuSoilPipeline::from_physics(food, taint, scent);
                    self.queue.write_buffer(&self.soil_buf, (idx * 16) as u64, bytemuck::bytes_of(&cell));
                }
            }
        }
    }
}
