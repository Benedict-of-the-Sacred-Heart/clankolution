//! GPU Birth, Mutation & PCG PRNG Pipeline
//!
//! Provides stateless GPU-compatible PCG hashing, symmetric triangular distribution sampling,
//! and parallel genome crossover/mutation math matching `birth_step.wgsl`, and the
//! indirect dispatch of the birth pass (`BirthDispatch`).

/// pcg3d (Jarzynski & Olano 2020), the stateless hash of (id, stream, tick) the shaders use.
#[inline]
pub fn pcg_hash(id: u32, stream: u32, tick: u32) -> u32 {
    let mix = |v: &mut [u32; 3]| {
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    };
    let mut v = [id, stream, tick].map(|x| x.wrapping_mul(1664525).wrapping_add(1013904223));
    mix(&mut v);
    v = v.map(|x| x ^ (x >> 16));
    mix(&mut v);
    v[0]
}

/// Uniform in [0, 1) from the top 24 bits, as `pcg3d_float`.
#[inline]
pub fn pcg_float(id: u32, stream: u32, tick: u32) -> f32 {
    (pcg_hash(id, stream, tick) >> 8) as f32 / 16_777_216.0
}

/// Evaluates zero-centered symmetric triangular distribution $T(-1, 0, 1)$ via $r_1 + r_2 - 1.0$.
#[inline]
pub fn pcg_triangular(id: u32, stream: u32, tick: u32) -> f32 {
    let r1 = pcg_float(id, stream, tick);
    let r2 = pcg_float(id, stream.wrapping_add(100), tick);
    r1 + r2 - 1.0
}

/// Mutates a single signed gene byte matching HTML reference:
/// `round((r1 + r2 - 1.0) * 100.0 * mut_rate)` clamped to `[-127, 127]`.
#[inline]
pub fn mutate_gene_byte(val: i8, id: u32, stream: u32, tick: u32, mut_rate: f32) -> i8 {
    let delta = (pcg_triangular(id, stream, tick) * 100.0 * mut_rate).round() as i32;
    let new_val = (val as i32 + delta).clamp(-127, 127);
    new_val as i8
}

/// Crossover between two 32-bit packed gene words with 48% inheritance from parent A.
#[inline]
pub fn crossover_genes(word_a: u32, word_b: u32, id: u32, tick: u32) -> u32 {
    let p = pcg_float(id, 42, tick);
    if p < 0.48 {
        word_a
    } else {
        word_b
    }
}

/// Mutates a 32-bit packed word containing 4 signed 8-bit gene weights,
/// applying triangular mutation and masking inactive dummy genes.
pub fn mutate_word(word: u32, id: u32, word_idx: u32, tick: u32, mut_rate: f32) -> u32 {
    let bytes = word.to_le_bytes();
    let mut mutated_bytes = [0u8; 4];
    for i in 0..4 {
        let stream = word_idx * 4 + i as u32;
        let b = mutate_gene_byte(bytes[i] as i8, id, stream, tick, mut_rate);
        mutated_bytes[i] = b as u8;
    }
    let mut result = u32::from_le_bytes(mutated_bytes);

    if word_idx < 70 && (word_idx % 7) == 6 {
        result &= 0x0000FFFF;
    }
    if word_idx >= 70 && ((word_idx - 70) % 3) == 2 {
        result &= 0x00FFFFFF;
    }

    result
}

/// The birth pass's indirect dispatch: one workgroup per queued birth. `encode_args` records
/// a one-thread dispatch (`assets/shaders/birth_args.wgsl`) that copies the birth count into
/// `args`, after the agent pass and before `dispatch_workgroups_indirect(args, 0)`.
pub struct BirthDispatch {
    pub args: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl BirthDispatch {
    pub fn new(device: &wgpu::Device, queue_buffer: &wgpu::Buffer) -> Self {
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("birth_args_buf"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
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
            label: Some("birth_args_bgl"),
            entries: &[entry(0, true), entry(1, false)],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("birth_args_bg"),
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
                    resource: args.as_entire_binding(),
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("birth_args_sm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../assets/shaders/birth_args.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("birth_args_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("birth_args_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("birth_args_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            args,
            pipeline,
            bind_group,
        }
    }

    pub fn encode_args(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
}
