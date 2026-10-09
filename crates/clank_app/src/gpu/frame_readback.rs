//! Non-blocking per-frame readbacks.
//!
//! Each GPU-mode frame copies its telemetry, audio voices and (optionally) one creature slot
//! into a staging buffer and maps it asynchronously. The CPU picks the results up on a later
//! frame instead of stalling until the GPU has finished every sub-tick.

use crate::gpu::types::{
    AudioVoice, GpuAgentAtomic, GpuAgentGenome, GpuAgentState, GpuAgentTrail, GpuChronicleEvent,
    GpuTelemetry,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Frames that may be in flight before the CPU waits for the oldest (back-pressure).
pub const RING_DEPTH: usize = 3;

const TELEMETRY_OFFSET: u64 = 0;
const AUDIO_OFFSET: u64 = 128;
const CHRONICLE_OFFSET: u64 = AUDIO_OFFSET + 256 * 16;
const STATE_OFFSET: u64 = CHRONICLE_OFFSET + 64 * 16;
const GENOME_OFFSET: u64 = STATE_OFFSET + 128;
const ATOMIC_OFFSET: u64 = GENOME_OFFSET + 352;
const TRAIL_OFFSET: u64 = ATOMIC_OFFSET + 16;
/// Bytes copied per frame: telemetry, the audio queue, the chronicle and one creature slot.
pub const FRAME_READBACK_BYTES: u64 = TRAIL_OFFSET + 96;

/// Where the queue buffer keeps telemetry and the audio voices.
pub const QUEUE_TELEMETRY_OFFSET: u64 = 0;
pub const QUEUE_AUDIO_OFFSET: u64 = 128 + 65536 * 16;
pub const QUEUE_CHRONICLE_OFFSET: u64 = QUEUE_AUDIO_OFFSET + 256 * 16;

/// The creature slot requested with a frame, as it was when that frame finished.
#[derive(Clone, Copy, Debug)]
pub struct SelectedSlotReadback {
    pub slot: u32,
    pub state: GpuAgentState,
    pub genome: GpuAgentGenome,
    pub atomic: GpuAgentAtomic,
    pub trail: GpuAgentTrail,
}

/// Everything one frame reported.
#[derive(Clone, Debug)]
pub struct FrameReadback {
    pub telemetry: GpuTelemetry,
    pub audio: Vec<AudioVoice>,
    /// The frame's chronicle events, in the order the GPU queued them.
    pub chronicle: Vec<GpuChronicleEvent>,
    pub selected: Option<SelectedSlotReadback>,
    /// The frame ran the picking query (tool 0), so `telemetry.selected_agent_*` is its answer.
    pub picking: bool,
}

/// GPU buffers a frame readback copies from.
pub struct ReadbackSources<'a> {
    pub queue_buffer: &'a wgpu::Buffer,
    pub agent_states: &'a wgpu::Buffer,
    pub agent_genomes: &'a wgpu::Buffer,
    pub agent_atomics: &'a wgpu::Buffer,
    pub agent_trails: &'a wgpu::Buffer,
    pub max_agents: u32,
}

struct Pending {
    buffer: usize,
    selected: Option<u32>,
    picking: bool,
    epoch: u64,
    ready: Arc<AtomicBool>,
}

pub struct ReadbackRing {
    buffers: Vec<wgpu::Buffer>,
    free: Vec<usize>,
    pending: VecDeque<Pending>,
    completed: VecDeque<FrameReadback>,
    /// Bumped when the GPU world is replaced; frames from an older epoch are dropped.
    epoch: u64,
}

impl ReadbackRing {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffers = (0..RING_DEPTH)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("frame_readback_{i}")),
                    size: FRAME_READBACK_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            buffers,
            free: (0..RING_DEPTH).rev().collect(),
            pending: VecDeque::with_capacity(RING_DEPTH),
            completed: VecDeque::with_capacity(RING_DEPTH),
            epoch: 0,
        }
    }

    pub fn in_flight(&self) -> usize {
        self.pending.len() + self.completed.len()
    }

    /// Forget every frame requested so far (their world was replaced).
    pub fn discard_all(&mut self) {
        self.epoch += 1;
        self.completed.clear();
    }

    /// Records and submits the copies for one frame. Returns false when every buffer is in
    /// flight; the caller must wait (`wait_for_oldest`) and retry.
    pub fn try_request(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &ReadbackSources,
        selected: Option<u32>,
        picking: bool,
    ) -> bool {
        let Some(buffer) = self.free.pop() else {
            return false;
        };
        let dst = &self.buffers[buffer];
        let selected = selected.filter(|&s| s < src.max_agents);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(
            src.queue_buffer,
            QUEUE_TELEMETRY_OFFSET,
            dst,
            TELEMETRY_OFFSET,
            128,
        );
        encoder.copy_buffer_to_buffer(
            src.queue_buffer,
            QUEUE_AUDIO_OFFSET,
            dst,
            AUDIO_OFFSET,
            256 * 16,
        );
        encoder.copy_buffer_to_buffer(
            src.queue_buffer,
            QUEUE_CHRONICLE_OFFSET,
            dst,
            CHRONICLE_OFFSET,
            64 * 16,
        );
        if let Some(slot) = selected {
            let slot = slot as u64;
            encoder.copy_buffer_to_buffer(src.agent_states, slot * 128, dst, STATE_OFFSET, 128);
            encoder.copy_buffer_to_buffer(src.agent_genomes, slot * 352, dst, GENOME_OFFSET, 352);
            encoder.copy_buffer_to_buffer(src.agent_atomics, slot * 16, dst, ATOMIC_OFFSET, 16);
            encoder.copy_buffer_to_buffer(src.agent_trails, slot * 96, dst, TRAIL_OFFSET, 96);
        }
        queue.submit([encoder.finish()]);

        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        dst.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
        self.pending.push_back(Pending {
            buffer,
            selected,
            picking,
            epoch: self.epoch,
            ready,
        });
        true
    }

    pub fn oldest_ready(&self) -> Option<bool> {
        self.pending
            .front()
            .map(|p| p.ready.load(Ordering::Acquire))
    }

    /// Moves finished frames, in request order, into the completed queue.
    pub fn collect_ready(&mut self) {
        while let Some(front) = self.pending.front() {
            if !front.ready.load(Ordering::Acquire) {
                break;
            }
            let p = self.pending.pop_front().unwrap();
            let buffer = &self.buffers[p.buffer];
            if p.epoch == self.epoch {
                let data = buffer.slice(..).get_mapped_range();
                self.completed
                    .push_back(parse(&data, p.selected, p.picking));
            }
            buffer.unmap();
            self.free.push(p.buffer);
        }
    }

    pub fn take_completed(&mut self) -> Vec<FrameReadback> {
        self.completed.drain(..).collect()
    }
}

fn parse(data: &[u8], selected: Option<u32>, picking: bool) -> FrameReadback {
    let at = |offset: u64, len: usize| &data[offset as usize..offset as usize + len];
    let telemetry: GpuTelemetry = bytemuck::pod_read_unaligned(at(TELEMETRY_OFFSET, 128));
    let voices = (telemetry.audio_voice_count as usize).min(256);
    let audio = bytemuck::pod_collect_to_vec::<u8, AudioVoice>(at(AUDIO_OFFSET, voices * 16));
    let events = (telemetry.chronicle_count as usize).min(64);
    let chronicle =
        bytemuck::pod_collect_to_vec::<u8, GpuChronicleEvent>(at(CHRONICLE_OFFSET, events * 16));
    let selected = selected.map(|slot| SelectedSlotReadback {
        slot,
        state: bytemuck::pod_read_unaligned(at(STATE_OFFSET, 128)),
        genome: bytemuck::pod_read_unaligned(at(GENOME_OFFSET, 352)),
        atomic: bytemuck::pod_read_unaligned(at(ATOMIC_OFFSET, 16)),
        trail: bytemuck::pod_read_unaligned(at(TRAIL_OFFSET, 96)),
    });
    FrameReadback {
        telemetry,
        audio,
        chronicle,
        selected,
        picking,
    }
}
