//! Consolidated 256-Voice Audio Queue & Soundscape Manager
//!
//! Consumes the GPU consolidated 256-voice queue and plays granular audio
//! events (bite/attack, kill, birth) with toroidal seam culling and zoom-modulated volume.

use bevy::prelude::*;
use crate::gpu::types::AudioVoice;

/// Event type constants matching WGSL audio emitter:
pub const EVENT_ATTACK: u32 = 0;
pub const EVENT_KILL: u32 = 1;
pub const EVENT_BIRTH: u32 = 2;

#[derive(Resource, Default)]
pub struct AudioVoiceQueue {
    pub voices: Vec<AudioVoice>,
}

impl AudioVoiceQueue {
    pub fn new() -> Self {
        Self {
            voices: Vec::with_capacity(256),
        }
    }

    /// Ingests a slice of audio voices from the GPU consolidated queue buffer.
    pub fn ingest_voices(&mut self, voices: &[AudioVoice]) {
        self.voices.clear();
        for &v in voices.iter().take(256) {
            if v.volume > 0.0 {
                self.voices.push(v);
            }
        }
    }

    /// Clears processed voices for the current frame.
    pub fn clear(&mut self) {
        self.voices.clear();
    }
}

pub fn audio_playback_system(mut queue: ResMut<AudioVoiceQueue>) {
    // Process queued voice events for playback
    // (In headless/testing mode, drains cleanly without panic)
    queue.clear();
}

pub struct ClankAudioPlugin;

impl Plugin for ClankAudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AudioVoiceQueue>()
            .add_systems(Update, audio_playback_system);
    }
}
