//! Tombstone Freelist Manager & Zero-Copy Storage Allocation
//!
//! Manages $O(1)$ zero-copy recycling of agent slots in VRAM storage buffers.
//! Incorporates atomic CAS death ownership to guarantee that concurrent predation
//! and starvation cannot cause double-frees or freelist corruption.

use std::sync::atomic::{AtomicU32, Ordering};

/// CPU-side manager and WGSL counterpart interface for the GPU tombstone freelist.
#[derive(Debug)]
pub struct TombstoneFreelistManager {
    capacity: usize,
    freelist: Vec<u32>,
    dead_claimed: Vec<AtomicU32>,
}

impl TombstoneFreelistManager {
    /// Creates a new freelist manager with the given total slot capacity.
    /// Initially all slots 0..capacity are available for allocation.
    pub fn new(capacity: usize) -> Self {
        // Pre-populate stack with slots 0..capacity in reverse so allocation returns 0, 1, 2...
        let freelist: Vec<u32> = (0..capacity as u32).rev().collect();
        let mut dead_claimed = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            dead_claimed.push(AtomicU32::new(0));
        }

        Self {
            capacity,
            freelist,
            dead_claimed,
        }
    }

    /// Allocates an agent slot from the top of the freelist stack.
    /// Returns None safely without unsigned underflow when carrying capacity is reached.
    pub fn allocate(&mut self) -> Option<u32> {
        let slot = self.freelist.pop()?;
        if let Some(claimed) = self.dead_claimed.get(slot as usize) {
            claimed.store(0, Ordering::Release);
        }
        Some(slot)
    }

    /// Returns a slot to the freelist stack for recycling.
    pub fn free(&mut self, slot: u32) {
        if (slot as usize) < self.capacity {
            self.freelist.push(slot);
        }
    }

    /// Returns the current number of available free slots.
    pub fn freelist_top(&self) -> u32 {
        self.freelist.len() as u32
    }

    /// Total capacity managed by this freelist.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Atomic CAS death claim and slot recycling.
    /// Guarantees slot is freed exactly once even if multiple threads/systems mark it dead.
    pub fn claim_death_and_free(&mut self, slot: u32) -> bool {
        if let Some(claimed) = self.dead_claimed.get(slot as usize) {
            let res = claimed.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire);
            if res.is_ok() {
                self.free(slot);
                return true;
            }
        }
        false
    }

    /// Access raw freelist slice for GPU buffer initialization.
    pub fn as_slice(&self) -> &[u32] {
        &self.freelist
    }
}
