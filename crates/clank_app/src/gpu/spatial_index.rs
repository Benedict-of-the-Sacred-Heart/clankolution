//! Hybrid Morton Grid Spatial Index
//!
//! Provides coordinate Morton encoding, cell offset indexing,
//! and spatial neighbor query acceleration for the $9 \times 6$ arena grid.

pub const MORTON_COLS: u32 = 9;
pub const MORTON_ROWS: u32 = 6;
pub const TOTAL_CELLS: usize = (MORTON_COLS * MORTON_ROWS) as usize; // 54
pub const CELL_SIZE: f32 = 100.0;
pub const SENTINEL: u32 = 0xFFFFFFFF;

/// Expands a 16-bit integer into 32-bit by inserting zeros between bits.
#[inline]
pub fn expand_bits(mut v: u32) -> u32 {
    v &= 0x0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    v
}

/// Computes a 32-bit Morton code (Z-order curve) by interleaving X and Y coordinates.
#[inline]
pub fn compute_morton_32(pos: [f32; 2]) -> u32 {
    let x_norm = ((pos[0] / 900.0).clamp(0.0, 1.0) * 65535.0).round() as u32;
    let y_norm = ((pos[1] / 600.0).clamp(0.0, 1.0) * 65535.0).round() as u32;
    expand_bits(x_norm) | (expand_bits(y_norm) << 1)
}

/// Maps arena position [x, y] to clamped cell ID 0..53 ($9 \times 6$ grid).
#[inline]
pub fn get_cell_id(pos: [f32; 2]) -> usize {
    let gx = ((pos[0].max(0.0) / CELL_SIZE).floor() as u32).min(MORTON_COLS - 1);
    let gy = ((pos[1].max(0.0) / CELL_SIZE).floor() as u32).min(MORTON_ROWS - 1);
    (gy * MORTON_COLS + gx) as usize
}

/// Table of start and end indices in the sorted agent list for each of the 54 spatial cells.
#[derive(Clone, Debug)]
pub struct CellOffsetsTable {
    pub offsets: [[u32; 2]; TOTAL_CELLS],
}

impl Default for CellOffsetsTable {
    fn default() -> Self {
        Self::new()
    }
}

impl CellOffsetsTable {
    /// Creates a table initialized with `[0xFFFFFFFF, 0xFFFFFFFF]` sentinels.
    pub fn new() -> Self {
        let mut table = Self {
            offsets: [[SENTINEL, SENTINEL]; TOTAL_CELLS],
        };
        table.clear();
        table
    }

    /// Clears the table back to sentinels, matching `morton_grid.wgsl: clear_cell_offsets`.
    pub fn clear(&mut self) {
        for entry in self.offsets.iter_mut() {
            *entry = [SENTINEL, SENTINEL];
        }
    }

    /// Retrieves the offset pair `[start, count]` or `[start, end]` for cell index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> [u32; 2] {
        self.offsets[i]
    }

    /// Sets the offset entry for cell index `i`.
    #[inline]
    pub fn set(&mut self, i: usize, start: u32, end: u32) {
        self.offsets[i] = [start, end];
    }
}
