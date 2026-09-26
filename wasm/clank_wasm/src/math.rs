pub struct Prng {
    pub s: u32,
}

impl Prng {
    #[inline(always)]
    pub fn new(seed: u32) -> Self {
        Self { s: seed }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        self.s = self.s.wrapping_add(0x6D2B79F5);
        let mut t = (self.s ^ (self.s >> 15)).wrapping_mul(1 | self.s);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        t ^ (t >> 14)
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / 4294967296.0
    }

    #[inline(always)]
    pub fn rand(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

#[inline(always)]
pub fn clamp(v: f64, min: f64, max: f64) -> f64 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

#[inline(always)]
pub fn clamp_f32(v: f32, min: f32, max: f32) -> f32 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

#[inline(always)]
pub fn wrap(x: f64, m: f64) -> f64 {
    ((x % m) + m) % m
}

extern "C" {
    fn host_sin(x: f64) -> f64;
    fn host_cos(x: f64) -> f64;
    fn host_atan2(y: f64, x: f64) -> f64;
}

#[inline(always)]
pub fn sin(x: f64) -> f64 {
    unsafe { host_sin(x) }
}

#[inline(always)]
pub fn cos(x: f64) -> f64 {
    unsafe { host_cos(x) }
}

#[inline(always)]
pub fn atan2(y: f64, x: f64) -> f64 {
    unsafe { host_atan2(y, x) }
}

#[inline(always)]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    // Compiles to native WebAssembly f64.sqrt instruction
    x.sqrt()
}
