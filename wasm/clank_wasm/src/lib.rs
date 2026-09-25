pub mod math;

#[no_mangle]
pub extern "C" fn test_sin(x: f64) -> f64 {
    math::sin(x)
}

#[no_mangle]
pub extern "C" fn test_cos(x: f64) -> f64 {
    math::cos(x)
}

#[no_mangle]
pub extern "C" fn test_atan2(y: f64, x: f64) -> f64 {
    math::atan2(y, x)
}

#[no_mangle]
pub extern "C" fn test_tanh(x: f64) -> f64 {
    math::tanh(x)
}

#[no_mangle]
pub extern "C" fn test_sqrt(x: f64) -> f64 {
    math::sqrt(x)
}

#[no_mangle]
pub extern "C" fn test_mulberry(seed: u32, iterations: u32) -> f64 {
    let mut rng = math::Prng::new(seed);
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += rng.next_f64();
    }
    sum
}
