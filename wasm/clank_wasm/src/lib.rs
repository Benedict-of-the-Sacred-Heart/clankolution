pub mod math;
pub mod agent;
pub mod soil;
pub mod archive;
pub mod world;

use agent::AgentData;
use world::World;

static mut GLOBAL_WORLD: Option<World> = None;
static mut ARCHIVE_BUFFER: Vec<u8> = Vec::new();

#[inline(always)]
fn get_world() -> &'static mut World {
    unsafe {
        let ptr = core::ptr::addr_of_mut!(GLOBAL_WORLD);
        if (*ptr).is_none() {
            *ptr = Some(World::new(1));
        }
        (*ptr).as_mut().unwrap()
    }
}

#[no_mangle]
pub extern "C" fn init_world(seed: u32) {
    unsafe {
        let ptr = core::ptr::addr_of_mut!(GLOBAL_WORLD);
        *ptr = Some(World::new(seed));
    }
}

#[no_mangle]
pub extern "C" fn evolve_ticks(ticks: u32) {
    let w = get_world();
    for _ in 0..ticks {
        w.evolve();
    }
}

#[no_mangle]
pub extern "C" fn get_agents_ptr() -> *const AgentData {
    let w = get_world();
    w.agents.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_agents_count() -> u32 {
    let w = get_world();
    w.agents.len() as u32
}

#[no_mangle]
pub extern "C" fn get_food_ptr() -> *mut f32 {
    let w = get_world();
    w.soil.food.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn get_taint_ptr() -> *mut f32 {
    let w = get_world();
    w.soil.taint.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn get_scent_ptr() -> *mut f32 {
    let w = get_world();
    w.soil.scent.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn get_tick() -> u32 {
    let w = get_world();
    w.tick
}

#[no_mangle]
pub extern "C" fn get_kills() -> u32 {
    let w = get_world();
    w.kills
}

#[no_mangle]
pub extern "C" fn get_births() -> u32 {
    let w = get_world();
    w.births
}

#[no_mangle]
pub extern "C" fn get_agent_size() -> u32 {
    std::mem::size_of::<AgentData>() as u32
}

#[no_mangle]
pub extern "C" fn create_snapshot() -> u32 {
    let w = get_world();
    let snap = archive::WorldSnapshot::new(
        w.tick,
        w.kills,
        w.births,
        w.roots,
        w.next_id,
        w.eclipse,
        &w.soil.food,
        &w.soil.taint,
        &w.soil.scent,
        &w.agents,
    );
    unsafe {
        let buf_ptr = core::ptr::addr_of_mut!(ARCHIVE_BUFFER);
        *buf_ptr = snap.to_bytes();
        (*buf_ptr).len() as u32
    }
}

#[no_mangle]
pub extern "C" fn get_archive_ptr() -> *const u8 {
    unsafe {
        let buf_ptr = core::ptr::addr_of!(ARCHIVE_BUFFER);
        (*buf_ptr).as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn get_agents_mut_ptr() -> *mut AgentData {
    let w = get_world();
    w.agents.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn set_max_capacity(cap: u32) {
    let w = get_world();
    w.set_max_capacity(cap);
}

#[no_mangle]
pub extern "C" fn get_max_capacity() -> u32 {
    let w = get_world();
    w.max_cap as u32
}

#[no_mangle]
pub extern "C" fn set_agents_count(count: u32) {
    let w = get_world();
    let count = (count as usize).min(crate::agent::ABSOLUTE_MAX_CAP);
    w.agents.resize(count, AgentData::default());
    w.pos_x.resize(count, 0.0);
    w.pos_y.resize(count, 0.0);
    w.ensure_grid_capacity(count);
}

#[no_mangle]
pub extern "C" fn sync_pos_cache() {
    let w = get_world();
    w.sync_pos_cache();
}

#[no_mangle]
pub extern "C" fn get_prng_state() -> u32 {
    let w = get_world();
    w.prng.s
}

#[no_mangle]
pub extern "C" fn set_prng_state(state: u32) {
    let w = get_world();
    w.prng.s = state;
}

#[no_mangle]
pub extern "C" fn get_roots() -> u32 {
    let w = get_world();
    w.roots
}

#[no_mangle]
pub extern "C" fn get_next_id() -> u32 {
    let w = get_world();
    w.next_id
}

#[no_mangle]
pub extern "C" fn get_eclipse() -> u32 {
    let w = get_world();
    w.eclipse
}

#[no_mangle]
pub extern "C" fn trigger_eclipse() {
    let w = get_world();
    w.eclipse = 210;
}

#[no_mangle]
pub extern "C" fn set_eclipse(eclipse: u32) {
    let w = get_world();
    w.eclipse = eclipse;
}

#[no_mangle]
pub extern "C" fn set_selective_pressures(mutation: f64, growth: f64, hostility: f64) {
    let w = get_world();
    w.mutation = mutation;
    w.growth = growth;
    w.hostility = hostility;
}

#[no_mangle]
pub extern "C" fn sync_params_to_wasm(
    tick: u32,
    kills: u32,
    births: u32,
    roots: u32,
    next_id: u32,
    eclipse: u32,
    mutation: f64,
    growth: f64,
    hostility: f64,
    prng_state: u32,
) {
    let w = get_world();
    w.tick = tick;
    w.kills = kills;
    w.births = births;
    w.roots = roots;
    w.next_id = next_id;
    w.eclipse = eclipse;
    w.mutation = mutation;
    w.growth = growth;
    w.hostility = hostility;
    w.prng.s = prng_state;
}

#[no_mangle]
pub extern "C" fn resize_world(w: f64, h: f64, cols: u32, rows: u32) {
    let world = get_world();
    world.resize(w, h, cols as usize, rows as usize);
}

#[no_mangle]
pub extern "C" fn get_grid_size() -> u32 {
    let w = get_world();
    w.soil.grid_size as u32
}

#[no_mangle]
pub extern "C" fn get_events_count() -> u32 {
    let w = get_world();
    w.events.len() as u32
}

#[no_mangle]
pub extern "C" fn get_events_ptr() -> *const world::SimEvent {
    let w = get_world();
    w.events.as_ptr()
}

#[no_mangle]
pub extern "C" fn clear_events() {
    let w = get_world();
    w.events.clear();
}

