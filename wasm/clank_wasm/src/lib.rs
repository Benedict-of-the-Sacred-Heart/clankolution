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
        GLOBAL_WORLD.as_mut().expect("World not initialized! Call init_world first.")
    }
}

#[no_mangle]
pub extern "C" fn init_world(seed: u32) {
    unsafe {
        GLOBAL_WORLD = Some(World::new(seed));
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
        ARCHIVE_BUFFER = snap.to_bytes();
        ARCHIVE_BUFFER.len() as u32
    }
}

#[no_mangle]
pub extern "C" fn get_archive_ptr() -> *const u8 {
    unsafe { ARCHIVE_BUFFER.as_ptr() }
}



