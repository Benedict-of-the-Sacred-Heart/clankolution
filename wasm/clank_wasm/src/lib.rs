pub mod math;
pub mod agent;
pub mod soil;
pub mod archive;
pub mod world;

use agent::AgentData;
use world::World;
use rkyv::Deserialize;

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
        w.w,
        w.h,
        w.soil.cols as u32,
        w.soil.rows as u32,
        w.mutation,
        w.growth,
        w.hostility,
        w.prng.s,
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
pub extern "C" fn alloc_archive_buffer(len: u32) -> *mut u8 {
    unsafe {
        let buf_ptr = core::ptr::addr_of_mut!(ARCHIVE_BUFFER);
        (*buf_ptr).resize(len as usize, 0);
        (*buf_ptr).as_mut_ptr()
    }
}

#[no_mangle]
pub extern "C" fn restore_snapshot() -> u32 {
    unsafe {
        let buf_ptr = core::ptr::addr_of!(ARCHIVE_BUFFER);
        let bytes = (*buf_ptr).as_slice();
        let archived = match rkyv::check_archived_root::<archive::WorldSnapshot>(bytes) {
            Ok(a) => a,
            Err(_) => return 1, // Corrupted / invalid archive
        };

        if archived.version != archive::WorldSnapshot::CURRENT_VERSION {
            return 2; // Unsupported version
        }

        let w = get_world();

        let target_w: f64 = archived.width.into();
        let target_h: f64 = archived.height.into();
        let target_cols: u32 = archived.cols.into();
        let target_rows: u32 = archived.rows.into();
        if w.w != target_w || w.h != target_h || w.soil.cols != target_cols as usize || w.soil.rows != target_rows as usize {
            w.resize(target_w, target_h, target_cols as usize, target_rows as usize);
        }

        w.tick = archived.tick.into();
        w.kills = archived.kills.into();
        w.births = archived.births.into();
        w.roots = archived.roots.into();
        w.next_id = archived.next_id.into();
        w.eclipse = archived.eclipse.into();
        w.mutation = archived.mutation.into();
        w.growth = archived.growth.into();
        w.hostility = archived.hostility.into();
        w.prng.s = archived.prng_state.into();

        let food_len = archived.food.len().min(w.soil.food.len());
        for i in 0..food_len {
            w.soil.food[i] = archived.food[i].into();
        }
        let taint_len = archived.taint.len().min(w.soil.taint.len());
        for i in 0..taint_len {
            w.soil.taint[i] = archived.taint[i].into();
        }
        let scent_len = archived.scent.len().min(w.soil.scent.len());
        for i in 0..scent_len {
            w.soil.scent[i] = archived.scent[i].into();
        }

        let snap_agents: Vec<crate::agent::AgentData> = match archived.agents.deserialize(&mut rkyv::Infallible) {
            Ok(ag) => ag,
            Err(_) => return 3,
        };

        if snap_agents.len() > w.max_cap {
            w.set_max_capacity(snap_agents.len() as u32);
        }

        w.agents.clear();
        w.agents.extend(snap_agents);

        w.build_grid();

        0 // Success
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
pub extern "C" fn get_mutation() -> f64 {
    let w = get_world();
    w.mutation
}

#[no_mangle]
pub extern "C" fn get_growth() -> f64 {
    let w = get_world();
    w.growth
}

#[no_mangle]
pub extern "C" fn get_hostility() -> f64 {
    let w = get_world();
    w.hostility
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

#[no_mangle]
pub extern "C" fn get_sparks_count() -> u32 {
    let w = get_world();
    w.spark_events.len() as u32
}

#[no_mangle]
pub extern "C" fn get_sparks_ptr() -> *const world::SparkEvent {
    let w = get_world();
    w.spark_events.as_ptr()
}

#[no_mangle]
pub extern "C" fn clear_sparks() {
    let w = get_world();
    w.spark_events.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_restore_snapshot_roundtrip() {
        init_world(42);
        evolve_ticks(15);

        let tick_saved = get_tick();
        let kills_saved = get_kills();
        let births_saved = get_births();
        let agents_saved = get_agents_count();
        let first_agent_x = get_world().agents[0].x;
        let first_agent_energy = get_world().agents[0].energy;

        let len = create_snapshot();
        assert!(len > 0);

        let mut saved_bytes = vec![0u8; len as usize];
        unsafe {
            saved_bytes.copy_from_slice(std::slice::from_raw_parts(get_archive_ptr(), len as usize));
        }

        // Simulate 50 more ticks to mutate world state
        evolve_ticks(50);
        assert!(get_tick() > tick_saved);

        // Upload and restore
        let buf_ptr = alloc_archive_buffer(len);
        unsafe {
            std::slice::from_raw_parts_mut(buf_ptr, len as usize).copy_from_slice(&saved_bytes);
        }

        let res = restore_snapshot();
        assert_eq!(res, 0);

        assert_eq!(get_tick(), tick_saved);
        assert_eq!(get_kills(), kills_saved);
        assert_eq!(get_births(), births_saved);
        assert_eq!(get_agents_count(), agents_saved);
        assert_eq!(get_world().agents[0].x, first_agent_x);
        assert_eq!(get_world().agents[0].energy, first_agent_energy);

        // Continue running after restore to ensure physics and lifecycle continue smoothly
        evolve_ticks(20);
        assert_eq!(get_tick(), tick_saved + 20);
    }
}


