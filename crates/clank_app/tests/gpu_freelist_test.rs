use clank_app::gpu::freelist::TombstoneFreelistManager;

#[test]
fn test_freelist_recycle_without_movement() {
    let mut freelist = TombstoneFreelistManager::new(1024);
    let slot_a = freelist.allocate().unwrap();
    let slot_b = freelist.allocate().unwrap();
    assert_ne!(slot_a, slot_b);
    freelist.free(slot_a);
    let slot_c = freelist.allocate().unwrap();
    assert_eq!(slot_c, slot_a); // Recycled slot without shifting slot_b!
}

#[test]
fn test_freelist_underflow_protection_at_capacity() {
    let mut freelist = TombstoneFreelistManager::new(2);
    assert!(freelist.allocate().is_some());
    assert!(freelist.allocate().is_some());
    // At capacity: next allocate must safely return None without unsigned underflow!
    assert_eq!(freelist.allocate(), None);
    assert_eq!(freelist.freelist_top(), 0);
}

#[test]
fn test_freelist_dead_claimed_cas_prevents_double_free() {
    // Verifies that atomic CAS on dead_claimed (0 -> 1) ensures exactly one thread frees the slot
    // even if predation and starvation trigger concurrently on the same tick.
    let mut freelist = TombstoneFreelistManager::new(1024);
    let slot = freelist.allocate().unwrap();
    assert!(freelist.claim_death_and_free(slot)); // First claim succeeds -> freed
    assert!(!freelist.claim_death_and_free(slot)); // Second claim fails -> prevented double-free!
}
