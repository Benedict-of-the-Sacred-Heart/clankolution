use clank_app::sim::SimWorld;
use clank_app::ui::MinimapCache;

#[test]
fn test_minimap_cache_stores_and_updates_clusters() {
    let sim = SimWorld::new(42);
    let mut cache = MinimapCache::default();

    // Cache should initially be empty or default
    assert!(cache.clusters.is_empty());

    // Update cache from sim
    clank_app::ui::update_minimap_cache_from_sim(&sim, &mut cache);

    // Living agents in default SimWorld should produce clusters
    assert!(!cache.clusters.is_empty());
    for cluster in &cache.clusters {
        assert!(cluster.count > 0);
        assert!(cluster.radius > 0.0);
        assert!(cluster.center[0] >= 0.0 && cluster.center[0] <= 950.0);
        assert!(cluster.center[1] >= 0.0 && cluster.center[1] <= 750.0);
    }
}
