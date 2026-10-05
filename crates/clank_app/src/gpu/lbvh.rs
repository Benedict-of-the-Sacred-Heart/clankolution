//! Karras 2012 Linear Bounding Volume Hierarchy (LBVH)
//!
//! Provides $O(\log N)$ spatial queries for mouse specimen picking with dynamic radius shrinking,
//! AoE tool application, and hierarchical Barnes-Hut / Minimap LOD clustering.

use super::types::{GpuAgentState, GpuLbvhNode};

/// Computes the longest common prefix (LCP) between two Morton keys.
/// Breaks ties on equal keys using the 32-bit unique agent slot ID.
#[inline]
pub fn common_prefix_length(i: i32, j: i32, keys: &[[u32; 2]]) -> i32 {
    let n = keys.len() as i32;
    if j < 0 || j >= n {
        return -1;
    }
    let key_i = keys[i as usize][0];
    let key_j = keys[j as usize][0];
    if key_i != key_j {
        return (key_i ^ key_j).leading_zeros() as i32;
    }
    let id_i = keys[i as usize][1];
    let id_j = keys[j as usize][1];
    32 + (id_i ^ id_j).leading_zeros() as i32
}

/// Evaluates branchless 1D shortest distance from point $p$ to interval $[b_{min}, b_{max}]$ on a toroidal domain of size $w$.
#[inline]
pub fn toroidal_aabb_dist_1d(p: f32, b_min: f32, b_max: f32, w: f32) -> f32 {
    if p >= b_min && p <= b_max {
        return 0.0;
    }
    let direct = if p < b_min { b_min - p } else { p - b_max };
    let wrapped = if p < b_min { w - b_max + p } else { w - p + b_min };
    direct.min(wrapped)
}

/// Evaluates Euclidean distance from point to AABB under toroidal boundary conditions.
#[inline]
pub fn distance_to_aabb(pos: [f32; 2], aabb_min: [f32; 2], aabb_max: [f32; 2]) -> f32 {
    let dx = toroidal_aabb_dist_1d(pos[0], aabb_min[0], aabb_max[0], 900.0);
    let dy = toroidal_aabb_dist_1d(pos[1], aabb_min[1], aabb_max[1], 600.0);
    (dx * dx + dy * dy).sqrt()
}

/// Toroidal Euclidean distance between two points on the 900x600 arena.
#[inline]
pub fn toroidal_dist(p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let dx = (p1[0] - p2[0]).abs();
    let x_dist = dx.min(900.0 - dx);
    let dy = (p1[1] - p2[1]).abs();
    let y_dist = dy.min(600.0 - dy);
    (x_dist * x_dist + y_dist * y_dist).sqrt()
}

/// Linear BVH Tree representation for fast CPU queries and GPU buffer testing.
#[derive(Clone, Debug, Default)]
pub struct LbvhTree {
    pub nodes: Vec<GpuLbvhNode>,
}

impl LbvhTree {
    /// Builds a Karras LBVH hierarchy from a slice of agents.
    /// Handles degenerate populations (N = 0 or N = 1) safely without underflow.
    pub fn build(agents: &[GpuAgentState]) -> Self {
        let mut keys: Vec<[u32; 2]> = agents
            .iter()
            .enumerate()
            .map(|(i, a)| [a.morton_code, i as u32])
            .collect();
        keys.sort_unstable_by(|a, b| a[0].cmp(&b[0]).then_with(|| a[1].cmp(&b[1])));

        Self::build_from_keys(&keys, agents)
    }

    /// Builds a Karras LBVH hierarchy from sorted keys (morton_key, slot_idx) and agents slice.
    pub fn build_from_keys(keys: &[[u32; 2]], agents: &[GpuAgentState]) -> Self {
        let n = keys.len();
        if n == 0 {
            return Self { nodes: Vec::new() };
        }
        if n == 1 {
            let agent_idx = keys[0][1] as usize;
            let a = &agents[agent_idx];
            let r = 2.0 + 3.0 * a.traits[0];
            let root_node = GpuLbvhNode {
                aabb_min: [a.pos_vel[0] - r, a.pos_vel[1] - r],
                aabb_max: [a.pos_vel[0] + r, a.pos_vel[1] + r],
                center_of_mass: [a.pos_vel[0], a.pos_vel[1]],
                count: 1,
                dominant_lineage: a.meta_flags & 0x0F,
                left_child: 0xFFFFFFFF,
                right_child: 0xFFFFFFFF,
                parent: 0xFFFFFFFF,
                leaf_idx: agent_idx as u32,
            };
            return Self {
                nodes: vec![root_node],
            };
        }

        let num_internal = n - 1;
        let total_nodes = num_internal + n;
        let mut nodes = vec![
            GpuLbvhNode {
                aabb_min: [f32::INFINITY, f32::INFINITY],
                aabb_max: [f32::NEG_INFINITY, f32::NEG_INFINITY],
                center_of_mass: [0.0, 0.0],
                count: 0,
                dominant_lineage: 0,
                left_child: 0xFFFFFFFF,
                right_child: 0xFFFFFFFF,
                parent: 0xFFFFFFFF,
                leaf_idx: 0xFFFFFFFF,
            };
            total_nodes
        ];

        // Initialize leaf nodes at index [num_internal .. total_nodes]
        for (i, k) in keys.iter().enumerate() {
            let agent_idx = k[1] as usize;
            let a = &agents[agent_idx];
            let r = 2.0 + 3.0 * a.traits[0];
            let leaf_node_idx = num_internal + i;
            nodes[leaf_node_idx] = GpuLbvhNode {
                aabb_min: [a.pos_vel[0] - r, a.pos_vel[1] - r],
                aabb_max: [a.pos_vel[0] + r, a.pos_vel[1] + r],
                center_of_mass: [a.pos_vel[0], a.pos_vel[1]],
                count: 1,
                dominant_lineage: a.meta_flags & 0x0F,
                left_child: 0xFFFFFFFF,
                right_child: 0xFFFFFFFF,
                parent: 0xFFFFFFFF,
                leaf_idx: agent_idx as u32,
            };
        }

        // Phase 1: Build Radix tree hierarchy (internal nodes 0..n-2)
        for i in 0..num_internal as i32 {
            let delta_next = common_prefix_length(i, i + 1, keys);
            let delta_prev = common_prefix_length(i, i - 1, keys);
            let d = if delta_next > delta_prev { 1 } else { -1 };
            let delta_min = common_prefix_length(i, i - d, keys);

            // Find upper bound l_max
            let mut l_max = 2;
            while common_prefix_length(i, i + l_max * d, keys) > delta_min {
                l_max *= 2;
            }

            // Binary search range length
            let mut l = 0;
            let mut step = l_max / 2;
            while step > 0 {
                if common_prefix_length(i, i + (l + step) * d, keys) > delta_min {
                    l += step;
                }
                step /= 2;
            }

            let j = i + l * d;
            let first = i.min(j);
            let last = i.max(j);
            let delta_node = common_prefix_length(first, last, keys);

            // Binary search split point (Karras 2012)
            let mut split = first;
            let mut step = last - first;
            loop {
                step = (step + 1) / 2;
                let new_split = split + step;
                if new_split < last {
                    if common_prefix_length(first, new_split, keys) > delta_node {
                        split = new_split;
                    }
                }
                if step <= 1 {
                    break;
                }
            }
            let gamma = split;

            let left = if gamma == first {
                num_internal as u32 + gamma as u32
            } else {
                gamma as u32
            };

            let right = if gamma + 1 == last {
                num_internal as u32 + (gamma + 1) as u32
            } else {
                (gamma + 1) as u32
            };

            nodes[i as usize].left_child = left;
            nodes[i as usize].right_child = right;
            nodes[left as usize].parent = i as u32;
            nodes[right as usize].parent = i as u32;
        }

        // Phase 2: Compute bounding boxes bottom-up
        let mut flags = vec![0u32; num_internal];
        for i in 0..n {
            let mut curr = nodes[num_internal + i].parent;
            while curr != 0xFFFFFFFF {
                flags[curr as usize] += 1;
                if flags[curr as usize] < 2 {
                    // First child arrived; terminate walk
                    break;
                }
                // Second child arrived: compute bounding box and metrics
                let left = nodes[curr as usize].left_child as usize;
                let right = nodes[curr as usize].right_child as usize;

                let min_x = nodes[left].aabb_min[0].min(nodes[right].aabb_min[0]);
                let min_y = nodes[left].aabb_min[1].min(nodes[right].aabb_min[1]);
                let max_x = nodes[left].aabb_max[0].max(nodes[right].aabb_max[0]);
                let max_y = nodes[left].aabb_max[1].max(nodes[right].aabb_max[1]);

                let total_count = nodes[left].count + nodes[right].count;
                let com_x = (nodes[left].center_of_mass[0] * nodes[left].count as f32
                    + nodes[right].center_of_mass[0] * nodes[right].count as f32)
                    / total_count.max(1) as f32;
                let com_y = (nodes[left].center_of_mass[1] * nodes[left].count as f32
                    + nodes[right].center_of_mass[1] * nodes[right].count as f32)
                    / total_count.max(1) as f32;

                let dominant = if nodes[left].count >= nodes[right].count {
                    nodes[left].dominant_lineage
                } else {
                    nodes[right].dominant_lineage
                };

                nodes[curr as usize].aabb_min = [min_x, min_y];
                nodes[curr as usize].aabb_max = [max_x, max_y];
                nodes[curr as usize].center_of_mass = [com_x, com_y];
                nodes[curr as usize].count = total_count;
                nodes[curr as usize].dominant_lineage = dominant;

                curr = nodes[curr as usize].parent;
            }
        }

        Self { nodes }
    }

    /// Returns the index of the root node (the node with parent == 0xFFFFFFFF).
    pub fn root_index(&self) -> u32 {
        if self.nodes.is_empty() || self.nodes.len() == 1 {
            return 0;
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if node.leaf_idx == 0xFFFFFFFF && node.parent == 0xFFFFFFFF {
                return i as u32;
            }
        }
        0
    }

    /// Evaluates mouse picking query matching `spatial_query.wgsl`.
    /// Returns (selected_agent_idx, selected_agent_id).
    pub fn pick_agent(
        &self,
        cursor: [f32; 2],
        agents: &[GpuAgentState],
        initial_search_radius: f32,
    ) -> (Option<u32>, Option<u32>) {
        if agents.is_empty() || self.nodes.is_empty() {
            return (None, None);
        }
        if agents.len() == 1 {
            let a = &agents[0];
            let d = toroidal_dist(cursor, [a.pos_vel[0], a.pos_vel[1]]);
            if d <= initial_search_radius {
                return (Some(0), Some(a.id));
            }
            return (None, None);
        }

        let mut search_r = initial_search_radius;
        let mut best_idx = None;
        let mut best_id = None;
        let mut best_priority = 2u32; // 0 = body hit, 1 = halo, 2 = none
        let mut best_dist = search_r;

        let mut stack = Vec::with_capacity(64);
        stack.push(self.root_index());

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            let box_dist = distance_to_aabb(cursor, node.aabb_min, node.aabb_max);
            if box_dist > search_r {
                continue; // Prune branch!
            }

            if node.leaf_idx != 0xFFFFFFFF {
                let agent_idx = node.leaf_idx as usize;
                let a = &agents[agent_idx];
                let d = toroidal_dist(cursor, [a.pos_vel[0], a.pos_vel[1]]);
                let visual_r = 2.0 + 3.0 * a.traits[0];
                let priority = if d <= visual_r { 0 } else { 1 };

                if d <= search_r {
                    if priority < best_priority || (priority == best_priority && d < best_dist) {
                        best_priority = priority;
                        best_dist = d;
                        best_idx = Some(agent_idx as u32);
                        best_id = Some(a.id);

                        // Dynamic Radius Shrinking:
                        if priority == 0 {
                            search_r = search_r.min(d);
                        }
                    }
                }
            } else {
                if node.right_child != 0xFFFFFFFF {
                    stack.push(node.right_child);
                }
                if node.left_child != 0xFFFFFFFF {
                    stack.push(node.left_child);
                }
            }
        }

        (best_idx, best_id)
    }

    /// Evaluates AoE tool query: returns all agents within radius of tool_pos.
    pub fn query_aoe(
        &self,
        tool_pos: [f32; 2],
        tool_radius: f32,
        agents: &[GpuAgentState],
    ) -> Vec<u32> {
        let mut results = Vec::new();
        if agents.is_empty() || self.nodes.is_empty() {
            return results;
        }

        let mut stack = Vec::with_capacity(64);
        stack.push(0u32);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            let box_dist = distance_to_aabb(tool_pos, node.aabb_min, node.aabb_max);
            if box_dist > tool_radius {
                continue;
            }

            if node.leaf_idx != 0xFFFFFFFF {
                let agent_idx = node.leaf_idx as usize;
                let a = &agents[agent_idx];
                let d = toroidal_dist(tool_pos, [a.pos_vel[0], a.pos_vel[1]]);
                if d <= tool_radius {
                    results.push(agent_idx as u32);
                }
            } else {
                if node.right_child != 0xFFFFFFFF {
                    stack.push(node.right_child);
                }
                if node.left_child != 0xFFFFFFFF {
                    stack.push(node.left_child);
                }
            }
        }

        results
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Viewport frustum culling: traverses the LBVH discarding entire subtrees
    /// that do not intersect the camera view rectangle [view_min, view_max].
    pub fn cull_frustum(
        &self,
        view_min: [f32; 2],
        view_max: [f32; 2],
        agents: &[GpuAgentState],
    ) -> Vec<u32> {
        let mut visible = Vec::new();
        if agents.is_empty() || self.nodes.is_empty() {
            return visible;
        }

        let mut stack = Vec::with_capacity(64);
        stack.push(self.root_index());

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx as usize];

            // AABB vs Viewport disjoint test
            if node.aabb_max[0] < view_min[0]
                || node.aabb_min[0] > view_max[0]
                || node.aabb_max[1] < view_min[1]
                || node.aabb_min[1] > view_max[1]
            {
                continue; // Discard off-screen subtree!
            }

            if node.leaf_idx != 0xFFFFFFFF {
                let agent_idx = node.leaf_idx as usize;
                let a = &agents[agent_idx];
                if a.pos_vel[0] >= view_min[0]
                    && a.pos_vel[0] <= view_max[0]
                    && a.pos_vel[1] >= view_min[1]
                    && a.pos_vel[1] <= view_max[1]
                {
                    visible.push(agent_idx as u32);
                }
            } else {
                if node.right_child != 0xFFFFFFFF {
                    stack.push(node.right_child);
                }
                if node.left_child != 0xFFFFFFFF {
                    stack.push(node.left_child);
                }
            }
        }

        visible
    }

    /// Extracts aggregated cluster discs from intermediate tree depths (e.g. depth 5–6)
    /// for hierarchical radar minimap LOD rendering.
    pub fn extract_minimap_clusters(&self, target_depth: usize) -> Vec<MinimapCluster> {

        let mut clusters = Vec::new();
        if self.nodes.is_empty() {
            return clusters;
        }

        let mut stack = Vec::new();
        stack.push((self.root_index(), 0usize)); // (node_idx, current_depth)

        while let Some((node_idx, depth)) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            if depth == target_depth || node.leaf_idx != 0xFFFFFFFF {
                if node.count > 0 {
                    let w = node.aabb_max[0] - node.aabb_min[0];
                    let h = node.aabb_max[1] - node.aabb_min[1];
                    let radius = (w.max(h) * 0.5).max(3.0);
                    clusters.push(MinimapCluster {
                        center: node.center_of_mass,
                        count: node.count,
                        dominant_lineage: node.dominant_lineage,
                        radius,
                    });
                }
            } else {
                if node.left_child != 0xFFFFFFFF {
                    stack.push((node.left_child, depth + 1));
                }
                if node.right_child != 0xFFFFFFFF {
                    stack.push((node.right_child, depth + 1));
                }
            }
        }

        clusters
    }
}

/// Aggregated cluster disc for hierarchical radar minimap LOD rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimapCluster {
    pub center: [f32; 2],
    pub count: u32,
    pub dominant_lineage: u32,
    pub radius: f32,
}

impl std::ops::Index<usize> for LbvhTree {
    type Output = GpuLbvhNode;
    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

