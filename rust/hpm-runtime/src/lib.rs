// PHCA v3.0 — HPM Grammar Runtime
// Implements v3.0 §3.2 Definition 3.4-3.7 (HPM Grammar)
// Phase 3.2 component — placeholder with basic structure

use std::collections::HashMap;
use phca_common::ResourceBounds;

/// HPM composition operator types (v3.0 §3.2 Definition 3.4).
#[derive(Debug, Clone, PartialEq)]
pub enum HPMOperator {
    Sequence,
    Parallel,
    Conditional,
    Hierarchy,
    Recurse,
    Interleave,
    TemporalInvariant,
    Reactive,
    AsiInput,
    Predict,
    Control,
}

/// A node in the HPM composition tree.
#[derive(Debug, Clone)]
pub struct HPMNode {
    pub operator: HPMOperator,
    pub children: Vec<HPMNode>,
    pub module_id: Option<String>,
    pub bounds: Option<ResourceBounds>,
}

impl HPMNode {
    pub fn leaf(operator: HPMOperator, module_id: impl Into<String>, bounds: ResourceBounds) -> Self {
        Self {
            operator,
            children: vec![],
            module_id: Some(module_id.into()),
            bounds: Some(bounds),
        }
    }

    pub fn composite(operator: HPMOperator, children: Vec<HPMNode>) -> Self {
        Self {
            operator,
            children,
            module_id: None,
            bounds: None,
        }
    }
}

/// HPM Grammar Runtime — validates and evaluates HPM compositions.
pub struct HPMRuntime {
    module_bounds: HashMap<String, ResourceBounds>,
}

impl HPMRuntime {
    pub fn new() -> Self {
        Self { module_bounds: HashMap::new() }
    }

    /// Register a leaf module with its resource bounds.
    pub fn register_module(&mut self, module_id: impl Into<String>, bounds: ResourceBounds) {
        self.module_bounds.insert(module_id.into(), bounds);
    }

    /// Validate an HPM composition expression.
    /// Returns the validated tree with composite bounds computed.
    /// Panics if validation fails.
    pub fn validate(&self, _spec: &str) -> HPMNode {
        // Phase 3.2: Full implementation with parser, type checker, resource verifier.
        // Phase 3.1: Stub that always passes for test module specs.
        todo!("HPM Grammar Runtime is Phase 3.2 — not yet implemented")
    }

    /// Compute the composite resource bounds for a validated tree.
    /// Uses v3.0 §3.6 corrected resource additivity rules.
    pub fn compute_composite_bounds(&self, node: &HPMNode) -> ResourceBounds {
        match node.operator {
            HPMOperator::Sequence => {
                // B_time = sum + τ_comp, B_mem = max + δ_shared
                let mut total_time: f64 = 0.0;
                let mut max_mem: f64 = 0.0;
                for child in &node.children {
                    let cb = self.compute_composite_bounds(child);
                    total_time += cb.b_time;
                    max_mem = max_mem.max(cb.b_mem);
                }
                ResourceBounds::new(
                    total_time + 0.001, // τ_comp = 1ms
                    max_mem + 1024.0,   // δ_shared = 1KB
                    0.0, 0.0,
                )
            }
            HPMOperator::Parallel => {
                // B_time = max + τ_sync, B_mem = sum + δ_comm
                let mut max_time: f64 = 0.0;
                let mut total_mem: f64 = 0.0;
                for child in &node.children {
                    let cb = self.compute_composite_bounds(child);
                    max_time = max_time.max(cb.b_time);
                    total_mem += cb.b_mem;
                }
                ResourceBounds::new(
                    max_time + 0.002, // τ_sync = 2ms
                    total_mem + 2048.0, // δ_comm = 2KB
                    0.0, 0.0,
                )
            }
            _ => {
                // Leaf node: return stored bounds
                node.bounds.clone().unwrap_or(ResourceBounds::new(0.0, 0.0, 0.0, 0.0))
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_node_creation() {
        let bounds = ResourceBounds::new(0.01, 1000.0, 5.0, 0.01);
        let leaf = HPMNode::leaf(HPMOperator::AsiInput, "ASI", bounds);
        assert_eq!(leaf.module_id.unwrap(), "ASI");
    }

    #[test]
    fn test_sequence_time_additivity() {
        let rt = HPMRuntime::new();
        let bounds1 = ResourceBounds::new(0.01, 1000.0, 5.0, 0.01);
        let bounds2 = ResourceBounds::new(0.02, 2000.0, 10.0, 0.01);

        let leaf1 = HPMNode::leaf(HPMOperator::Predict, "P1", bounds1);
        let leaf2 = HPMNode::leaf(HPMOperator::Predict, "P2", bounds2);
        let seq = HPMNode::composite(HPMOperator::Sequence, vec![leaf1, leaf2]);

        let composite = rt.compute_composite_bounds(&seq);
        assert!((composite.b_time - 0.031).abs() < 0.001); // 0.01 + 0.02 + 0.001
    }

    #[test]
    fn test_parallel_time_is_max() {
        let rt = HPMRuntime::new();
        let bounds1 = ResourceBounds::new(0.01, 1000.0, 5.0, 0.01);
        let bounds2 = ResourceBounds::new(0.05, 2000.0, 10.0, 0.01); // slower

        let leaf1 = HPMNode::leaf(HPMOperator::Predict, "P1", bounds1);
        let leaf2 = HPMNode::leaf(HPMOperator::Predict, "P2", bounds2);
        let par = HPMNode::composite(HPMOperator::Parallel, vec![leaf1, leaf2]);

        let composite = rt.compute_composite_bounds(&par);
        assert!((composite.b_time - 0.052).abs() < 0.001); // max(0.01, 0.05) + 0.002
    }
}
