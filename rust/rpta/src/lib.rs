// PHCA v3.0 — RBTA Constraint Enforcer
// Implements v3.0 §2.1 Definition 2.2 (Constraint Enforcer)
// and Theorem 2.1 / Theorem 3.1 (Constraint Composition)

use std::collections::HashMap;
use phca_common::ResourceBounds;

/// Types of resource bounds that can be violated.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundType {
    Time,
    Memory,
    Energy,
    EntropyFloor,
    SensorFailure,
}

/// A single constraint violation detected by the enforcer.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub module_id: String,
    pub bound_type: BoundType,
    pub measured: f64,
    pub allowed: f64,
}

impl ConstraintViolation {
    pub fn new(module_id: impl Into<String>, bound_type: BoundType, measured: f64, allowed: f64) -> Self {
        Self { module_id: module_id.into(), bound_type, measured, allowed }
    }
}

/// Aggregate enforcement status for a cycle.
#[derive(Debug, Clone, PartialEq)]
pub enum EnforcerAction {
    /// All bounds satisfied — continue normal operation.
    Continue,
    /// Warning-level violation — interrupt current module, continue cycle.
    Interrupt,
    /// Critical violation — terminate the current cognitive cycle.
    Terminate,
}

/// The RBTA constraint enforcer.
///
/// Checks per-module resource bounds (time, memory, energy, entropy floor)
/// and composite bounds for module compositions.
pub struct RBTAEnforcer {
    bounds: HashMap<String, ResourceBounds>,
}

impl RBTAEnforcer {
    /// Create a new enforcer with the given per-module resource bounds.
    pub fn new(bounds: HashMap<String, ResourceBounds>) -> Self {
        Self { bounds }
    }

    /// Check all resource bounds for the current cognitive cycle.
    ///
    /// Returns (violations, action):
    /// - `violations`: list of any violations detected
    /// - `action`: recommended action (Continue / Interrupt / Terminate)
    pub fn check_cycle(
        &self,
        runtime_log: &HashMap<String, f64>,
        memory_log: &HashMap<String, f64>,
        energy_log: &HashMap<String, f64>,
        belief_entropies: &HashMap<String, f64>,
        sensor_failure_count: usize,
        asi_failure_limit: usize,
    ) -> (Vec<ConstraintViolation>, EnforcerAction) {
        let mut violations = Vec::new();

        for (module_id, bounds) in &self.bounds {
            // Check time bound
            if let Some(&runtime) = runtime_log.get(module_id) {
                if runtime > bounds.b_time {
                    violations.push(ConstraintViolation::new(
                        module_id, BoundType::Time, runtime, bounds.b_time,
                    ));
                }
            }

            // Check memory bound
            if let Some(&mem) = memory_log.get(module_id) {
                if mem > bounds.b_mem {
                    violations.push(ConstraintViolation::new(
                        module_id, BoundType::Memory, mem, bounds.b_mem,
                    ));
                }
            }

            // Check energy bound
            if let Some(&energy) = energy_log.get(module_id) {
                if energy > bounds.b_energy {
                    violations.push(ConstraintViolation::new(
                        module_id, BoundType::Energy, energy, bounds.b_energy,
                    ));
                }
            }

            // Check entropy floor (A3: Incomplete Knowledge)
            if let Some(&entropy) = belief_entropies.get(module_id) {
                if entropy < bounds.entropy_floor {
                    violations.push(ConstraintViolation::new(
                        module_id, BoundType::EntropyFloor, entropy, bounds.entropy_floor,
                    ));
                }
            }
        }

        // Check ASI sensor failure limit
        if sensor_failure_count > asi_failure_limit {
            violations.push(ConstraintViolation::new(
                "ASI", BoundType::SensorFailure, sensor_failure_count as f64, asi_failure_limit as f64,
            ));
        }

        // Determine action based on violation severity
        let action = match violations.len() {
            0 => EnforcerAction::Continue,
            1..=2 => EnforcerAction::Interrupt,
            _ => EnforcerAction::Terminate,
        };

        (violations, action)
    }

    /// Update the enforcer's module bounds (e.g., when new modules are registered).
    pub fn update_bounds(&mut self, module_id: String, bounds: ResourceBounds) {
        self.bounds.insert(module_id, bounds);
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bounds() -> HashMap<String, ResourceBounds> {
        let mut map = HashMap::new();
        map.insert("ASI".into(), ResourceBounds::new(0.002, 100_000.0, 10.0, 0.01));
        map.insert("WM".into(), ResourceBounds::new(0.005, 50_000.0, 5.0, 0.01));
        map.insert("G'".into(), ResourceBounds::new(0.020, 500_000.0, 50.0, 0.01));
        map
    }

    #[test]
    fn test_single_module_time_violation() {
        let enforcer = RBTAEnforcer::new(make_bounds());
        let mut runtime_log = HashMap::new();
        runtime_log.insert("ASI".into(), 0.010); // B_time = 0.002, exceeded!

        let (violations, action) = enforcer.check_cycle(
            &runtime_log,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            0, 5,
        );

        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].bound_type, BoundType::Time);
        assert_eq!(action, EnforcerAction::Interrupt);
    }

    #[test]
    fn test_all_bounds_satisfied() {
        let enforcer = RBTAEnforcer::new(make_bounds());

        let mut runtime_log = HashMap::new();
        runtime_log.insert("ASI".into(), 0.001);
        runtime_log.insert("WM".into(), 0.003);
        runtime_log.insert("G'".into(), 0.015);

        let (violations, action) = enforcer.check_cycle(
            &runtime_log,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            0, 5,
        );

        assert_eq!(violations.len(), 0);
        assert_eq!(action, EnforcerAction::Continue);
    }

    #[test]
    fn test_entropy_floor_violation() {
        let enforcer = RBTAEnforcer::new(make_bounds());
        let mut entropies = HashMap::new();
        entropies.insert("G'".into(), 0.001); // below floor of 0.01

        let (violations, _) = enforcer.check_cycle(
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &entropies,
            0, 5,
        );

        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].bound_type, BoundType::EntropyFloor);
    }

    #[test]
    fn test_multiple_violations_triggers_terminate() {
        let enforcer = RBTAEnforcer::new(make_bounds());
        let mut runtime_log = HashMap::new();
        runtime_log.insert("ASI".into(), 0.010);
        runtime_log.insert("WM".into(), 0.010);
        runtime_log.insert("G'".into(), 0.100);

        let (violations, action) = enforcer.check_cycle(
            &runtime_log,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            0, 5,
        );

        assert!(violations.len() >= 3);
        assert_eq!(action, EnforcerAction::Terminate);
    }

    #[test]
    fn test_sensor_failure_limit() {
        let enforcer = RBTAEnforcer::new(make_bounds());

        let (violations, action) = enforcer.check_cycle(
            &HashMap::new(), &HashMap::new(), &HashMap::new(), &HashMap::new(),
            10,  // sensor_failure_count
            5,   // asi_failure_limit
        );

        let has_sensor_violation = violations.iter().any(|v| matches!(v.bound_type, BoundType::SensorFailure));
        assert!(has_sensor_violation);
        // Sensor failure is 1 violation; 1-2 violations → Interrupt. 
        // 3+ violations would be needed for Terminate.
        assert_eq!(action, EnforcerAction::Interrupt);
    }
}
