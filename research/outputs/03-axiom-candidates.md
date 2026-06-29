# Axiom Candidates — Universal Principles of Intelligence

**Output:** Phase 1 Deliverable
**File:** outputs/03-axiom-candidates.md
**Status:** COMPLETE
**Cross-refs:** [01-intelligence-constraints-model](01-intelligence-constraints-model.md), [outputs/04-architecture-readiness-report.md](04-architecture-readiness-report.md)

---

## Purpose

A filtered list of candidate **universal principles** — statements that appear to be true of any intelligence in any physical universe. These are candidates for axioms that could guide architecture design.

---

## Selection Criteria

Each candidate axiom must satisfy:
1. **Universatility:** Would it apply to any intelligent system (biological, artificial, alien)?
2. **Non-triviality:** Is it more than a definitional tautology?
3. **Evidence:** Is there strong theoretical and/or empirical support?
4. **Usefulness:** Would it guide architecture design?

---

## Tier 1: Strong Candidates (Well-Supported)

These axioms have strong theoretical and empirical support and appear truly universal.

### Axiom 1: Resource Boundedness

> Every intelligent system operates under finite computational, memory, and energy resources.

- **Status:** VERIFIED
- **Evidence:** Thermodynamics, information theory, computer science
- **Corollary:** Tradeoffs between resource allocation are inevitable
- **Corollary:** Optimal solutions are not always reachable

### Axiom 2: Temporal Causality

> Causes precede effects; information propagates at finite speed.

- **Status:** VERIFIED
- **Evidence:** Physics (causality, relativity)
- **Corollary:** All decisions use outdated information
- **Corollary:** Prediction is necessary to compensate for delay

### Axiom 3: Incomplete Knowledge

> No finite system can have complete knowledge of its environment or itself.

- **Status:** VERIFIED
- **Evidence:** Gödel's incompleteness, complexity theory, FLP impossibility
- **Corollary:** Uncertainty is irreducible
- **Corollary:** Self-modeling is always approximate

### Axiom 4: Prediction as Primary

> Intelligence is fundamentally about predicting future states to guide action.

- **Status:** STRONGLY SUPPORTED
- **Evidence:** Predictive coding (neuroscience), control theory, reinforcement learning, information theory
- **Corollary:** Systems learn from prediction errors
- **Corollary:** Action can be understood as prediction fulfillment

### Axiom 5: Feedback-Driven Adaptation

> Adaptive intelligence requires closed-loop feedback between action and perception.

- **Status:** VERIFIED
- **Evidence:** Control theory, cybernetics, reinforcement learning
- **Corollary:** Open-loop systems cannot adapt
- **Corollary:** Feedback delay constrains stability

---

## Tier 2: Probable Candidates (Supported but Conditional)

These axioms have strong support but may apply only under certain conditions.

### Axiom 6: Compression = Understanding

> Understanding the world is equivalent to finding compressed representations that enable prediction.

- **Status:** STRONGLY SUPPORTED
- **Evidence:** Information bottleneck principle, MDL, Kolmogorov complexity
- **Reservation:** May conflate understanding with compression; some forms of understanding may not be compressible
- **Corollary:** The best model is the simplest that fits the data

### Axiom 7: Hierarchical Organization

> Intelligent systems organize knowledge hierarchically across multiple levels of abstraction.

- **Status:** STRONGLY SUPPORTED
- **Evidence:** Neuroscience (cortical hierarchy), cognitive architectures (Soar subgoaling), deep learning
- **Reservation:** Alternative flat architectures may exist
- **Corollary:** Lower levels handle detail; higher levels handle abstraction

### Axiom 8: Criticality Optimality

> Optimal information processing occurs at the edge of chaos.

- **Status:** SUPPORTED
- **Evidence:** Langton (1990), self-organized criticality, Zhang et al. (2024)
- **Reservation:** Not proven for all forms of intelligence; some systems may operate far from criticality
- **Corollary:** Systems must self-tune to remain in the critical regime

### Axiom 9: Multiple Memory Systems

> Effective cognition requires multiple memory systems with different timescales and access characteristics.

- **Status:** STRONGLY SUPPORTED for biological cognition
- **Evidence:** Neuroscience (Squire, Baddeley), cognitive architectures
- **Reservation:** Not proven necessary for all possible intelligences
- **Corollary:** No single memory mechanism suffices for all cognitive functions

### Axiom 10: Emergence from Interaction

> Complex intelligence emerges from the interaction of simpler components, not from a single monolithic mechanism.

- **Status:** STRONGLY SUPPORTED
- **Evidence:** Complex systems theory, neuroscience, multi-agent systems
- **Reservation:** Some cognitive functions may be substrate-specific; emergence may not explain everything
- **Corollary:** Intelligence is not reducible to any single algorithm

---

## Tier 3: Speculative Candidates (Promising but Unverified)

These axioms are intellectually compelling but lack sufficient evidence.

### Axiom 11: Embodiment Necessity

> All intelligence is grounded in sensorimotor interaction with an environment.

- **Status:** [UNVERIFIED] — challenged by purely symbolic AI
- **Evidence:** Embodied cognition, Brooks, Maturana & Varela
- **Counterevidence:** LLMs exhibit intelligent behavior without embodiment
- **Need:** More evidence on whether LLM intelligence is "genuine" or merely simulated

### Axiom 12: Conservation of Intelligence

> Total intelligence in a system is conserved (cannot be created or destroyed, only redistributed).

- **Status:** [UNVERIFIED] — purely speculative
- **Evidence:** None direct; analogy to conservation laws in physics
- **Status:** Philosophical speculation

### Axiom 13: Intelligence Scales With Complexity

> Intelligence capacity scales with system complexity (component count × interaction diversity).

- **Status:** [UNVERIFIED] — observed in AI scaling but mechanism unclear
- **Evidence:** Scaling laws in LLMs, network complexity in brains
- **Counterevidence:** Diminishing returns observed in scaling

---

## Axiom Refinement Summary

| ID | Axiom | Tier | Status | Use in Architecture |
| :--- | :--- | :--- | :--- | :--- |
| A1 | Resource Boundedness | 1 | VERIFIED | Design for tradeoffs |
| A2 | Temporal Causality | 1 | VERIFIED | Prediction is mandatory |
| A3 | Incomplete Knowledge | 1 | VERIFIED | Handle uncertainty explicitly |
| A4 | Prediction as Primary | 1 | STRONG | Prediction error as learning signal |
| A5 | Feedback-Driven Adaptation | 1 | VERIFIED | Closed-loop design |
| A6 | Compression = Understanding | 2 | STRONG | MDL principle as guide |
| A7 | Hierarchical Organization | 2 | STRONG | Multiple abstraction levels |
| A8 | Criticality Optimality | 2 | SUPPORTED | Self-tuning mechanisms |
| A9 | Multiple Memory Systems | 2 | STRONG | Separate memory types |
| A10 | Emergence from Interaction | 2 | STRONG | Design interactions, not just components |
| A11 | Embodiment Necessity | 3 | UNVERIFIED | Conditional design choice |
| A12 | Conservation of Intelligence | 3 | SPECULATIVE | Not yet useful |
| A13 | Intelligence Scales with Complexity | 3 | UNVERIFIED | Tentative scaling guidance |
