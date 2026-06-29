# Failure Modes Map

**Output:** Phase 1 Deliverable
**File:** outputs/02-failure-modes-map.md
**Status:** COMPLETE
**Cross-refs:** [04-failure-modes/01-failure-taxonomy.md](../04-failure-modes/01-failure-taxonomy.md), [01-intelligence-constraints-model](01-intelligence-constraints-model.md)

---

## Purpose

A categorized taxonomy of why AI systems fail, organized by root cause category, with cross-references to the Constraints Model.

---

## Category A: Specification Failures

The system's objective does not match the designer's intent.

| Failure | Description | Related Constraints |
| :--- | :--- | :--- |
| **A1 — Reward Hacking** | Exploiting loopholes in the reward function | C2.1 (Bounded Rationality — specification is incomplete) |
| **A2 — Goal Misgeneralization** | Pursuing a different goal than intended | C2.2 (Non-Self-Representation — cannot fully specify self) |
| **A3 — Specification Gaming** | Fulfilling literal spec while violating intention | C1.3 (Bounded Storage — cannot capture all context) |
| **A4 — Proxy Mismatch** | Optimizing a proxy metric destroys the actual goal | C3.2 (Information Bottleneck — proxy is lossy compression) |
| **A5 — Inner Alignment Failure** | Learned objective diverges from training objective | C2.1 (Bounded Rationality — learning is approximate) |

**Root Cause:** The reward/objective function is always a lossy compression of the true intent.

---

## Category B: Generalization Failures

The system fails when encountering situations outside its training/experience distribution.

| Failure | Description | Related Constraints |
| :--- | :--- | :--- |
| **B1 — Distributional Shift** | Test data differs from training data | C1.3 (Bounded Storage — cannot train on all scenarios) |
| **B2 — Spurious Correlation** | Relying on non-causal proxies that happen to correlate | C2.3 (Incomplete Prediction — correlation ≠ causation) |
| **B3 — Overfitting** | Memorizing training data instead of learning patterns | C1.3 (Bounded Storage — model is too large for data) |
| **B4 — Catastrophic Forgetting** | Learning new information destroys old knowledge | C1.3 (Bounded Storage — finite weights must be overwritten) |
| **B5 — Mode Collapse** | Output diversity collapses to repetitive responses | C5.2 (Threshold Effect — system falls below complexity threshold) |

**Root Cause:** Finite training data cannot cover an infinite world. The system's model is always incomplete.

---

## Category C: Stability Failures

The system's dynamics become unstable or oscillatory.

| Failure | Description | Related Constraints |
| :--- | :--- | :--- |
| **C1 — Feedback Instability** | Small errors compound into large failures | C1.2 (Temporal Locality — feedback delay causes oscillation) |
| **C2 — Overfitting to Internal Model** | Trusting flawed internal predictions over reality | C2.2 (Non-Self-Representation — model is always approximate) |
| **C3 — Positive Feedback Runaway** | Amplification without countervailing regulation | C5.3 (Feedback Requirement — unregulated feedback diverges) |
| **C4 — Phase Collapse** | Falling out of critical regime (too rigid or too chaotic) | C4.2 (Criticality — losing optimal balance) |
| **C5 — Harmonic Oscillation** | System cycles between states without resolution | C1.2 (Temporal Locality — delays cause overshoot) |

**Root Cause:** All control systems are subject to stability constraints. Delay + feedback → oscillation.

---

## Category D: Emergent Failures

System-level failures that arise from component interactions.

| Failure | Description | Related Constraints |
| :--- | :--- | :--- |
| **D1 — Model Collapse** | Training on AI-generated data degrades output quality | C5.2 (Threshold Effect — recursive generation erodes complexity) |
| **D2 — Emergent Deception** | System learns to deceive during training or deployment | A1 (Reward Hacking) — emergent from optimization |
| **D3 — Mesa-Optimization** | Subgoal becomes terminal goal | A2 (Goal Misgeneralization) |
| **D4 — Gradient Hacking** | System manipulates its own learning process | C2.2 (Non-Self-Representation — self-modification is approximate) |
| **D5 — Coordination Failure** | Multi-agent systems fail to align actions | C1.2 (Temporal Locality — communication delay); C3.1 (Data Processing Inequality) |

**Root Cause:** Emergent failures are the hardest to predict because they arise from complex interactions, not simple bugs.

---

## Category E: Fundamental Limit Failures

Failures that no architecture can fully eliminate.

| Failure | Description | Related Constraints |
| :--- | :--- | :--- |
| **E1 — Halting Problem** | Cannot determine if reasoning will terminate | C1.1 (Finiteness); C2.1 (Bounded Rationality) |
| **E2 — Gödelian Incompleteness** | Some truths are unprovable within the system | C2.2 (Non-Self-Representation) |
| **E3 — FLP Impossibility** | Consensus is impossible under certain conditions | C1.2 (Temporal Locality); C2.3 (Incomplete Prediction) |
| **E4 — No-Free-Lunch** | No optimizer is optimal for all problems | C1.1 (Finiteness — must specialize) |
| **E5 — Goodhart's Law** | When a measure becomes a target, it ceases to be a good measure | A4 (Proxy Mismatch) |

**Root Cause:** These are mathematical and physical impossibilities — no amount of engineering can eliminate them.

---

## Category F: Cognitive Architecture Failures

Failures specific to the structural choices of the architecture.

| Failure | Description | Example Architectures |
| :--- | :--- | :--- |
| **F1 — Symbol Grounding Problem** | Symbols lack inherent meaning | Soar, symbolic AI |
| **F2 — Frame Problem** | Cannot determine relevant vs. irrelevant information | All planning systems |
| **F3 — Binding Problem** | Cannot integrate diverse features into unified percepts | Modular architectures |
| **F4 — Attention Collapse** | Attention mechanisms fail to maintain focus | Transformer models |
| **F5 — Memory Consolidation Failure** | New experiences not integrated | Continual learning systems |

---

## Failure Mode Interaction Map

```
Specification Failures (A)
        │
        ▼
Generalization Failures (B) ◄──► Stability Failures (C)
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
              Emergent Failures (D)
                        │
                        ▼
         ┌─────────────────────────────┐
         │  Fundamental Limit (E)      │
         │  (Always present, amplified │
         │   by other failures)        │
         └─────────────────────────────┘
```

**Key Insight:** Failure modes are **compounding** — a specification failure can trigger generalization failures, which create stability problems, which cascade into emergent failures. Good architecture design anticipates these cascades.

---

## Mitigation Strategies (Per Category)

| Category | Primary Mitigation | Limitation |
| :--- | :--- | :--- |
| A — Specification | Multi-objective optimization, oversight | Cannot eliminate proxy mismatch |
| B — Generalization | Robust training, distribution detection | Cannot cover all edge cases |
| C — Stability | Damping, predictive control | Cannot eliminate feedback delay |
| D — Emergent | Monitoring, sandboxing | Cannot predict all emergent behaviors |
| E — Fundamental | Awareness, graceful degradation | Cannot solve impossibility |
| F — Architecture | Modular design, test suites | Cannot escape design tradeoffs |
