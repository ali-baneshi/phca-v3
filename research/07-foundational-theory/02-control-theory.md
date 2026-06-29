# Control Theory and Cognitive Architecture

**Domain:** Foundational Theory
**File:** 07-foundational-theory/02-control-theory.md
**Status:** DRAFT
**Cross-refs:** [01-information-theory](01-information-theory.md), [04-complex-systems](04-complex-systems.md), [02-system-structure/02-action-and-feedback.md](../02-system-structure/02-action-and-feedback.md)

---

## 1. Core Question

How do the principles of control theory — feedback, homeostasis, and regulation — inform the design of intelligent systems?

---

## 2. Key Control Theory Concepts

### 2.1 Feedback Loops

The core mechanism: a system monitors its output and feeds that information back to adjust input.

**Closed-loop (Negative Feedback):**
```
Reference ──► Error ──► Controller ──► Plant ──► Output
    ▲                                                      │
    └─────────────────── Feedback ─────────────────────────┘
```

- System continuously compares actual output to desired reference
- Error signal drives corrective action
- Enables stability despite disturbances

**Open-loop:**
```
Reference ──► Controller ──► Plant ──► Output
```
- No feedback; outputs based on predetermined inputs
- Simpler but cannot self-correct

### 2.2 PID Control

A widely-used control algorithm with three components:

| Component | Function | Cognitive Analogy |
| :--- | :--- | :--- |
| **Proportional (P)** | React to current error | Immediate corrective action |
| **Integral (I)** | Accumulate past errors | Learning from history |
| **Derivative (D)** | Predict future error based on rate of change | Anticipatory adjustment |

### 2.3 Homeostasis

The process by which systems maintain internal stability despite external fluctuations.

- **Set Point:** The target value the system maintains
- **Disturbance:** External changes that push the system away from set point
- **Corrective Action:** Internal processes that restore equilibrium

**Role in Intelligence:** Homeostasis provides the fundamental "drives" — the system acts to reduce discrepancies between current and desired states.

### 2.4 Predictive Control (Model Predictive Control)

Instead of reacting to current errors, the controller:
1. Uses an internal model to predict future states
2. Calculates optimal action sequence over a time horizon
3. Executes the first action, then recalculates

**Role in Intelligence:** MPC is analogous to planning — the system simulates future outcomes before acting.

---

## 3. The Cybernetic Metaphor

The **cybernetic view** treats the mind as a feedback control system:

- **Perception** = measuring current state
- **Goals** = reference signals
- **Cognition** = controller computing actions
- **Action** = modifying environment
- **Learning** = updating the controller

**Limitations of the Cybernetic Metaphor:**
- Overly computational — misses the phenomenological, experiential aspects of mind
- Assumes goals are fixed and given — does not explain goal generation
- Does not account for creativity and novelty

---

## 4. The Ecological Metaphor

An alternative: intelligence is not internal computation but an **emergent property** of agent-environment interaction.

- **Mind is not in the head** — it is distributed across brain, body, and environment
- **Cognition is for maintaining viability** — not for representing the world
- **Goals are not given but enacted** — they emerge from the system's organization

**Synthesis:** Modern cognitive architectures increasingly integrate both metaphors — using feedback principles from cybernetics while grounding them in the ecological context of an embodied agent.

---

## 5. Implications for Architecture

1. **Reference signals (goals) are fundamental** — without goals, control has no direction
2. **Feedback delay is critical** — the system must either tolerate oscillation or predict
3. **Multiple nested loops** — fast loops for reflexes, slow loops for learning
4. **Homeostasis as motivation** — drives to reduce discrepancies between current and desired
5. **Prediction enables proactivity** — not just reaction

---

## 6. Assumptions

1. **Intelligent systems are goal-directed.** This may not hold for all forms of intelligence.
2. **Control theory applies to cognition.** The cybernetic metaphor is productive but may be incomplete.
3. **Feedback is necessary for adaptation.** Not seriously challenged.

---

## 7. Open Questions

1. Where do goals come from? Are they intrinsic (homeostatic) or acquired?
2. Can a purely open-loop system exhibit intelligence?
3. How does predictive control scale to unstructured, non-physical environments (e.g., social cognition)?
4. What is the control-theoretic equivalent of creativity?

---

## 8. Sources

- Ashby, W.R. (1956): *An Introduction to Cybernetics*
- Wiener, N. (1948): *Cybernetics: Or Control and Communication in the Animal and the Machine*
- Powers, W.T. (1973): *Behavior: The Control of Perception*
- Friston, K. (2010): *The Free-Energy Principle: A Unified Brain Theory?*
