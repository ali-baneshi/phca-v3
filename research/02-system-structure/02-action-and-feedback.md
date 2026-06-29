# Action and Feedback Loops

**Domain:** B. System Structure
**File:** 02-system-structure/02-action-and-feedback.md
**Status:** DRAFT
**Cross-refs:** [01-state-and-memory](01-state-and-memory.md), [07-foundational-theory/02-control-theory.md](../07-foundational-theory/02-control-theory.md), [08-neuroscience-principles/02-prediction-processing.md](../08-neuroscience-principles/02-prediction-processing.md)

---

## 1. Action Selection Mechanisms

### 1.1 Types of Action Selection

| Mechanism | Description | Example |
| :--- | :--- | :--- |
| **Reactive** | Direct mapping from state to action | Reflex, stimulus-response |
| **Deliberative** | Explicit reasoning about future states | Planning, search |
| **Utilitarian** | Selection based on expected utility | Reinforcement learning |
| **Habituated** | Automatic, learned sequences | Skill execution |
| **Competitive** | Parallel proposals compete for selection | LIDA's coalitions, neural action selection |

### 1.2 The Selection Problem

Action selection is non-trivial because:

- **Multiple actions are always possible** — the system must commit to one
- **Consequences are uncertain** — outcomes depend on an unpredictable environment
- **Time pressure** — deliberation competes with reactivity
- **Conflicting goals** — actions that serve one goal may harm another

### 1.3 Resolution Strategies

1. **Hierarchical:** Goals are prioritized; higher-priority goals subsume lower
2. **Winner-Take-All:** Parallel action proposals compete; strongest wins
3. **Voting/Integration:** Multiple systems contribute; action is an ensemble
4. **Sequential:** Actions are serialized; one at a time in order of priority

---

## 2. Types of Feedback Loops

### 2.1 Negative Feedback (Error Correction)

The system measures the difference between current state and desired state, and acts to reduce that difference.

```
Desired State ──► Error ──► Action ──► Environment ──► Current State
                    ▲                                          │
                    └──────────────────────────────────────────┘
```

**Properties:**
- Tends toward stability and homeostasis
- Requires a reference signal (goal)
- Delay in feedback can cause oscillation

### 2.2 Positive Feedback (Amplification)

A change in one direction is amplified, pushing the system further in that direction.

```
Small Change ──► Amplification ──► Larger Change ──► Further Amplification
```

**Properties:**
- Tends toward instability and rapid change
- Can drive phase transitions
- Dangerous without countervailing negative feedback

### 2.3 Predictive Feedback (Feedforward)

The system predicts the outcome of its actions and uses that prediction to pre-emptively adjust.

```
State ──► Predict Outcome ──► Compare to Goal ──► Pre-emptive Action
```

**Properties:**
- Reduces effects of feedback delay
- Requires an accurate internal model
- Can produce overcorrection if model is wrong

---

## 3. Loop Dynamics

### 3.1 Damping and Oscillation

- **Underdamped:** System oscillates around target before settling
- **Critically damped:** System approaches target optimally
- **Overdamped:** System approaches target slowly, no oscillation

### 3.2 Coupling Strength

Loops can be:
- **Tightly coupled:** Fast, direct feedback (e.g., proprioception)
- **Loosely coupled:** Slow, indirect feedback (e.g., long-term consequences)

### 3.3 Loop Nestedness

Feedback loops nest within each other at different timescales:
- Fast loops: real-time motor control (ms)
- Medium loops: action selection and learning (seconds-minutes)
- Slow loops: structural and architectural change (hours-years)

---

## 4. Implications for Architecture Design

1. **Multiple feedback types are necessary** — no single loop type suffices for all situations
2. **Loop delays must be matched** — fast loops cannot depend on slow feedback
3. **Positive and negative feedback must coexist** — stability without stagnation, change without collapse
4. **Predictive feedback requires models** — and models require updating, which requires learning loops

---

## 5. Assumptions

1. **All intelligent systems require feedback.** No known counterexample.
2. **Hierarchical action selection is computationally efficient.** Supported by neuroscience and AI practice.
3. **Positive feedback without negative regulation leads to instability.** Well-established in control theory.

---

## 6. Open Questions

1. Can a purely feedforward system exhibit anything recognizable as intelligence?
2. What is the optimal topology of nested feedback loops?
3. How does the system resolve conflicts between feedback from different timescales?

---

## 7. Sources

- Ashby, W.R. (1956): *An Introduction to Cybernetics* — Foundational work on feedback and regulation
- Powers, W.T. (1973): *Behavior: The Control of Perception* — Perceptual control theory
- Beer, R. (1995): *A Dynamical Systems Perspective on Agent-Environment Interaction*
