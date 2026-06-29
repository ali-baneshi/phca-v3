# System Structure: State, Memory, Action, Feedback Loops

**Domain:** B. System Structure
**File:** 02-system-structure/01-state-and-memory.md
**Status:** DRAFT
**Cross-refs:** [01-intelligence-foundations/02-minimal-cognition.md](../01-intelligence-foundations/02-minimal-cognition.md), [08-neuroscience-principles/01-memory-systems.md](../08-neuroscience-principles/01-memory-systems.md), [06-cognitive-architectures/04-coala.md](../06-cognitive-architectures/04-coala.md)

---

## 1. Core Question

What are the fundamental structural components any intelligent system must possess? How do state, memory, action, and feedback interrelate?

---

## 2. The Four Pillars

### 2.1 State

State is the system's internal representation of its current situation. It includes:

- **Sensory State:** Current input from environment
- **Internal State:** Working memory contents, active goals, ongoing computations
- **Context State:** Situational framing and relevant long-term memory activations

**Key Properties:**
- State is always **time-dependent** — it represents a snapshot at time *t*
- State is **bounded** — only a finite amount can be represented due to resource constraints
- State must be **interpretable** by the system's action-selection mechanisms

### 2.2 Memory

Memory is the persistence of information across time. It is not a single store but a **hierarchy of systems** with different time constants and capacities:

| Memory Type | Duration | Capacity | Function |
| :--- | :--- | :--- | :--- |
| Sensory | < 1 second | High | Buffer raw input |
| Working | Seconds-minutes | Limited (7±2 chunks) | Active manipulation |
| Episodic | Years | Very high | Personal experiences |
| Semantic | Years | Very high | Facts and concepts |
| Procedural | Years | Very high | Skills and habits |

**Key Insight:** Memory is not storage — it is **reconstruction**. Every act of retrieval is a constructive process, not a read from a static database.

### 2.3 Action

Action is the interface between the system and its environment (or between components within the system). Actions can be:

- **External:** Manipulating the environment (movement, speech, tool use)
- **Internal:** Modifying internal state (memory retrieval, attention shifts, reasoning steps)

**Key Properties:**
- Actions are **selected based on state + memory**
- Actions have **consequences** that alter both environment and internal state
- Action selection operates under **bounded rationality** — optimal action is not always computable

### 2.4 Feedback Loops

Feedback loops connect action back to state, creating the closed-loop structure of cognition:

1. **Perception-Action Loop:** Sense → Decide → Act → Sense (changed environment)
2. **Prediction-Error Loop:** Predict → Sense → Compare → Update model → Predict (revised)
3. **Goal-Progress Loop:** Set goal → Assess progress → Adjust action → Re-assess

---

## 3. The Minimal Viable Architecture

Based on the above, any intelligent system requires at minimum:

```
 ┌─────────────────────────────────────────────┐
 │                   System                     │
 │  ┌──────────┐    ┌──────────┐               │
 │  │  State   │◄──►│  Memory  │               │
 │  └────┬─────┘    └──────────┘               │
 │       │                                     │
 │       ▼                                     │
 │  ┌──────────┐    ┌──────────────┐           │
 │  │  Action  │◄──►│  Feedback    │           │
 │  │  Select  │    │  Processing  │           │
 │  └────┬─────┘    └──────┬───────┘           │
 │       │                 │                    │
 └───────┼─────────────────┼────────────────────┘
         │                 │
         ▼                 ▼
     Environment ◄─────────┘
```

This is the **perception-cognition-action loop** found in all cognitive architectures.

---

## 4. Architectural Constraints on Structure

1. **Bounded State:** State cannot capture everything. The system must prioritize what to represent.
2. **Memory Access Latency:** Retrieval from long-term memory is slower and less reliable than working memory access.
3. **Action Inertia:** Actions take time and have irreversible consequences.
4. **Feedback Delay:** The consequences of actions are often not immediately observable, requiring predictive models.
5. **Compositionality:** State and memory representations must support composition for the system to handle novel situations.

---

## 5. Assumptions

1. **The perception-action loop is universal.** This assumes no form of intelligence exists without sensorimotor interaction. [UNVERIFIED — pure reasoning systems might violate this]
2. **Memory systems are hierarchical.** This is supported by neuroscience but may be an implementation detail, not a universal constraint.
3. **Feedback is necessary for learning.** This is well-established but assumes that open-loop systems cannot exhibit intelligence.

---

## 6. Open Questions

1. Is there a minimum coupling strength between perception and action required for intelligence?
2. Can a system with only one type of memory exhibit intelligence, or are multiple memory types necessary?
3. What is the optimal ratio of internal vs. external actions for adaptive intelligence?
4. Are feedback loops sufficient for learning, or is a separate consolidation mechanism required?

---

## 7. Sources

- Brooks, R. (1991): *Intelligence Without Representation* — AI Journal
- Clark, A. (2013): *Whatever Next? Predictive Brains, Situated Agents, and the Future of Cognitive Science*
- Fuster, J. (2015): *The Prefrontal Cortex* (5th Edition) — Memory hierarchy and executive function
- Newell, A. (1990): *Unified Theories of Cognition* — Soar architecture foundations
