# Soar Cognitive Architecture

**Domain:** Cognitive Architectures
**File:** 06-cognitive-architectures/01-soar.md
**Status:** DRAFT
**Cross-refs:** [02-act-r](02-act-r.md), [03-lida](03-lida.md), [04-coala](04-coala.md), [02-system-structure/01-state-and-memory.md](../02-system-structure/01-state-and-memory.md), [05-architectural-constraints/01-bounded-computation.md](../05-architectural-constraints/01-bounded-computation.md)

---

## 1. Overview

**Soar** (originally *SOAR*, standing for State, Operator, And Result) is a symbolic cognitive architecture developed by John Laird, Allen Newell, and Paul Rosenbloom. It implements the **Problem Space Hypothesis** — the claim that all goal-oriented behavior can be represented as search through a problem space defined by states and operators.

**Key Principle:** Soar represents all deliberate behavior as the application of operators to states to achieve goals. When knowledge is insufficient, it uses **universal subgoaling** to automatically create substates for problem-solving.

---

## 2. Core Components

### 2.1 Memory Systems

| Memory Type | Content | Access |
| :--- | :--- | :--- |
| **Working Memory** | Current situation, sensory data, active goals | Direct, symbolic graph structures |
| **Procedural Memory** | Production rules (if-then) | Parallel pattern matching |
| **Semantic Memory** | Facts and knowledge | Retrieval cues |
| **Episodic Memory** | Past experiences | Temporal cues |
| **Spatial Memory** | Visual-spatial information | Spatial reasoning system |

### 2.2 Decision Cycle

Soar's processing is organized as a sequence of **decision cycles**:

1. **Elaboration Phase:** Productions fire in parallel, adding knowledge to working memory
2. **Decision Phase:** A single operator is selected based on preferences accumulated during elaboration
3. **Application Phase:** The selected operator is applied, changing working memory state
4. **Impasse Handling:** If no operator can be selected (knowledge insufficient), an **impasse** triggers automatic subgoaling

### 2.3 Universal Subgoaling

When Soar cannot make a decision (impasse), it automatically:
- Creates a **substate** in working memory
- Sets a **subgoal** to resolve the impasse
- Applies the same decision cycle to the subproblem
- When resolved, results are incorporated into the parent state

This creates a **dynamic hierarchical goal structure** — subgoals are created and resolved as needed, without predetermined plans.

### 2.4 Learning Mechanisms

| Mechanism | Description |
| :--- | :--- |
| **Chunking** | Automatically creates new productions that summarize problem-solving episodes. When a subgoal is resolved, the architecture compiles the reasoning into a single rule. |
| **Reinforcement Learning** | Updates numeric preferences for operators based on delayed reward |
| **Episodic Learning** | Stores experiences for future retrieval |
| **Semantic Learning** | Acquires new facts from experience |

---

## 3. Key Principles

1. **Universal Architecture:** Soar aims to be a general architecture for all cognition — not task-specific
2. **Emergent Goals:** Goals are not pre-planned; they arise from impasses during problem-solving
3. **All Learning from Experience:** Chunking compiles problem-solving episodes into long-term knowledge
4. **Symbolic Representation:** All knowledge is represented symbolically (symbolic graph structures)
5. **Integration without Compromise:** Modules (memory, learning, reasoning) are integrated within a single framework

---

## 4. Strengths

- **Generality:** Same architecture for diverse tasks (reasoning, planning, natural language)
- **Transparency:** Symbolic representations are interpretable
- **Learning Integration:** Chunking integrates learning naturally into the decision cycle
- **Graceful Impasse Handling:** Automatically breaks down problems when stuck

---

## 5. Weaknesses

- **Symbol Grounding Problem:** Symbols lack inherent meaning; meaning must be provided by the designer
- **Limited Perception:** Original Soar had minimal perceptual processing
- **Scalability:** Symbolic reasoning does not scale to high-dimensional real-world perception
- **Rigid Knowledge Representation:** Production rules are brittle compared to neural representations

---

## 6. Assumptions

1. **All cognition can be modeled as problem-space search.** This may not hold for instinctive, emotional, or unconscious processes.
2. **Symbolic representation is sufficient for intelligence.** This is challenged by connectionist and embodied approaches.
3. **Impasses are the primary driver of learning.** Learning may also occur through other mechanisms.

---

## 7. Open Questions

1. Can Soar's symbolic architecture be integrated with neural representations for perception and motor control?
2. Is chunking sufficient for all forms of learning, or are qualitatively different learning mechanisms needed?
3. How does Soar handle creativity and insight — phenomena that do not resolve neatly through impasse-driven problem-solving?

---

## 8. Sources

- Newell, A. (1990): *Unified Theories of Cognition* — The foundational text
- Laird, J. (2012): *The Soar Cognitive Architecture* — Comprehensive overview
- Laird, J., Newell, A., & Rosenbloom, P. (1987): *SOAR: An Architecture for General Intelligence* — Original paper
