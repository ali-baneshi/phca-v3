# LIDA Cognitive Architecture

**Domain:** Cognitive Architectures
**File:** 06-cognitive-architectures/03-lida.md
**Status:** DRAFT
**Cross-refs:** [01-soar](01-soar.md), [02-act-r](02-act-r.md), [04-coala](04-coala.md), [08-neuroscience-principles/03-attention-mechanisms.md](../08-neuroscience-principles/03-attention-mechanisms.md)

---

## 1. Overview

**LIDA** (Learning Intelligent Distribution Agent) is a biologically inspired cognitive architecture developed by Stan Franklin and his team at the University of Memphis. It models human-like cognition by implementing **Bernard Baars' Global Workspace Theory (GWT)** — a model of consciousness as a "global workspace" for information integration.

**Key Principle:** Cognition unfolds as a continuous stream of **cognitive cycles** (~10 Hz), each involving perception, conscious broadcast, and action selection. Consciousness, in LIDA, is a **functional mechanism** for information dissemination.

---

## 2. The LIDA Cognitive Cycle

Each cognitive cycle operates in three phases:

### Phase 1: Understanding (Perception)

1. **Sensory stimuli** enter **Sensory Memory** (raw buffer)
2. Stimuli activate feature detectors in **Perceptual Associative Memory**
3. Higher-level detectors identify objects, events, categories
4. The result is a **Situational Model** — the agent's current understanding

### Phase 2: Consciousness (Attention)

1. **Attention Codelets** (autonomous threads) monitor the situational model
2. They identify important information and form **coalitions**
3. Coalitions **compete** for access to the **Global Workspace**
4. The winning coalition becomes the current **content of consciousness**
5. This content is **broadcast globally** to all system modules

### Phase 3: Action Selection (Learning & Behavior)

1. The global broadcast triggers **learning** in all memory systems
2. Relevant **action schemes** are activated from Procedural Memory
3. The selected behavior is **executed**

---

## 3. Memory Systems

LIDA implements multiple memory types based on psychological theory:

| Memory Type | Function | Neural Analogy |
| :--- | :--- | :--- |
| **Sensory Memory** | Buffers raw sensory input | Sensory cortices |
| **Perceptual Associative Memory** | Recognizes objects/categories from features | Temporal cortex |
| **Transient Episodic Memory** | Stores recent events | Hippocampus |
| **Declarative Memory** | Stores factual knowledge | Cortex |
| **Procedural Memory** | Stores action schemes and skills | Basal Ganglia |

---

## 4. Consciousness as a Functional Process

LIDA treats consciousness as:

- **A computational mechanism** — not subjective experience (phenomenal consciousness)
- **A competition** — information coalitions compete for access to global broadcast
- **A broadcast** — winning coalition is disseminated to all modules for learning and coordination
- **A binding mechanism** — it allows disparate modules to coordinate responses

**Important:** LIDA models *functional* consciousness — behaviors that require consciousness in humans. It does not attempt to solve the "hard problem" of subjective awareness (qualia).

---

## 5. Learning

The global broadcast triggers learning in multiple memory systems:

- **Perceptual Learning:** Updating Perceptual Associative Memory
- **Episodic Learning:** Storing events in Transient Episodic Memory
- **Declarative Learning:** Consolidating to Declarative Memory
- **Procedural Learning:** Updating action selection mechanisms

---

## 6. Strengths

- **Biologically Plausible:** Models conscious cognitive cycle based on neuroscience
- **Functional Consciousness:** Provides a mechanism for consciousness without solving philosophy
- **Modular Memory:** Multiple memory types support diverse cognitive functions
- **Continuous Operation:** Uninterrupted cognitive cycles enable real-time interaction

---

## 7. Weaknesses

- **Consciousness Mechanism Unproven:** Global Workspace Theory is one model; not universally accepted
- **Complexity:** Large number of interacting modules makes analysis difficult
- **Implementation Gaps:** Many components are specified conceptually rather than computationally
- **Perception Limitations:** LIDA implementations typically use simplified perceptual systems

---

## 8. Assumptions

1. **Consciousness serves a functional role** — it is not epiphenomenal.
2. **A 10 Hz cognitive cycle is sufficient** for modeling human cognition. [UNVERIFIED — some processes may operate at different timescales]
3. **Global broadcast is how integration occurs.** This assumes a central integration point, which may not exist in all architectures.

---

## 9. Open Questions

1. Can LIDA's functional consciousness account for subjective experience, or is a different mechanism required?
2. How does the architecture handle non-conscious processing (skills, habits) that operate below the cognitive cycle?
3. Is the Global Workspace model necessary, or are there alternative mechanisms for information integration?

---

## 10. Sources

- Franklin, S. et al. (2007): *The LIDA Architecture: Adding New Modes of Learning*
- Baars, B. (1988): *A Cognitive Theory of Consciousness* — Global Workspace Theory
- Baars, B. & Franklin, S. (2009): *How Conscious Experience and Working Memory Interact*
