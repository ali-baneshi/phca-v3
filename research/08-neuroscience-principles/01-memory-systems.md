# Neuroscience of Memory Systems

**Domain:** Neuroscience Principles
**File:** 08-neuroscience-principles/01-memory-systems.md
**Status:** DRAFT
**Cross-refs:** [02-prediction-processing](02-prediction-processing.md), [03-attention-mechanisms](03-attention-mechanisms.md), [02-system-structure/01-state-and-memory.md](../02-system-structure/01-state-and-memory.md), [06-cognitive-architectures/02-act-r.md](../06-cognitive-architectures/02-act-r.md)

---

## 1. Core Question

How does the brain implement memory? What architectural lessons can we learn from neuroscience's multi-system memory model?

---

## 2. The Multiple Memory Systems Model

Memory is not a single faculty but a collection of **distinct systems** with different neural substrates and functional properties.

### 2.1 Classification by Duration

| System | Duration | Capacity | Function |
| :--- | :--- | :--- | :--- |
| **Sensory Memory** | < 1 second | Very high | Buffers raw sensory data |
| **Working Memory** | Seconds to minutes | Limited (7±2 items) | Active manipulation of information |
| **Long-Term Memory** | Days to years | Effectively unlimited | Persistent storage |

### 2.2 Classification by Type

```
                    ┌─────────────────────┐
                    │      Memory         │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌───────────────┐             ┌───────────────┐
        │  Declarative  │             │Non-Declarative│
        │  (Explicit)   │             │  (Implicit)   │
        └───────┬───────┘             └───────┬───────┘
                │                             │
        ┌───────┴───────┐             ┌───────┴───────┐
        ▼               ▼             ▼               ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Episodic │  │ Semantic │  │Procedural│  │Priming / │
   │ (Events) │  │ (Facts)  │  │ (Skills) │  │Condition.│
   └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## 3. Neural Substrates

### 3.1 Hippocampus (and Medial Temporal Lobe)

- **Primary Role:** Consolidation of new declarative memories; spatial navigation
- **Mechanism:** Binds sensory and conceptual elements into coherent episodes
- **Clinical Evidence (Patient H.M.):** Bilateral hippocampal resection → severe anterograde amnesia (cannot form new declarative memories) but intact procedural learning and working memory

### 3.2 Neocortex

- **Primary Role:** Long-term storage of declarative memories
- **Mechanism:** After consolidation (via hippocampus), memories are distributed across cortical regions, stored near the sensory areas that originally processed them
- **Prefrontal Cortex:** Essential for working memory — maintains and manipulates information

### 3.3 Basal Ganglia

- **Primary Role:** Procedural memory, habit learning, stimulus-response associations
- **Mechanism:** Gradual, incremental learning using dopamine signals for reinforcement
- **Clinical Evidence (Parkinson's):** Dopamine loss → deficits in habit learning; declarative memory intact

### 3.4 Double Dissociation

| Structure | Damaged → | Impaired | Intact |
| :--- | :--- | :--- | :--- |
| **Hippocampus** | Patient H.M. | New declarative memories | Procedural learning |
| **Basal Ganglia** | Parkinson's | Habit learning | Declarative memories |

This double dissociation confirms that declarative and procedural memory are **neurologically distinct systems**.

---

## 4. Memory Processes

| Process | Description | Mechanism |
| :--- | :--- | :--- |
| **Encoding** | Transforming perception into memory trace | Hippocampal binding |
| **Consolidation** | Stabilizing memory over time | Hippocampus → cortex transfer; sleep-dependent |
| **Storage** | Maintaining memory over time | Synaptic changes, distributed cortical networks |
| **Retrieval** | Accessing stored information | Pattern completion in hippocampus; cue-based recall |
| **Reconsolidation** | Updating retrieved memory | Retrieved memories become labile, then re-stabilize |
| **Forgetting** | Inability to retrieve | Decay, interference, or failed retrieval cues |

---

## 5. Key Architectural Lessons

1. **Multiple memory systems are necessary** — no single memory type suffices for all cognitive functions
2. **Memory is reconstruction, not playback** — retrieval is a constructive process, not database access
3. **Consolidation takes time** — new memories are fragile; stabilization requires time and sleep
4. **Forgetting is functional** — not all information can or should be retained
5. **Memory and action are coupled** — episodic memory informs future action; procedural memory enables skilled action

---

## 6. Assumptions

1. **The multiple memory systems model is correct.** Extensively supported but not the only memory theory.
2. **Consolidation is necessary for long-term storage.** Supported by strong empirical evidence.
3. **Memory is distributed, not localized.** Supported by lesion studies and imaging.

---

## 7. Open Questions

1. How exactly is memory transferred from hippocampus to cortex during consolidation?
2. Can artificial systems benefit from sleep-like consolidation mechanisms?
3. What is the capacity limit of working memory, and can it be expanded?
4. Are there additional memory systems not captured by the standard taxonomy?

---

## 8. Sources

- Squire, L. (2004): *Memory Systems of the Brain — A Brief History and Current Perspective* — PMC
- Eichenbaum, H. (2012): *The Cognitive Neuroscience of Memory* (2nd Edition)
- The Neuroanatomical, Neurophysiological and Psychological Basis of Memory — [PMC5491610](https://pmc.ncbi.nlm.nih.gov/articles/PMC5491610/)
- Memory Systems — MSU Open Books Introduction to Neuroscience
