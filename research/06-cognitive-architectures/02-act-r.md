# ACT-R Cognitive Architecture

**Domain:** Cognitive Architectures
**File:** 06-cognitive-architectures/02-act-r.md
**Status:** DRAFT
**Cross-refs:** [01-soar](01-soar.md), [03-lida](03-lida.md), [04-coala](04-coala.md), [08-neuroscience-principles/01-memory-systems.md](../08-neuroscience-principles/01-memory-systems.md)

---

## 1. Overview

**ACT-R** (Adaptive Control of Thought — Rational) is a hybrid cognitive architecture developed by John R. Anderson at Carnegie Mellon University. It serves as both a theory of human cognition and a computational framework for building cognitive models.

**Key Principle:** ACT-R models cognition as a **production system** (symbolic rules) modulated by **subsymbolic equations** (continuous, probabilistic processes). This hybrid architecture bridges the gap between symbolic AI and neural/statistical approaches.

---

## 2. Core Components

### 2.1 Modules

ACT-R consists of specialized, largely independent modules:

| Module | Function | Analogy (Brain) |
| :--- | :--- | :--- |
| **Visual Module** | Perceives the environment | Visual cortex |
| **Manual Module** | Executes motor actions | Motor cortex |
| **Declarative Memory** | Stores facts (chunks) | Hippocampus / Cortex |
| **Procedural Memory** | Stores rules (productions) | Basal Ganglia |
| **Goal Module** | Tracks current objectives | Prefrontal Cortex |
| **Imaginal Module** | Manipulates mental representations | Parietal Cortex |

### 2.2 Buffers

Each module has a dedicated **buffer** — a limited-capacity interface that holds the module's current state. Together, all buffer contents represent the system's "state of mind."

- **Buffer Capacity:** Typically holds one chunk at a time
- **Buffer Updates:** Modules (except procedural) can only be accessed via their buffers
- **Pattern Matching:** Productions match against the current contents of all buffers

### 2.3 Production System

The procedural memory module and pattern matcher together form a **production system**:

1. **Match:** All productions whose conditions match current buffer contents are identified
2. **Conflict Resolution:** Among matching productions, the one with highest **utility** is selected
3. **Fire:** The selected production executes its actions, modifying buffers
4. **Repeat:** The cycle continues, producing sequential cognition

---

## 3. Hybrid Architecture: Symbolic + Subsymbolic

### 3.1 Symbolic Level

- **Discrete and deterministic** logic of task performance
- **Chunks** represent facts (declarative knowledge)
- **Productions** represent skills (procedural knowledge)

### 3.2 Subsymbolic Level

Continuous mathematical equations modulate symbolic operations:

| Process | Subsymbolic Mechanism | Effect |
| :--- | :--- | :--- |
| **Memory Retrieval** | **Activation** — a chunk's activation level determines retrieval speed and probability | More recently/useful chunks retrieved faster |
| **Production Selection** | **Utility** — expected value of a production's outcome | Higher utility productions preferred |
| **Learning** | **Base-level learning** — activation increases with use, decays with time | Spacing effect, forgetting curve |
| **Conflict Resolution** | **Noise** — random noise added to utility values | Explains variability and errors |

### 3.3 Key Insight

The subsymbolic equations are not "add-ons" — they are integral to how ACT-R produces human-like behavior. Without them, ACT-R would be a deterministic production system incapable of modeling the variability, learning curves, and error patterns of human cognition.

---

## 4. Learning Mechanisms

| Mechanism | Description |
| :--- | :--- |
| **Base-Level Learning** | Declarative chunks increase in activation with use, decay with time |
| **Strengthening** | Production utilities update based on reward history |
| **Compilation** | Sequences of productions can be merged into single more efficient productions |
| **Attentional Learning** | The system learns where to direct visual attention |

---

## 5. Strengths

- **Empirically Grounded:** Extensively validated against human behavioral and fMRI data
- **Hybrid:** Bridges symbolic and statistical approaches
- **Predictive Power:** Generates quantitative predictions (reaction times, error rates, brain activity)
- **Modular:** Independent modules with clear interfaces

---

## 6. Weaknesses

- **Task-Specific:** Requires manual modeling for each task; not an autonomous agent
- **Limited Perception:** Perceptual modules are simplified compared to actual vision/audition
- **No Autonomous Learning:** Cannot autonomously decide what to learn; learning depends on modeler-defined productions
- **Rigid Module Boundaries:** Some cognitive phenomena may require more integrated processing

---

## 7. Assumptions

1. **Cognition is sequential** — one production fires at a time. This may not hold for parallel unconscious processes.
2. **All declarative knowledge is chunk-based.** Some knowledge may be distributed and non-symbolic.
3. **Utility maximization drives procedural learning.** This assumes rational action selection in learning.

---

## 8. Open Questions

1. Can ACT-R be extended to handle truly autonomous, open-ended learning?
2. How does ACT-R's hybrid approach compare to end-to-end neural approaches for modeling cognition?
3. Is the module/buffer architecture a necessary feature of cognition or specific to human cognition?

---

## 9. Sources

- Anderson, J.R. (2007): *How Can the Human Mind Occur in the Physical Universe?*
- Anderson, J.R. et al. (2004): *An Integrated Theory of the Mind* — Psychological Review
- ACT-R Website: [act-r.psy.cmu.edu](https://act-r.psy.cmu.edu/)
