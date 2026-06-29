# Prediction Processing in the Brain

**Domain:** Neuroscience Principles
**File:** 08-neuroscience-principles/02-prediction-processing.md
**Status:** DRAFT
**Cross-refs:** [01-memory-systems](01-memory-systems.md), [03-attention-mechanisms](03-attention-mechanisms.md), [07-foundational-theory/02-control-theory.md](../07-foundational-theory/02-control-theory.md), [06-cognitive-architectures/04-coala.md](../06-cognitive-architectures/04-coala.md)

---

## 1. Core Question

How does the brain make predictions, and how does prediction error drive learning and behavior?

---

## 2. The Predictive Brain Hypothesis

The brain is not a passive processor of sensory input but an active **inference engine** that constantly generates models to anticipate incoming information.

**Core Idea:** Perception is not bottom-up (sensation → meaning) but top-down (expectation → modulated by sensation). What we perceive is a **controlled hallucination** — a model of the world constrained by sensory evidence.

---

## 3. Three Interconnected Theories

### 3.1 The Bayesian Brain Hypothesis

The brain represents information **probabilistically** — as probability distributions rather than fixed values.

- **Internal beliefs** are probability distributions over possible states of the world
- **Incoming evidence** updates beliefs via Bayes rule (posterior ∝ prior × likelihood)
- **Uncertainty is represented** — the brain knows what it doesn't know

```
P(hypothesis | data) ∝ P(data | hypothesis) × P(hypothesis)
   (Posterior)              (Likelihood)         (Prior)
```

### 3.2 Predictive Coding Theory

A specific neural mechanism for implementing Bayesian inference.

**Hierarchical Architecture:**
```
Higher Areas
    │
    ├─── Generate predictions (top-down)
    │
    ▼
Lower Areas
    │
    ├─── Compute prediction errors (bottom-up)
    │
    ▼
Sensory Input
```

**Process:**
1. Higher-level areas predict what lower-level areas should perceive
2. Lower-level areas compare actual input to prediction
3. **Prediction error** (mismatch) is sent back up
4. Higher areas update their model to reduce prediction error

### 3.3 The Free Energy Principle (Friston)

A unifying principle: all living systems minimize **free energy** — a measure of surprise or prediction error.

- **Free Energy** = the long-term average of prediction error
- **Minimization** occurs through two paths:
  1. **Inference** (Perception): Update internal model to better predict input
  2. **Active Inference** (Action): Change the environment to match predictions

**Key Insight:** The Free Energy Principle unifies perception, action, and learning under a single principle (minimizing surprise).

---

## 4. Prediction Error and Learning

### 4.1 Immediate Resolution (Inference)

When a prediction error occurs:
- The brain can update its **current perception** — reinterpret what it sees
- This is fast and does not change long-term knowledge

### 4.2 Long-Term Change (Learning)

When prediction errors **persist**:
- The brain updates its **internal model** — synaptic plasticity changes future predictions
- This is slower but produces lasting change
- Learning = minimizing future prediction errors

### 4.3 Precision Weighting (Attention)

Not all prediction errors are equally important. The brain must decide which to trust:

- **High precision** (reliable signal) → error signal is trusted, model is updated
- **Low precision** (noisy signal) → error signal is ignored, prediction is maintained
- **Attention = precision weighting** — selecting which prediction errors to learn from

---

## 5. Active Inference

The brain does not only update models to match the world — it also changes the world to match models.

**Example (Reaching for a cup):**
- The brain predicts the proprioceptive sensation of the hand at the cup's location
- It sends motor commands that fulfill that prediction
- If proprioception matches the prediction, the action succeeded

**Active inference unifies:**
- **Perception:** Update model to match world
- **Action:** Change world to match model
- Both serve the same goal: minimizing prediction error

---

## 6. Implications for Architecture

1. **Prediction is primary** — intelligence is fundamentally about prediction, not reaction
2. **Error drives learning** — the system learns when predictions fail
3. **Hierarchical processing** — predictions at multiple levels of abstraction
4. **Active inference** — action is a form of prediction fulfillment
5. **Precision weighting** — attention modulates what is learned

---

## 7. Assumptions

1. **The brain is fundamentally a prediction engine.** Well-supported but not the only theory of brain function.
2. **The Free Energy Principle is a valid unifying framework.** [UNVERIFIED — debated in neuroscience]
3. **Prediction error is sufficient to explain learning.** Some learning may occur without prediction error (e.g., Hebbian learning).

---

## 8. Open Questions

1. Is the Free Energy Principle a genuine scientific theory or a mathematical re-description?
2. How exactly do neurons implement precision weighting?
3. Can predictive coding be implemented in artificial systems to achieve brain-like efficiency?
4. What is the relationship between prediction error and reward?

---

## 9. Sources

- Friston, K. (2010): *The Free-Energy Principle: A Unified Brain Theory?* — Nature Reviews Neuroscience
- Clark, A. (2013): *Whatever Next? Predictive Brains, Situated Agents, and the Future of Cognitive Science*
- Wikipedia: *Free Energy Principle*, *Predictive Coding*
- Wellcome Centre for Human Neuroimaging: *The Bayesian Brain* — [fil.ion.ucl.ac.uk](https://www.fil.ion.ucl.ac.uk/archive/bayesian-brain/index.html)
