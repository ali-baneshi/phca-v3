# Information Theory and Intelligence

**Domain:** Foundational Theory
**File:** 07-foundational-theory/01-information-theory.md
**Status:** DRAFT
**Cross-refs:** [02-control-theory](02-control-theory.md), [04-complex-systems](04-complex-systems.md), [05-architectural-constraints/01-bounded-computation.md](../05-architectural-constraints/01-bounded-computation.md)

---

## 1. Core Question

How does information theory — the mathematics of communication and uncertainty — inform our understanding of intelligence?

---

## 2. Key Information-Theoretic Quantities

### 2.1 Shannon Entropy (H)

Measures the average uncertainty or "surprise" of a random variable.

```
H(X) = -∑ p(x) log₂ p(x)
```

- **High entropy** → high uncertainty, high information potential
- **Low entropy** → predictable, low information content

**Role in Intelligence:** Sensory input entropy drives the need for processing. A system in a high-entropy environment needs more information processing capacity.

### 2.2 Mutual Information (I)

Measures the amount of information shared between two variables — how much knowing X reduces uncertainty about Y.

```
I(X; Y) = H(X) - H(X|Y)
```

**Role in Intelligence:**
- **Sensory processing:** Mutual information between stimulus and neural response measures coding efficiency
- **Learning:** Mutual information between input and output constrains what can be learned
- **Attention:** Maximizing mutual information between attended features and goals

### 2.3 Kolmogorov Complexity (K)

Measures the length of the shortest program that produces a given string.

- **Random strings** → high K (cannot be compressed)
- **Patterned strings** → low K (simple program can generate them)

**Role in Intelligence:**
- Intelligence can be framed as finding the shortest (simplest) program that explains observations
- The **Minimum Description Length (MDL)** principle: the best model is the one that most compresses the data
- Links to Occam's razor and scientific discovery

### 2.4 Distinctions

| Measure | Perspective | Focus | Cognitive Analogy |
| :--- | :--- | :--- | :--- |
| **Shannon Entropy** | Statistical | Ensemble behavior / uncertainty | Sensory noise, predictability |
| **Kolmogorov Complexity** | Algorithmic | Individual data structure | Pattern recognition, insight |
| **Mutual Information** | Relative | Dependency between signals | Sensory integration, inference |

---

## 3. The Information Bottleneck Principle

An intelligent system must compress input data while preserving information relevant to future goals.

```
Input → [Compression] → Representations → [Prediction] → Output
         ↓                                ↑
    Irrelevant info                  Relevant info
        discarded                    maintained
```

This creates a fundamental tradeoff:
- **More compression** → better generalization, less detail
- **Less compression** → better accuracy on training, overfitting risk

---

## 4. Algorithmic Information Theory and AI

### 4.1 Solomonoff Induction

A theoretical framework for prediction: the probability of a future observation is proportional to the sum over all programs that explain past observations, weighted by their complexity.

```
P(future | past) ∝ ∑_{programs} 2^{-K(program)} × [program predicts future]
```

**Implication:** The optimal predictor is one that considers *all* possible explanations, weighted by simplicity.

### 4.2 AIXI (Hutter)

A theoretical agent that uses Solomonoff induction to maximize expected reward over all possible futures. Represents the **theoretical upper bound** of intelligence.

**Limitation:** AIXI is not computable — it requires infinite computation.

---

## 5. Implications for Architecture

1. **Intelligence requires compression** — transforming high-entropy sensory input into low-complexity models
2. **Model selection is inevitable** — every system must choose what to compress and what to retain
3. **Prediction and compression are dual** — a system that compresses well predicts well, and vice versa
4. **Bounded information processing** — finite systems cannot maintain all information; forgetting is mandatory

---

## 6. Assumptions

1. **Intelligence can be formalized as compression + prediction.** This is one view (Hutter, Legg) but not universally accepted. [UNVERIFIED]
2. **Kolmogorov complexity is the right measure of model quality.** Practical systems cannot compute K.
3. **The information bottleneck is unavoidable.** Supported by the data processing inequality. [UNVERIFIED — follows from DPI but assumes optimal compression is the only relevant objective]

---

## 7. Open Questions

1. Is intelligence equivalent to compression? Are there aspects of intelligence that compression does not capture?
2. How do biological systems implement efficient compression in practice?
3. Can Kolmogorov complexity be approximated well enough for practical use in cognitive architectures?
4. Is the Information Bottleneck optimal, or are there alternative principles?

---

## 8. Sources

- Shannon, C. (1948): *A Mathematical Theory of Communication*
- Kolmogorov, A.N. (1965): *Three Approaches to the Quantitative Definition of Information*
- Hutter, M. (2005): *Universal Artificial Intelligence*
- Tishby, N. & Zaslavsky, N. (2015): *Deep Learning and the Information Bottleneck Principle*
