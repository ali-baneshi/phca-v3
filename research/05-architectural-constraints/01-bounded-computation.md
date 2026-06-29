# Architectural Constraints: Bounded Computation, Time, and Resources

**Domain:** E. Architectural Constraints
**File:** 05-architectural-constraints/01-bounded-computation.md
**Status:** DRAFT
**Cross-refs:** [04-failure-modes/01-failure-taxonomy.md](../04-failure-modes/01-failure-taxonomy.md), [07-foundational-theory/01-information-theory.md](../07-foundational-theory/01-information-theory.md)

---

## 1. Core Question

What constraints must any intelligent architecture accept as fundamental? What limits are imposed by computation, time, and resources?

---

## 2. The Three Fundamental Constraints

### 2.1 Bounded Computation

Any physical system has finite computational capacity.

- **Finite State Machines:** Every physical computer has finite memory and states
- **Halting Problem:** No finite system can perfectly predict the behavior of all other finite systems
- **Complexity Classes:** Some problems are inherently intractable (NP-hard, undecidable)
- **Kolmogorov Bound:** The shortest program describing a system cannot be arbitrarily short

**Implication:** An intelligent system cannot:
- Model itself perfectly (self-reference problem)
- Compute optimal action in all situations
- Achieve perfect prediction
- Represent all possible states of its environment

### 2.2 Time Dependence

All physical processes take time.

- **Computation Latency:** Decisions take non-zero time
- **Communication Latency:** Information transfer is finite (speed of light limit)
- **Feedback Delay:** Actions have delayed consequences
- **Temporal Binding:** Events must be temporally correlated to be meaningful

**Implication:** An intelligent system must:
- Act before optimal decisions can be computed (bounded rationality)
- Handle asynchronous information arrival
- Maintain temporal coherence across delays
- Predict future states to compensate for latency

### 2.3 Resource Limitations

Computation, energy, memory, and attention are finite.

| Resource | Limitation | Architectural Impact |
| :--- | :--- | :--- |
| **Energy** | Thermodynamic limit on computation | Must minimize unnecessary computation |
| **Memory** | Storage capacity limits | Must forget, compress, prioritize |
| **Attention** | Processing bandwidth limits | Must select what to process |
| **Information** | Channel capacity limits | Must filter, summarize, abstract |

**Implication:** An intelligent system must:
- Trade off between resource costs and decision quality
- Compress and abstract information
- Forget or archive less important information
- Prioritize which computations to perform

---

## 3. Derived Constraints

### 3.1 The Bounded Rationality Constraint

Because of bounded computation and time, systems cannot always find optimal solutions. Instead, they satisface — find solutions that are "good enough" within resource limits.

**Formalization:** An agent with bounded rationality chooses action *a* from a set *A*, maximizing expected utility *U(a)* but subject to computational cost *C(a)*:

```
a* = argmax_a [U(a) - C(a)]
```

Where *C(a)* includes time, energy, and memory costs of computing *a*.

### 3.2 The Non-Self-Representation Constraint

No finite system can contain a complete representation of itself (Russell's paradox analogue).

- A complete self-model would require more states than the system has
- Self-reference leads to incompleteness (Gödelian)
- Recursive self-improvement faces fundamental limits

### 3.3 The Uncertainty Constraint

Because of finite information and computation, all knowledge is probabilistic.

- **Epistemic Uncertainty:** Lack of knowledge about the world
- **Aleatoric Uncertainty:** Inherent randomness in the world
- **Model Uncertainty:** Imperfect internal models

### 3.4 The Compression-Adaptivity Tradeoff

- More compression (simpler models) → better generalization but less detail
- Less compression (more detailed models) → better prediction on seen data but overfitting
- Optimal balance depends on environment stability

---

## 4. Constraint Interactions

```
                ┌─────────────────────┐
                │  Bounded            │
                │  Computation        │
                └──────────┬──────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Time         │  │  Resources    │  │  Uncertainty  │
│  Dependence   │◄─┤  (Memory,     │◄─┤  (No perfect  │
│  (Latency,    │  │   Energy,     │  │   knowledge)  │
│   delay)      │  │   Attention)  │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

All constraints interact and compound. Bounded computation forces resource tradeoffs, which are constrained by time, which creates uncertainty.

---

## 5. Universal Invariants (Axiom Candidates)

Based on the constraints analysis, the following appear to be **universal constraints** that any intelligence in a physical universe must respect:

1. **Finiteness:** No system has infinite capacity
2. **Temporal Locality:** Effects follow causes; information propagates at finite speed
3. **Resource Tradeoff:** Optimization on one axis incurs cost on another
4. **Self-Ignorance:** No system can fully model itself
5. **Probabilistic Knowledge:** All knowledge about the external world is uncertain
6. **Bounded Optimality:** Optimal solutions are not always reachable; satisficing is mandatory

---

## 6. Assumptions

1. **The constraints are universal** — they apply to any physical intelligence, biological or artificial. This assumes physicalism.
2. **Quantum computing does not fundamentally change these constraints.** Quantum algorithms may change complexity bounds but do not eliminate finiteness.
3. **Bounded rationality is a feature, not a bug** — it enables generalizability by preventing over-optimization.

---

## 7. Open Questions

1. Could a system with very different resource tradeoffs (e.g., a galaxy-sized computer) exhibit qualitatively different intelligence?
2. Are there architectural constraints that apply specifically to digital systems but not biological ones (and vice versa)?
3. Can bounded rationality be formalized into a design methodology?
4. Is the non-self-representation constraint absolute, or can asymptotic self-models be sufficient?

---

## 8. Sources

- Simon, H. (1956): *Rational Choice and the Structure of the Environment* — Origins of bounded rationality
- Gödel, K. (1931): *On Formally Undecidable Propositions* — Incompleteness theorems
- Chaitin, G. (1987): *Algorithmic Information Theory* — Kolmogorov complexity bounds
- Russell, S. & Wefald, E. (1991): *Do the Right Thing: Studies in Limited Rationality*
