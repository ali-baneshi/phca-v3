# Knowledge Graph — Cross-File Dependency Map

**File:** knowledge-graph.md
**Status:** COMPLETE

---

## Domain Dependency Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE FOUNDATIONS                         │
│  (01-what-is-intelligence, 02-minimal-cognition)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SYSTEM STRUCTURE                              │
│  (01-state-and-memory, 02-action-and-feedback)                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌────────────────┐ ┌────────────┐ ┌────────────────┐
│  EMERGENCE     │ │ FAILURES   │ │ CONSTRAINTS    │
│  (one file)    │ │ (one file) │ │ (one file)     │
└───────┬────────┘ └───────┬────┘ └───────┬────────┘
        │                  │              │
        └──────────────────┼──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FOUNDATIONAL THEORY                            │
│  (info theory, control theory, distributed systems, complex systems)│
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  COGNITIVE ARCHITECTURES                            │
│  (Soar, ACT-R, LIDA, CoALA)                                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  NEUROSCIENCE PRINCIPLES                           │
│  (memory, prediction, attention)                                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  MULTI-AGENT SYSTEMS                                │
│  (one file)                                                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUTS                                      │
│  (constraints model, failure map, axioms, readiness report)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File-to-File Cross-References

### 01 — Intelligence Foundations
| File | Cross-refs to |
| :--- | :--- |
| 01-what-is-intelligence | 01-intellect-foundations/02, 07-foundational-theory/01 |
| 02-minimal-cognition | 01-intellect-foundations/01, 03-emergence-conditions/01 |

### 02 — System Structure
| File | Cross-refs to |
| :--- | :--- |
| 01-state-and-memory | 01-intellect-foundations/02, 08-neuroscience/01, 06-cognitive-arch/04 |
| 02-action-and-feedback | 02-system/01, 07-foundational-theory/02, 08-neuroscience/02 |

### 03 — Emergence Conditions
| File | Cross-refs to |
| :--- | :--- |
| 01-emergence-in-systems | 01-intellect-foundations/02, 07-foundational-theory/04, 05-constraints/01 |

### 04 — Failure Modes
| File | Cross-refs to |
| :--- | :--- |
| 01-failure-taxonomy | 05-constraints/01, 07-foundational-theory/02 |

### 05 — Architectural Constraints
| File | Cross-refs to |
| :--- | :--- |
| 01-bounded-computation | 04-failures/01, 07-foundational-theory/01 |

### 06 — Cognitive Architectures
| File | Cross-refs to |
| :--- | :--- |
| 01-soar | 06-cognitive-arch/02,03,04; 02-system/01; 05-constraints/01 |
| 02-act-r | 06-cognitive-arch/01,03,04; 08-neuroscience/01 |
| 03-lida | 06-cognitive-arch/01,02,04; 08-neuroscience/03 |
| 04-coala | 06-cognitive-arch/01,02,03; 02-system/01 |

### 07 — Foundational Theory
| File | Cross-refs to |
| :--- | :--- |
| 01-information-theory | 07-foundational/02,04; 05-constraints/01 |
| 02-control-theory | 07-foundational/01,04; 02-system/02 |
| 03-distributed-systems | 07-foundational/04; 09-multi-agent/01 |
| 04-complex-systems | 07-foundational/01,02,03; 03-emergence/01; 09-multi-agent/01 |

### 08 — Neuroscience Principles
| File | Cross-refs to |
| :--- | :--- |
| 01-memory-systems | 08-neuroscience/02,03; 02-system/01; 06-cognitive-arch/02 |
| 02-prediction-processing | 08-neuroscience/01,03; 07-foundational-theory/02; 06-cognitive-arch/04 |
| 03-attention-mechanisms | 08-neuroscience/01,02; 06-cognitive-arch/03 |

### 09 — Multi-Agent Systems
| File | Cross-refs to |
| :--- | :--- |
| 01-multi-agent-theory | 07-foundational/03,04; 03-emergence/01 |

### Outputs
| File | Cross-refs to |
| :--- | :--- |
| 01-constraints-model | 05-constraints/01; 03-emergence/01 |
| 02-failure-modes-map | 04-failures/01; outputs/01 |
| 03-axiom-candidates | outputs/01; outputs/04 |
| 04-readiness-report | outputs/01,02,03 |
