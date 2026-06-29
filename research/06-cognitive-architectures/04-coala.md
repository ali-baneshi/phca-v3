# CoALA — Cognitive Architectures for Language Agents

**Domain:** Cognitive Architectures
**File:** 06-cognitive-architectures/04-coala.md
**Status:** DRAFT
**Cross-refs:** [01-soar](01-soar.md), [02-act-r](02-act-r.md), [03-lida](03-lida.md), [02-system-structure/01-state-and-memory.md](../02-system-structure/01-state-and-memory.md)

---

## 1. Overview

**CoALA** (Cognitive Architectures for Language Agents) is a conceptual framework proposed by Sumers et al. (2023) at Princeton. It provides a structured vocabulary and design space for building LLM-based autonomous agents by bridging classical cognitive architecture theory with modern language model capabilities.

**Key Principle:** CoALA systematizes the design of language agents by organizing them around three pillars: **Memory**, **Action Space**, and **Decision-Making**. It argues that LLMs can serve as the "engine" for realizing classical cognitive architectures (Soar, ACT-R) in modern AI.

---

## 2. The Three Pillars

### 2.1 Memory

CoALA divides memory into multiple systems with different functions and timescales:

| Memory Type | Duration | Function | Analogy |
| :--- | :--- | :--- | :--- |
| **Working Memory** | Short (within context window) | Immediate reasoning, current goals | LLM context window |
| **Episodic Memory** | Long-term | Past experiences | Stored conversation history |
| **Semantic Memory** | Long-term | Facts and knowledge | Knowledge base / vector DB |
| **Procedural Memory** | Long-term | How-to knowledge (skills) | Tools, APIs, code |


### 2.2 Action Space

CoALA defines a structured space for agent actions, divided into two categories:

**Internal Actions** (operating on internal state):
- Memory retrieval / storage
- Self-reflection / reasoning steps
- Attention / focus switching
- Goal decomposition

**External Actions** (operating on the environment):
- Tool use (APIs, calculators, search)
- Web browsing
- Environmental manipulation (robotics)
- Communication

### 2.3 Decision-Making

The decision process is an interactive loop:

1. **Observe** — Receive input from environment
2. **Retrieve** — Access relevant memory (internal action)
3. **Reason** — Plan or infer using the LLM's capabilities
4. **Decide** — Select an action (internal or external)
5. **Act** — Execute the action
6. **Learn** — Update memory based on outcome

---

## 3. Relationship to Classical Architectures

| Classical Architecture | CoALA Analogy | Translation |
| :--- | :--- | :--- |
| **Soar** (Problem Space) | Working Memory + Goals | LLM decomposes tasks via prompts |
| **ACT-R** (Buffer-based) | Memory types + Retrieval | Vector DB as declarative memory |
| **LIDA** (Cognitive Cycle) | Observe → Act → Learn loop | LLM tool-use loop |
| **Global Workspace** | Context window | What's currently "conscious" to the LLM |

---

## 4. Strengths

- **Synthesis:** Unifies decades of cognitive architecture research with modern LLM practice
- **Modularity:** Clear separation of concerns (memory, action, decision)
- **Comparability:** Provides a common vocabulary for comparing different agent designs
- **Extensibility:** Can incorporate new LLM capabilities without changing the framework

---

## 5. Weaknesses

- **Dependence on LLM Core:** Framework quality depends on underlying LLM quality
- **No Learning Mechanism Specified:** Does not address how agents improve from experience
- **No Perception Layer:** Assumes the LLM's text input is sufficient
- **No Embodiment:** No account of sensorimotor grounding

---

## 6. Assumptions

1. **Language is a sufficient interface for intelligence.** This is a major assumption inherited from LLM-based approaches. [See 01-intelligence-foundations/01-what-is-intelligence.md](../01-intelligence-foundations/01-what-is-intelligence.md)]
2. **Classical cognitive architectures provide a useful design vocabulary.** This may not hold for non-symbolic forms of intelligence.
3. **Memory can be effectively retrieved via similarity search.** Vector similarity may not capture contextual relevance.

---

## 7. Open Questions

1. Can CoALA be extended to handle truly embodied, sensorimotor intelligence?
2. How does learning occur within the CoALA framework? What is the equivalent of chunking or base-level learning?
3. Is the LLM a suitable "engine" for implementing classical architectures, or does it introduce qualitatively different properties?

---

## 8. Sources

- Sumers, T.R. et al. (2024): *Cognitive Architectures for Language Agents* — [arXiv:2309.02427](https://arxiv.org/abs/2309.02427)
- Cognee Blog: *CoALA Explained* — [cognee.ai](https://www.cognee.ai/blog/fundamentals/cognitive-architectures-for-language-agents-explained)
