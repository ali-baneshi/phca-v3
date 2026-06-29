# Multi-Agent Systems and Collective Intelligence

**Domain:** Multi-Agent Systems
**File:** 09-multi-agent-systems/01-multi-agent-theory.md
**Status:** DRAFT
**Cross-refs:** [07-foundational-theory/03-distributed-systems.md](../07-foundational-theory/03-distributed-systems.md), [07-foundational-theory/04-complex-systems.md](../07-foundational-theory/04-complex-systems.md), [03-emergence-conditions/01-emergence-in-systems.md](../03-emergence-conditions/01-emergence-in-systems.md)

---

## 1. Core Question

How does collective intelligence emerge from the interaction of multiple agents? What architectural patterns enable effective multi-agent cognition?

---

## 2. Agent Architectures

### 2.1 Types of Agents

| Type | Internal Model | Complexity | Example |
| :--- | :--- | :--- | :--- |
| **Reactive** | None — direct stimulus-response | Low | Simple robots, reflex systems |
| **Model-Based** | Internal state representation | Medium | Soar, ACT-R agents |
| **Goal-Based** | Goals + planning | High | BDI (Belief-Desire-Intention) agents |
| **Utility-Based** | Utility function for tradeoffs | High | RL agents |
| **Learning** | Can improve over time | Very high | LLM agents |

### 2.2 BDI Architecture (Belief-Desire-Intention)

A prominent cognitive model for agents:

- **Beliefs:** What the agent knows about the world (state representation)
- **Desires:** What the agent wants to achieve (goals)
- **Intentions:** What the agent commits to doing (selected plans)

**Key Insight:** BDI separates reasoning about the world from reasoning about goals and actions. This modularity is useful for multi-agent systems where different agents may have different beliefs, desires, and intentions.

---

## 3. Communication Protocols

### 3.1 Engineered Protocols

Formal, deterministic frameworks for agent communication:

- **FIPA Standards:** Standardized agent communication language
- **Message Types:** Inform, request, propose, accept-proposal, reject-proposal
- **Ontologies:** Shared vocabularies for semantics

### 3.2 Emergent/Learned Protocols

Agents develop their own communication strategies (often via RL):
- **DIAL:** Differentiable inter-agent learning
- **TarMAC:** Targeted multi-agent communication
- **Gossip Protocols:** Scalable, fault-tolerant information dissemination

### 3.3 Tradeoffs

| Approach | Flexibility | Reliability | Interpretability |
| :--- | :--- | :--- | :--- |
| **Engineered** | Low | High | High |
| **Emergent** | High | Variable | Low |

---

## 4. Coordination Mechanisms

| Mechanism | Description | When to Use |
| :--- | :--- | :--- |
| **Market-Based** | Agents bid on tasks using price signals | Heterogeneous capabilities, dynamic tasks |
| **Consensus** | Agents agree on shared state (Raft, Paxos) | Safety-critical coordination |
| **Hierarchical** | Leader determines actions | Clear chain of command |
| **Swarm / Stigmergy** | Indirect coordination via environment | Large numbers of simple agents |
| **Negotiation** | Agents bargain for resources | Conflicting goals |

---

## 5. Emergent Behavior

### 5.1 Conditions for Emergence

Intelligence beyond individual capacity emerges when:

1. **Agents are diverse** — different specialized capabilities
2. **Agents interact** — communication and coordination
3. **Feedback exists** — actions affect other agents and the environment
4. **No central controller** — intelligence is distributed

### 5.2 Measuring Emergence

Recent frameworks use **information theory** (Partial Information Decomposition) to measure emergence:

- **Synergy:** System predictive power > sum of individual predictive powers (true emergence)
- **Redundancy:** System predictive power = individual predictive powers (no emergence)
- **Unique:** Individual contributions without synergy (no emergence)

### 5.3 Synergy vs. Redundancy

- **Good emergence (synergy):** Agents complement each other, producing intelligence beyond any individual
- **Bad emergence (redundancy):** Agents duplicate effort, producing no additional intelligence
- **Pathological emergence:** Agents interfere with each other, reducing total intelligence

---

## 6. Implications for Architecture

1. **Distributed intelligence requires intentional design** — emergence is not automatic
2. **Communication overhead must be balanced** — too much costs resources; too little reduces coordination
3. **Diversity is valuable** — homogeneous agents create redundancy, not synergy
4. **Partial knowledge is the default** — agents cannot know everything; coordination mechanisms must handle uncertainty
5. **Scalability requires locality** — agents cannot process all other agents' information

---

## 7. Assumptions

1. **Collective intelligence can exceed individual intelligence.** Well-established (ant colonies, human teams).
2. **Communication is necessary for coordination.** [UNVERIFIED — some coordination occurs without communication (stigmergy)]
3. **Diverse agents are better than homogeneous agents.** True for some tasks; not universal.

---

## 8. Open Questions

1. What is the optimal ratio of agent specialization to generality?
2. Can multi-agent systems produce forms of intelligence that single-agent systems cannot?
3. How do you prevent emergent misalignment — where collective behavior veers from intended goals?
4. Is there a universal coordination mechanism, or is coordination fundamentally domain-specific?

---

## 9. Sources

- Jennings, N.R. (2000): *On Agent-Based Software Engineering* — Foundations of MAS
- Wooldridge, M. (2002): *An Introduction to Multiagent Systems*
- Emergent Coordination in Multi-Agent Language Models — [arXiv:2510.05174](https://arxiv.org/html/2510.05174v1)
- Balch, T. & Parker, L. (2002): *Robot Teams: From Diversity to Polymorphism*
