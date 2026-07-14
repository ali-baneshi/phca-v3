# PHCA Course — Lessons 2 to 10

*(Reminder: The full course map is in Lesson 1. This file covers Lessons 2–10 — the remainder of "Part 1: Foundation and Base", plus the start of "Part 2: The Building Skeleton" and the beginning of "Part 3: The Predictive Brain".)*

---

# Lesson 2: First Principle — Resource Constraints (A1)

## A Simple Picture

Imagine you are the architect of a building where one person will live, but that person has only **one small battery** that gets charged daily, only **one small refrigerator** for food, and only **a few hours** to get things done. If you design the building as if this person had unlimited resources (e.g., a massive heating system that drains the entire battery in one hour), the building looks beautiful on paper but is unlivable in practice.

PHCA applies exactly this philosophy to a "cognitive agent": **No module is allowed to assume it has unlimited resources.**

## What Does "Constraint" Mean Exactly?

In PHCA's code (`config.py`), each module has a small object called `ResourceBounds` with four numbers:

- **B_time** — the maximum time this module may spend in each "cycle" (each run). For example, the sensor module (ASI) has only 5 ms; the predictor module (G′) has a bit more, 20 ms.
- **B_mem** — the maximum memory it is allowed to use (e.g., G′ up to 500 KB).
- **B_energy** — an estimate of "energy" based on the number of mathematical computations performed (not real electrical energy, but a conventional unit proportional to the number of computational operations).
- **entropy_floor** — this one is a bit different and will be explained fully in Lesson 4; for now, just know it means "this module is never allowed to claim it is 100% certain."

## Who Ensures No One Exceeds Their Budget?

A special module called **RBTA** (Resource-Bounded Turing Supervisor) checks every cycle, after all modules have done their work: "Did anyone exceed their budget?" If yes, depending on how many violations occurred:

- **0 violations** → nothing happens, the cycle continues normally.
- **1 or 2 violations** → the system says "INTERRUPT" — meaning the next cycle becomes more restricted (e.g., it may only evaluate one action option instead of several).
- **3 or more violations, or one very severe violation** → the system says "TERMINATE" — meaning the agent is forced to perform a completely safe and simple action (e.g., "stay put") until things calm down.

The important architectural point is: **These reactions are real, not just a warning in a log.** When RBTA says TERMINATE, the agent's actual behavior in the next cycle changes — this is different from many other systems that only print a warning message but do not change behavior.

## Why This Design Is Smart (and Why It Is Hard)

The benefit is that the agent is forced to **prioritize** — just like a living being that, when tired or hungry, cannot do everything and must choose. But the difficulty is that if you set the budgets incorrectly (too strict or too lenient), or if this supervisor itself has a bug, the entire agent's behavior can become unpredictable. In our reviews, this is exactly what happened: a small change in how "violation severity" was calculated caused entropy violations (one of the four budget types) to go undetected for a while — meaning the supervisor was completely blind to one type of risk, without anyone noticing, until it was carefully tested.

## Lesson 2 Summary
- Each module has an explicit budget: time, memory, energy, and (as we will see) a minimum uncertainty floor.
- A central supervisor (RBTA) checks these budgets every cycle and, if violated, **actually changes behavior**, not just warns.
- This supervisor must be designed very carefully, otherwise it may become blind to certain types of risk without being noticed.

**Next Lesson:** Second Principle — Temporal Causality: why the execution order of modules matters and what happens if this order breaks down.

---

# Lesson 3: Second Principle — Temporal Causality (A2)

## A Simple Picture

Suppose you want to decide whether to take an umbrella. Logically, you first look at the sky (observation), then predict "it will probably rain" (prediction), and then decide (action). If the order is reversed — you decide first and *then* look at the sky — your decision is meaningless because it was not based on any information.

This is exactly what "Temporal Causality" guarantees in PHCA: **Each module should only use the output of modules that actually executed before it.**

## Why Should This Be a Separate "Principle"?

You might think this is obvious — can you even write code that uses the future? In practice, in complex systems with multiple modules that each maintain "state," such bugs happen very easily. For example: imagine the action selection module, instead of using the prediction from *this cycle*, mistakenly uses the prediction from the *previous cycle* (which is still in memory). The code runs, no error is raised, but logically "old data" replaced "fresh data" — this is precisely a violation of temporal causality, just much subtler than the umbrella example.

## How Does PHCA Guarantee This?

With a **fixed and documented sequence** for each cycle (fully detailed in Lesson 7): first input sanitization, then memory writing, then prediction, then action selection, then action execution in the real world, and only *after that* error computation and learning. Each of these stages is only allowed to use the output of *previous* stages.

An interesting detail is that the builders themselves also created a deliberate testing tool: a flag called `desync` that, when activated, intentionally messes up the order — precisely to verify that their detection system actually understands when this principle is violated. This is a very good engineering technique: **to make sure you are checking a rule correctly, intentionally build a broken version and see if you can detect that it is broken.**

## A Subtler Architectural Point

In the official cycle diagram, some modules (e.g., MDIM, APC, Attention, HPM) are set up "before action selection," while PEU and TSPL (which measure error and learn) work "after action execution in the real world." This order is exactly logical: you cannot compute prediction error unless you know what *actually* happened, and you only know that after executing the action.

## Lesson 3 Summary
- Temporal causality means: no module should use data that has not yet been produced.
- This seems obvious in simple systems, but in multi-module systems it is very easily and subtly violated.
- PHCA has a fixed sequence and even built a deliberate "sabotage" tool to ensure violation detection actually works.

**Next Lesson:** Third Principle — Incomplete Knowledge: why this agent is never allowed to say "I am 100% certain."

---

# Lesson 4: Third Principle — Incomplete Knowledge (A3)

## A Simple Picture

Think of a very confident friend who never says "I don't know" — they always answer with 100% certainty, even when they have no idea about the topic. This person is dangerous because you cannot tell when to actually trust them and when not to.

PHCA wants to avoid this state. The third principle states: **The agent should never pretend its knowledge is complete.** There must always be some degree of "uncertainty" remaining in its beliefs.

## How Is This Uncertainty Measured?

Through a concept called **belief entropy** — a mathematical number that tells "how spread out is my belief probability distribution?" If the agent is completely certain, this number approaches zero. The rule is: this number must never reach absolute zero — it has a "floor" (e.g., 0.01) that must always be maintained.

The practical measurement in the prediction model (G′) is interesting: it uses a technique called **MC-Dropout**. The idea is: feed the same input to the model several times, but each time randomly turn off a small subset of the network's neurons (dropout) and see how much the answers differ. If the answers are very similar, the model is "self-consistent" (high certainty). If the answers are scattered, the model is uncertain.

*(A quick reminder since we saw this in detail in our earlier discussions: this type of "certainty" only tells "the model is self-consistent," not necessarily "the model is correct." In the lesson on action selection, we will see in detail what a big consequence this difference had.)*

## Why Must This Floor Always Be Maintained?

If the agent is allowed to be 100% certain, two risks arise:
1. It has no motivation to explore or investigate further — because it thinks it knows everything.
2. If that 100% certainty is wrong (which is always possible), there is no mechanism to detect this mistake.

RBTA (the supervisor from the previous lesson) checks this entropy floor just like time/memory/energy — if a module's entropy falls below the floor, this counts as a "violation."

## An Honest Note from Our Own Review

When we examined this part in the actual code, we noticed that this entropy check **in practice is only meaningful for one or two modules** (like the prediction model and the motivational system), while for the rest a fixed symbolic number is set that can never be violated. This means on paper the principle is defined "for the whole system," but in practice only part of the system is actually under this supervision. This is a good example of how a beautiful principle at the design level may only be partially implemented at the implementation level — exactly the kind of thing an architect should always look for.

## Lesson 4 Summary
- The agent must never pretend its knowledge is complete; this is measured by a mathematical number (belief entropy) with a minimum floor.
- Practical measurement is done via MC-Dropout — which measures "model self-consistency," not necessarily "model correctness."
- This check is in practice only truly active for a subset of modules — a point every architect should consider when distinguishing between "complete design" and "complete implementation."

**Next Lesson:** Fourth Principle — Prediction as the Core: the beating heart of the entire system, and the source of the biggest challenge in this project.

---

# Lesson 5: Fourth Principle — Prediction as the Core (A4)

## A Simple Picture

Imagine two types of drivers. The first driver only follows fixed rules: "If the light is red, stop." The second driver constantly simulates in their mind: "If I press the gas now, where will I be in three seconds? Will the car in front brake?" and decides based on these ongoing predictions.

PHCA aims to be the second type. This principle states: **Every cycle, the agent must guess "if I do this right now, what will the world look like in the next step?" and this guess must actually influence its decision.**

## Why This Principle Was the Most Important and Eventful of the Entire Project

In the first four principles, the issue was mostly "how to implement this correctly." Here, the question is deeper: **Can this principle even be implemented correctly in the real world?**

Let me tell the story simply, because this is the best example of "thinking like an architect" that we saw throughout this project:

1. **Step one:** The design says every decision must come from the model's prediction. But in the early version, when the agent had a specific goal (e.g., "go to that corner of the room"), instead of using the model's prediction, it used a classical pathfinding algorithm (similar to what powers Google Maps) — without ever looking at the learned model. This meant the principle was effectively violated, but no one noticed because a number always came out from somewhere.

2. **Step two:** This was discovered and fixed — the decision was forced to always go through the model's prediction.

3. **Step three (this is where it gets interesting):** When they tried this in a larger and harder world, **it was a disaster.** The agent almost never reached its goal — worse than a completely random agent! Because the learned model was not yet good enough, and when decision-making relied entirely on it, bad decisions were made.

4. **Step four:** The implementers tried a middle ground: "only when the model is confident, use its prediction; otherwise fall back to the classical method." But when they tried this too, it turned out that the **"confidence" measure itself was not reliable** — the model always claimed high confidence, even when it was completely wrong (exactly what I warned about MC-Dropout in the previous lesson).

## The Architectural Lesson of This Story

This story illustrates a very important point: **A beautiful design principle (prediction must drive decision-making) can be completely correct at the philosophical level, but implementing it correctly requires many things that are not obvious at first glance** — including that the prediction model must actually be good, and the "confidence" measure must actually be meaningful, not just a nice number that always stays high.

The current result? In continuous, physical worlds (like controlling an arm or a pendulum), this principle actually works and the model truly drives decisions. In discrete grid worlds with specific goals, a classical algorithm currently drives decisions, and the learned model is more of an "observer" than a "decision-maker" — and this is honestly documented as such, not hidden.

## Lesson 5 Summary
- This principle says: decisions must come from the model's ongoing prediction, not from fixed rules.
- The history of this principle in the project went through a complete cycle: hidden violation → fix → disaster at larger scale → attempt at middle ground → discovery that the confidence measure itself is not reliable.
- The lesson for an architect: no principle, however beautiful, is reliable without rigorous testing at different scales.

**Next Lesson:** Fifth Principle — Learning from Feedback: how the agent actually learns from its mistakes (and a similar story of a "disconnected wire").

---

# Lesson 6: Fifth Principle — Learning from Feedback (A5)

## A Simple Picture

When a child learns to walk, every time they fall they receive an "error signal" — their body figures out "that movement didn't work" — and next time they adjust slightly. No one from the outside says "good job" or "no"; the result itself (falling or not falling) is the feedback.

The fifth principle states: **The agent should work exactly the same way — the difference between "what it predicted" and "what actually happened" must be its learning engine.**

## The Learning Chain in PHCA

Let's trace it step by step, because this chain itself is a good example for understanding "layered architecture":

1. **PEU** (Prediction Error Unit) computes the difference between prediction and reality — a simple number: "how wrong was I?"
2. **TSPL** (a small learning module) takes this error and makes a minor adjustment to the model.
3. **The prediction model (G′)** itself, separately, adjusts its main weights using a standard machine learning method (gradient descent on recent data).

The interesting point is: these two learning paths (TSPL and G′ itself) work **separately**. TSPL only adjusts a small auxiliary vector (like a fine-grained final correction), while the main model weights (where almost all "intelligence" is stored) are learned by the model itself, independent of TSPL.

## An Extra Layer: Attention

The design says that in addition to overall error, there should be an "attention" layer that determines "which part of the information is more important?" and this importance should affect the intensity of learning — meaning when something that received a lot of attention is predicted incorrectly, it should get a stronger correction.

## The Story of a "Disconnected Wire" — Another Example of the Same Pattern

When we examined this part carefully in the code, we found something interesting (which was fortunately later fixed): the attention layer actually computed its weights and even "wrote" them onto the prediction model — but the model itself never "read" these weights! Exactly like a power cord plugged into a socket but with the other end connected to nothing. Electricity flows through the wire (computation happens), but no light turns on (no effect on learning).

This is a good example of a key point for any architect: **Seeing that "the code exists and runs" is not enough; you must look for proof that this code actually reaches its destination.** This bug was later found and fixed — currently, attention weights actually affect learning intensity.

## How Do We Know This Principle Is Working Correctly?

There is a very simple and elegant test for this: if you turn learning "off," the model weights should not change at all (relative change should be exactly zero). If learning is "on," you should see a measurable change. This is exactly what is checked in the project's formal tests — a simple but decisive test.

## Lesson 6 Summary
- Learning must come from the difference between "prediction" and "reality," not from an external judgment.
- This chain has multiple layers: PEU (error measurement) → TSPL (fine adjustment) → model itself (main learning) → attention (importance weighting).
- A real "disconnected wire" example (attention was computed but had no effect) showed why you must always demand practical proof, not just seeing the code.

**Next Lesson:** Now that we have seen all five principles, it is time to see the complete skeleton — the overview of the 12-step cycle that executes all these principles together.

---

# Lesson 7: Overview of the 12-Step Cycle

## Why Now Is a Good Time for This Lesson

So far we have seen the five principles separately — like showing you the electrical plan, the plumbing plan, the structural plan, etc. separately. Now it is time to put them all together and see the complete building plan at once.

## An Important Note Before Starting: Two Types of "Numbers" That Should Not Be Confused

Before diving into details, let me clarify a common confusion. In this project's documentation, you sometimes see "12 steps" and sometimes numbers like "Phase 6" or "Phase 19." These are **two completely different things:**

- **"12 cognitive steps"** — this is what we see in this lesson: the fixed sequence that executes *every time* the agent takes a step. This is a "runtime" concept.
- **"Phases 1 to 19/20"** — these refer to the **construction history of the project itself** (like chapters of a book that the author wrote sequentially) — e.g., "Phase 5" means the period when execution speed was worked on, "Phase 6" means the period when scientific validation was worked on. This is a "development history" concept, not something that executes every time. The project is currently in periods of "hardening and maturation," which are a continuation of those same phases.

Whenever you see these two numbers, ask yourself "What time frame is this number talking about — every time the agent moves, or during the construction of the entire project?"

## Now: The 12-Step Sequence

Let me narrate it like a story — a "day in the life of the agent" in one step:

**1. Input Sanitization (ASI).** The agent sees something from the world (e.g., its position on a grid). First, this input is sanitized — checked for strange numbers (like infinity or "not a number").

**2. Memory Writing (M1/M2).** This new observation is recorded in short-term and working memory.

**3. Prediction (G′).** The agent guesses "if I perform each of these possible actions, what will happen next?"

**4. Action Selection.** Based on these predictions (and the goals we will see in the next step), the agent selects an action.

*— Here the action is actually executed in the world (`env.step`) —*

**5. Error Computation (PEU).** Now that the actual result is known, it is compared with the prediction from step 3.

**6. Learning (TSPL + model itself).** Based on this error, the model is slightly adjusted.

*— Concurrently with steps 5 and 6, a parallel path also executes —*

**7. Motivational Adjustment (MDIM, APC, Attention, HPM).** The system decides "what is most important to me right now?" and adjusts its parameters. (These also affect action selection beforehand, although they are drawn after it in the diagram — we will see why in their own lesson.)

**8. Safety Check (RBTA).** The supervisor we saw in Lesson 2 checks whether anyone exceeded their budget.

**9. Logging.** Everything is saved for later review.

**10. Memory Consolidation.** Every so often, episodic memories are converted into more "general knowledge" (like how our brain categorizes the day's memories when we sleep).

**11. Counter Increment.** The cycle number increases by one.

**12.** And everything repeats from the beginning — with one difference: this time the model is slightly "more experienced."

## An Architectural Image to Help Visualization

If you want to imagine this as a building plan: input enters through the "main door" (ASI), passes through the "hallway" of memory, enters the "thinking room" (prediction), goes from there to the "decision room," goes through the "exit door" to the world, and when the world reacts, comes back through a "return hallway" (error and learning) to improve the internal map of the building slightly. This cycle repeats thousands of times a day, and each time, the internal map becomes slightly more accurate.

## Lesson 7 Summary
- The 12-step cycle is the fixed sequence that runs every time the agent moves — not to be confused with "project development phases" (which are a completely separate, historical concept).
- The sequence is: sanitization → memory → prediction → action selection → execution in the world → error → learning → (parallel: motivational adjustment) → safety check → logging → consolidation → repeat.
- From this lesson onward, we enter the individual rooms of this building and examine each one up close.

**Next Lesson:** The first room — Input Sanitization Module (ASI). The simplest module, but an important guardian.

---

# Lesson 8: Input Sanitization Stage (ASI)

## Why the First Room of the Building Should Be a "Guard"

Imagine you own a factory where raw materials come in from outside. If a contaminated or defective shipment enters the production line, it could ruin the entire line. That is why every good factory has an incoming inspection station — before anything enters the main line.

**ASI** plays exactly this role in PHCA. Its name is an acronym for something technical, but its job is very simple and vital: **Everything that comes from the outside world (an observation, a number, a vector of information) is checked before entering the rest of the system.**

## What Exactly Is It Looking For?

Mostly, it looks for two types of "contamination" that are very dangerous in mathematical computation:

- **NaN** (Not a Number) — an invalid mathematical result, e.g., when you divide zero by zero.
- **Inf** (Infinity) — a number that exceeds the representable limit.

Why are these dangerous? Because if just *one* of these enters the neural network computations (the prediction model), it acts like a drop of poison in a pool — it spreads quickly and ruins everything. A NaN that enters a computation almost always causes all subsequent computations to also become NaN. If this enters the model's weights, it may permanently corrupt the entire model — the model will never return to a healthy state.

## A Simple Module, But Why Is It Important?

ASI might be the simplest module in the entire system — it does nothing complex, just checks and sanitizes. But this simplicity carries an important point: **In software architecture, the most important components are not always the most complex ones.** A front-door guard that does its job correctly can prevent greater disasters deeper in the system — just like a simple blood test can prevent a larger disease.

## Its Connection to Previously Seen Principles

The interesting point is that ASI also has its own budget (per the first principle — Lesson 2): only 5 ms of time and a small amount of memory. This shows that even the simplest module is not exempt from the general rule that "no one has unlimited resources."

## Lesson 8 Summary
- ASI is the first inspection station — every input passes through it before entering the rest of the system.
- It looks for invalid numbers (NaN) and infinity (Inf), because these can spread like poison through the entire system.
- The simplicity of this module does not mean it is unimportant — rather, it is a good example of how the best defenses are sometimes the simplest.

**Next Lesson:** Now that the input is clean, it must be stored somewhere — we enter the short-term and working memory room (M1 and M2).

---

# Lesson 9: Short-Term and Working Memory (M1 and M2)

## Why Does an Agent Need Two Types of Memory Instead of One?

Imagine you are cooking. You have a "work table" in front of you on which you are currently working (the onion you just chopped, the pot on the stove) — these are things you need *right now*. Separately, you have a "refrigerator" that stores things you might need later.

PHCA has the same division, with two technical names:

**M1 (Sensory/Short-Term Memory):** A very small, very short-lived buffer — like the kitchen work table. It only keeps the last 50 observations and works like a queue (FIFO — First In, First Out: the oldest thing leaves first to make room for something new).

**M2 (Working Memory):** Slightly "smarter" than M1. Instead of simply discarding the oldest item, it has a concept called **salience** — meaning it can determine "this memory is more important than that one" and decide what to keep based on that.

## Why Is This Difference Important?

If you only had a simple memory that always "discards the oldest," a problem would arise: imagine a very important moment happened (e.g., an unexpected collision with an obstacle), but because that moment is slightly older, it gets forcibly discarded just when you still need it. M2, with its concept of "salience," tries to prevent this problem — it keeps more important things longer, even if they are older.

## What Role Do These Memories Play in the Rest of the System?

Everything the prediction model (G′) or the motivational system (MDIM) needs to "understand the current situation" comes from these memories. If these memories are corrupted or incomplete, all subsequent decisions will be built on incomplete data — just like a building where if the foundation is weak, no matter how beautiful the upper floors are, the entire structure is at risk.

## An Architectural Point for Further Thought

When you look at these two memory layers, a good question is: "When do these two layers become misaligned?" For example, what happens if M1 (which blindly discards the oldest) discards something that M2 still needs? These types of questions are exactly what an experienced architect should always ask — not just "what does each part do," but "where might these parts interfere with each other?" (In the lesson on long-term memory — M3 — we will see a real example of this type of interference that actually happened in the project.)

## Lesson 9 Summary
- M1 is a very short-term, simple memory (a 50-item queue, oldest out first).
- M2 is a slightly smarter memory that decides what to keep based on "importance," not just "age."
- These are the informational foundation of the entire system — every subsequent decision is built on data that these provide.

**Next Lesson:** Now we reach the beating heart of the entire system — the World Model (G′). What it is, why it is so central, and a first look at three different methods tried for building it.

---

# Lesson 10: The World Model (G′) — What It Is and Why It Is So Important

## A Simple Picture

Imagine a professional chess player who, in their mind, before moving a piece, simulates several moves ahead: "If I put my knight here, how will my opponent react?" This mental simulation is a "world model" — a compressed and simplified version of the world's rules that lives in the player's mind and allows them to predict the future without actually moving the piece.

**G′** (its name derived from the mathematical symbol for "transition function") plays exactly this role in PHCA: **A compact internal model that learns "if I am in state X and perform this action, what will the next state be?"**

## Why Is This Module the Beating Heart of the Entire System?

If you recall, in Lesson 5 (Fourth Principle) we saw that the entire philosophy of this project revolves around decisions coming from prediction. G′ is what actually produces this prediction. Almost every other module in this system — from the motivational system to the safety supervisor — somehow uses G′'s output:

- The motivational system (MDIM) uses prediction error magnitude to understand "how confused am I right now?"
- The safety supervisor (RBTA) uses G′'s certainty to check the entropy floor.
- Action selection (which we will see fully in later lessons) uses G′'s prediction to score options.

If you pull G′ out of the system, almost everything else collapses — exactly like the main pillar of a building.

## Three Different Ways of Building This "Predictive Brain"

The interesting point is that PHCA does not have just one version of G′; it has implemented three different approaches (each will be seen separately in later lessons, but here is a general overview):

1. **Gaussian version:** A classical statistical model that assumes relationships roughly follow a "normal distribution" — fast and simple, but limited.
2. **Discrete graph version:** A map-like structure that connects possible states like nodes in a network — suitable for discrete worlds like a grid.
3. **Neural network version (MLP):** A small neural network that improves through gradual learning (which we saw in Lesson 6) — this is the version used in most of the project's main experiments, and we will cover it in full detail in the next lesson.

## An Architectural Question You Should Always Ask

When you see a system with three different implementations for one role, the good question is: "Are these three truly equivalent, or is one the 'real default' and the others more experimental?" This question is not unfounded — in PHCA, we saw that the MLP version is what is used in most formal experiments, while a "hybrid" version (mixing all three) is only activated in specific experimental modes. Knowing this difference is important, because when someone says "G′ works like this," you should ask "Which G′? The one actually used in the main experiments, or a side version?"

## Lesson 10 Summary
- G′ is the agent's "world model" — a compact, learned version of the world's rules that allows it to simulate the future without actually acting.
- This module is the centerpiece of the entire system — almost everything else depends on its output.
- Three different implementations exist (Gaussian, graph, neural network), but the neural network version (MLP) is the one most used in practice.

**Next Lesson:** Now that we understand what G′ is and why it is important, it is time in Lesson 11 to look under the hood of its real version (the MLP neural network) — exactly how many layers it has, how it learns, and how it measures its own "certainty."
