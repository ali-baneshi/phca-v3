# PHCA Course — Lessons 11 to 20

*(Reminder: Lessons 1 to 10 are in the two previous files. Here we continue from where Lesson 10 ended — we open the hood of the world model, then enter the motivational and adjustment system, and finally reach long-term memory.)*

---

# Lesson 11: Under the Hood of G′ — The Real Neural Network

## A Simple Picture

In the previous lesson, we said G′ is like a "mental simulator." Now it is time to see what this simulator is made of. Imagine a student who wants to learn "if I throw the ball at this angle, where will it land?" — they learn this not through precise physics formulas, but through **repeated trial and gradual adjustment of their guess**. A neural network works exactly the same way: a very flexible mathematical function that gradually adjusts by seeing many examples.

## Precise Architecture

The main version of G′ in PHCA is a relatively small and simple network: **three layers** — one input layer (which takes the current state + the chosen action), two hidden layers of 128 neurons each, and one output layer (which predicts the next state). The entire network has about 38,000 parameters (adjustable numbers) — a very small number compared to today's large neural networks (which have billions of parameters), but sufficient for this agent's small world.

## Two Smart Design Points

**1. How does it measure "certainty"?** In Lesson 4, we saw that certainty measurement is done with the MC-Dropout technique. There is a subtler point here: the designers intentionally decided **not to use direct comparison of the prediction with the current state for measuring certainty.** Why? Because if they did, the model could learn a lazy trick: "always say nothing changes" — since this guess is usually close to the current state and would get a "high certainty score" without actually having learned anything. This is exactly an example of "reward system cheating" that we discussed in Lesson 1 — and interestingly, right here in the heart of a small module, the same concern appears again and has been thoughtfully prevented.

**2. Two learning modes.** The model has two ways to learn: at the beginning ("warm-up") it learns one by one from each experience. After a while, it enters a "stable" mode and instead of immediate learning, it reviews small batches of past memories (stored in episodic memory — Lesson 19) and learns from them. The latter is similar to what happens in our sleep: the brain "reviews" the day's memories to consolidate them.

## An Important Architectural Point Discovered in Our Review

When we carefully examined this part, a very important point was found that later became key: **This model's "certainty" measure only measures "is the model self-consistent," not "is the model reliable for *multi-step* tasks (e.g., navigating a long path)."** A model can be very accurate and "self-consistent" in predicting "the next single step," but when you chain these small predictions together to form a long path, small errors accumulate and the final result can be completely wrong. This is exactly what we saw in Lesson 5 (when we discussed the model-based decision-making disaster) — and its root lies precisely here, in the design of the certainty measure.

## Lesson 11 Summary
- G′ is a relatively small neural network (three layers, 38k parameters).
- Its design intentionally avoids a common trap (learning "nothing changes" to get a false certainty score) — a good example of smart design.
- But its certainty measure is only "one-step," not "multi-step" — and this limitation later had major consequences.

**Next Lesson:** The Prediction Error Unit (PEU) — the thing that measures the difference between guess and reality and powers the entire learning engine.

---

# Lesson 12: Prediction Error Unit (PEU)

## A Simple Picture

Imagine a marksman who, before each shot, guesses "where will the arrow hit?" and after shooting, measures the distance between the guess and the actual impact point. This "distance" is exactly what PEU computes — not for archery, but for every prediction the agent makes.

## What Does PEU Actually Do?

Its job is relatively simple and concise: it compares G′'s prediction (which was produced in step 3 of the cycle) with the *actual* state of the world after the action is executed and produces an "error" number. This error is weighted — meaning some dimensions are given more importance (the name is "precision-weighted error"; the idea is that if a dimension of the state is usually very variable, a small mistake in it should be given less importance than when a usually stable dimension unexpectedly changes).

## Why Is This Number So Important?

This "error" number is exactly what spreads throughout the system and has effects:

- It goes to **TSPL** (next lesson) to guide learning.
- It goes to **MDIM** (Lesson 14) so the motivational system understands "how confused am I right now" — because one of the six intrinsic drives is exactly this "prediction error reduction."
- It is even carried into final reports and evaluation metrics (which we will see in later lessons, Part 7 of the course).

It can be said that PEU is a "main vein" in this architecture — small and simple, but everything feeds from it.

## An Architectural Point: Why Error Computation Must Be *After* Action Execution

We mentioned this in Lesson 3 (Temporal Causality), but let's go deeper here: you cannot compute the error earlier, because error requires the "final reality," and the final reality is only known after the world has reacted to the agent's action. This means PEU always works "one step behind" G′ — G′ guesses, the world reacts, and then PEU compares. This mandatory sequence is itself one of the guarantees of temporal causality.

## Lesson 12 Summary
- PEU measures the distance between "what was predicted" and "what actually happened."
- This number is broadly distributed throughout the system — from learning to motivation to final reports.
- Due to temporal causality, this computation must always be done *after* the actual execution of the action.

**Next Lesson:** TSPL — where this error is actually turned into an adjustment, with an interesting story about "what exactly this adjustment affects."

---

# Lesson 13: Learning from Error (TSPL)

## A Simple Picture

Imagine a tailor sewing a garment based on a pre-made pattern. Every time a measurement turns out slightly wrong, the tailor makes a "small correction" to the pattern — not rebuilding the entire pattern, just a tiny adjustment. TSPL is exactly this "fine adjuster."

## An Important Point to Keep in Mind from Now On

The name TSPL is somewhat misleading — you hear "learning" and think this is what builds the entire model. But when we carefully examined this part, an interesting fact emerged: **TSPL does not work on the main weights of the neural network.** Those main weights (the same 38k parameters we saw in Lesson 11) are learned by G′ itself, directly and independently, using the standard gradient descent method.

TSPL maintains something else: a **small auxiliary vector** (sized to the dimensions of the state, not the size of the entire network) that acts like a "final correction" added to the model's output. This vector is adjusted with a simpler rule (not the full network gradient descent, but a simpler "delta" rule), and also has a constraint: if this correction becomes too large (exceeds a threshold), it is automatically zeroed out — a kind of "safety valve" that prevents this small correction from gaining excessive power.

## Why Does This Separation Exist? (And Its Architectural Lesson)

This separation is logical in itself — similar to having a large system (the main network learning) and a smaller, faster system (TSPL) working together, each at its own speed. But the architectural point you should learn is: **When a module has a big, meaningful name (like "Learning"), always ask how much of the actual work this name truly represents.** Here, the majority of "intelligence" is stored in the main network weights, not in TSPL — TSPL is more of a fine-grained side layer, even though its name suggests it is the heart of learning.

## Connection to "Protection of Old Knowledge" (More in Later Lessons)

An additional role of TSPL is: when the agent moves from one "task" to another (e.g., the goal changes), TSPL can mark part of this small vector as "protected" so it does not change too quickly in subsequent adjustments — a simple technique to help prevent sudden forgetting. This works together with a larger mechanism (episodic memory, which we will see in Lesson 19) to reduce forgetting.

## Lesson 13 Summary
- TSPL does not work on the main network weights; it only adjusts a small auxiliary vector with a simple rule.
- The main, heavy learning happens elsewhere (the G′ network itself).
- The general lesson: always compare a module's name with its actual work, not just with what the name suggests.

**Next Lesson:** Now we enter the motivational section — the six-drive MDIM system, perhaps the most eventful module of this entire project.

---

# Lesson 14: The Six-Drive Motivational System (MDIM)

## A Simple Picture

Imagine a person who at any moment has several simultaneous "needs" — hungry, curious, tired, wanting to see something new. Their brain constantly weighs these needs against each other and decides "which one is most urgent right now?" MDIM (Multi-Drive Intrinsic Motivation) does the same thing for the PHCA agent.

## What Are the Six "Needs"?

Each of these six drives represents an aspect of "what matters to this agent" — things like: reduction of prediction error (i.e., "I want to understand the world better"), exploration of new states (i.e., "I want to see something new"), energy conservation, and several others. Each drive has a "deficit" number indicating "how unsatisfied this need is right now" — exactly like hunger that increases over time.

Each cycle, these six deficit numbers are converted into probabilities using a mathematical method called **softmax** (the larger the deficit, the higher the chance of being selected), and one is chosen as the "current goal" — with some randomness, not always exactly the highest.

## The Eventful Story of This Module

This module is one of the most eventful parts of the entire project history, and let me narrate it step by step because it has many architectural lessons:

**Chapter One:** In the early version, when the agent had an external goal (e.g., a specific point on the grid it needed to reach), this beautiful six-drive system was completely bypassed and replaced by a fixed rule: "always select only the error-reduction drive." Since almost all test scenarios have a specific goal, this meant **in practice, the six-drive system almost never actually worked** — it was only activated in very specific, goal-free situations.

**Chapter Two:** This was discovered and that fixed rule was removed — now all six drives always genuinely compete, regardless of whether an external goal exists or not.

## Why This Story Matters for General Understanding of the Architecture

This is exactly the same pattern we saw in Lesson 5 (Fourth Principle), just in a different module: **A complex and beautiful system designed on paper, but a small condition in the code bypassed it in most real situations.** This repetition of the pattern tells us something important: this type of problem was not accidental, but rather a **common pattern in the development of complex systems** — when multiple teams or stages work on a project, temporary solutions (written to solve a small problem) can silently replace the main architecture, unless someone regularly and carefully checks.

## Lesson 14 Summary
- MDIM has six internal "needs" that compete each cycle to determine which is the "goal of the moment."
- In this project's history, a fixed rule bypassed this competition in most situations — which was later discovered and fixed.
- This story shows how seemingly insignificant small conditions can neutralize the entire philosophy of a system in practice.

**Next Lesson:** APC — the controller that automatically adjusts system parameters, like a thermostat that keeps the room temperature constant.

---

# Lesson 15: Adaptive Parameter Controller (APC)

## A Simple Picture

Imagine you have a thermostat at home. If the temperature drops too low, it turns on the heater; if it gets too high, it turns it off. This is a "controller" — something that constantly measures the situation and adjusts a parameter (here: heater on/off) to keep the system balanced.

**APC** (Adaptive Parameter Control) plays this role for PHCA, but instead of temperature, what it controls is **prediction error instability**.

## How Does It Work Exactly?

APC uses a classic and very well-known technique in control engineering called **PID** (which is used everywhere in industry, from furnace temperature control to autonomous vehicle control). The general idea is: the controller looks at three things — "how far am I from the ideal state right now?" (P), "how far have I cumulatively been over time?" (I), and "how fast am I moving toward or away from the ideal?" (D). By combining these three, a smooth adjustment without severe oscillation is achieved.

APC uses this to adjust parameters such as learning rate or decision thresholds — if prediction error is very unstable and oscillatory (meaning the environment is changing rapidly or the model is getting confused), APC can adjust parameters so the system behaves more cautiously.

## Why Is This Layer Necessary?

Without such a controller, the system would have to set its parameters with fixed (hard-coded) numbers — which works well for a static environment, but when the environment changes (e.g., the goal moves or noise levels increase), those fixed numbers are no longer optimal. APC allows the system to adapt itself to changing conditions without anyone having to manually change parameters.

## An Architectural Point

This module is a good example of a general system design principle: **Instead of setting each parameter as a fixed constant, build a "meta" layer that adjusts parameters based on the current situation.** This makes the system more flexible, but has a cost: you must now ensure this meta-layer itself works correctly — because if APC is misconfigured, instead of making the system more stable, it may create instability (a well-known risk in control engineering called "oscillation due to improper control").

## Lesson 15 Summary
- APC acts like a thermostat — it adjusts system parameters based on prediction error instability.
- It uses the classic PID technique, the same one used in many industrial control systems.
- Its benefit is flexibility, but it itself must be carefully designed and tested so it does not become a source of instability.

**Next Lesson:** Attention — the module that decides "which part of the information is more important right now," and the same place where we saw the "disconnected wire" story in Lesson 6 — this time with full details.

---

# Lesson 16: Attention

## A Simple Picture

When you are talking to someone in a noisy, crowded room, your brain automatically "highlights" their voice and makes the background noise fainter. This ability is called "attention" — focusing on the part of the information that is more important, and relatively ignoring the rest.

## How Does Attention Work in PHCA?

This module produces an "importance weight" for each dimension of the state information — it says "how important is this dimension right now?" This computation is **sparse**, meaning instead of giving equal importance to everything, it tries to focus attention on a few important dimensions, rather than spreading attention uniformly.

An interesting technical detail: in computing these weights, a specific type of noise called **Gumbel noise** is used. Without diving into the mathematics, its purpose is: to allow the system to sometimes introduce some randomness instead of always selecting exactly the same part — this prevents a type of "early locking onto one choice."

## The "Disconnected Wire" Story — This Time with Full Details

In Lesson 6, we alluded to this, but now that we have seen both Attention and G′ up close, let me complete it. The chain was:

1. Attention computes the importance weights.
2. These weights are written onto a variable in the G′ model.
3. When G′ wants to learn (i.e., adjust its network weights), it *should* use these weights to more strongly correct errors related to "more important" dimensions.

The problem was that step 3 did not happen. The weights were written (step 2), but nowhere in the actual learning computation were they read. This means from the outside, everything "appeared" to work correctly — numbers were computed, nothing threw an error — but in terms of actual effect, this weight computation was **completely useless**.

## Why Is This Type of Bug So Hard to Find?

Because none of the usual tests catch it — the code runs, output is produced, no error occurs. The only way to find it is to ask "Is this variable actually *read* anywhere?" — a type of check that goes beyond the usual testing of "does the code work"; you must ask "Does this code have the effect it claims to have?" This is exactly the method that found this bug: tracing data from source to destination, not just checking that the code runs without error.

Fortunately, this bug is now fixed — the attention weights are actually multiplied into G′'s learning gradient computation and have a real effect.

## Lesson 16 Summary
- Attention decides which dimension of information is more important right now, and this importance should affect learning.
- A real example of a "disconnected wire" — computation happened but had no effect on learning — was discovered and fixed in this section.
- The general lesson: to find this type of bug, you must trace data from its source to its final destination, not just check that the code runs without error.

**Next Lesson:** HPM — Hierarchical Procedural Memory, the module that builds the tree structure of the entire cycle for resource budget computation.

---

# Lesson 17: Hierarchical Procedural Memory (HPM)

## A Simple Picture

Imagine you want to estimate the total time for a construction project. Some tasks must be done **sequentially** (first foundation, then walls, then roof — you cannot build the roof before the walls), and some tasks can be done **in parallel** (interior painting and exterior gardening can proceed simultaneously). The total project time depends on which tasks are sequential and which are parallel.

**HPM** (Hierarchical Procedural Memory) does exactly this for the cognitive cycle: it builds a tree structure that shows which stages of the cycle (which we saw in Lesson 7) execute sequentially and which in parallel, and from this structure, it computes the **total time and energy of the cycle.**

## Two Types of Composition

- **SEQUENCE:** total time = sum of all parts' times (plus a small coordination overhead).
- **PARALLEL:** total time = maximum time among parallel parts (not the sum, because they execute concurrently) — but the interesting point is that **energy always adds up**, even in parallel mode, because both paths consume energy simultaneously, even if their times overlap.

This distinction (time: maximum in parallel mode / energy: always summed) is a subtle but correct mathematical point — and when we examined it, we saw it was correctly implemented according to a formal theorem documented in the project's documentation.

## What Is This Computation Used For?

HPM's output is directly given to RBTA (the resource supervisor, Lessons 2 and 18) to determine whether the entire cycle, given its sequential/parallel composition, exceeds the time and energy budget or not.

## An Architectural Point: When a Structure Is Replicated in Multiple Places

When we examined this part, an interesting point emerged: in an older version of the code, this tree structure was **manually and separately rewritten in several different places in the code** — one version for a lighter cycle mode, and another for the final check. This meant that if someone wanted to change this structure, they would have to change each version separately and in sync — a classic software engineering risk (when a logic is copied in multiple places, sooner or later one version falls behind the others). Later, these duplicate versions were consolidated into a single function — a simple but important improvement for code maintainability.

## Lesson 17 Summary
- HPM builds a tree structure of the sequential/parallel stages of the cycle and computes total time and energy from it.
- Its rule: time in parallel mode is "max," but energy is always "summed" — a subtle and physically correct distinction.
- The history of this module is a good example of the danger of "copying logic in multiple places" and the importance of consolidating it into a single source.

**Next Lesson:** We return to RBTA — this time with full details, including an interesting story about how an attempt to make the safety supervisor "more precise" temporarily made it more blind.

---

# Lesson 18: Resource Supervisor (RBTA) — Deep Dive

## Quick Reminder

In Lesson 2, we were introduced to RBTA: the supervisor that checks every cycle whether anyone has exceeded their time/memory/energy/entropy budget, and if violated, changes the actual behavior of the next cycle (INTERRUPT or TERMINATE). Now that we have seen how HPM computes time and energy, it is time to see how RBTA itself decides "how serious is this violation?"

## The Interesting Story of "Violation Severity"

The simple question this module must answer: if multiple violations happen simultaneously, how do we decide how severe the system's reaction should be?

**First version (simple):** Just count the number of violations. Zero violations = continue. One or two = be cautious. Three or more = completely stop and go to safe mode.

This is simple, but has a logical flaw: a very, very minor violation (e.g., one millisecond over the limit) has exactly the same weight as a catastrophic violation (e.g., a thousand times over the limit) — because only "count" matters, not "magnitude."

**Second version (attempt at precision):** Instead of simple counting, compute a "severity" (between zero and one) for each violation, based on how far over the limit it went, and sum these severities.

This seems logically better — but right here, a subtle bug entered the system: the severity formula assumed that a violation always means "the measured number exceeded the *upper* limit" (like time or memory). But entropy violation is the opposite — it means the number has crossed a *lower* limit (became too small, not too large). Because the formula did not account for this opposite direction, **every entropy violation always got a severity of zero** — meaning from the supervisor's perspective, nothing important ever happened, even if the agent's uncertainty had completely collapsed.

**Third version (fix):** They changed the formula to account for the violation direction (going above or going below), and also, instead of relying entirely on "severity," returned to the simpler counting method, with one added rule: "if even a single violation is extremely severe, immediately TERMINATE, even if the count is low." This combination — the reliable simplicity of counting, plus an exception for truly catastrophic cases — was the version that ultimately remained.

## The Architectural Lesson of This Story

This is one of the best examples in the entire course of an important principle: **When you want to make a simple system "more precise," you must ensure the new version still covers all the cases the old version covered.** Here, the attempt to be more precise (from simple counting to weighted severity) unintentionally completely ignored one type of violation. The solution was not to give up on being more precise, but to write a precise test that directly checks "when *only* an entropy violation occurs, does the system actually respond correctly?" — not just "is the violation object created?"

## Lesson 18 Summary
- RBTA must decide how serious violations are — a simpler approach (counting) versus a more precise one (severity).
- An attempt to be more precise unintentionally rendered one type of violation (entropy, whose direction is opposite to others) completely ineffective.
- The final version is a combination of reliable simplicity and a smart exception for truly severe cases.

**Next Lesson:** Episodic Memory (M3) — where the agent's long-term memories are stored, and a story about how a simple "delete oldest" policy could destroy exactly what needed to be preserved.

---

# Lesson 19: Episodic Memory (M3)

## A Simple Picture

Remember M1 and M2 (Lesson 9) — the work table and a slightly smarter memory. Now it is time to go to the main "warehouse": **M3**, memory that stores complete experiences (not just the last few) for a long time — on disk, in a real database (SQLite), not just in temporary memory.

## Why Is This Memory Necessary?

If you recall, in Lesson 11 we said that in "stable mode," instead of learning immediately from each experience, the prediction model reviews small batches of past memories. Where do these memories come from? Exactly from M3. This technique is called "replay" — a very common and proven technique in machine learning that helps the model not forget things it learned long ago, because it periodically "reviews" them.

It also has another smart feature: instead of replaying memories completely randomly, it uses a technique called **Prioritized Experience Replay** — memories that still have high prediction error (meaning the model has not yet learned from them well) have a higher chance of being replayed. This is logical: why spend time reviewing something you already know well?

## The Story of "Who Should Be Forgotten?"

Every memory has a capacity limit — M3 is no different (e.g., maximum 10,000 memories). When this limit is reached, something must be removed to make room for new memories. The question is: **Which memory should be removed?**

The initial simple solution was: "delete the oldest" — seems logical, right? But it has a subtle problem: in an agent that is supposed to **sequentially learn multiple different tasks** (e.g., first task 1, then task 2, through task 10), "the oldest memories" are exactly the memories of the **first tasks**. This means the seemingly neutral "delete the oldest" policy systematically and specifically destroys the memories of the early tasks — exactly what the "replay to prevent forgetting" mechanism was supposed to preserve!

This was a subtle architectural paradox: M3's goal was "preventing forgetting," but its deletion policy could wipe out exactly what needed to be preserved, fastest of all — provided the number of memories exceeded capacity.

## The Solution: Fair Quota for Each Task

The solution ultimately adopted was: instead of globally deleting "the oldest," each task gets a **specific quota** of the total capacity. If a task has more memories than its quota, only memories from that same task (and only the oldest of that task) are deleted; other tasks remain untouched. This guarantees that no task is completely "starved."

## Lesson 19 Summary
- M3 is long-term memory, stored on disk, and is the foundation of the "replay to prevent forgetting" mechanism.
- The simple "delete oldest" policy can violate the main goal of this memory (preserving memories of all tasks).
- The solution: each task gets a fair quota of capacity, rather than a global competition based on age.

**Next Lesson:** The last lesson of this section — Memory Consolidation: how raw experiences are turned into "more general knowledge," similar to what happens in our sleep.

---

# Lesson 20: Memory Consolidation

## A Simple Picture

They say that when we sleep, our brain "reviews and categorizes" the day's memories — turning scattered, detailed experiences into "more general rules." For example, if today you touched a hot stove three times, your brain converts these into a general rule: "hot stove hurts, be careful." This process is called **consolidation**.

Consolidation in PHCA has the same role: every few cycles, it examines the raw, detailed memories stored in M3 (previous lesson) and tries to extract **semantic facts** from them — more general and compact rules.

## How Does It Determine Which Memories Should Be Combined?

It uses a simple and common mathematical technique called **cosine similarity** — a method for measuring "how similar two experiences are in pattern" regardless of their absolute magnitude. If several different memories have a similar pattern, they are combined into a more "general fact."

## What Are These Extracted Facts Used For?

They are fed back to the motivational system (MDIM) to form part of the context on which decisions are based — meaning the agent uses not only the experience of *right now*, but also a compact summary of *all* past experiences for making decisions.

## Why Is This Separate Layer Necessary? (Its Difference from Normal Learning)

You might ask: doesn't the G′ model (which we saw in Lesson 11) already learn by reviewing memories? Why is a separate "consolidation" layer needed?

The difference is: G′'s learning affects the **neural network weights** — a type of compressed, non-directly readable knowledge (no one can look at a neural network's weights and understand "what does this mean"). Consolidation builds something else: **explicit, retrievable facts** — things that can be directly queried: "what facts do you know about this situation?" This is similar to the difference between "unconscious skill of riding a bicycle" (which resides in muscles and reflexes) and "conscious knowledge that you can express in words" — both types of knowledge are useful, but they are of a different nature.

## Lesson 20 Summary
- Consolidation transforms raw M3 memories into semantic, compact facts — similar to what happens in our brain during sleep.
- It uses pattern similarity (cosine similarity) to determine which memories should be combined.
- This type of "explicit knowledge" differs from the "implicit knowledge" stored in neural network weights, and each plays a different role in decision-making.

---

## Brief Summary of Lessons 11 to 20

With this lesson, we have completed Part 3 (The Predictive Brain), Part 4 (Motivation and Adjustment), and Part 5 (Safety and Long-Term Memory) of the course. You now have a complete picture of the **internal components** of the building — from the input sensor to long-term memory, from the prediction model to the motivational system, from the safety supervisor to memory consolidation. The only thing we have not yet fully seen is: **with all these components, how exactly does this agent decide which action to take?** This is precisely the topic of Part 6 of the course — Lessons 21 and 22 — where we put all these bricks together to see how the final moment of decision-making actually happens, and there we will finally arrive, in full detail, at the same eventful story we alluded to in Lessons 5 and 14.
