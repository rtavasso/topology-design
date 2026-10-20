# Foundational questions, retained in the researcher's words

> **Note (19 Oct 2026):** This file is unchanged from v0 apart from this note. Its references to `docs/03…`, `docs/04…` etc. now live in [`archive/v0/docs/`](../archive/v0/docs/). The current plan is in [SPEC.md](SPEC.md). Several questions here now have sharper formal homes:
> - §5 "hidden learning" → hidden-progress measures (Phase 2);
> - §6 "productive frustration" → saddle/plateau dynamics (SPEC §2);
> - §7 "memorization vs generalization" → gradient coherence and localization (Phase 3).

This document preserves selected **verbatim question fragments** from the originating discussion (October 2026). They are not commitments that the initial framing was correct: the rest of the documentation contains objections, derivations, and pivots. Their purpose is to preserve what the researcher was *trying to understand* before our working language became more formal.

## 1. The original conceptual puzzle: what does routing allow numerically?

> “are there reverse residual connections in deep nns? that flow from later layers backwards?”

> “i am thinking of residual connections as letting neurons learn gating functions rather than discriminatory functions. like it adds a dof to what neuron can compute.”

> “what can a residual conneciton neuron not stably express or is hard to learn to express reliably? what behavior?”

**Resulting clarification:** routing changes how easy it is for a network to preserve or modify a representation without necessarily expanding scalar linear expressivity. Some finite-width nonlinear function classes do change. An exact linear residual often changes initialization bias and path-degree dependence, not parameter dimension. See docs/03_first_principles_math.md.

**What must not be lost:** the intention was to reason **from the numerical expression of a neuron**, not to blindly transplant a fashionable architecture.

## 2. Experimental cleanliness must remain the default

> “To make this experimentally clean on the topological effects of information routing (like residual connections), it's important we don't introduce new parameters as a gate.”

> “start from first principles of the actual numeric expression of a neuron, using a residual connection as an example of what topology can change, and think about the expressability of different numeric behavior directly as a result of topology.”

> “avoid recurrence in our topologies because that conflates extra inference compute with topological advantage. This is purely a question of information flow, and numeric expression.”

**Resulting constraints:** same trained operations and activations, parameter budget, softmax CE, data/optimizer and single-pass feed-forward execution; vary parameter-free edges. Use learned gates, recurrent loops, branch losses or normalization changes only as **labeled diagnostic controls**, never silently.

## 3. The enduring core question about paths

> “That final question is exactly the framing I want you to keep in mind as we are exploring this problem.”

The referenced question was:

> “Can we construct a parameter-free routing topology that simultaneously offers short paths for preserving information, short paths for destroying information, and short paths for constructing new information—without forcing those operations to compete through the same learned weights?”

This question evolved rather than disappeared. The term “independently controllable” was challenged because **productive cooperation and friction between paths may matter more than path separation**.

## 4. Landscape and efficiency of single-example learning

> “the gradient landscape and how well a single example can teach a model.”

> “What requires LLMs seeing trillions of tokens compared to humans with higher learning efficiency? Is it a gradient landscape question, a gradient update/circuit formation question, or something else?”

**What survived:** investigate parameter-space conditioning, local gradient access, which other inputs one example affects, circuit-formation dynamics and information interference as distinct questions. Residuals do not generally convexify the loss. Human one-shot learning is not the same as one SGD update and has many other explanations (priors, grounding, memory, active learning, in-context adaptation).

## 5. Hidden learning and softmax CE

> “training on an example will lower the models loss on predicting that token in the future, but there is some other effect in the model that masks this learning such that the model doesn't ever predict what it was trained on.”

> “for this discussion, let's operate under the constraint that we optimize with softmax CE loss.”

> “can we disentangle the loss such that the short path can learn without muddying the signal to the long path? The different parts of the network are in tension with each other during the learning phase”

**Correction:** lower CE for identical $(x,y)$ means greater probability for $y$, not necessarily top-1. Subsequent updates, other contexts and competing logits can hide apparent acquisition/retention/transfer. Additive branches share $p-e_y$ and hence can starve one another. Exact token grouping factors the loss, but was judged a problematic imposed division of labor; see docs/04_gradient_dynamics.md.

## 6. Productive machine 'frustration' as a research intuition

> “The machine analog of frustration is something I am considering as a useful signal not to reason away.”

> “frustration can perhaps be thought of as short-long-path friction but long path learning. Independence between these paths isn't what we are truly after.”

> “promoting the stability of short term friction for long-path learning is something we need to address.”

> “I would offer some pushback on the idea if it requires any a priori groupings.”

**Resulting formulation:** useful unresolved error lies outside the shallow computation's current effective directions but can drive new generalizable computation. Do **not** maximize branch disagreement or impose semantic token groups. Disagreement can also be noise, cancellation or interference. Measure whether long computations learn anything useful while simple features are already strong.

## 7. Finding mechanisms, not leaderboard outcomes

> “The real framing for these experiments isn't about CE loss, it is about characterizing the topologies that affect memorization and generalization.”

> “should inform topology design such that we could control the structures or allocate parameters or training to different types of learning rather than the current approach of mixing everything together.”

> “we really need a way of measuring long path learning and utility.”

> “If we probe for structure we expect, like a 2 layer circuit, we are blind to what we don't know to measure”

**Resulting methodology:** discover unanticipated input-output functions before proposing a mechanism, then causally intervene on trained graph pathways, track temporal changes, and separately measure expression versus acquisition. Do not call any single route semantic memorization/generalization before discovering its actual contribution.

## 8. Scale and polysemantic representations

> “models and neurons can appear polysemantic when the number of parameters is insufficient to allow for each neuron to be monosemantic.”

> “This can make the interpretability of networks scale dependent such that there is some covariant variable alongside scale that we might be blind to.”

> “You can see why experimental design is so critical.”

**Resulting plan:** width is not the only axis; scale feature demand, importance, correlations and sparsity separately. Interpretability itself has resolution and coordinate-dependence confounds. Validate measurement on planted features, exact basis changes and different SAE dictionary sizes.

## 9. Why this repo exists

> “I need to preserve the conversation and insights we have had over the course of the conversation, not just where we have ended up.”

> “I want to preserve not just what we have decided, but why, and also what we have pivoted away from.”

> “these documents will form the nascent structure for the coding agent to begin creating experiments, and this agent will have only those documents as context”

This is why the repo contains an intellectual history, explicit decisions and rejected paths, worked mathematics, measurement definitions, calibration tests, a literature map, and an agent implementation contract—**not merely a final architecture proposal**.

## One-sentence research mandate

**Study how parameter-free DAG information routing changes the creation, expression, coexistence and causal utility of computations supporting memorization and generalization, by using first-principles math, source-agnostic mechanism discovery, controlled training histories and scaling-aware interventions—before optimizing architectures.**
