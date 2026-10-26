# Designing a generator of worlds — what the literature did, and why

**Status:** a design note, 26 Oct 2026. Discussion with the researcher; no experiment is designed or run yet. It follows from F19 (the data gap is the prior) and the discussion in history Entries 26–28: a library of uniform, composable blocks, trained on a generated distribution of worlds (PFN-style), with a separate store for facts about this world.

**Goal of this note.** Before designing a world generator, read how earlier generators were designed and why, and decide:
- what carries over to our setting;
- what we want to show;
- what is needed to see it.

---

## 1. Generators in the literature

### Prior-data fitted networks and TabPFN — a prior over data-generating processes
- **What** (Müller et al. 2022; Hollmann et al. 2023, 2025): each training dataset comes from one randomly sampled world. TabPFN's worlds are mostly structural causal models (a random DAG with random functions and noise), mixed with random Bayesian neural networks.
- **The generator includes:**
  - continuous targets cut into a random number of classes;
  - class labels shuffled;
  - feature and label indices rotated during training.
- **Their reasoning:**
  - "performance crucially depends on the specification of a suitable prior";
  - Occam's razor and the speed prior: graphs with few nodes and parameters are sampled more often;
  - causal structure is assumed to be common in real tabular data;
  - hyperparameters are distributions, so predictions integrate over them;
  - label and feature shuffles remove meaning from positions and identities, so the network must infer roles from the data.
- **Limits they report:**
  - bounded by the generator's support (dataset size, number of features, number of classes);
  - weaker on data types the prior covers poorly (categorical, missing values);
  - the SCM part of the prior mattered more than the BNN part.

### Learning universal predictors — a generator that is as universal as possible
- **What** (Grau-Moya et al. 2024): sequences from random programs run on a small universal Turing machine. Transformers and LSTMs are meta-trained to predict them, aiming to approximate Solomonoff induction.
- **Their reasoning:** a length-weighted program prior is the most general prior there is; a network that approximates prediction under it should transfer widely.
- **Found:**
  - larger models get closer to the ideal;
  - transfer to Markov sources and Chomsky-hierarchy tasks;
  - **transformers fail to generalize to longer sequences**;
  - the authors say the right datasets and architectures are open.

### Meta-learned in-context learners — task diversity decides what is learned
- **Kirsch et al. 2022:**
  - tasks were made by randomly projecting MNIST inputs and permuting labels, so one dataset yields unboundedly many tasks;
  - as the number of tasks grows, the model moves from **memorizing tasks**, to **identifying** which seen task it's in, to **general learning to learn** that works on unseen tasks (≥ 2^13 tasks in their setup);
  - accessible state (attention memory) mattered more than parameter count;
  - meta-training had loss plateaus, shortened by bigger meta-batches.
- **Raventós et al. 2023:** linear-regression tasks drawn from a finite pool. Below a diversity threshold, the transformer behaves like Bayesian inference over the pool and fails on new tasks. Above it, it behaves like ridge regression, the general algorithm.
- **Garg et al. 2022:**
  - function classes (linear, sparse linear, small ReLU nets, decision trees), each chosen *because it has a known optimal or standard estimator to compare with*;
  - transformers match those estimators in context;
  - they are robust to several distribution shifts, but not to skewed inputs;
  - a curriculum (small dimension and short prompts first) made training feasible.

### Compositional generators — composition as the object of study
- **MLC** (Lake & Baroni 2023): each episode samples a new **interpretation grammar**:
  - primitive words map randomly to output symbols;
  - function words get random argument structure and a random output pattern, evaluated recursively;
  - one rule (left-to-right concatenation) is fixed across episodes.

  The network sees 14 study examples and must answer queries.
  - *Reasoning:* word meanings change every episode, so the network must learn *how to infer meanings and compose them*, not the meanings themselves.
  - *Found:* human-level systematicity, and very low error on SCAN's systematic splits.
  - *But:* 100% error on SCAN's length split and on novel COGS structures (productivity). Episodes outside the meta-training distribution are not handled.
- **PCFG SET** (Hupkes et al. 2020): string-edit functions composed by a probabilistic grammar.
  - The functions: copy, reverse, shift, swap, repeat, echo; and binary append, prepend, remove-first, remove-second.
  - Grammar probabilities were tuned so sequence lengths and tree depths match a natural corpus.
  - Five tests, each built into the generator: systematicity (pairs held out together), productivity (longer compositions), substitutivity (synonymous functions), localism, and overgeneralization (pairs with an exceptional meaning).
  - *Found:* systematicity 0.53–0.72 and productivity 0.30–0.50, against 0.79–0.92 task accuracy. All models overgeneralize rare exceptions (below about 0.5%) and struggle to hold a rule and its exceptions at once.
  - The authors: models "may find unexpected solutions, even when the data is very carefully constructed."
- **CFQ and DBCA** (Keysers et al. 2020): split by **compound divergence** while keeping **atom divergence** low. Test examples use only familiar primitives (atoms) in unfamiliar combinations (compounds).
  - *Reasoning:* if atoms are shared, any drop must come from combining them.
  - *Found:* accuracy falls steeply as compound divergence rises.

### Synthetic pretraining — which structure transfers
- **LIME** (Wu et al. 2021): generated tasks of deduction, abduction and induction from rule, case and result, with fresh random symbols every example.
  - *Reasoning:* encode an inductive bias as a dataset, with content-free symbols so only reasoning, not knowledge, can be learned.
  - *Found:* about 2 GPU-hours of synthetic pretraining (≤ 2.5% of downstream compute) beat pretraining on 40× more in-domain data on theorem-proving benchmarks.
- **Edge of chaos** (Zhang et al. 2024/25): next-state prediction on elementary cellular automata.
  - Transfer peaked at **intermediate complexity** (class IV, e.g. rule 110).
  - Simple rules allow a trivial local solution; chaotic rules show no structure.
  - Models trained on complex rules attend further back.
- **Pre-pretraining on formal languages** (Hu et al. 2025):
  - **Helps:** hierarchical languages (Dyck, and especially Shuffle-Dyck), saving about 33% of natural-language tokens at 1B parameters.
  - **Doesn't help:** copy languages.
  - **Hurts:** random strings.
  - Each language has an **optimal duration**; more hurts.
  - *Reasoning:* the data should have dependencies like the target domain's *and* be learnable by the architecture (C-RASP).

### Automata and groups — what gradient descent actually finds
- **Shortcuts to automata** (Liu et al. 2023):
  - shallow transformers learn **shortcut** solutions (logarithmic or constant depth) instead of step-by-step simulation;
  - over 99% accuracy in distribution;
  - brittle under distribution shift and on longer sequences;
  - non-abelian groups needed more depth, and training was unstable;
  - scratchpad training pushed models to recurrent solutions that generalize in length.

---

## 2. Design principles that recur

| | principle | where it comes from | why |
|---|---|---|---|
| G1 | **Randomize surface per world:** symbols, labels, positions | MLC, LIME, TabPFN, Kirsch | forbids memorizing content, so roles must be inferred from the examples in context |
| G2 | **Sample worlds from a simplicity-weighted structured space** | TabPFN (Occam, speed prior), universal predictors (length prior) | in-prior worlds stay learnable; matches the tendency of real data to be simple |
| G3 | **Aim for a complexity sweet spot:** no worlds solvable by a local lookup, no pure noise | edge of chaos; Hu et al. (random hurts, copy useless) | trivial worlds teach shortcuts; noise teaches nothing |
| G4 | **Pair the generator with a known optimal predictor** | Garg, Raventós, universal predictors (CTW) | says what success is, and how far from it the learner is |
| G5 | **Sweep the number of distinct worlds** | Kirsch, Raventós | the memorize → identify → generalize transition is set by task diversity |
| G6 | **Evaluate on held-out compositions with matched atoms, and on longer compositions** | DBCA/CFQ, PCFG SET, MLC | the only honest tests of composition. Productivity (length) is the near-universal failure |
| G7 | **Build exceptions in at controlled rates** | PCFG SET overgeneralization | rule-versus-memory interaction becomes measurable |
| G8 | **Expect shortcuts; test for them out of distribution** | Liu et al., Hupkes et al. | gradient descent finds whatever works in distribution |
| G9 | **Curriculum and meta-batch size matter for meta-training** | Garg, Kirsch | meta-training has its own plateaus |
| G10 | **Mix synthetic with real data, in limited amounts** | Hu et al., LIME | short synthetic pretraining transfers; too much hurts |

---

## 3. What carries over to our setting

- **G1, directly.** Each world gets a random bijection between tokens and residues (and, for groups, elements). This turns our earlier caveat into the point: one-hot symbols carry no geometry, so the meaning of every token must be inferred from the context examples.
- **G2 and G4 together, and we already have both halves.** `program_prior.py` enumerates short arithmetic programs and computes the exact Bayes predictor (E1, k\* = 3). Sample worlds from the same length-weighted program space, and the ideal learner for the generator is computable *exactly*. Few synthetic setups have that.
- **G3:**
  - drop worlds a single lookup solves (e.g. constant functions);
  - prefer programs whose output depends on both inputs through composition;
  - avoid random tables as worlds, except as a memory test.
- **G5 is the meta-level of our data floor.** The number of distinct training worlds plays the role that training pairs played in E1 and Q1. The Q1 protocol (keep the solution while shrinking the training pool) can be asked one level up.
- **G6 and G8:** held-out operation pairs with matched primitive frequencies (DBCA), and longer programs (productivity). Liu et al. predict shortcuts and length failure in shallow transformers. Routed, weight-tied composition with halting is our candidate for the recurrent-like solution that generalizes in length.
- **G7:** rule-plus-exceptions worlds (I5, `modadd_exc`) with exception rates around PCFG SET's critical 0.5%, for the memory-store question.
- **Scale constraint:** in-context worlds must fit a context. With small moduli (p ≈ 5–17) and contexts of ≤ 64 examples, an in-prior world needs only a handful of examples under the ideal learner.

---

## 4. What we want to show, and what is needed to see it

| claim | how it could fail | needed |
|---|---|---|
| **C1: past a world-diversity threshold, a learner trained on many worlds needs near-ideal numbers of examples on a new in-prior world** (vs ~870 for single-task gradient descent) | stays in the memorize or identify phase; or reaches the threshold only beyond our compute | G1, G2, G4 (exact Bayes reference), G5 (sweep of distinct worlds), the E1 scoreboard (predict-then-learn code length) on held-out worlds |
| **C2: a library of uniform routed blocks reaches the general phase with fewer distinct worlds, and generalizes better to held-out compositions and longer programs, than a monolithic network of equal size** | monolithic equals or beats modular (then modules aren't needed); or both fail on length | compositional generator with decorrelated primitives; DBCA splits; a depth sweep; a monolithic baseline at matched parameters and compute (I23) |
| **C3: blocks become reusable primitives** (the same block serves the same operation across contexts) | routes don't align with primitives; ablating a block breaks unrelated things | ground-truth decomposition per example (a synthetic advantage); routing statistics against primitive labels; block ablations |
| **C4: facts specific to a world go to a memory store without bending the rule** | the memory becomes a universal sink, or rules overgeneralize exceptions (PCFG SET) | rule-plus-exception worlds at controlled rates; a memory block with a per-entry cost (the two-part description length) |
| **C5 (optional): is the world-diversity threshold prior or search?** (Q1's protocol one level up) | — | train past the threshold, then shrink the world pool, and compare with training from scratch on the smaller pool |

**Honesty check (L12):** C1 on worlds *inside* the generator is expected to succeed. MLC, Garg and Kirsch all show it, given enough worlds. So C1's informative parts are:
- where the threshold sits;
- the distance from the exact Bayes reference.

The tests that can fail in interesting ways are C2 (modular against monolithic, compositions held out, length) and C4 (memory).

---

## 5. Open decisions for the researcher

1. **The primitives.** Hand-chosen arithmetic and group operations (analysable, but built in), or programs sampled from a minimal universal language (closer to the bitter lesson, harder to interpret)?
2. **How compositional the worlds are.** C2 needs explicitly compositional worlds. A broader, TabPFN-like family is a different study.
3. **How the real world enters.** Grounding and memory (C4) need at least one "world" with persistent facts. In our toy, that could be a fixed world with exceptions that recurs across training.
4. **The architecture's state.** Kirsch's result says the block must read the context, so it needs attention.

**Sources** are linked inline in the chat discussion of 26 Oct 2026 and in history Entry 29.
