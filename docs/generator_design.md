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

## 5. Open decisions, their options and implications (26 Oct 2026)

### D-1. Primitives: what worlds are built from

| option | exact Bayes reference | ground-truth decomposition (C3) | evidence | fit to the bitter lesson |
|---|---|---|---|---|
| **A. Hand-chosen operations:** modular +, −, ×, negation, constants; group products (abelian Z_p; non-abelian such as dihedral or S_5) | yes, by enumeration (`program_prior.py`: 155 functions at size ≤ 5, 1,015 at size 7) | yes | groups: shallow transformers learn shortcuts; non-abelian ones need depth and train unstably (Liu et al.) | weakest |
| **B. A small generic language:** composition, conditionals, a bounded iteration, over small integers | yes, for small program sizes | yes | PCFG SET: models "find unexpected solutions" | middle |
| **C. A minimal universal language** (a tiny Turing machine, combinators) | no; only approximate baselines | weak: program steps aren't reusable operations | transfer, but transformers fail on length (Grau-Moya et al.) | strongest |

**Grounding fact that matters most here:** composition in a *non-solvable* group (S_5, A_5) has no constant-depth shortcut unless TC0 = NC1 (Liu et al.). Abelian composition does have one. So products of S_5 elements *force* genuinely serial computation, and abelian sums don't. That's the cleanest way to make "serial composition is necessary" true by construction, which is what C2 tests.

### D-2. How compositional the worlds are

| option | what it can test | risk |
|---|---|---|
| **A. Explicitly compositional:** each world is a program of depth 1–D over the primitives, with pairs and orders decorrelated | C2, C3; held-out combinations (DBCA); length (train ≤ D, test > D) | without decorrelation, the learner stores frequent composites as single blocks |
| **B. Broad, unstructured** (TabPFN-like: random functions or tables over Z_p) | generic in-context learning | cannot tell modular from monolithic; composition claims impossible |
| **C. Mostly A, plus a small share of random-table worlds** | as A, plus memorization pressure that resembles "facts" | the share of table worlds is a new knob |

Sub-decisions:
- maximum training depth and the held-out depths;
- number of primitives (4–8);
- arity;
- inputs as pairs (a, b) → y, or as sequences (for group products);
- exclusion of trivial worlds (constants, a single identity), the complexity sweet spot.

Evidence:
- MLC: random grammars per episode give systematicity but not productivity.
- Kirsch, Raventós: the number of distinct worlds sets the phase. Combinatorics gives thousands of programs from a few primitives.

### D-3. Grounding and memory: how "this world" enters

| option | what it tests | risk |
|---|---|---|
| **A. Pure synthetic,** with no persistent world | the library only | no test of memory (C4) |
| **B. A "home world" that recurs throughout training:** fixed symbols, a fixed program, plus exceptions that are never shown in context | C4: is the home rule served by the library (few examples on unseen home pairs) while its exceptions are recalled from storage, without interference in other worlds? | rare exceptions get overgeneralized (PCFG SET, below about 0.5%), so the exception rate must be swept; memory becomes a sink unless entries cost something |
| **C. Mix in real data** (e.g. decimal arithmetic in text) | the real world's prior over which compositions occur | many confounds for a first round; Hu et al. and LIME suggest synthetic first, then real data in limited amounts |

Memory mechanism:
- implicit in weights;
- a designated memory block (key–value, charged per entry: the two-part description length);
- external retrieval.

Grounding:
- complementary learning systems: fast hippocampal storage, slow cortical structure;
- F14: confidently wrong predictions are the signal that should route an item to memory;
- X10 / MemSinks: separation needs input-dependent routing.

### D-4. Architecture: where composition lives

| option | what it isolates | risk |
|---|---|---|
| **A. Monolithic transformer** (PFN baseline) | the reference every claim is measured against (I23) | — |
| **B. One weight-tied block, applied recurrently with halting** (a library of size 1; universal-transformer / Mixture-of-Recursions style) | serial composition plus halting, *without* routing | may learn shortcuts anyway |
| **C. A routed library of K uniform blocks** (Mixture-of-Modules style) | reusable primitives (C3) | routing's failure modes: collapse, instability, overfitting (Rosenbaum et al.) |
| **D. C plus a memory block** | C4 | memory as a sink |

- Every block needs attention over the context (Kirsch: state matters more than parameters).
- Compare at matched parameters and compute.
- Halting options:
  - a fixed-depth sweep;
  - learned halting with a cost per step;
  - stop when the layer-to-layer prediction stabilizes.
- Evidence that the comparison is worthwhile: every monolithic model in this literature fails at length (MLC, PCFG SET, universal predictors, shortcuts to automata). Training that encourages step-by-step solutions repairs it (Liu et al.). If B or C repair length without that, it's a genuine result.

### D-5. Scale and measurement (practical)

- **Moduli** 5–17 and **contexts** of ≤ 64 examples, which is enough for the ideal learner.
- **A sweep of distinct training worlds** (e.g. 2^6 to 2^14; Kirsch needed about 2^13).
- **Models sized for T4s.**
- **A curriculum on depth** (Garg).
- **The scoreboard:** predict-then-learn code length per held-out world against the exact Bayes reference.

### How the decisions interact

- C2 needs D-1 ∈ {A, B}, D-2 ∈ {A, C} and D-4 comparing A, B and C.
- C3 needs ground-truth decompositions (D-1 A or B).
- C4 needs D-3 B and D-4 D.
- C1 works under any choice, but is only informative about where the threshold sits.

**A minimal coherent first round, if the researcher agrees:**
- **primitives:** modular operations plus one non-abelian group (D-1 A, leaning B);
- **worlds:** compositional and decorrelated, with a depth split (D-2 A or C);
- **grounding:** a home world with exceptions (D-3 B);
- **architectures:** monolithic, recursive and routed library (+ memory), at matched compute (D-4);
- **reference:** the exact Bayes learner.

## 6. Decisions taken (26 Oct 2026, the researcher)

1. **Primitives with exact solutions known in advance.** Hand-chosen operations (D-1 A, possibly B), so every block's intended function is verifiable. The aim is to show that blocks learned the *general* circuit, not memorized, and that blocks learned on some tasks compose, *at inference*, in combinations never seen in training.
2. **Chunking is allowed.** If frequently co-occurring operations are stored as one block, that is program compression (library learning), not a defect. It becomes a measured outcome. It muddies the experiment only if a composite *replaces* its parts, so that new combinations needing a part alone fail. Held-out compositions test exactly that.
3. **No memory store or home world at this stage** (D-3 A). Focus on composition and generality.
4. **Architectures:** a monolithic PFN against the same PFN broken into a routed library of uniform blocks (D-4 A vs C). No memory block.
   - The shared-block recursion (B) is not a separate arm. It is the K = 1 end of a sweep over library size, kept only as a diagnostic.
   - **Instrumentation is a primary deliverable,** so that a failure of composition can be diagnosed rather than taken as a refutation (L6).

### What "learned the general circuit" can mean, operationally

The domains are finite, so some claims are provable by exhaustive evaluation, and others can only be measured:
- **Exhaustive correctness:** on held-out worlds (fresh symbol bijections, unseen moduli in range), every query in the domain is correct. For a finite domain this *proves* the computed function, within that domain.
- **Specificity:** each block's use aligns with ground-truth sub-programs (primitive or chunk). Ablating a block breaks exactly the programs that contain its sub-program.
- **Novel composition at inference:** held-out combinations (DBCA-style, atoms matched) are correct with zero gradient steps.
- **A shared interface:** the intermediate value after each step is decodable by one fixed probe across contexts and steps. If decodability decays along a route, the blocks don't share an interface.
- **Beyond the domain** (longer compositions, moduli outside the range): measured, not proven.

Known exact solutions do not *ensure* the general circuit is learned; they make it *verifiable*. Memorization and shortcuts remain possible, and are detected by the held-out world × composition design.

### Instrumentation for composition (to diagnose, not to judge)

- **Per block:**
  - activation norm in and out;
  - gradient and update norms;
  - the block's gain, ‖∂h_out/∂h_in‖ (spectral estimate). Repeated application compounds gains: H7 and F17, where chains were unstable;
  - utilization;
  - effective rank of its outputs.
- **Along a route:**
  - residual-stream norm against step;
  - tuned-lens prediction and entropy at each step, and the KL divergence between steps (the researcher's stability signal);
  - probe decodability of the intermediate value.
- **Router:**
  - entropy and load per block;
  - the gradient reaching the router;
  - how often routes switch over training (non-stationarity);
  - collapse indicators.
- **Outcomes:**
  - accuracy and code length by world depth, by primitive, and by split (in-distribution, held-out combination, longer);
  - the distribution of route lengths and halting points.

## 7. Second literature check: has this been done, and what sharpens (26 Oct 2026)

### Closest prior work

| work | what it did | relation to our design |
|---|---|---|
| **Schug et al. 2024, *Discovering modular solutions that generalize compositionally*** | Modular teacher: a hypernetwork in which a task code combines modules. **Theorem:** modules are identifiable (up to a linear map) given compositional support (every module seen), connected support (no group of modules seen only in isolation) and no overparameterization. A linear number of combinations suffices. Modular students generalize; monolithic ones (ANIL, MAML) don't. | Gives the support conditions for our training mix. But the composition is additive, in weight space, not serial |
| **"When can transformers compositionally generalize in-context?"** (Schug et al. 2024) | A vanilla transformer fits in-distribution tasks but fails on held-out combinations, **even though the task code is linearly decodable** from its residual stream. A bottleneck separating *task inference* from *task execution* (transformer → code → hypernetwork) fixes it. | Directly supports splitting a PFN into router (inference) and blocks (execution). Gives a diagnostic: decode the true program on held-out tasks to tell an inference failure from an execution failure |
| **Ramesh et al. 2024, compositional capabilities of transformers** | Serial compositions of bijections and permutations, with the program given as task tokens. Training on a small subset generalizes to exponentially many unseen compositions. **Step-by-step formats (intermediate outputs) help greatly.** Attention selects the function, the MLP computes it. Order biases in training block some compositions. | The serial-composition anchor. Monolithic, and the program is given rather than inferred from examples |
| **Neural Interpreters** (Rahaman et al. 2021) | Functions with signatures; tokens routed by inferred type; repeated function iterations. Comparable to ViT with fewer parameters; better on abstract-reasoning held-out splits. Signatures collapse unless fixed at initialization. | The nearest architecture to a routed library. Parallel routing per token rather than a chosen serial program; no exact-solution verification |
| **Alet et al. 2018, modular meta-learning** | A library of modules shared across tasks; per-task composition found by simulated annealing; better than MAML with few examples. | The "library plus search over compositions" idea, with explicit search instead of a learned router |
| **Mittal et al. 2022, *Is a modular architecture enough?*** | Modular beats monolithic only slightly. The benefit of specialization grows with the number of rules. Learned modules mostly **fail to specialize**. Defines collapse and alignment metrics. | Expect the failure; adopt their metrics |
| **Jarvis et al. 2023; Boopathy et al. 2025** | Theory: modules help only if they *perfectly* segregate the compositional information; modular sample complexity is independent of input dimension. But learning the right modules is hard, and naive modular networks did *worse* than monolithic ones. | Specialization must be induced or diagnosed, not assumed |
| **Wiedemer et al. 2023** | Sufficient conditions: compositional and sufficient support, and a model that shares the true composition function. | The formal frame for "provably composable" |
| **Lippl & Stachenfeld 2025** | Kernel-regime models generalize only "conjunction-wise additively". Deep ReLU networks drift towards lookup tables (memorization leak). | Predicts that monolithic models fail on non-additive (serial, non-abelian) composition, and matches F19 |
| **Zhao et al. 2024; *From reasoning traces to reusable modules* (2026)** | Training on 2–3-skill combinations transfers to 4–5 and to held-out skills. **Composability must be learned from composed experience:** atom-only data doesn't repair a missing atom's compositions. Skills that always co-occur stay entangled. | Direct evidence on the training-mix variable and on decorrelation |

**Gap that remains.** No study found does all of the following together:
- serial composition of a routed library of uniform blocks;
- the program *inferred in context* (PFN-style), rather than given or additive;
- exact solutions to verify blocks;
- a monolithic PFN of equal size as the comparison;
- sweeps of how much composition is in the training mix;
- per-block instrumentation of composition stability.

### Sharpenings to carry into the design

1. **Closure is a trap. Composition must be *necessary*.** Many algebraic families are closed under composition. A composite affine map over Z_p is another affine map, and a product of S_5 elements is one of 120 permutations. A learner can then *identify the composite directly* (task identification) without composing. Composition is necessary only when the composites leave any small family the learner could index, *and* the test uses composite functions never seen as worlds. For S_5, composition is forced only if the program (the sequence of operations) is part of the input, or the held-out set is defined over unseen products.
2. **Count tasks as distinct functions, not programs.** For K primitives and depth L there are K^L unary programs, but far fewer distinct functions (identities, commutation, closure). Over small Z_p, the reachable set at each depth can be enumerated exactly (breadth-first search over function tables, as `program_prior.py` does). That gives:
   - the number of distinct functions at each minimal depth;
   - the depth at which the reachable set saturates;
   - the fraction of all functions reachable;
   - for every held-out function, a *certificate* of composability: its shortest program and the fact that it never appeared in training. That's the "provably composable" set.
3. **The training mix as a measured variable,** parameterized by:
   - the fraction of composed worlds (versus single-primitive worlds);
   - the distribution of minimal depth;
   - the number of distinct combinations relative to K (Schug: linear can suffice);
   - the connectivity of the co-occurrence graph (connected versus disconnected support);
   - decorrelation of operation pairs and orders.

   Prior evidence: atoms alone don't teach composition; connected support is needed; order biases block compositions; 2–3-step training can extrapolate.
4. **Library size against true primitive count.** Schug: more modules than the structure justifies lets the student memorize one module per task. Sweep K around the true number of primitives, and include K = 1.
5. **Separate inference from execution, and instrument both.** Decode the ground-truth program from the router's state on held-out worlds:
   - decodable but wrong → an execution failure;
   - not decodable → an inference failure.
6. **New open decision: intermediate supervision.** Ramesh found intermediate outputs help composition greatly, but supervising intermediate values hands the decomposition to the learner. Options: none (cleanest test); for a fraction of worlds (another mix variable); or diagnostic only.
7. **Adopt Mittal's metrics** (collapse, alignment to the nearest permutation, plus mutual information between routes and ground-truth sub-programs) alongside the instrumentation in §6.
8. **Prediction from kernel theory (Lippl):** monolithic models should do well on additive or abelian composites and fail on non-additive serial ones. The primitive families should include both, so the comparison can discriminate.

## 8. State, control and how the library is built (26 Oct 2026)

**Decision (researcher): no intermediate supervision, and no sweep of it.** The claim to prove is instead that each grown block's output carries two things:
1. **the value:** the primitive's output on its input;
2. **control:** where the computation should go next, learned from the training mix.

### Data and control in the state
- In a serial routed library, the state after each block must carry the *value* and the *control*: the remaining program, or at least the next operation.
- In a PFN world the program is inferred from the context, so a first step has to write the inferred program into the state. Each block then executes one operation and passes the rest along (it "pops" the program). That is a learned interpreter's state: a value register plus a program register.
- **Whether blocks must carry control depends on what the router reads:**
  - **state only** (the researcher's original idea: "the output of a layer routes to the next"): blocks *must* carry control, so control decodability in the state is a testable claim, and width is a meaningful knob;
  - **state plus the context:** the router can re-derive control at every step, and blocks need not carry it.
- **What to measure, per step:**
  - a probe decoding the value;
  - a probe decoding the next operation and the remaining program;
  - interference between the two (does adding control degrade the value readout?).
- **Width:** in bits the need is small (value ≈ log2 p, control ≈ L·log2 K). The real limits are linear decodability and interference (superposition), not raw capacity.

### Building the library: three options

| option | gives | doesn't give |
|---|---|---|
| **A. Train each primitive block alone** to exhaustive correctness on its finite domain, verify it, freeze it, add it to the library; then train routing and control on the mix | blocks provably compute their primitives; no collapse or specialization failure | library *discovery* (we defined the decomposition); control is still learned only in the composed stage; blocks trained apart must share an interface (value code) |
| **B. Train library and router jointly** on the mix | the real test of discovery | likely to fail at first (Mittal; Boopathy); needs the instrumentation to diagnose |
| **C. Staged:** A as a positive control, then B on the same generator, comparing discovered blocks with the known ones | splits "composition fails" into router, interface and stability problems (if A fails) versus discovery problems (if A works and B fails) | costs two stages |

**Note on "grokking" a primitive.** Over a finite value domain, a block trained on *all* inputs simply fits the function; there's nothing to generalize to within the block. Generality across worlds comes from the symbol randomization being handled *before* the blocks: an encoder maps context symbols to a shared value code, and blocks operate on values, not symbols. Grokking-style generalization matters only if a block sees part of its domain.

### Norms, interfaces and stability
- **Interface drift:** each block's output must land in the distribution the next block was trained on. Norm growth along a route moves later blocks off their training distribution, which is especially damaging for blocks trained alone.
- **Residual addition against function composition:** f(g(x)) *replaces* the value, but a residual stream *adds*. Additive updates help optimization; composition needs overwriting. Options:
  - a value subspace that each block replaces;
  - a learned erase;
  - normalization at block boundaries so the state's scale is fixed.
- **A fixed value code:** for Z_p, a unit-norm Fourier (cos/sin) code is natural, and it's what networks learn for modular arithmetic anyway. It fixes the scale and geometry of the interface.
- **Error accumulation at inference:** with per-step error ε, accuracy decays with length unless the code has margin and a clean-up step (snap to the nearest code word, as in vector-symbolic architectures). Some failures to generalize to longer programs may be exactly this, not a failure of composition.
- **Instrumentation to add:**
  - norm and distance-to-codebook of the state at every step;
  - gain on the value subspace (≈ 1 for stable composition);
  - the accuracy-versus-step curve with and without clean-up.

## 9. Simplified design: a frozen primitive library plus a learned composer (26 Oct 2026, the researcher)

**Stage 1:** a library of single-layer primitive blocks, each trained alone on a known task to exhaustive correctness, then frozen. The primitives are chosen to compose serially into a wide variety of programs, as higher-level abstractions, not necessarily fully general yet.

**Stage 2:** a meta-model trained on a second mix learns to compose programs from the frozen blocks. Routing is learned here. Composition must be contextual (attention over the examples), so it can generalize to compositions unseen in training.

Joint discovery is deferred.

### Where control lives
- With frozen blocks trained alone, blocks cannot carry control: they never saw composition. **Control lives in the meta-model:**
  - at each step it reads the context examples (attention), the current value state (the last block's output) and its own execution history;
  - it chooses the next block or halts.
- This is the separation of task inference from execution that fixed compositional generalization in "When can transformers compositionally generalize in-context?".
- Value-dependent routing (conditionals) is still possible, because the meta-model reads the value state.
- **The provable claims become:**
  - blocks provably compute their primitives (Stage 1, exhaustive);
  - the inferred program is decodable from the meta-model's state on held-out worlds, which separates inference failures from execution failures.

### The machine model (what the state is)
- Serial composition of value → value blocks covers only unary chains. Binary operations need several values in the state, so the state should be a small **register or stack machine** over a fixed value code:
  - blocks are instructions (operations on registers, plus moves or copies);
  - a program is a straight-line instruction sequence.

  With enough registers and length, straight-line programs express any finite function. Loops would need halting.
- With mixed operations (add, multiply, compare/select), composites leave any small family, which avoids the closure trap (§7). Reachable functions per program length are enumerable.
- Higher-level abstractions (chunks) can be added to the library later as compressed sub-programs.

### Training signal for the composer (a new decision)

| signal | effect |
|---|---|
| **Program supervision** (the known program as target) | easiest, but hands in programs. Equivalent programs must be accepted: supervise against *any* minimal program |
| **Final output only** | soft routing with clean-up, straight-through, or policy gradient; the hardest credit assignment |
| **Search plus self-training (expert iteration)** | the composer proposes; exact frozen blocks execute candidates *on the context examples*; verified programs become training targets (DreamCoder's recognition model). No hand-given programs. |

### Verification and search at inference
- Frozen exact primitives make verification cheap and exact. Run any candidate program on the context inputs and compare with the context outputs.
- The composer proposes; search branches over candidates in parallel (VRAM); mismatches are pruned. Generalization to unseen compositions then doesn't depend on the composer having seen them.
- **Evaluation axis:** first-proposal accuracy ("seeing") against accuracy within k verified samples ("searching"). The gap measures how much of the competence is the learned prior and how much is search.

### What the experiment now tests
Given a verified primitive library, can a meta-model infer and compose programs in context, and generalize to unseen compositions and longer programs, compared with a monolithic PFN of equal compute on the same worlds? Measured with:
- first-proposal and with-search accuracy;
- program decodability;
- sweeps of the composition mix.

Joint discovery comes after.

## 10. The composer as a PFN over programs (26 Oct 2026, the researcher)

**Researcher's position:**
- train the composer the PFN way: it learns support over the program prior seen in training, and generalizes at inference;
- routing should *not* depend on block output values, which are pure functional outputs with no routing information;
- routing comes from structure learned from the training mix: which primitives compose, in which structures, how often;
- the one-to-many problem (many programs explain the same observations) is resolved as PFNs resolve it, by predicting over worlds with support in training.

**What the PFN literature supplies.** For data generated by sampling a latent from a prior and then observations given the latent, the loss E[−log q(latent | observations)] is minimized by the exact posterior. This is the forward-KL argument behind PFNs (Müller et al. 2022), inference compilation (Le et al. 2017) and neural posterior estimation. Two composer objectives follow, and both are PFN-consistent:

1. **The latent is the target:** cross-entropy of the composer's program distribution against the program that generated the world. The optimum is the posterior over programs given the context, P(π | D) ∝ prior(π) · 1[π reproduces D].
   - **Correction to §9:** this is not "handing in programs". It is amortized posterior estimation. It spreads mass over every program consistent with the observations, in proportion to the prior, which is exactly the one-to-many resolution.
   - Evaluation must be functional (do the predicted outputs match), not an exact match to one program.
2. **Observations only:** predict the query output through a mixture over the composer's programs, each executed by the frozen library: −log Σ_π q(π | D) · 1[π(x_query) = y_query]. Same optimum, but only output data is used, at higher variance.

**Routing is open-loop from the context.** For straight-line programs, the whole program is inferred from the context and then executed by the frozen blocks. Routing doesn't read intermediate values. Data-dependent behaviour, if wanted, lives inside instructions (a select instruction), not in routing. An autoregressive program decoder learns the structural prior: which instruction tends to follow which.

**Exact checks this makes possible:**
- **posterior fidelity:** with the generator's prior known and programs executable, the true posterior over programs (or over functional-equivalence classes) given any context is computable by enumeration. The composer's q(π | D) can be compared with it directly (KL), which tests whether it learned the PFN posterior;
- **inference against execution:** execution is exact by construction (verified frozen blocks), so any error is an inference error.

**Support against samples (to state precisely).**
- **In support:** held-out programs that the training prior could generate, but that weren't sampled. PFN theory predicts success given enough distinct worlds (the Kirsch / Raventós thresholds).
- **Out of support:** compositions *excluded by design* (DBCA-style held-out pairs, longer programs) have zero probability under the training prior. PFN theory promises nothing there. Generalization would be an empirical property of the learned solution (Raventós: past the diversity threshold, the learner behaves like the general algorithm). This is the test where structure should matter.
- Both are evaluated, and reported separately.

**Prior art on program-synthesis generators (Shin et al. 2019).** Synthesis models trained on randomly generated programs scored well on their own test distribution and collapsed under shifts in *salient variables*: program length, nesting, input statistics. Karel accuracy fell from 73.5% to 27.9%. Resampling the generator to make salient variables roughly uniform restored most of it. For us:
- list the salient variables (program length, instruction frequencies, pair and order statistics, operand patterns);
- control their distributions in the mix, and include those that are swept;
- report accuracy along each.

Related: DeepCoder (Balog et al. 2017; a network predicts which functions appear, to guide search) and RobustFill (Devlin et al. 2017; neural synthesis from examples), both trained on synthetic program priors.

## 11. Third literature check: the design is close to neural program synthesis (26 Oct 2026)

With frozen, verified primitives and a composer that infers programs from examples, the design sits inside **neural program synthesis and induction**. What that literature already shows:

| work | finding | consequence for us |
|---|---|---|
| **DeepCoder, RobustFill** (2017) | Synthesizers trained on randomly generated programs from a DSL work well in distribution; DeepCoder uses the network to guide search | our Stage 2 is this setup with a neural-implemented DSL |
| **Shin et al. 2019** | Collapse under shifts in salient variables (Karel 73.5% → 27.9%); fixed by resampling those variables to be uniform | control and report the salient variables |
| **ExeDec** (Shi et al. 2024) | Compositional-generalization benchmark for synthesis with five splits. A Transformer gets 90.9% in distribution vs 42.4% on the held-out average (string domain); 53.6% vs 5.7% (list domain). **Predicting execution subgoals step by step** raises the held-out average to 87.0% and 23.1% | adopt the five splits as our out-of-support taxonomy (below). The strongest known lever is conditioning on execution |
| **Execution-guided synthesis** (Chen et al. 2019), **REPL synthesis** (Ellis et al. 2019), **latent execution** (LaSynth 2021) | Conditioning the next choice on the execution state of the partial program improves synthesis. A learned latent executor nearly matches a real interpreter | evidence against purely open-loop composition (see the decision below) |
| **Neural Execution Engines** (Yan et al. 2020) | Transformer subroutines (compare, add, min) learned **in isolation**, then **reused without retraining** inside sorting and shortest paths. Trained on size 8, near-perfect at size 100. Critical ingredients: **binary (structured) value codes**, and masks separating *where* from *what*. Without them, accuracy fell below 10% at 100 | strong support for the frozen-library stage, and evidence that the **value code is a first-order design choice** |
| **NPI, and NPI with recursion** (Reed & de Freitas 2016; Cai et al. 2017) | A learned controller over a library of learned subprograms, trained with execution traces. Recursion makes guarantees about generalization tractable | the controller-plus-library lineage. Supervision there was much stronger than we plan |
| **TerpreT** (Gaunt et al. 2016) | Gradient-based program induction loses to discrete search and constraint solvers | supports verification and search over a purely gradient-based composer |
| **Hypernetwork in-context** (Schug et al. 2024) | A monolithic transformer infers the task but can't execute held-out combinations; an inference/execution bottleneck fixes it | the closest monolithic-vs-bottleneck comparison |

**Honest framing of Stage 1.** Once the primitives are frozen and verified exact, the neural library behaves like a symbolic interpreter. Stage 1 is then a neural-guided program synthesizer: a positive control and a measurement framework, not the novel part. What is not standard, and is ours:
1. **A head-to-head with a monolithic PFN on identical worlds,** scored by predict-then-learn code length against the *exact* Bayes posterior.
2. **A posterior-fidelity check** of the composer against the enumerated posterior over programs.
3. **Controlled sweeps of composition in the mix,** with certified counts of composable functions.
4. **The "seeing vs searching" gap:** first proposal against within k verified samples.
5. **Stage 3:** discovered rather than defined primitives, where the neural implementation actually matters.

**Out-of-support taxonomy (from ExeDec), adopted:**
- length generalization;
- compositions across concept groups seen only separately;
- a switched order of concepts;
- an operation seen only in isolation, used inside compositions;
- a held-out functionality of an operation.

**Decision raised by the literature: open-loop or execution-conditioned composition.**
- The researcher's position: routing should not read block outputs, because they carry no routing information.
- The literature's strongest lever (ExeDec, execution-guided synthesis) conditions each step on the *execution state relative to the target outputs*: what remains to be achieved. That isn't routing information carried by a block. It is the specification shrinking as the program runs.
- **Options:**
  - open-loop only (the cleanest test of the learned structural prior);
  - execution-conditioned as a variant (the known lever);
  - both, reported separately.

**Value code.** Neural Execution Engines show that structured value codes decide generalization. For residues, a Fourier code is the natural choice. Whether to fix the code or compare two is open.

## 12. Open-loop vs execution-conditioned, the value code, and sizing the instruction set (26 Oct 2026)

### Open-loop vs execution-conditioned composition
- **Open-loop:** the composer reads the context examples and emits the whole program; the frozen blocks then run it.
- **Execution-conditioned:** after each instruction, the composer also sees the intermediate values for every context example, next to the target outputs.
- **For an ideal Bayesian composer these are equivalent.** Intermediate values are a deterministic function of the examples and the partial program, so they carry no new information.
- **The difference is computational.** An open-loop composer must *internally simulate* the library: it has to know what each primitive does in order to predict what a program will do. An execution-conditioned composer offloads that simulation to the blocks. Open-loop therefore tests whether the composer learned an internal model of its primitives' semantics. Errors in that internal model should compound with program length.
- **Prediction:** the two are similar in distribution; open-loop degrades faster on longer programs and on held-out compositions. This matches ExeDec and execution-guided synthesis.
- **The comparison is cheap:** the same composer, with or without an execution-trace input.

### The value code
The value code is how a value (e.g. "3 mod 7") is written as a vector in the state, the format every frozen block reads and writes. Options:
- **one-hot / table** (no geometry);
- **binary** (positional bits);
- **Fourier** (cos and sin of 2πkv/p for a few k; modular addition becomes rotation, so it builds in modular structure);
- **learned** (an embedding learned jointly with the blocks in Stage 1).

It matters for four things:
1. **The interface:** blocks trained apart must agree on it.
2. **Generalization within a block:** this matters only if a block sees part of its domain. Stage 1 trains on full domains, so it largely drops out.
3. **Stability over long programs:** a discrete code allows clean-up (snap to the nearest code word) after each block, so errors cannot accumulate. Without clean-up, drift and norm effects become observable.
4. **A built-in prior:** Fourier builds in modular structure; one-hot builds in none.

**Default:** one-hot with clean-up for the main experiment, which gives an exact, discrete interface and isolates the composer. A soft interface without clean-up is the variant for studying stability.

**Related simplification:** keep value semantics fixed across worlds (no per-world symbol randomization) in this round. Randomizing symbols adds a second hidden variable (which symbol means which value, identifiable only up to automorphisms) on top of program inference.

### Sizing the instruction set: counts from `experiments/worlds/count_isa.py`
Setup: two registers (r0 = x, r1 = y), output r0, instructions dest ← src1 op src2 or dest ← f(src).

| instruction set | L = 1 | 2 | 3 | 4 | 5 | 6 | examples to identify |
|---|---|---|---|---|---|---|---|
| p = 5, {add, sub, mul, neg, inc} (32 instructions), new functions by minimal length | 13 | 74 | 475 | 3,214 | 21,777 | 145,201 | 8 → 36% unique, 12 → 86% |
| p = 5, additive only {add, sub, neg, inc} | 10 | 30 | 57 | 27 | 0 | 0 (saturates at **125** = p³ affine maps) | 3 → 86%, 4 → 100% |
| p = 7, mixed | 13 | 80 | 552 | 4,179 | 32,741 | — | 6 → 48%, 8 → 87%, 12 → 98% |

**What the counts say:**
- **The closure trap, measured.** Additive-only programs of any length collapse to 125 affine functions, identifiable from 3–4 examples, so a learner can identify the function without composing. Adding multiplication makes the number of distinct functions grow about 6–7× per step, so composition is necessary.
- **Diversity against thresholds.** Distinct functions at length ≤ 4: about 3.8k (p = 5) or 4.8k (p = 7). At ≤ 5: about 25k or 38k. At ≤ 6: about 171k (p = 5). Kirsch's general-learning threshold was about 2^13 ≈ 8k tasks, so training up to length 5 gives a few times that, with length 6 as the held-out length split (145k functions that provably need 6 steps).
- **Context size.** Pinning a function down takes 8–12 examples (p = 5) or 6–8 (p = 7). p = 7 leaves more held-out query inputs (49 inputs per world).
- **Concept groups fall out naturally:**
  - additive-only programs form a closed family (affine maps);
  - multiplication-only programs are near-closed (monomials);
  - mixed programs reach polynomial space.

  So ExeDec's "compose different concepts" split is certifiable: train on single-concept programs, test on mixed ones.

**Still to fix:** p, the number of registers (more registers raise diversity per length), whether to add a constant or comparison/select instruction, and the program prior (uniform over programs vs uniform over distinct functions; Shin's salient variables).

## 13. Decisions (26 Oct 2026, the researcher)

- **Program prior:** sample uniformly over *distinct functions*, then a random program that computes the chosen function, so frequently computed functions are not over-weighted.
  - **Refinement:** draw the program from the function's *minimal-length* programs (or minimal + a small slack). Otherwise long redundant programs (no-op or cancelling instructions) dominate each function's set, and the composer learns to emit them.
- **Domain:** mod 7, mixed operations ({add, sub, mul, neg, inc}).

**Settled so far:**
- frozen primitive library, each block verified exhaustively;
- composer trained as a PFN over programs;
- held-out splits defined over functions, using ExeDec's taxonomy plus single-concept-train / mixed-test;
- training lengths ≤ 5, with length 6 held out;
- a monolithic PFN of equal compute as the baseline;
- instrumentation per §6 and §8.

**Tentative:** one-hot value code with clean-up as the main interface (a soft interface as the stability variant); fixed value semantics across worlds.

**Open before preregistration:**
- the number of registers;
- whether to add a constant or compare/select instruction;
- open-loop vs execution-conditioned (both as arms?);
- the composition-mix sweep levels;
- model sizes and compute.

## 14. Remaining design criteria (26 Oct 2026, for discussion)

**Decided (researcher):**
- run both open-loop and execution-conditioned composers and compare them;
- sweep training mixes, and stratify test performance by composition;
- size every model for T4 GPUs.

**Registers (counts from `count_isa_registers.py`, mod 7):**
- three registers (99 instructions) reach 6,257 distinct functions by length 4;
- two registers (32 instructions) reach 4,825.

So a third register adds about 30% diversity at three times the composer's choice set, and more equivalent programs per function. Leaning towards two registers.

**Mix sweep:** separate the three things that "more composition" bundles together:
1. the maximum training depth;
2. the number of distinct training functions (diversity, the Kirsch / Raventós threshold);
3. concept mixing (single-concept versus mixed programs).

Depth and diversity are confounded unless diversity is subsampled at a fixed depth cap.

**T4 sizing:**
- each example is one token (x, y, f(x, y)), with a context of 12–16 examples and programs of ≤ 6 instruction tokens;
- composer and monolithic PFN are small transformers (≈ 1–5M parameters), with parameters and training episodes matched;
- the frozen blocks are tiny maps on one-hot values;
- several runs should share a GPU (fleet-style batching), given Kaggle's weekly GPU quota.

**Sources** are linked inline in the chat discussion of 26 Oct 2026 and in history Entry 29.
