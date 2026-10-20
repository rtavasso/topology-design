# Active research spec

**Status:** active as of 19 October 2026, replacing the v0 plan in [`archive/v0/`](../archive/v0/). Revised the same day after the [literature review](literature_review.md). The v0 motivations are preserved in [00_questions_in_original_words.md](00_questions_in_original_words.md); the change log is in [history.md](history.md).

## 1. Question

> **How does parameter-free feed-forward connectivity shape which structures emerge during learning** — which features get preserved, erased or constructed, which computations get shared versus specialized, and which updates transfer versus interfere — **and through what mechanism?**

Constraints carried over from v0, all still binding for primary comparisons:
- same learned operators and parameter count;
- softmax cross-entropy;
- single-pass, acyclic graphs;
- no learned gates, no recurrence, no per-branch losses.

## 2. The causal chain we study

We study a chain in which every link has existing theory, so each link yields a prediction before any training:

```
topology
  ├─► function class            (expressivity: what can be represented at fixed size)
  └─► parameterization          (a path polynomial in the weights)
         └─► global parameter-to-function map ─► effective preconditioner on the learned function
                                                        └─► learning order (plateaus, which features first)
                                                        └─► implicit bias  (which solution among many)
                                                               └─► emergent structure
                                                                   (shared vs specialized, composite vs shortcut,
                                                                    coherent vs idiosyncratic updates)
```

### Where topology can and cannot act

Two results locate the topology effect:
- **Prop. 1:** every graph in the parent-selection family $h_i=h_{i-1}+F_i(h_{k_i})$ is first-order identical at zero branch weights.
- **Marcotte, Peyré & Gribonval (ICML 2025):** residual blocks have exactly the same conservation laws as the same blocks without the skip.

So **topology cannot act through new local invariants.** It acts through how blocks are composed into the global parameter-to-function map. That makes it a *dynamical* effect that grows with how far learning travels. Two consequences organize everything below.

1. **Richness is the independent variable, alongside topology.** Topology effects in residual families are higher order in branch weight, so they are gated by:
   - initialization scale;
   - a fixed (parameter-free) branch multiplier $\alpha$;
   - the task's required distance from identity.

   The strongest possible result is a **phase boundary**: an effect that is zero in the lazy limit and grows in a predicted direction as features move. Width is a secondary axis. The branch multiplier also selects which infinite-depth theory applies (ODE, SDE, or degenerate limit; see [review §1](literature_review.md)), so it must be swept, never fixed silently.
2. **"Productive frustration" is a saddle.** A loss that is stationary in the shallow directions while the deep route carries unresolved error at second order is a strict saddle (the v0 product-starvation toy is exactly this; see `test_product_path_starvation`).
   - Plateau length, escape direction and hidden progress during the plateau are measurable.
   - There is theory for them: saddle-to-saddle dynamics, leap complexity.
   - "Productive" means that escaping the saddle produces a feature that generalizes.

## 3. Inference protocol (binding for every study)

The replication literature shows most claimed architecture effects shrink under equal tuning, shared code and counted seed variance (Melis 2018; Lucic 2018; Narang 2021; Musgrave 2020). Structure claims are likely *more* fragile than accuracy claims, because the metrics are less standardized. The protocol follows Platt-style strong inference: eliminate the boring explanations first.

**Rival explanations every study must rule out**, and how:

| Rival | How to rule it out |
|---|---|
| (a) Effective learning rate | Per-arm learning-rate tuning |
| (b) Lazy–rich regime shift | Richness phase diagram; matched kernel movement |
| (c) Effective depth / path length | Depth-matched controls; dose–response over mutants |
| (d) Capacity / expressivity | Over-provisioned width, with fit reported per arm and target |
| (e) Metric artifact | Two metric families plus nulls |
| (f) Seed noise | Paired, powered seeds |

### Rules

1. **Directional, preregistered prediction** with at most **one or two primary endpoints**, written before the main runs together with explicit kill criteria. Example: "if the effect on the lag is below 2 seed SDs, flips sign across the top three LR points, or disappears at matched training loss, H2 is rejected for this target." Name in advance which outcome favors the regime hypothesis over the mechanism. All other metrics are exploratory, with Holm or Benjamini–Hochberg correction.
2. **Per-arm tuning as a nuisance.**
   - Every arm gets the same search algorithm and budget.
   - Report expected-best-vs-trials curves (Dodge et al. 2019) for loss *and* the structural endpoint.
   - Check that each LR optimum is interior to the sweep.
   - Report results at each arm's optimum *and* at a shared LR.
   - Use a fully decayed schedule in every arm (Kaddour et al. 2023).
3. **Richness phase diagram.**
   - Sweep initialization scale and $\alpha$ for every topology.
   - **Built-in control:** the paired effect must go to 0 as $\alpha\to0$ and grow with richness. A nonzero effect at vanishing richness signals a bug, an LR artifact or a confound.
   - Report zero-branch and random-branch initialization separately; at random init the kernels already differ at $O(\epsilon)$.
   - Log kernel movement relative to initialization and weight norm, so arms can be compared at a matched feature-learning regime.
4. **Over-provision width for expressivity.** Widen until every arm fits every target to a stated tolerance. Report fit per arm and target, and compute structural endpoints only for arms that fit.
5. **Three clocks.** Report structure at matched steps, at several matched training-loss thresholds, and at the end of the decayed schedule. If structure-vs-loss trajectories overlap across topologies, only speed differs.
6. **Seeds and statistics.**
   - **Pilot:** 50–100 seeds of one arm to measure the null distribution of each primary endpoint. No published estimate of seed variance for structural metrics exists, so this pilot is itself a result.
   - **Reported results:** ≥10 paired seeds.
   - **Flagship headline:** ≈29 runs per arm, judged by $P(A>B)$ with a run-level bootstrap. Claim A>B only if the lower CI bound is >0.5 and the upper bound is >0.75 (Bouthillier et al. 2021).
   - Randomize data seeds independently of init seeds; data sampling dominates variance.
   - The unit of analysis is the training run.
7. **Dose–response and reversal, not just endpoints.**
   - Order the one-edge mutants by a graph statistic (longest chain feeding the output; path-length distribution) and predict a monotone trend.
   - Add a **random-routing control** with the same operators and parameters.
   - Show the effect can be **induced** (add the edge that creates the composition to a parallel arm) and **eliminated** (cut it from a sequential arm).
8. **Nulls.**
   - Identical-graph null and operator-label permutation.
   - Mask-preserving shuffles for any weight-graph statistic. Otherwise "parallel is more modular" is true by construction.
   - Density-matched rewiring as the null for wiring claims (Frankle et al. 2020).
9. **Derive before you measure.** If a quantity has a closed form in the setting (as one-example transfer does in linear chains, Prop. 4), test the closed form and measure only the residual. For nonlinear parallel arms, predict plateau structure from two-layer theory first (§4 Phase 2).
10. **Fix or cross confounding architecture factors.** Normalization placement flips which layers become redundant (Pre-LN vs Post-LN). Fix it, or cross it as a factor. Record activation memory and critical path as well as parameters and FLOPs.

## 4. Study sequence

| Phase | Question | Directional prediction | Kill / pivot criterion |
|---|---|---|---|
| **0. Linear theory** | Flows, implicit bias and parameter degeneracy of linear parent-selection DAGs | Props. 1–7 (done/partly done); open: Prop. 7 analysis, separable limits, RLCT | Simulation disagrees with derivation → bug or wrong derivation |
| **1. Gated linear networks** | Does fixed routing predict which pathways win, i.e. shared vs specialized representations? | Pathways active in more contexts win the race and carry shared structure (Saxe et al. 2022); branches are recruited sequentially | Race theory fails on routing variants with matched parameters |
| **2. Construction by composition** (flagship) | Does sequential routing change *when* composite features form relative to their constituents? | Sequential shortens the constituent→composite lag; parallel follows its two-layer leap-complexity prediction; the gap grows with richness and is monotone over mutants | No lag difference at per-arm-tuned LR; no richness scaling; no dose–response |
| **3. Reuse, interference, memorization** | Does topology change gradient coherence and where exceptions are stored? | Graphs whose effective kernel adapts show rising within-rule coherence; skip-only differences in forgetting are null (Mirzadeh 2022) | Coherence and localization indistinguishable across topologies |
| **4. Preserve / erase; transformers** | How do topologies handle removing information after use? Does parallel attention block composition? | Composed stacks erase more slowly than shortcut-rich graphs; a decaying identity coefficient speeds erasure; parallel attention delays or blocks induction heads | No erasure-cost difference; induction heads form at matched time |
| **5. Scale** | Do Phase 2–4 effects survive principled scaling? | Effects persist under μP / depth-μP at fixed feature load | Effects are artifacts of untransferred hyperparameters |

Advancing a phase requires the previous phase's prediction to be *tested*, not *confirmed*. A clean failed prediction is a result and changes the next prediction.

The [literature review](literature_review.md#test-predictions-that-prior-theory-already-makes) lists 13 further predictions from prior theory, each with its phase and falsifier. Studies should test the relevant ones alongside their own hypothesis.

### Phase 0 — finish the linear theory

**Done** (tested in `tests/`; see [phase0_linear_theory.md](phase0_linear_theory.md) for which parts restate prior work):
- first-order universality;
- parallel graph equals one matrix;
- balanced-chain flow and mode-wise preconditioner;
- closed-form NTK;
- first single-example step equals a scalar rescale;
- orientation lock of chains (anticipated by Bartlett et al. 2018);
- non-separable endpoint equivalence;
- exact chain-plus-shortcut reduction (Prop. 7).

**Open**, in priority order. Items 1–2 are the program's best theory-novelty bets ([review, niche map](literature_review.md#the-niche-map-separates-occupied-ground-from-open-territory)).
1. **Gradient-flow implicit bias for linear parent-selection DAGs.**
   - Analyse the coupled $(P,W_3)$ flow and explain the per-mode exceptions in the [probe](../experiments/phase0/prop7_pathway_probe.py).
   - Classify which parent vectors give *commuting* path-liftings, i.e. which are mirror-descent-equivalent (Li, Wang, Lee & Arora 2022).
   - Use intrinsic path-space dynamics (Marcotte et al., ICLR 2026) for graphs that don't split into chains plus shortcuts.
   - We found no published treatment of these general DAGs: tensor views explicitly exclude branched networks (Yun et al. 2021).
2. **RLCT / learning coefficient of residual and parallel parameterizations** of the same function class.
   - Use the fiber-codimension method (Pepin Lehalleur & Rimányi 2024).
   - Conjecture: residual maps near identity targets are less singular than plain chains.
   - Calibrate the LLC estimator against the exact plain-chain RLCT here before using it as a stage detector in later phases.
3. **Separable-data directional limits.** Prediction: each arm converges to the KKT point of the margin problem defined by *its own* parameter map, so parallel and sequential limits differ (Cai et al. 2025).
4. **Weight-decay endpoint contrast.** The chain with decay toward identity is Boix-Adsera's (2025) residual model, with a log-type penalty on large singular values. The parallel arm under the same decay is ridge-to-$I$ at strength $\lambda/L$ (Prop. 2). Check the predicted spectra at finite $L$.
5. **Imbalanced initialization** (Kunin et al. 2024; Dominé et al. 2025).
6. **Robustness only:** finite-step drift, Adam, minibatch cross-terms.

**Deliverable.** A per-topology table: preconditioner, which modes are accelerated or slowed, implicit bias (with and without weight decay), and RLCT. Its predictions feed Phases 2 and 4.

### Phase 1 — gated deep linear networks as the analyzable bridge

GDLNs route inputs through pathways selected by **fixed, non-learned gates**. Learning reduces to a race in which pathways shared across contexts win, yielding abstraction (Saxe, Sodhani & Lewallen 2022). Feature-learning ReLU networks behave like an equivalent GDLN (Jarvis et al. 2025). Cross-architecture saddle-to-saddle theory (Zhang, Saxe & Latham 2026) predicts that parallel topologies recruit branches one at a time.

1. **Replicate** the neural-race headline result: shared-pathway dominance, and generalization to held-out context–item combinations.
2. **Vary gating topology at matched parameters.** Use the same weight matrices but different assignments of which contexts open which nodes. Predict which representation wins and the order of branch recruitment, then train.
3. **Bridge.** Train small ReLU networks on the same tasks and test whether the GDLN predictions transfer.

**Decision for the researcher:** fixed input-dependent gates add no parameters and no compute. Are they inside the "parameter-free routing" constraint? This spec uses them in Phase 1 only, labeled as analyzable surrogates. Primary nonlinear comparisons remain edge-only rewires.

### Phase 2 — construction by composition (flagship nonlinear study)

**State this framing up front.** With ReLU MLP blocks, the parallel graph computes $x+\sum_i W_2^{(i)}\mathrm{ReLU}(W_1^{(i)}x)$. That is exactly **one two-layer network of width $L\cdot m$ plus an identity skip**. The flagship contrast is therefore partly shallow-wide versus deep-residual, and the sample-complexity version of H2 is occupied by 2- vs 3-layer separations (Nichani et al. 2023; Allen-Zhu & Li 2019).

What stays new:
- the **dynamical** endpoint;
- its **richness scaling**;
- the **one-edge mutants as a dose–response** between the two arms.

The reduction is also an asset. The parallel arm's plateau sequence is *predictable before training* from two-layer mean-field and leap-complexity theory (Abbe, Boix-Adserà & Misiakiewicz 2022/2023), so every deviation of the sequential arm is attributable to composition.

**Architecture:**
- sequential $(0,1,\dots)$, parallel $(0,0,\dots)$, and one-edge mutants ordered by longest chain to the output;
- a random-routing control;
- width over-provisioned until every arm fits every target;
- a fixed branch multiplier $\alpha$ as the richness knob.

**Targets.** Boolean functions $y=h(g_1(x_{S_1}),g_2(x_{S_2}))$:
- the intermediates have low-order correlation with $y$, so they are learned first;
- the composite has high leap complexity from raw inputs.

**H2 (directional, primary endpoint = lag).**
- After $g_1,g_2$ are represented, the sequential graph acquires the composite faster than the parallel graph predicts.
- The effect is monotone across mutants, grows with richness, and is ≈0 in the lazy limit.
- **Secondary prediction:** sequential arms revise intermediate features after the composite emerges ("backward feature correction"; Allen-Zhu & Li 2023).

**Controls:**
- a pure high-order parity with no learnable constituents (prediction: no difference);
- a sum of independent features (prediction: parallel ≥ sequential);
- the induce/eliminate edge reversal;
- three clocks;
- the richness phase diagram.

**Measurements** (tiers as in §5):
- Walsh–Fourier atlas of centered logits over the full input domain at every checkpoint, with emergence defined on *continuous* coefficient trajectories or across a range of thresholds (thresholded "emergence" manufactures mirages; Schaeffer et al. 2023);
- per-example output agreement against within-arm seed spread;
- plateau lengths and hidden-progress measures;
- layerwise computation profile;
- linear decodability of $g_1,g_2$;
- LLC (calibrated in Phase 0).

### Phase 3 — reuse, interference, memorization

**Setups:**
1. Two tasks sharing a subcomputation ($f\circ g$, $h\circ g$): measure transfer and module specialization (Jarvis et al. 2023).
2. Rule-plus-exceptions with a correlated shortcut (v0 generator, [archive doc 07 §C](../archive/v0/docs/07_datasets_and_scale.md)).
3. Sequential task switch: measure interference and forgetting.

**Measurements:**
- per-example gradient coherence and stiffness;
- NTK change relative to initialization;
- the counterfactual one-step transfer matrix $T(i,j)$ (closed-form-checked in the linear case);
- localization of exception storage (Maini et al. 2023);
- per-branch output share and ablate-and-rescue for branch specialization.

**Predictions:**
- **H3:** graphs whose effective kernel adapts show coherence among same-rule examples rising faster than raw-input graphs do.
- **From prior theory:** identical parallel branches homogenize under weight decay or SGD noise (Ziyin 2024) and collapse onto a dominant branch unless symmetry is broken (Alimaskina 2026; Béna & Goodman 2025). Skip-only effects on forgetting are expected to be null (Mirzadeh 2022); a replicated null is a result.

### Phase 4 — preserve and erase; tiny transformers

**Erasure tasks** ("use then erase"):
- Measure erasure lag and leakage of the erased feature, using both a linear probe *and* projection ablation on the natural distribution.
- Add a **fixed, decaying identity coefficient** $\alpha_l$ as a second parameter-free knob (Zhang et al., CVPR 2026). This separates erasure through the identity path from erasure through shortcut branches, two mechanisms the parent-selection family otherwise conflates.
- **Predictions:** composed stacks erase more slowly than shortcut-rich graphs (Prop. 3, Prop. 7); decaying $\alpha_l$ speeds erasure and lowers effective rank.

**Tiny transformers.** Sequential vs parallel attention/MLP blocks, at matched parameters and FLOPs, with Pre-LN vs Post-LN fixed or crossed.
- **Primary target: induction-head emergence.** Induction heads need cross-layer K-composition, and a 1-layer solution is exponentially larger than a 2-layer one (Sanford, Hsu & Telgarsky 2024). So placing the two attention blocks in parallel should delay or block formation.
- **Readouts:** phase-change step and prefix-matching score (Singh et al. 2024).
- **Secondary:** layerwise computation profile, and progressive neural collapse by layer.
- **Causal-tool requirements:** validate on Tracr or InterpBench first. Use one *restricted* map class (linear DAS at fixed rank) for all arms. Report interchange-intervention accuracy (IIA) against random-init versions of each topology, because unrestricted maps make IIA vacuous (Sutter et al. 2025).

### Phase 5 — scale

Use μP / depth-μP (Yang et al.; Bordelon et al. 2024) so hyperparameters transfer. No μP result exists for parallel or mixed graphs, so check transfer empirically first. Vary separately:
- width at fixed latent feature load;
- feature load at fixed width;
- data support.

Two sizes do not make a scaling claim.

## 5. Measurement stack, by validity across topologies

The core difficulty is **correspondence**: topologies with matched parameters share no units or coordinates that mean the same thing. A measure is valid across topologies only if it is computed on something both share — the function, the data, a reference algorithm, or activations at a symmetry-matched site. Full rationale and failure modes: [review, measurement section](literature_review.md#valid-measurement-starts-in-function-space-and-climbs-toward-mechanism).

| Tier | Use for | Instruments | Required guards |
|---|---|---|---|
| **1** | Primary endpoints | Walsh–Fourier / purified ANOVA atlas of centered logits; per-example output agreement (JSD, disagreement) vs within-arm seed spread; algorithm-discriminating behavioral probes | Fixed input measure; continuous metrics |
| **1 (synthetic)** | Mechanism claims | Causal abstraction against a *shared* reference algorithm (e.g. the $g_1,g_2,h$ program) | Restricted map class; held-out pairs; random-init baselines; projection-ablation confirmation |
| **2** | Developmental timing | LLC trajectories / susceptibilities; hidden-progress measures; NTK change relative to init | SGLD settings tuned and reported per arm; LLC calibrated in Phase 0 |
| **2–3** | Representation comparisons | Procrustes, debiased CKA, mutual k-NN, direct-matching stitching — **only at symmetry-matched sites**; always paired with linear decodability of known intermediates | Permutation nulls corrected for layer search; never raw CKA alone |
| **3** | Single-network descriptors | Bias-corrected participation ratio; functional (output-based) modularity; layerwise profile: lesion/swap curves, gradient-alignment cosine $\cos(F_i(h_i),-\nabla_hL)$, Jacobian alignment across depth; progressive NC1 | Topology-aware ablation: resample ablation normalized to a matched ablation of all branches at the same depth |
| **4** | Corroboration only | SAE/crosscoder counts, training-data attribution | — |
| **Within one topology only** | — | Raw sharpness, linear mode connectivity, weight-graph clusterability without mask-preserving nulls | — |

**Claim rule.** A between-topology claim needs:
- the difference to exceed within-topology seed spread;
- a null for every score;
- agreement across at least two tiers.

**Deferred indefinitely:** retained-subgraph causal-depth scores $D_k,U_k$ (v0), SAE pipelines, architecture search.

## 6. Emergent-structure targets, prioritized

From the [review](literature_review.md#prioritize-four-emergent-structure-targets-that-share-one-experimental-grid), ranked by cheap operationalization, prior evidence that *connectivity* moves the target, and theory that predicts direction:

1. **Order and timing of interaction acquisition** (Walsh atlas). Phase 2 primary.
2. **Layerwise computation profile** (lesion/swap curves, gradient-alignment cosine, Jacobian alignment). These have the largest documented connectivity effects (Residual Alignment; Jastrzębski 2018; Raghu 2021) and test whether sequential arms re-represent while parallel arms only sum.
3. **Effective rank and singular-value structure.** Closed-form chain-vs-parallel contrast available from Phase 0 item 4. Note that input–output complexity and embedding rank are different quantities; measure both.
4. **Branch symmetry and specialization.** Riskier, but makes a sharp falsifiable prediction.
5. **Induction-head emergence** (transformer extension).

Cheap secondary targets with open niches: progressive neural collapse in non-residual and dense nets, and skip topology vs grokking delay. Superposition counts stay deferred, because they are instrument-dependent.

## 7. Out of scope until a phase earns it

Recurrence, learned gates, LLM-scale runs and architecture search are all out of scope. The v0 guardrails on claim types (expressivity vs optimization vs mechanism; [archive charter](../archive/v0/docs/02_research_charter.md)) remain good practice and are summarized in [AGENTS.md](../AGENTS.md).
