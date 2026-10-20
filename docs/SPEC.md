# Active research spec

**Status:** active as of 19 October 2026, replacing the v0 plan in [`archive/v0/`](../archive/v0/). The v0 motivations are preserved in [00_questions_in_original_words.md](00_questions_in_original_words.md); the change log is in [history.md](history.md).

## 1. Question

> **How does parameter-free feed-forward connectivity shape which structures emerge during learning** — which features get preserved, erased or constructed, which computations get shared versus specialized, and which updates transfer versus interfere — **and through what mechanism?**

Constraints carried over from v0, all still binding for primary comparisons:
- same learned operators and parameter count;
- softmax cross-entropy;
- single-pass, acyclic graphs;
- no learned gates, no recurrence, no per-branch losses.

## 2. The causal chain we study

v0 framed topology effects as "pathways" and "circuits" and planned to discover them empirically. We instead study a chain in which every link has existing theory, so each link yields a prediction before any training:

```
topology
  ├─► function class            (expressivity: what can be represented at fixed size)
  └─► parameterization          (a path polynomial in the weights)
         └─► symmetries ─► conserved quantities ─► effective preconditioner on the learned function
                                                        └─► learning order (plateaus, which features first)
                                                        └─► implicit bias  (which solution among many)
                                                               └─► emergent structure
                                                                   (shared vs specialized, composite vs shortcut,
                                                                    coherent vs idiosyncratic updates)
```

Two consequences of this framing are proved in [phase0_linear_theory.md](phase0_linear_theory.md) and organize everything below.

1. **Topology effects in residual families are higher-order in branch weight.** Every graph in the parent-selection family $h_i=h_{i-1}+F_i(h_{k_i})$ shares the same first-order dynamics near zero branch weights (Prop. 1). Topology matters in proportion to how far learning moves the branches. The primary experimental axis is therefore **richness**:
   - initialization scale;
   - a fixed (parameter-free) branch multiplier $\alpha$;
   - the task's required distance from identity.

   Width is a secondary axis.
2. **"Productive frustration" is a saddle.** A loss that is stationary in the shallow directions while the deep route carries unresolved error at second order is a strict saddle (the v0 product-starvation toy is exactly this; see `test_product_path_starvation`). Plateau length, escape direction and hidden progress during the plateau are measurable, and there is theory for them (saddle-to-saddle dynamics, leap complexity). "Productive" means that escaping the saddle produces a feature that generalizes.

## 3. Rules for every study

1. **Directional, pre-registered prediction.** Each study states which graph should do what, by roughly how much, and why (citing a Phase 0/1 derivation or a named theory). "Some regimes will differ" is not a hypothesis.
2. **Per-arm learning-rate tuning.** Different topologies induce different preconditioners, so a shared learning rate is not a fair control. Sweep LR for each arm and report best-vs-best plus full LR curves. Optimizer type, schedule and batch order stay shared.
3. **Richness sweep.** Report every effect as a function of initialization scale and branch multiplier. A topology effect that only appears at one richness setting is a regime effect, not a principle.
4. **Over-provision width for expressivity.** For nonlinear comparisons, widen until every arm fits every target to a stated tolerance, and report the fit. Do not restrict to targets both arms can express at small width (that removes the phenomenon). Do not compare arms when one cannot fit.
5. **Two clocks.** Compare at matched optimizer steps *and* at matched training loss. They answer different questions: speed versus a different solution.
6. **Replication.** At least 5 paired seeds per arm, plus independent data seeds. Uncertainty is reported at the run level, not across correlated probes.
7. **Nulls.** Every study includes an identical-graph null (two copies of the same graph under different labels) and an operator-label permutation.
8. **Derive before you measure.** If a quantity has a closed form in the setting (as one-example transfer does in linear chains, Prop. 4), test the closed form; measure only the residual.

## 4. Study sequence

| Phase | Question | Directional prediction | Kill / pivot criterion |
|---|---|---|---|
| **0. Linear theory** | Conserved quantities, flows and implicit bias of the linear parent-selection family | Done for parallel and chain (Props. 1–6). Open: mixed graphs (Prop. 7), separable-data bias, imbalanced init | Simulation disagrees with derivation → bug or wrong derivation |
| **1. Gated linear networks** | Does fixed routing predict which pathways win, i.e. shared vs specialized representations? | Pathways active in more contexts win the race and carry shared structure (Saxe et al. 2022) | Race theory fails on routing variants with matched parameters |
| **2. Construction by composition** (flagship) | Does sequential routing make it easier to build new features from learned ones? | Sequential shortens the lag between learning constituents and learning their composite; parallel does not; the gap grows with richness | No lag difference at per-arm-tuned LR, or the difference vanishes across richness |
| **3. Reuse, interference, memorization** | Does topology change gradient coherence and where exceptions are stored? | Graphs whose effective kernel adapts show rising within-rule coherence; parallel/raw-input graphs stay flatter | Coherence and localization indistinguishable across topologies |
| **4. Preserve / erase** | How do topologies handle removing information after use? | Composed stacks erase more slowly than shortcut-rich graphs (Prop. 3, Prop. 7 linear evidence) | No measurable erasure-cost difference |
| **5. Scale** | Do Phase 2–4 effects survive principled scaling? | Effects persist under μP / depth-μP at fixed feature load | Effects are artifacts of untransferred hyperparameters |

Advancing a phase requires the previous phase's prediction to be *tested*. It does not need to be *confirmed*: a clean failed prediction is a result and changes the next prediction.

### Phase 0 — finish the linear theory (≈1–2 weeks)

**Done** (tested in `tests/`):
- first-order universality across the family;
- parallel graph equals one matrix;
- balanced-chain flow and mode-wise preconditioner;
- closed-form NTK, which makes one-example transfer a function of $W$;
- first single-example step equals a scalar rescale;
- non-separable endpoint equivalence (numerical).

**Open**, in priority order:
1. **Prop. 7, mixed graphs.** Chain-plus-shortcut graphs already reduce exactly to a coupled $(P,W_3)$ flow (proved and tested). Analyse that ODE and explain the per-mode exceptions seen in the [probe](../experiments/phase0/prop7_pathway_probe.py). For graphs that don't split into chains plus shortcuts, use the Marcotte–Gribonval–Peyré algorithm. This is the first candidate *novel* theory result. We found no published treatment of implicit bias for general linear residual DAGs; check further before claiming novelty.
2. **Imbalanced (small random) initialization.** Compare against Kunin et al. 2024 and Dominé et al. 2025.
3. **Separable-data directional limits** for parallel vs sequential with fixed readout.
4. **Robustness only:** finite-step drift, Adam, weight decay toward identity (shown to break balancedness), minibatch cross-terms.

**Deliverable.** A per-topology table: preconditioner, which modes are accelerated or slowed, and implicit bias. Its predictions feed Phase 2 and Phase 4.

### Phase 1 — gated deep linear networks as the analyzable bridge

Gated deep linear networks (GDLNs) route inputs through pathways selected by **fixed, non-learned gates**. Learning reduces to a race between pathways in which pathways shared across contexts win, yielding abstraction (Saxe, Sodhani & Lewallen 2022). Jarvis et al. 2025 show feature-learning ReLU networks behave like an equivalent GDLN. This is the most direct existing theory of "routing shapes emergent structure."

1. **Replicate** the neural-race headline result in our code: shared-pathway dominance and generalization to held-out context–item combinations.
2. **Vary gating topology at matched parameters.** Use the same weight matrices but different assignments of which contexts open which nodes. Predict, from race theory, which representation wins and how held-out generalization changes. Then train.
3. **Bridge.** Train small ReLU networks on the same tasks and test whether the GDLN predictions transfer, following Jarvis et al.

**Decision for the researcher:** fixed input-dependent gates add no parameters and no compute. Are they inside the "parameter-free routing" constraint? This spec uses them in Phase 1 only, labeled as analyzable surrogates. Primary nonlinear comparisons remain edge-only rewires.

### Phase 2 — construction by composition (flagship nonlinear study)

**Architecture.** Parent-selection family with ReLU MLP blocks:
- sequential $(0,1,\dots)$, parallel $(0,0,\dots)$, and one-edge mutants;
- width over-provisioned until every arm fits every target;
- a fixed branch multiplier $\alpha$ as the richness knob.

**Targets.** Boolean functions $y=h(g_1(x_{S_1}),g_2(x_{S_2}))$:
- the intermediates $g$ have low-order correlation with $y$, so they are learned first;
- the composite term has high leap complexity from raw inputs (Abbe, Boix-Adserà & Misiakiewicz 2023).

**Hypothesis H2 (directional).** After $g_1,g_2$ are represented, the sequential graph acquires the composite term faster than the parallel graph, because later blocks see the intermediates as inputs and the composite has low leap *relative to them*. The size of the effect grows with richness and is ~0 in the lazy regime (Prop. 1 logic).

**Controls:**
- a pure high-order parity with no learnable constituents (prediction: no difference);
- a sum of independent features with no composition (prediction: parallel ≥ sequential);
- a matched-loss comparison;
- the richness sweep.

**Measurements:**
- full Walsh–Fourier atlas of centered logits over the input domain at every checkpoint;
- emergence time of each coefficient;
- **lag** between emergence of the intermediate coefficients and the composite coefficient (primary endpoint);
- plateau lengths;
- hidden-progress measures during plateaus (Barak et al. 2022);
- local learning coefficient (LLC) as a source-agnostic stage detector (Hoogland et al. 2024).

**Falsifiers.** No lag difference at per-arm-tuned LR; a difference that does not scale with richness; or a difference explained by fitting floor or learning rate.

### Phase 3 — reuse, interference, memorization

**Setups:**
1. Two tasks sharing a subcomputation ($f\circ g$, $h\circ g$): measure transfer from training on one to the other, and module specialization (Jarvis et al. 2023).
2. Rule-plus-exceptions with a correlated shortcut (v0 task generator, [archive doc 07 §C](../archive/v0/docs/07_datasets_and_scale.md)).
3. Sequential task switch: measure interference and forgetting.

**Measurements:**
- per-example gradient coherence and stiffness over training (Chatterjee 2020; Fort et al. 2019);
- the counterfactual one-step transfer matrix $T(i,j)$ from v0, now validated against its closed form in the linear case;
- how localized exception storage is (Maini et al. 2023).

**H3 (directional).** Graphs whose effective kernel adapts (sequential) show coherence among same-rule examples rising faster over training than raw-input graphs do. Exception storage is not predicted yet: state a prediction after Phase 2.

### Phase 4 — preserve and erase

**Tasks** where a feature must affect the output early and then be removed from the representation ("use then erase"). Measure erasure lag and residual leakage of the erased feature (linear probe *and* causal patching).

**Prediction** from Props. 3 and 7: composed stacks erase more slowly than graphs with shortcut paths.

**Then tiny transformers.** Sequential vs parallel attention/MLP blocks. This is a real-world parallel-vs-composed topology; PaLM reports a scale-dependent quality cost for the parallel form. Read the result through the residual-stream read/write framework (Elhage et al. 2021). Validate any circuit tooling on Tracr or InterpBench models with known ground truth before using it.

### Phase 5 — scale

Use μP / depth-μP (Yang et al.; Bordelon et al. 2024) so hyperparameters transfer. Vary separately:
- width at fixed latent feature load;
- feature load at fixed width;
- data support.

Two sizes do not make a scaling claim.

## 5. Instruments (use existing, validated ones first)

| Need | Instrument | Notes |
|---|---|---|
| What function was learned | Walsh–Fourier / ANOVA atlas of centered logits | finite domains; v0 doc 06 §2 still applies |
| When structure forms, without a template | LLC and refined LLC (devinterp) | detects developmental stages |
| Progress during plateaus | hidden-progress measures | Barak et al. 2022 |
| Example-to-example transfer | gradient coherence, stiffness, counterfactual $T(i,j)$ | check against closed form where one exists |
| Parameterization geometry | conserved-quantity residuals (e.g. balancedness), preconditioner spectra | Phase 0 code |
| Causal localization (Phase 4+) | activation patching with multiple baselines, validated on Tracr / InterpBench | v0 doc 06 §§4–5 caveats apply |

**Deferred indefinitely:** retained-subgraph causal-depth scores $D_k,U_k$ (v0), SAE pipelines, architecture search.

## 6. Out of scope until a phase earns it

Recurrence, learned gates, LLM-scale runs and architecture search are all out of scope. The v0 guardrails on claim types (expressivity vs optimization vs mechanism; [archive charter](../archive/v0/docs/02_research_charter.md)) remain good practice and are summarized in [AGENTS.md](../AGENTS.md).
