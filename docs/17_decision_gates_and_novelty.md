# Decision gates, novelty boundary, and when to stop

**Status: active study-order and decision policy (19 October 2026).** This supersedes any earlier impression that a large interpretability framework or architecture search must be built as the next milestone.

## 1. Research principle

We are not guaranteed that fixed skip topology is a useful independent driver of generalizable circuit formation. A meaningful scientific program can either:
- identify a topology property that predicts an acquired functional mechanism under known conditions, **or**
- establish that the effect is explainable by capacity/initialization/optimization/data and specify the boundary where topology offers no additional explanation.

Avoid retrofitting “productive frustration” to every outcome. Commit to a specific hypothesis and at least one plausible alternative explanation **before running** a discriminating experiment.

## 2. Five-stage decision ladder

### Gate 0 — Algebra and implementation audit

**Objective:** exact fixtures: same params and learned evaluations; graph acyclicity; shape and edge counts; identical logits/Jacobian at intended matched starting point; same first-step gradient for both models; $BAx$ second-order effect; Hessian difference accounted for.

**Go if:** symbolic, finite-difference and autograd checks pass; computational budget and source of difference are independently verified.

**Pause/revise if:** graph pairs quietly change learned capacity, activation count, output head, or cannot reproduce analytic equality. **Do not train a larger benchmark to hide the bug.**

### Gate 1 — Is there a measurable topology-dependent learning effect?

**Objective:** train the matched linear pair from [Study 1](15_matched_linear_topology_study.md) with softmax CE, fixed readout, controlled linear teacher modes; compare full learned functions, mode evolution and one-example transfer under paired seeds.

**Go if:** a repeatable difference emerges beyond implementation/numerical noise, and its sign/shape can be tied to the graph's higher-order interaction and quantified under multiple task conditions.

**Revise if:** effect disappears when initialization, output scale, learning rate or optimizer are controlled. That indicates a narrow parameterization regime, not a robust principle.

**Stop *this branch* if:** difference is trivial, entirely reducible to a simple known rescaling/regularization effect without useful new predictions, or only arises from accidental unfair matching. Preserve the negative report and choose a new graph contrast only with a distinct falsifiable hypothesis.

**Explicit caveat:** this is linear function learning, not evidence about semantic circuits or memorization/generalization division.

### Gate 2 — Can it change nonlinear function acquisition without mere expressivity advantage?

**Objective:** choose small nonlinear operator pairs whose target task is constructively realizable by both at the experimental width; track complete output functions on a finite probe space, exact example transfer and broad latent-rule generalization.

**Go if:** a reproducible, nontrivial functional interaction emerges differently under a routing mutation **after accounting for unequal approximation capacity and optimization budget**.

**Revise if:** one graph cannot fit the target even with favorable optimization/training. This is an expressivity study, not an optimization/circuit study; report it separately.

**Stop *this branch* if:** topology only predicts lower CE on training, but no stable difference in learned function and transfer across task seeds; don't invest in massive causal extraction on an unvalidated phenomenon.

### Gate 3 — Is the discovered computation causally attributable, including acquisition?

**Objective:** target a detected functional change; use small-graph causal interventions, faithful behavior reproduction, alternative patch baselines and joint lesions; distinguish inference-time route use from gradient-route importance during training.

**Go if:** at least some observed functional changes have stable causal support and the manipulated topology edge changes mechanism acquisition or implementation as predicted.

**Revise if:** apparent circuits disappear or reverse under small intervention changes, or explanation residuals remain large. De-emphasize canonical depth/count claims, record uncertainty, improve instrument only as needed.

**Stop *mechanistic attribution claim* if:** no robust extraction can be found. Functional learning differences may still be real and publishable without a complete circuit ontology.

### Gate 4 — Does the relationship persist under changes in scale and data?

**Objective:** factorial *subset* varying width at fixed feature demand, feature demand at fixed width, both jointly, sparsity, task complexity, data support, optimizer. Use new seeds and hold out entire generator families.

**Go if:** the graph property predicts a measured learning mechanism or transfer pattern beyond a single model/data regime, with stated boundary conditions.

**Revise if:** strong dependence on feature load or width appears; describe a phase region rather than a universal claim.

**Stop *universal design claim* if:** effect is idiosyncratic to the discovery family or reverses unpredictably. Do not initiate NAS or LLM-scale engineering based on it.

### Gate 5 — Only then automate discovery/search or move toward attention

**Objective:** test whether validated structural descriptors predict out-of-sample mechanistic outcomes in a broader graph grammar and small attention network.

**Go if:** a surrogate or search objective has out-of-sample predictive validity for functional/circuit measures, with random/local search baseline.

**Stop if:** search consistently optimizes early CE or path counts without improving the scientifically targeted dynamics. Do not let validation benchmark wins override mechanistic falsifications.

## 3. Novelty and literature replication plan

**Before claiming novelty** compare an observed finding to:
- [Saxe, McClelland & Ganguli (2014)](https://arxiv.org/abs/1312.6120), exact deep linear mode-learning dynamics.
- [Shi, Shea-Brown & Buice (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html), common-loss competing pathways and feature specialization.
- [Qiu, Kuang & Goel (ICML 2024)](https://proceedings.mlr.press/v235/qiu24e.html), spurious/core complexity, phase overlap and separate subnetworks.
- [Xu & Ziyin (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9941833e8327910ef25daeb9005e4748-Abstract-Conference.html), exact finite-width kernel-vs-feature modes.
- [Chen, Luo & Pan (ICML 2026)](https://proceedings.mlr.press/v306/chen26cv.html), training-origin attribution for interpretable units.
- [Miller et al. (2024)](https://arxiv.org/abs/2407.08734), causal circuit faithfulness depends on ablation method.
- [Anani et al. (ICML 2026)](https://proceedings.mlr.press/v306/anani26a.html) and [Hadad et al. (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html), circumscribed stability/minimality guarantees for selected circuit-discovery settings.

**Replication hierarchy:**
1. reproduce the relevant mathematical/empirical observation, in a compatible setting;
2. establish what changes *only* when a fixed edge is rewired;
3. show an outcome not already predicted by pathway depth, expressivity, kernel conditioning, weight scaling or task statistics;
4. confirm the effect on a fresh task family and scale regime;
5. only then make a topology design principle.

A replicated existing result with better controls or a meaningful negative boundary remains worthwhile. Do not relabel it original.

## 4. Report matrix: separate kinds of contribution

Each paper/report must include:
- **Expressivity:** target realizability for the compared graphs or explicitly unresolved competence mismatch.
- **Optimization:** exact initial matching and the parameterization differences left after matching.
- **Function:** complete/held-out temporal output changes, mode emergence and reuse.
- **Causal mechanism:** what was actually intervened on; acquisition vs expression; coverage and alternate baselines.
- **Generalization:** which new inputs/task families support the conclusion.
- **Resources:** trainable params, learned-op evals, fixed arithmetic, train steps/tokens and wall time.
- **Uncertainty:** independent seeds, task seeds, selection bias, and effect-size intervals.
- **Negative regimes:** task/scale/optimizer variants where the claim fails.

Do not collapse these into a single score.

## 5. Stop, pivot, or continue as a scientific decision

**Continue:** new observation is stable and discriminates against plausible alternative mechanisms; next experiment follows an articulated falsifiable prediction.

**Pivot:** measurement isn't identifiable, training regime is too simple/too capacity-limited, or expressive competence isn't matched. Change *one control* and document why.

**Stop a line:** repeated nulls across well-powered experiments or effects reducible to known trivially parameterized geometry, without new mechanistic predictions. Record closure; don't broaden the search hoping to find a positive.

**Keep exploring** even after negative results if they falsify an important motivating belief (e.g. all short paths necessarily starve deep learning) and lead to a simpler testable principle.

## 6. Not yet success criteria

The following are **not** sufficient to cross Gate 2–5:
- faster train CE or better benchmark accuracy alone;
- more high-order Fourier terms (could be gain or parameterization);
- larger Jacobian/gradient norm;
- path count or polynomial degree without causal output effect;
- a convincing SAE visualization with unknown reconstruction/ablation fidelity;
- a single ablation ranking with no baseline sensitivity;
- interpretability that works on one known algorithm but misses unexpected functions;
- one width, one data seed or one optimizer.

## 7. Engineering boundary

Phase 0–1 requires small reproducible graph compiler, CE trainer, exact functional observation and per-example update-transfer. **Do not build** a general NAS controller, an SAE training farm, or a complete automated circuit-tracing framework. Introduce tools only to test a hypothesis already supported by lower-level evidence.

## 8. Documentation policy

[History](01_intellectual_history.md) and [original questions](00_questions_in_original_words.md) remain historical. Append new dated decisions to [decision record](10_decision_log_and_hypotheses.md), mark earlier plans explicitly as superseded, and link active programs from README and AGENTS. Never quietly rewrite past deliberation to imply the conclusion was settled from the start.
