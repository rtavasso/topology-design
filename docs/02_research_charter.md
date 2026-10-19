# Research charter: scope, object, and standards of evidence

## Core research question

Can **parameter-free information routing in feed-forward computational graphs** favor stable preservation of useful representations, controlled destruction of inadequate ones, and construction of novel reusable mechanisms, **while keeping unresolved errors accessible to the computations capable of resolving them**?

Our ultimate goal is a predictive understanding of how topology controls:
- which mechanisms form during learning;
- their representational distribution (modular, overlapping, superposed);
- how training examples change behavior, including retention, interference and transfer;
- when learned behavior relies on memorization versus structure;
- how these properties vary with model capacity, feature demand, task complexity and data;
- whether we can ultimately allocate operations/parameters/training by function more deliberately.

**We are not trying to discover a universal best model architecture in one experiment.**

## Terms and distinctions

**Topology:** directed acyclic connections among fixed learned operators and parameter-free transforms (identity, addition, optional fixed projections only when carefully controlled). Always specify whether the compared graph changes arithmetic count or output normalization.

**Learned-operation depth:** number of learned transformations along a path. **Nonlinear compositional depth:** sequential nonlinear transformations on which an output causally depends. These are not interchangeable. **Parameter-interaction degree:** minimum/highest polynomial degree in trainable weights for linear graphs, likewise different.

**Memorization:** fitting information about observed training instances that does not transfer to relevant unseen instances, operationalized using randomized exception mappings and matched rule/exception splits. Memorization is not necessarily local, shallow, or isolated in parameter space.

**Generalization:** correct behavior on held-out combinations/rules with controlled data-generating factors. No generalization metric is completely prior-free; state its distributional assumptions.

**Circuit:** an operationally defined causally involved subgraph or subspace of computations with explicit intervention boundaries and tolerance for behavior reproduction. An extracted circuit need not be unique, semantic, neuron aligned or consistent across inputs.

**Circuit formation:** detectable emergence/evolution of causally relevant computations during training, distinct from merely decoding features that were already present or observing lower CE. Must connect longitudinal functional changes to interventions when feasible.

**Utility:** causal contribution to a *specified* behavior and evaluation distribution, not activation magnitude, gradient size, or output contribution alone.

**Frustration:** an exploratory term for unresolved prediction error beyond the shallow model's current effective functional directions that remains usefully accessible to developing more expressive computations. Avoid anthropomorphic interpretations; test whether it predicts novel generalizable circuits.

**Feature load / representational pressure:** a family of experimentally controlled latent feature count, sparsity, correlations and importance relative to width/capacity, not one universal measured scalar.

## Research constraints

### Primary comparisons
1. Same softmax CE over the same token vocabulary; binary prediction uses a 2-logit softmax.
2. Same count and parameterization of learned operations; hold activations, widths, output heads, optimizer and datasets fixed unless a controlled factorial study varies them.
3. Acyclic single-pass execution. No recurrent inference.
4. No extra trainable gates, per-path losses, distillation, hand-specified token groups or semantic routing targets as *candidate architectures*.
5. Measured differences must be caused by **parameter-free topology**, not a changed param budget, unrecorded optimizer tweak, or extra learned-operation evaluation.
6. Record differences in fixed operations, activation memory, runtime and critical path; they are real resource costs.
7. Keep instrumentation out of the model during ordinary training unless testing a clearly labeled training intervention.

### Diagnostic/secondary controls
May include: plain networks, learned gates, branch-specific supervision, artificially fixed features, grouped conditional CE, gradient detach, synthetic planted circuits, or retrained pruned models. Label these as interventions outside the pure-topology comparison. They can test causal hypotheses, not establish a strictly parameter-free architectural advantage.

## Evidence hierarchy

**Tier 0 — algebraic invariant:** exact for graph class; show derivation and assumptions, symbolic/numeric tests.

**Tier 1 — verified implementation consequence:** forward, backward, Hessian and path-count tests reproduce predicted behavior.

**Tier 2 — measured association:** paired topology runs consistently differ on a behavioral/mechanistic statistic across seeds.

**Tier 3 — mechanistic intervention:** targeted graph/activation/gradient intervention changes an identified computation as predicted, survives different baselines and redundancy controls.

**Tier 4 — generalizable design principle:** effect replicates across widths, feature loads, data regimes, task families and optimizers, with interpretable boundary conditions.

No tier can be skipped in a published claim. Tier 4 requires reporting negative regimes.

## Primary scientific measurements

1. **Residual accessibility:** CE residual alignment with shallow and deep local trainable directions, with realistic parameter-step/conditioning regularization; access is not creation or utility.
2. **One-example transfer/interference:** actual loss/logit changes on labeled, semantically related/unrelated and shortcut-conflicting probes after one update; paired with gradient approximations.
3. **Mechanistic emergence/utility:** time-varying functional atlas, discovered subgraphs, causal depth/necessity/sufficiency/redundancy, rule-vs-exception distributions.
4. **Representation geometry:** spectra, effective dimensionality, sharing, interference, basis-robust comparison and SAE resolution sensitivity where appropriate.
5. **Resource and scale relationships:** data/compute to acquire known mechanisms, model width and feature load, nonlinear complexity, shortcut reliability, training time.

Aggregate CE/accuracy are mandatory validity signals but cannot substitute for the mechanistic outcomes.

## What would falsify motivating intuitions?

- Matching Jacobian conditioning and initialization entirely erases claimed routing differences.
- Distinct graphs have different path counts but indistinguishable functional mechanisms and intervention profiles.
- Long computational routes become important only for memorization or do not contribute to generalization.
- Functional interaction discoveries disappear when probe distribution or logit normalization changes.
- 'Circuit formation' is an artifact of SAE dictionary width, ablation baseline or broken subgraph accounting.
- The routing effect reverses at width or data regimes after feature load is controlled.
- Larger apparent residual accessibility fails to yield actual training improvements or transfer.
- An edge change impacts accuracy but no measured mechanism changes after adequate measurement validation (could mean incomplete instrument, not automatic absence of mechanism).

## Non-goals and deferred work

- No claim about matching human/LLM token efficiency from tiny synthetic tasks. Humans differ in priors, multimodal grounding, active interaction, objectives, working memory and in-context learning; topology is only one possible contributing factor.
- No LLM-scale or large search effort until validation instrumentation exists.
- No reliance on single neuron labels, linear probes, one SAE, or a single gradient-landscape visualization as 'the mechanism'.
- No assumption all computations admit a clean path/feature attribution.
- No forced decomposition into memorization and generalization subnetworks.
- No imposed token partition or branch independence as the presumed solution.

## Deliverables for an eventual scientific report

A credible result should supply: an explicit graph mutation and matched control; formal predicted consequence; independent measured dynamics; a causally confirmed behavior-related mechanism; per-example and OOD distribution effects; conditioning and measurement-confound controls; a scale/task regime map; negative cases; and reproducible configs/raw data.

**Preferred first result:** demonstrate a reliable, causally validated link between one parameter-free routing choice and emergence/utility of a computation, even on a toy model. That is more valuable than any initial benchmark win.
