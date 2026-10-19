# Mechanistic measurement framework: discover first, explain causally second

This is the central scientific instrumentation specification. **The measured outcome is not chiefly CE or accuracy.** Those are training signals and validity checks. We seek topology-dependent differences in emergent computations, memorization, generalization, interference, and circuit acquisition.

## 1. Observe multiple levels, rather than insisting on a neuron-level ontology

**Functional level:** changes in input-output function over a stated probe distribution, canonically represented to remove softmax gauge freedom.

**Representation level:** dimensionality, subspaces, feature interference and evolution; optional SAEs/feature dictionaries with acknowledged nonuniqueness.

**Causal graph level:** necessary/sufficient or interacting subgraphs under specified inference-time interventions.

**Learning level:** which routes and updates support *acquisition* of behaviors over training, distinguished from routes merely needed to *express* them.

**Outcome level:** memorization of arbitrary observed exceptions, generalization to unseen rule instances, robust transfer under broken shortcuts, and effects of single-example updates.

An extracted object at one level is not automatically a circuit at another. For example, a Fourier coefficient of the output function is not a hidden path, and an SAE feature is not automatically causal.

## 2. Functional atlas before hypothesis-directed probes

For tiny finite domains, exhaustively evaluate $z_t(x)$ at each checkpoint. Use centered logits $\tilde z=z-\mathrm{mean}_j z_j$ or logit differences to remove softmax common-shift ambiguity. Track $p_t(x)$ and CE as complementary functions; don't infer unseen-input behavior from train metrics.

For balanced Boolean inputs and a **scalar logit contrast** $m_t(x)$:

$$m_t(x)=\sum_{S\subseteq[n]}\hat m_t(S)\prod_{i\in S}x_i,\quad
\hat m_t(S)=2^{-n}\sum_xm_t(x)\prod_{i\in S}x_i.$$

This spans all interaction orders without deciding which circuit to look for. Track:
- energy per input-interaction order $\sum_{|S|=k}\hat m_t(S)^2$;
- temporal appearance/movement of coefficients;
- novel basis components vs those present at initialization;
- effects by feature subset, including interactions not specified by the task generator;
- coefficient stability across seeds and shifts in probe measure.

Walsh–Fourier spectra depend on input encoding and measure. For correlated feature distributions, uniform-domain Fourier coefficients and empirical-data ANOVA are different quantities. Record both when informative. General categorical domains can use exhaustive contrast matrices, orthogonal polynomial/basis decompositions or functional ANOVA, with documented interaction of factors.

Functional novelty is not necessarily true algorithmic novelty. A change in logits could be a simple gain/temperature effect. Normalize/compare spectral *shape* and use causal checks.

For larger domains, use stratified fixed probe sets with held-out probes, uncertainty and sketching; do not pretend samples are exhaustive.

## 3. Source-agnostic behavior discovery

Instead of preselecting 'parity circuit at depth 2':
1. Identify transitions in the functional atlas across checkpoints (e.g. emerging interaction clusters, changed generalization on families of inputs, nontrivial transfer groupings).
2. Cluster inputs by **changes in logits/one-example influence**, not by only hand-assigned semantics.
3. Generate candidate behavioral contrasts automatically.
4. Select diverse contrasts for causal mapping; retain a blind or held-out pool to test whether discovery overfit.
5. Reveal the synthetic generator's latent truth only after the mechanism-discovery pass, to audit which effects were missed.

This is not fully assumption-free: input representation, intervention operations, selected distributions and contrast metrics define what can be discovered. Include an 'unknown/underexplained' category, and never equate detector failure to absence of computation.

## 4. Long-path learning: exact and operational profiles

### A. Exact linear expansion

For $h_L=(I+W_L)\cdots(I+W_1)x$, expand into sums of products involving 0 through $L$ learned operators. This **exactly** decomposes by learned-operator degree *in linear residual networks*. Track each order's change through training and cancellations, and compare against symbolic oracles.

But high product degree does not imply nonlinear functional complexity; do not use as the sole circuit measure.

### B. Causal depth profile for nonlinear trained graphs

Define a set $\mathcal C_k(G)$ of allowable **retained subgraphs** of a fixed trained graph $G$, such that every retained input-output route contains at most $k$ learned transformations. Specify subgraph semantics and *how severed edges are replaced* (zero, conditional mean, resampling, or other controlled source). **Do not retrain** the candidate retained subgraphs for this metric.

For probe distribution $Q$, define:

$$D_k^{(Q)}(G)=\min_{C\in\mathcal C_k(G)}
\mathbb E_{x\sim Q} D_{\rm KL}\!\left(p_G(\cdot|x)\|p_C(\cdot|x)\right).$$

This measures approximate depth needed to **reproduce the trained function**, not correctness or the unique true circuit. Separately define

$$U_k^{(Q)}(G)=
\min_{C\in\mathcal C_k(G)}\mathbb E_{(x,y)\sim Q}[-\log p_C(y|x)]
-
\mathbb E_{(x,y)\sim Q}[-\log p_G(y|x)].$$

Positive $U$ says the *best* retained depth-$k$ subgraph underperforms the full network on $Q$; negative indicates a subgraph improves on it. Evaluate on seen arbitrary exceptions, unseen latent-rule cases, shortcut-broken cases and broad unbiased probes separately.

**Mathematical detail:** when $\mathcal C_k\subseteq\mathcal C_{k+1}$ with fixed intervention conventions and exact minima, $D_k$ and $U_k$ are nonincreasing in $k$; $D_k\ge0$. Empirical heuristic subgraph search may show apparent nonmonotonicity from search error, which is not a true property of the defined minimum. Use exact enumeration for tiny graphs to validate heuristics.

**Crucial limitations:** subgraph outputs under ablation may be off-distribution; retained-subgraph class may favor some topologies; skip additions and nonlinear activations make attribution non-additive; sharing/redundancy complicates identifying a unique minimum; downstream layer normalization may change after masking. Report baseline sensitivity, coverage (how many subgraphs searched), and forward fidelity.

A no-intervention **functional composition lower bound** or known theoretical expressivity limit can complement $D_k$, but do not present a subgraph search result as a fundamental minimum circuit depth without assumptions.

### C. Contribution to *acquisition* versus expression

An inference-time removal tests expression. To test acquisition:
- clone a checkpoint, replay identical data/optimizer state;
- selectively mask gradients through a specified edge or operation *during training only*, with a sham-control of matching arithmetic if possible;
- later compare functional atlas and causal profiles;
- document changed learning trajectory and compensation; this intervention violates a normal model train but is *diagnostic*.
Other options: train from identical seeds with an edge removed from the start, or surgically freeze subset updates, always distinguish altered optimization from fixed-graph inference.

A component with little inference-time ablation effect may still have assisted acquisition earlier. Conversely a vital output path may not have formed the underlying rule.

## 5. Necessity, sufficiency, synergy and redundancy

For trained graph $G$ on $Q$, let $L_G$ be expected target CE. For a removed set $S$ of graph components with a defined ablation baseline, $A(S)=L_{G\setminus S}-L_G$.

- **Necessity of S:** large positive $A(S)$ under multiple plausible interventions.
- **Sufficiency of S:** keeping S and ablating other components preserves behavior relative to full G.
- **Interaction:** $I(A,B)=A(A\cup B)-A(A)-A(B)$; strong positive interaction may signal redundancy/compensation when two removals together are more harmful than the sum of each.
- **Synergy:** some routes only work jointly, which can yield other sign patterns. Do not label sign mechanically without checking joint behavior.
- **Negative ablation effects:** a part can hurt the objective under a particular distribution; removing it may help. Do not discard such findings.

Single-edge ablation can fail to reveal redundant circuits. Use joint interventions, Shapley-style attribution on very small graphs, and matched fidelity/error propagation. In larger graphs estimate subsets, not all $2^E$ possibilities.

No ablation is guaranteed causal under arbitrary off-distribution replacements. Compare several patch baselines and use counterfactual activation interchange when possible.

## 6. Representational geometry and superposition

No default assumption that circuits are single neurons. A useful representation may be distributed, rotated, sparse, superposed, or shared among behaviors.

Track per checkpoint:
- centered activation covariance eigenvalues, singular values;
- effective dimension $d_{\rm eff}=(\mathrm{tr}\Sigma)^2/\mathrm{tr}(\Sigma^2)$ (participation ratio), which is **not feature count**;
- CKA (orthogonal rotation and isotropic-scaling invariant, not general-invertible invariant), subspace overlap/CCA where well-conditioned and appropriate;
- feature sparsity, co-activation and correlation on controlled latent factors;
- reconstruction of known features from trained representation, with held-out fit;
- optional SAEs at multiple dictionary sizes, with reconstruction and intervention-fidelity checks.

A general invertible change of basis at a compatible linear interface can leave model function unchanged when adjacent weights are adjusted. Test whether the interpreter reports spurious circuit shifts for such equivalent models. Elementwise nonlinear layers are not freely invariant under arbitrary basis transformations: only apply symmetry tests where valid.

Superposition depends not just on width, but on feature count, importance, sparsity, co-occurrence, competition and task distribution. Vary these explicitly. SAEs do not provide a canonical unit count; instrument-induced splits, merges and missed inhibitory/inactive features are expected hazards.

## 7. Time-resolved matching of mechanisms

Save fixed probe inputs and checkpoints; measure functional fingerprints of each extracted subgraph: vector of **logit/probability changes under intervention** across probes. Compare fingerprints across time and seeds using distribution-normalized similarity, not exact neuron IDs.

Report birth/strengthening/relocation/merging/redundancy changes cautiously: attribution graph construction itself may be unstable. Track uncertainty by resampling probes and intervention controls.

Crucial: a new high-order functional interaction need not have arisen from a new long graph path; a deep causal path can contribute to a low-order rule. Only infer linking after causally mapping both.

## 8. Residual access and one-example measurements

See docs/04_gradient_dynamics.md. Include in longitudinal record:
- shallow functional projection of CE residual and separately deep incremental projection, regularized for finite update magnitude;
- actual one-example SGD update effects on identical inputs, semantically related unseen inputs, contradictory shortcut cases and previously memorized exceptions;
- change in train/test functional atlas coefficients after cloned update;
- gradient norm, alignment and projection are explanatory, not primary discovery targets.

**Watch for pathological overparameterization:** Jacobian columns may span all train-logit residual directions yet not yield a useful finite update or generalize. Use ridge/bounded-step estimates and OOD probes.

## 9. Validation of the *instrument* before interpreting new graphs

Build a calibration suite with planted:
- a linear known depth-$k$ path;
- a two-step nonlinear composition;
- a short shortcut and true interaction;
- two redundant mechanisms (single-edge lesion should fail, joint lesion should work);
- deliberately superposed sparse features;
- exact change-of-basis-equivalent representations;
- a circuit whose activation is large but causal utility is zero;
- a mechanism not in the 'expected' feature dictionary;
- a strongly interfering example update.

Measure recall/precision of discovered behaviors or subgraphs where ground truth is available; false positives and unidentifiable cases matter. Assess sensitivity to probe distribution, checkpoint selection, ablation baseline, graph search budget, SAE size and random seed.

Only promote a causal-depth or circuit count trend to an architectural claim if the instrument succeeds on relevant calibration cases.

## 10. Reporting templates

For every topology × seed × task × scale, produce:
1. **Functional evolution:** logit-contrast spectra, newly appearing interactions, rule/exception changes.
2. **Causal-depth reproduction/utility:** $D_k$ and $U_k$ by evaluation distribution, plus searches/baselines/coverage.
3. **Mechanism atlas:** candidate subgraphs, causal fingerprint, necessity/sufficiency/redundancy with uncertainty.
4. **Learning dependencies:** what training-time route interventions changed about circuit acquisition.
5. **Geometry:** effective dimension, spectra, sparsity, basis/SAE robustness, feature demand.
6. **Dynamics:** gradient access, example transfer/interference and optimizer/time comparisons.
7. **Unexplained behavior fraction:** where extracted circuits do not adequately reproduce the trained model.

Do not compress all of this to a single weighted score. The goal is a *mechanistic history linked to the topology mutation*.
