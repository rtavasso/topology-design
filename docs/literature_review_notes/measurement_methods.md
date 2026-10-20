# Measurement Methods for Emergent Structure, Valid Across Network Topologies

Scope: methods for describing the structure a trained or training network converges to, judged by whether they still mean something when the compared networks have different connectivity graphs (residual vs plain, parallel vs composed, dense routing) at matched parameter count. The brief listed items that are already known (CKA, LLC, Barak progress measures, gradient coherence, stiffness, Tracr, InterpBench, patching best practices, Miller 2024, Leask 2025). They are mentioned only where a newer critique or extension changes how to use them.

Shared vocabulary used throughout:

- **"Cross-topology valid":** the measure can be computed on objects that exist in both networks, such as the input-output function, a chosen cut point's activations, or the training data. It does not depend on a correspondence between parameters or units that the topology change destroys.
- **Invariance classes for representation measures** (Klabunde et al. hierarchy): permutation ⊂ orthogonal ⊂ invertible-linear; plus isotropic scaling, translation, and affine.

---

## Q1. Function-space measures (parameterization-independent)

### Takeaway
Function-space measures are the only family that is valid across topologies by construction, because they never touch the parameterization. The three families covered here are output and logit agreement, spectral (Fourier/Walsh) and functional-ANOVA decompositions, and behavioural probes. Their weaknesses are a different kind of artifact:
- dependence on the input distribution;
- metric choice (discontinuous metrics create spurious "emergence");
- a many-to-one map from mechanism to function, so identical functions can hide different structure.

They should be the primary outcome variables. They cannot on their own justify claims about internal structure.

### Cited Findings
- **Klabunde et al.'s survey taxonomy of functional measures.** The survey splits functional measures into five groups:
  - performance-based;
  - hard-prediction (e.g., disagreement);
  - soft-prediction (e.g., JSD between output distributions);
  - gradient/adversarial-example-based;
  - stitching-based.

  It states that "both perspectives [functional and representational] only provide a partial view on neural network similarity." Its functional measures focus on multi-class classification. — [Klabunde et al., arXiv 2305.06329 v3 2024; ACM Computing Surveys 2025, doi 10.1145/3728458](https://arxiv.org/html/2305.06329v3)
- **ReSi grounds representational measures against functional ones.** It does so by "grounding by prediction": correlating representational similarity with accuracy differences, prediction disagreement, and output JSD. This makes output agreement the de facto ground truth for benchmarking. — [ReSi, Klabunde et al., ICLR 2025](https://arxiv.org/html/2408.00531v2)
- **Equal accuracy does not mean equal function.** Wide and deep networks with similar overall accuracy "make distinctive errors and show different patterns across classes." Per-example disagreement detects architecture effects that aggregate accuracy hides. — [Nguyen, Raghu, Kornblith, ICLR 2021](https://arxiv.org/abs/2010.15327v2)
- **Metric choice can manufacture apparent emergence.** Nonlinear or discontinuous metrics (exact-match accuracy) produce sharp "emergent" transitions. Continuous metrics (token edit distance, Brier/log-likelihood) applied to the same models show smooth change. The paper title is "Are Emergent Abilities of LLMs a Mirage?" — [Schaeffer, Miranda, Koyejo, NeurIPS 2023](https://arxiv.org/abs/2304.15004v1)
- **Spectral bias as a function-space structural descriptor.** Networks fit low-frequency Fourier components of the target function first. This gives a function-space way to describe what a network has "converged toward" that is independent of the parameterization. — [Rahaman et al., ICML 2019](https://proceedings.mlr.press/v97/rahaman19a)
- **Walsh-Hadamard (Boolean Fourier) analysis.** It quantifies the low-degree spectral bias of networks on discrete inputs, and that spectrum can be targeted with a scalable Walsh-Hadamard regularizer. The degree profile of the Walsh spectrum is therefore a measurable, parameterization-free summary of the learned function. — [Gorji et al., UAI 2023](https://arxiv.org/abs/2305.09779v1)
- **Functional ANOVA needs a purification step.** Additive or interaction decompositions of a learned function f are not unique unless an identifiability constraint is imposed. Lengerich et al. give a "purification" algorithm that moves mass from interaction terms into lower-order terms, so the decomposition into main effects and pure interactions is identifiable. Without it, interaction strengths are artifacts of the fitting procedure. — [Lengerich, Tan, Chang, Hooker, Caruana, AISTATS 2020](https://arxiv.org/abs/1911.04974v2)
- **Behaviour can split by algorithm even at matched function.** On modular addition, small changes to hyperparameters and initialization lead networks to learn different algorithms:
  - the "Clock" algorithm;
  - the "Pizza" algorithm;
  - parallel mixtures of several algorithms.

  So two networks can reach the same input-output function with qualitatively different internal mechanisms. Telling them apart needed targeted function-space probes beyond accuracy (behavioural signatures specific to each algorithm). — [Zhong, Liu, Tegmark, Andreas, NeurIPS 2023](https://arxiv.org/abs/2306.17844)
- **Influence-based generalization profiles are a behavioural probe.** In large LMs, influence "decays to near-zero when the order of key phrases is flipped." This kind of directional behaviour is not visible from aggregate loss. — [Grosse et al. 2023](https://arxiv.org/abs/2308.03296)

### Inferences
- **Invariances.** Function-space measures are invariant to every reparameterization, including unit permutations, rescaling symmetries (Dinh-type), and changes in the computational graph. They are *not* invariant to the following:
  - **The evaluation input distribution.** Fourier/Walsh and ANOVA terms are defined relative to an input measure. Uniform vs data distribution gives different decompositions.
  - **Output parameterization.** Comparing logits vs probabilities vs argmax gives different answers. Logits also carry a per-example additive shift freedom under softmax, so compare centered logits or log-probabilities.
  - **Metric discontinuity** (Schaeffer 2023).
- **Recommended practice for cross-topology work.**
  - Report soft agreement (JSD or KL on held-out data) and hard disagreement, each against a **seed-to-seed baseline within each topology**. A topology effect is only meaningful if the between-topology distance clearly exceeds the within-topology seed spread.
- **Spectral and ANOVA decompositions** (Walsh degree profile, purified interaction order) are a natural fit for the project's question of whether topology changes the *order of interactions* a network represents. Compositional or parallel branches plausibly bias toward lower-order additive structure.
  - Inputs need to be designed so the decomposition is tractable: Boolean or synthetic tasks, or sampled estimators on real inputs.
  - The decomposition has to be estimated under a stated input measure.
- **Cost.** Output agreement is cheap (forward passes only). Exact Walsh spectra are exponential in input bits, so use sampled or low-degree estimators. ANOVA purification is cheap on tabulated function values but needs dense sampling of the input space.

### Gaps
- I found no paper that uses Walsh/Fourier or purified-ANOVA spectra specifically to compare residual vs non-residual (or parallel vs composed) topologies at matched parameter count. This appears to be an open niche.
- I did not verify a canonical reference for "functional similarity across architectures" beyond prediction agreement and stitching. The Klabunde survey is the best entry point.

---

## Q2. Representational similarity across architectures (CKA, SVCCA/PWCCA, Procrustes/shape metrics, RSA, stitching) and their critiques

### Takeaway
Every representational similarity measure encodes a choice of invariance group and a weighting over principal components. The choice drives conclusions:
- No measure dominates across benchmarks (ReSi, ICLR 2025).
- CKA is dominated by top principal components, is manipulable, and is biased upward at high width and low sample size.
- Stitching scores can be inflated by out-of-distribution exploitation.
- Uncalibrated layer-max scores are confounded by width and depth.

The defensible practice for cross-topology work:
- Use several measures from different families.
- Calibrate against permutation nulls and seed baselines.
- Prefer metrics (true distances) such as Procrustes shape distance, plus neighbourhood measures.
- Ground every comparison in decodable task information.

### Cited Findings
**Families, invariances, preprocessing**
- **Invariance classes.** Klabunde et al. list permutation, orthogonal, isotropic scaling, invertible linear, translation, and affine. Most measures are permutation-invariant.
  - Raghu et al. (SVCCA) argued for invariance to invertible linear maps.
  - Kornblith et al. argued that orthogonal invariance captures practical differences better.
  - Mean-centering implies translation invariance.
  - Representations of different width are compared after zero-padding or PCA.
  - CNN feature maps require flattening, with matched or upsampled spatial dimensions.

  — [Klabunde et al. survey](https://arxiv.org/html/2305.06329v3)
- **Generalized shape metrics.** Williams et al. modified CCA- and Procrustes-type measures so that they satisfy the triangle inequality and are therefore true metrics. They added a convolution-aware metric and approximate Euclidean embeddings of many networks. They applied this to NAS-Bench-101 (many architectures) and found relationships interpretable in terms of model performance. — [Williams, Kunz, Kornblith, Linderman, NeurIPS 2021](https://arxiv.org/abs/2110.14739)
- **Similarity measures as statements about linear decoding.**
  - CKA and CCA measure the average alignment between optimal linear decoders across a distribution of decoding tasks.
  - Procrustes shape distance upper-bounds the distance between optimal linear readouts.
  - The converse bound holds when the participation ratio is low.

  — [Harvey, Lipshutz, Williams, PMLR v285 (2024)](https://arxiv.org/abs/2411.08197)

**Benchmarks: no measure dominates**
- **ReSi** benchmarks 24 measures × 6 tests × 14 architectures × 7 datasets (graphs, language, vision).
  - "No measure dominates."
  - CKA ranks 1st for language but 8th for vision and 11th for graphs.
  - Jaccard (k-NN overlap) is strongest for vision. Neighbourhood measures are strong on graphs.
  - Orthogonal Procrustes is consistently in the top half. Its "size-and-shape" preprocessing variant gives substantially different results, so preprocessing changes what is measured.
  - Neither RSA nor SVCCA stands out in aggregate.

  — [ReSi, ICLR 2025](https://arxiv.org/html/2408.00531v2)
- **Ding, Denain, Steinhardt ("Grounding Representation Similarity")** propose sensitivity and specificity tests tied to functional behaviour (probing accuracy, OOD robustness).
  - PWCCA fails specificity to random seed.
  - CKA fails sensitivity to principal-component deletion: about 97% of BERT last-layer PCs had to be removed before CKA exceeded the seed baseline, while probe accuracy fell from 80% to 63%. The cause is that CKA is a quartic function of singular values and is dominated by the largest.
  - Orthogonal Procrustes "attained consistently good performance."
  - No measure handled their challenge set.

  — [Ding et al., NeurIPS 2021](https://ar5iv.labs.arxiv.org/html/2108.01661)

**CKA-specific critiques**
- **CKA is sensitive to outliers and to transformations that preserve linear separability**, and "CKA scores can often be shifted with little effect on what the models do." — [Davari et al., ICLR 2023](https://arxiv.org/abs/2210.16156v2)
- **High scores do not imply task information.** Optimizing synthetic data to maximize a similarity score can reach CKA near 1, or high unregularized linear-regression similarity, without encoding task variables as decodably as the reference data.
  - Measures capture principal components in different orders. All first learn the top-variance PC. Angular Procrustes captures low-variance PCs much earlier than CKA.
  - CKA's sensitivity to a PC's variance is quadratic.
  - There is no universal "good score" threshold.
  - High angular Procrustes implies high CKA, but not the reverse.

  — [Cloos et al., ICLR 2025](https://arxiv.org/html/2407.07059v3)
- **CKA is biased in the low-sample, high-dimension regime.**
  - Biased CKA can report high similarity between random matrices, and can be pushed to its maximum with independent random data of different sample-to-feature ratios.
  - It depends on the feature-to-sample ratio, so representations of different sizes are not directly comparable.
  - Shuffling sample pairings of real data barely changes biased CKA.
  - Only debiased CKA reliably detects true stimulus-driven alignment.
  - This matters directly when comparing topologies with different hidden widths at a cut point.

  — [Murphy, Zylberberg & Fyshe, arXiv 2405.01012 (2024)](https://arxiv.org/abs/2405.01012v1)
- **Width and depth confounders, and a calibration fix.**
  - Linear CKA and RV have non-zero null baselines that grow roughly with d/n, so wider models look more aligned even when independent.
  - Taking the maximum over many layer pairs inflates scores, and deeper models have more pairs.
  - Mutual k-NN has a dimension-independent null of k/(n−1).
  - The fix is a permutation null (shuffle row correspondences) with an aggregation-aware (maxT-style) correction.
  - After calibration, the reported global convergence between models "largely disappears," but local neighbourhood agreement survives.

  — [Gröger, Wen, Brbić, "Revisiting the Platonic Representation Hypothesis: An Aristotelian View," arXiv 2602.14486 v2, June 2026 (preprint)](https://arxiv.org/html/2602.14486)
- **Block structure from a dominant principal component.** In high-capacity models, many consecutive layers propagate a dominant PC, producing "block structure." Similarity *outside* the block is shared across widths and depths, but the block structure is "unique to each model." Global CKA heatmaps across architectures are therefore partly measuring one dominant direction. — [Nguyen, Raghu, Kornblith, ICLR 2021](https://arxiv.org/abs/2010.15327v2)
- **System identification.** When responses were generated by models of known architecture, linear encoding and CKA did not reliably identify the true generating architecture. Results depended on factors unrelated to architecture, such as stimulus set. This is a direct warning against inferring architecture or topology identity from similarity scores. — [Han, Poggio, Cheung, ICML 2023](https://arxiv.org/abs/2302.06677v1)

**Model stitching**
- **Stitching origins.** Lenc & Vedaldi introduced measuring equivalence of representations by learning a map ("stitching") between networks. — [Lenc & Vedaldi, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Lenc_Understanding_Image_Representations_2015_CVPR_paper.html)
- **Bansal, Nakkiran, Barak revived stitching** as a functional-substitutability test.
  - Same-architecture networks trained differently (supervised vs self-supervised) stitch with no performance drop.
  - Representations learned with more data or width can improve weaker models' top halves ("more is better").
  - They argue stitching reveals aspects of representations that CKA cannot.

  — [Bansal, Nakkiran, Barak, NeurIPS 2021](https://arxiv.org/abs/2106.07682)
- **Stitching across different-shaped ResNet layers** reached high accuracy whenever the sender layer was later than the receiver, even across distant layers. The authors conclude that "naively implemented model stitching may not always be an accurate measure of similarity." — [Hernandez, Dangovski, Lu, Soljačić, arXiv 2303.11277 (NeurIPS 2022 workshop)](https://ar5iv.labs.arxiv.org/html/2303.11277)
- **Task-loss-matching stitching exploits the receiver.**
  - It makes distant layers look more similar than corresponding layers. TLM picked the same layer only 63.75% of the time for ResNet-18 and 24.17% for ViT-Ti.
  - The stitched activations are out-of-distribution: energy-based OOD detectors gave AUROC close to 1.
  - Recommendation: use direct matching of activations, and check that stitched representations are in-distribution.
  - OPD (the Ding et al. recommended baseline) was inconsistent in their settings.

  — [Balogh & Jelasity, AAAI 2025](https://arxiv.org/html/2412.11299v1)

**RSA and the broader alignment literature**
- **Getting aligned on representational alignment** proposes a unifying framework spanning cognitive science, neuroscience, and ML for representational alignment research, and lays out open problems. — [Sucholutsky et al., arXiv 2310.13018 (2023, rev. 2024)](https://arxiv.org/abs/2310.13018)
- **RSA** (Kriegeskorte et al. 2008) compares representational dissimilarity matrices. It is invariant to any transformation that preserves the RDM, given the chosen distance. In ReSi it is not a standout in aggregate but performs well in specific settings, such as output-difference correlation on graphs. — [ReSi](https://arxiv.org/html/2408.00531v2); original RSA: [Kriegeskorte, Mur, Bandettini 2008, Frontiers](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full)

### Inferences
- **Invariance summary** (derived from definitions in Klabunde et al.):

  | Measure | Invariant to | Not invariant to |
  |---|---|---|
  | Linear CKA | Orthogonal maps, isotropic scaling (and translation after centering) | General invertible linear maps |
  | SVCCA / PWCCA | Invertible linear maps on the retained subspace | Choice of truncation (SVCCA) or weighting (PWCCA) |
  | Orthogonal Procrustes / angular shape metric | Orthogonal maps (+ scale and translation, depending on preprocessing) | General linear maps (the regularized generalized shape metric interpolates toward linear invariance) |
  | RSA | Whatever the RDM distance is invariant to | — |
  | k-NN / Jaccard | Any transformation preserving neighbourhood ranks | Local distances and global geometry (captures only local rank order) |

- **Implication for topology comparisons.** Residual and dense routing changes which basis is "privileged" (e.g., residual streams lack an elementwise nonlinearity at the stream itself). Elementwise-nonlinear hidden layers break rotational symmetry. So the right invariance group may differ *by site* in the two networks.
  - Compare at sites with matched symmetry: residual stream vs residual stream, or post-nonlinearity vs post-nonlinearity.
  - Otherwise report measures at two invariance levels (orthogonal and linear).
- **Width differences at the cut point** are a first-order confounder for CKA and RV (Gröger 2026; Murphy et al. 2024). Use debiased CKA or a permutation-calibrated score. Report mutual k-NN, whose null does not depend on dimension.
- **Stitching is the representational measure closest to functional meaning.** For different graphs it requires choosing cut points, and that choice is itself an instrument decision.
  - Use direct matching rather than task-loss matching (Balogh & Jelasity 2025).
  - Use a low-capacity stitch (affine).
  - Run in-distribution checks on stitched activations.
  - Report the asymmetry (A→B vs B→A) as signal: "more is better" in Bansal et al.
- **Validation practice that should be mandatory** (synthesizing Ding 2021, Cloos 2025, Gröger 2026, Han 2023):
  - seed-to-seed within-topology baselines;
  - permutation nulls with correction for layer search;
  - PC-deletion sensitivity tests;
  - reporting decodability of task variables alongside any similarity score.
- **Cost.** All these measures are cheap: CKA and Procrustes are O(n·d²) or O(n²·d), and minibatch CKA is used at scale (Raghu 2021). Stitching requires training small layers per cut-pair, which is moderate cost.

### Gaps
- I did not retrieve the original SVCCA/PWCCA papers in this session. Their known critique is PWCCA's failure of seed specificity, per Ding et al. 2021.
- I did not retrieve Kriegeskorte's later critiques of RSA variants (e.g., whitened RSA, the cross-validated distance estimators in rsatoolbox). They would need direct sourcing.
- No benchmark like ReSi tests measures on *same-parameter-count, different-connectivity* pairs specifically.

---

## Q3. Structure measures (modularity/clusterability, effective rank/dimensionality, spectral, specialization, compositionality, superposition)

### Takeaway
Structure measures quantify properties of a single network (how modular, how high-dimensional, how compositional, how superposed), which can then be compared across topologies. This sidesteps the correspondence problem entirely.

Their validity hinges on four things:
1. Whether the graph or statistic is defined on parameters (topology-dependent) or on activations and gradients (more comparable).
2. Upstream vs downstream definitions disagree.
3. Finite-sample bias in dimensionality estimates.
4. Dependence on an instrument such as an SAE (superposition counts).

### Cited Findings
- **Clusterability (Filan et al. 2021).**
  - Treat the network as a weighted graph over neurons (edges = weight magnitudes).
  - Run spectral clustering (normalized cut).
  - Compare with a shuffled-weights null to test whether trained networks are more clusterable than chance.
  - This is weight-space modularity.

  — [Filan, Casper, Hod, Wild, Critch, Russell, arXiv 2103.03386](https://arxiv.org/abs/2103.03386)
- **Local specialization (Hod et al.).** Two proxies:
  - **importance:** how crucial a neuron set is to performance;
  - **coherence:** how consistently it responds to input features.

  Partitions come from spectral clustering on either weights or activation correlations. Even weight-only partitions reveal important and coherent groups. — [Hod, Filan, Casper, Critch, Russell, arXiv 2110.08058](https://arxiv.org/abs/2110.08058)
- **Upstream vs downstream modularity.**
  - Upstream: similar responses to input variation.
  - Downstream: similar effects on outputs when perturbed.

  Methods agree *within* a family but "largely disagree" *across* families. Dropout markedly increases modularity, and other regularizers have smaller effects. Input-mirroring (disentanglement-like) and output-mirroring (compositionality-like) modularity are distinct targets. — [Lange, Rolnick, Kording, TMLR 2022](https://arxiv.org/abs/2203.11815v1)
- **Path-ensemble structure of residual networks (a topology-specific structural measure).**
  - Lesioning shows that ResNet paths "do not strongly depend on each other."
  - Effective paths are short: for a 110-layer ResNet most gradient comes from paths 10 to 34 layers deep.
  - Path-length distribution and lesion sensitivity are therefore structural descriptors tied directly to skip topology.

  — [Veit, Wilber, Belongie, NeurIPS 2016](https://arxiv.org/abs/1605.06431v2)
- **Skip connections shape representation structure directly.** Removing a skip connection in a ViT block splits the layer-similarity map into two groups (before and after the block) and breaks the uniform grid structure. Removing middle-block skips cost about 4% accuracy. — [Raghu et al., NeurIPS 2021](https://ar5iv.labs.arxiv.org/html/2108.08810)
- **Dimensionality estimation is biased.** The naive participation ratio (PR) is biased because both its numerator and denominator are biased. It is highly sensitive to the number of stimuli and neurons sampled. Chun et al. derive an unbiased PR estimator, a noise-corrected version, and a local (curved-manifold) variant, validated on synthetic data and LLM activations. — [Chun, Canatar, Chung, Lee, arXiv 2509.26560 (ICLR 2026 poster)](https://arxiv.org/html/2509.26560v1)
- **Superposition as a measurable quantity.**
  - Effective feature count F = exp(H(p)), where p is the share of activation magnitude per SAE latent.
  - Superposition ψ = F/N, with N = neurons.
  - Validated on toy models: r = 0.99 vs ground truth from weights, r = 0.94 using SAE outputs.
  - F correlates with LLC during grokking (r = 0.908).
  - Caveats: it depends heavily on SAE hyperparameters (L1 strength, dictionary size, feature splitting) and is "best suited for comparisons under identical settings rather than as an absolute feature count."

  — [Bereska et al., TMLR 2025](https://arxiv.org/html/2512.13568v1)
- **Compositionality (Tree Reconstruction Error).** Andreas scores compositionality by how well the true representation-producing model is approximated by a model that *composes inferred primitives* according to a known derivation structure. He relates the score to learning dynamics, human judgments, representational similarity, and generalization. — [Andreas, ICLR 2019](https://arxiv.org/abs/1902.07181)
- **SAE-feature universality across architectures** (detail in Q7): Wang et al. used SAE feature similarity to compare Transformers and Mamba. — [Wang et al., ICLR 2025](https://arxiv.org/abs/2410.06672v1)

### Inferences
- **Weight-space modularity is not cross-topology valid as-is.** The neuron graph itself differs between topologies. A residual network's weight graph has implicit identity edges, and a parallel-branch network has block-diagonal structure *imposed by design*.
  - Any weight-graph clusterability score must be compared against a **topology-matched null**: shuffle weights while preserving the fixed connectivity mask, as in Filan's shuffle control but respecting the architecture's sparsity pattern.
  - Otherwise "topology X is more modular" is tautological.
- **Activation- and gradient-based (functional) modularity are more comparable.** Lange et al.'s upstream and downstream associations, and Hod et al.'s coherence, are defined on unit behaviour. They are computed on any network with the same inputs and outputs.
  - Report upstream and downstream modularity separately, since they disagree.
- **Effective rank and PR should use bias-corrected estimators** with matched sample counts. Compare at matched sites, because width differs.
- **Superposition (ψ) and SAE-based counts are instrument-dependent** (Bereska 2025; also Leask 2025, already known).
  - Use identical SAE recipes and sweep dictionary size.
  - Report plateau behaviour before comparing topologies.
- **Compositionality metrics like TRE** require a hypothesized derivation structure. They suit synthetic compositional tasks where the ground-truth composition is known, which is a natural testbed for parallel vs composed branch topologies.
- **Cost.** Spectral clustering of neuron graphs is cheap up to about 10⁵ neurons. Bias-corrected PR is cheap. SAE-based superposition measures need SAE training per site and topology (moderate to expensive). TRE is cheap on toy tasks.

### Gaps
- I did not fetch spectral measures of weight matrices (e.g., heavy-tailed self-regularization / WeightWatcher, Martin & Mahoney). Their cross-topology validity is unclear because they are per-layer weight statistics.
- I found no work that systematically tests modularity metrics against topology-preserving nulls.

---

## Q4. Developmental and dynamical measures (LLC/susceptibilities, NTK alignment, progress measures, phase transitions, reparameterization-invariant landscape measures, mode connectivity)

### Takeaway
Developmental measures describe *when* and *how* structure forms. That makes them powerful for topology studies (does a skip connection shift or remove a phase transition?), and they carry a range of invariance properties:
- Fully invariant: LLC and susceptibilities are geometric and Bayesian quantities, invariant to reparameterizations that preserve the function and the prior locally.
- Partly invariant: NTK measures are function-space kernels, but the kernel depends on the parameterization's tangent map.
- Not invariant: raw Hessian sharpness fails under reparameterization unless the metric is tracked.
- Topology-entangled: linear mode connectivity is a weight-space notion and cannot directly compare different graphs. It is evidence of mechanistic sameness *within* a topology family.

### Cited Findings
- **Susceptibilities (SLT).** The derivative of a localized posterior expectation with respect to a data-distribution perturbation q_h = (1−h)q + hq′. It equals minus the covariance between the observable and the loss change.
  - Localized to a component (e.g., an attention head).
  - PCA on per-token susceptibilities of a 2-layer attention-only transformer recovered word segmentation (PC1) and the induction circuit (PC2) across 4 seeds.
  - Cost: roughly $2k and $5k per checkpoint for head-level analysis of Pythia-410M and 1.4B.
  - Caveats: SGLD hyperparameters are poorly understood, magnitudes depend on perturbation scale (compare within a single probe setup), and PCA may not suffice at scale.

  — [Baker, Wang, Hoogland, Murfet, "Structural Inference," ICLR 2026](https://arxiv.org/pdf/2504.18274)
- **Further Timaeus work.** It extends susceptibility analysis to developmental "embryology" of LMs and to "patterning" (the dual of interpretability). — [Embryology of a Language Model, arXiv 2508.00331](https://alphaxiv.org/abs/2508.00331); [Patterning: The Dual of Interpretability, arXiv 2601.13548](https://arxiv.org/pdf/2601.13548). These were not fetched, so details are unverified.
- **NTK evolution.**
  - Training has a "highly chaotic rapid initial transient" of about 2 to 3 epochs that determines the final linearly connected basin.
  - The NTK changes rapidly in this window, after which it changes at roughly constant velocity.
  - Loss-landscape and NTK measures evolve in highly correlated ways across architectures.

  — [Fort et al., NeurIPS 2020](https://arxiv.org/abs/2010.15110v1)
- **Kernel alignment.** Baratin et al. show that the tangent kernel aligns with the target function during training, giving implicit regularization via neural feature alignment. — [Baratin et al., AISTATS 2021](https://arxiv.org/pdf/2008.00938)
- **Silent alignment.** Atanasov, Bordelon, Pehlevan show the kernel can align early, while loss is still flat, then grow in scale, so networks can act as kernel machines with a learned kernel. — [Atanasov et al., ICLR 2022](https://arxiv.org/abs/2111.00034v1)
- **Sharpness is not reparameterization-invariant.** Equivalent reparameterizations (layer-wise rescaling in ReLU nets) can make minima arbitrarily sharp without changing the function. — [Dinh et al., ICML 2017](https://arxiv.org/abs/1703.04933)
- **Invariance by tracking the metric.** Reparameterization invariance is "an inherent property of any neural net if one explicitly represents the metric." Plain Hessian-based flatness measures are inconsistent under reparameterization. — [Kristiadi, Dangel, Hennig, NeurIPS 2023](https://arxiv.org/abs/2302.07384v3)
- **Even adaptive sharpness predicts generalization poorly.** Adaptive (scale-invariant) sharpness is a weak predictor of generalization in modern settings. It mainly tracks hyperparameters such as learning rate, and sometimes correlates *negatively* with OOD error. — [Andriushchenko et al., ICML 2023](https://arxiv.org/abs/2302.07011)
- **Linear mode connectivity and stability.** Networks become stable to SGD noise, meaning linearly connected after training from a shared early checkpoint, after a short early phase. — [Frankle et al., ICML 2020](https://ar5iv.arxiv.org/html/1912.05671)
- **Permutation alignment (Git Re-Basin)** gives zero-barrier LMC for independently trained ResNets on CIFAR-10. The authors note shortcomings of the single-basin hypothesis and give a counterexample. — [Ainsworth, Hayase, Srinivasa, ICLR 2023](https://arxiv.org/abs/2209.04836v6)
- **Mechanistic mode connectivity.** Lack of linear low-loss connectivity indicates dissimilar mechanisms, defined as different invariances to input transformations. Naive fine-tuning often fails to change mechanisms. — [Lubana et al., ICML 2023](https://arxiv.org/abs/2211.08422v3)
- **Phase-transition detectors and LLC as cross-measure anchors.** Bereska's superposition count correlates with LLC during grokking (r = 0.908). — [Bereska et al., TMLR 2025](https://arxiv.org/html/2512.13568v1)

### Inferences
- **LLC and susceptibilities are the most principled cross-topology developmental measures.** They are defined via the loss and posterior and are invariant to smooth reparameterizations. The LLC is a scalar "effective complexity," which makes it comparable across graphs at equal parameter count.
  - Caveats:
    - SGLD estimator hyperparameters (step size, localization strength γ, number of steps) must be **tuned per architecture**, because loss geometry differs.
    - Report estimator diagnostics.
    - Susceptibility magnitudes are only comparable within a probe setup (Baker 2026).
- **NTK alignment and kernel velocity.** These are function-space objects: Gram matrices over inputs, so the input × input kernels from different graphs can be compared with CKA or alignment. They depend on parameterization through ∂f/∂θ (e.g., NTK vs μP scaling). So topologies should be compared under the same parameterization convention. A skip connection also changes the NTK's layer-sum structure even at initialization.
  - Recommendation: measure kernel *change* relative to initialization, both kernel distance and target alignment over time, to factor out the topology's prior kernel.
- **Landscape sharpness.** Use metric-aware (Fisher or Riemannian) or function-space curvature, not raw Hessian eigenvalues, when comparing graphs. Treat even adaptive sharpness as weak evidence (Andriushchenko 2023).
- **Mode connectivity.** LMC and permutation alignment require the same parameter space, so they are invalid across topologies. They are useful *within* each topology as evidence for single-basin, mechanism-unique solutions (Lubana 2023). The cross-topology question then becomes: "Does topology X produce seeds that are more often linearly connected (after alignment) than topology Y?" That is a comparable within-family statistic.
- **Cost.**
  - LLC: SGLD chains, about 10² to 10³ steps per checkpoint (moderate).
  - Susceptibilities: per component × per probe set; expensive at scale (Baker 2026).
  - Empirical NTK: O(n² · params) at full size; use subsampling or random projections.
  - LMC: cheap after alignment.

### Gaps
- I did not retrieve a specific saddle or plateau detection method paper (e.g., Hessian negative-eigenvalue tracking). Only indirect evidence via Fort 2020 transients and the silent-alignment plateaus.
- The Fisher-geometric sharpness preprint ([arXiv 2606.20469](https://arxiv.org/pdf/2606.20469)) appeared in search but was not fetched.

---

## Q5. Causal/mechanistic measures (causal abstraction, interchange interventions, DAS, critiques; adapting across graphs)

### Takeaway
Causal abstraction is the best formal tool for cross-topology comparison, because it compares each network to the *same high-level algorithm* instead of to each other. That converts a correspondence problem into two separate alignment problems.

Three cautions apply:
1. **Map class:** with unrestricted non-linear alignment maps, interchange-intervention accuracy (IIA) becomes vacuous. Random networks can reach 100% IIA.
2. **Subspace illusions:** DAS-found subspaces can be "dormant pathway" artifacts.
3. **Instrument dependence:** circuit and feature-discovery results depend on the method. MIB finds that SAEs do not beat neurons for causal-variable localization.

### Cited Findings
- **Causal abstraction framework.** Geiger et al. formalize mechanistic interpretability as causal abstraction: a high-level causal model is an abstraction of the network if interventions align. This unifies activation patching, interchange interventions, path patching, and related methods. — [Geiger et al., JMLR 26, 2025](https://jmlr.org/papers/v26/23-0058.html)
- **Distributed Alignment Search (DAS)** learns a (rotated) subspace aligned with a high-level causal variable. It is evaluated by interchange-intervention accuracy. — [Geiger, Wu, Potts, Icard, Goodman, CLeaR 2024](https://proceedings.mlr.press/v236/geiger24a.html)
- **Subspace activation patching can create an interpretability illusion.** A subspace intervention can change model behaviour by activating a dormant parallel pathway, not by manipulating the feature the model actually uses. DAS-style results can therefore appear without the variable being represented in that subspace. — [Makelov, Lange, Nanda, ICLR 2024](https://arxiv.org/abs/2311.17030)
- **The non-linear representation dilemma.** If alignment maps can be arbitrarily non-linear, any algorithm can be aligned with any network (under input-injectivity and similar assumptions).
  - Randomly initialized Pythia reaches 100% IIA on IOI with complex enough maps.
  - A random MLP reaches over 80% IIA on hierarchical equality.
  - Linear maps reach near-perfect alignment only in trained models.
  - Remedies: constrain map complexity when linear encoding is plausible, trade off accuracy vs map complexity (as in probing), and treat generalization of the map as central.

  — [Sutter, Minder, Hofmann, Pimentel, NeurIPS 2025](https://arxiv.org/pdf/2507.08802)
- **MIB benchmark** (circuit localization and causal-variable localization; 4 tasks, 5 models).
  - Attribution and mask-optimization methods are best for circuit localization.
  - Supervised DAS is best for causal-variable localization.
  - SAE features do not outperform neurons.

  — [Mueller et al., ICML 2025](https://arxiv.org/abs/2504.13151v1)
- **Crosscoders (model diffing) carry sparsity artifacts.** Standard L1 crosscoders make shared concepts falsely appear model-specific. The authors propose Latent Scaling diagnostics and BatchTopK training. — [Minder, Dumas, Juang, Chughtai, Nanda, NeurIPS 2025](https://arxiv.org/abs/2504.02922v2)
- **Mechanistic sameness via shared input invariances.** Lubana et al. define mechanistic similarity as shared invariances to input transformations. This is graph-agnostic. — [Lubana et al., ICML 2023](https://arxiv.org/abs/2211.08422v3)
- **Residual ensembles under lesioning.** Ablation and lesion results in residual networks behave like path ensembles. A single lesion has small effects that do not imply the absence of function. — [Veit et al., NeurIPS 2016](https://arxiv.org/abs/1605.06431v2)

### Inferences
- **Cross-topology protocol.**
  1. Fix a high-level algorithm H. This is best on synthetic tasks with a known algorithm (Tracr- or InterpBench-style ground truth, already known).
  2. Find an alignment of H to each network separately, using the **same restricted map class** (linear or orthogonal DAS, low-rank) and the same budget.
  3. Report IIA on **held-out** base/source pairs.
  4. Compare *which* high-level variables are localizable, *where* (relative depth or path), and *with what dimensionality*.
  5. Include random-init and shuffled-label controls of both topologies, per Sutter et al. A topology difference counts only if trained-vs-random IIA gaps differ.
- **Dormant pathways matter more for routed topologies.** Residual and dense networks have more parallel paths, so the dormant-pathway risk (Makelov 2024) is plausibly greater.
  - Validate subspace findings by checking that the subspace is causally used on the **natural distribution**: projection-ablation effects, not only interchange effects.
- **Ablation baselines** (mean vs zero vs resample) interact with topology. Zero-ablating a residual branch leaves the identity path intact, while zero-ablating a layer in a plain net cuts all flow. So ablation effect sizes are not comparable across graphs unless normalized, e.g., to the effect of ablating the whole path set at matched depth. Resample ablation on the natural distribution is the safest common baseline (consistent with known Zhang & Nanda 2024 guidance).
- **Map generalization** is the key validity control (Sutter 2025). Report alignment-map complexity next to IIA for both networks.
- **Cost.** DAS: train rotation per site and variable (moderate). Path patching and attribution patching: cheap. Exhaustive interchange search over sites scales with the number of sites, which differs between topologies, so correct for multiple comparisons (search-size effects analogous to Gröger's layer-max confound).

### Gaps
- I found no published protocol explicitly for comparing causal-abstraction results across networks with different graphs. The protocol above is an inference.
- I did not fetch newer post-2025 critiques of attribution patching faithfulness beyond MIB.

---

## Q6. Training-data attribution and influence as structure measures

### Takeaway
Training-data attribution (TDA) gives a function-of-data descriptor (which examples shape which predictions). It is defined on the shared dataset, so it is comparable across topologies in principle. In practice, attribution estimates for deep nets approximate a proximal response, not leave-one-out retraining, and depend on curvature approximations that are architecture-specific. The most robust cross-topology use is to compare attribution *patterns* (rank correlations, sparsity, cross-task transfer), each with its own validation such as the linear datamodeling score.

### Cited Findings
- **What influence functions actually estimate.** In nonlinear nets, influence estimates poorly match leave-one-out retraining. They do approximate the "proximal Bregman response function" (PBRF), which still supports finding influential or mislabeled examples. — [Bae, Ng, Lo, Ghassemi, Grosse, NeurIPS 2022](https://arxiv.org/abs/2209.05364)
- **EK-FAC influence at scale (up to 52B parameters).**
  - Influence patterns are sparse.
  - Generalization becomes more abstract with scale.
  - Influence decays to near-zero when key-phrase order is flipped.

  — [Grosse et al. 2023](https://arxiv.org/abs/2308.03296)
- **TRAK** uses random projections of per-example gradients plus ensembling for scalable attribution. It is evaluated by the linear datamodeling score (LDS): correlation between predicted and actual outputs when training on random data subsets. — [Park et al., ICML 2023](https://proceedings.mlr.press/v202/park23c.html)
- **Datamodels** (Ilyas et al. 2022) fit linear predictors of model outputs from training-subset membership by retraining many models. This is the ground-truth-style reference behind LDS. — [Ilyas et al., arXiv 2202.00622](https://arxiv.org/abs/2202.00622) (not fetched in this session)

### Inferences
- **Invariances.** Datamodels (retraining-based) are fully function-level, so they are cross-topology valid, at very high compute cost. Gradient-based TDA (influence, TRAK) depends on the parameterization through gradients and curvature approximations (K-FAC structure assumes layer-wise blocks, which residual or dense routing complicates).
- **Use.** Compute per-test-example attribution vectors over a shared training set for each topology. Compare with rank correlation, top-k overlap, and sparsity, and cluster test examples by their attribution profiles. Seed-to-seed attribution agreement within a topology must be reported as a baseline, since TDA is known to be noisy across seeds.
- **Cost.** Datamodels: thousands of retrains, only feasible on small models. TRAK: a few model ensembles plus projections (moderate). EK-FAC: fitting curvature plus per-query gradient sweeps (moderate to high).

### Gaps
- I did not find studies that use TDA explicitly to compare architectures or topologies.
- I did not retrieve recent TDA benchmarks (2025–2026) on seed-robustness of attributions.

---

## Q7. Measures used specifically to compare different architectures, and lessons on validity

### Takeaway
Cross-architecture comparisons have relied mainly on CKA heatmaps (ViT vs CNN; wide vs deep), SAE feature matching (Transformer vs Mamba), stitching, and circuit-motif analysis. The validity lessons are consistent:
- Pair similarity heatmaps with interventions (skip removal, probes).
- Beware layer-alignment and width confounds.
- Treat SAE-based "universality" percentages as instrument-dependent.
- Report qualitatively different motifs even when aggregate similarity is high.

### Cited Findings
- **ViT vs CNN (Raghu et al.).**
  - Measure: minibatch CKA (unbiased HSIC, batches of 1024, 10,240 examples, 20 repeats).
  - ViTs have more uniform representations across layers.
  - Self-attention aggregates global information early.
  - Residual connections strongly propagate features upward.
  - Removing skip connections in middle ViT blocks splits the similarity structure into before and after groups and costs about 4% accuracy.
  - Caveats noted: CKA is a single scalar; channel-count mismatch for probes; cross-depth layer alignment shown without sensitivity analysis.
  - The authors paired CKA with interventions and linear probes.

  — [Raghu et al., NeurIPS 2021](https://ar5iv.labs.arxiv.org/html/2108.08810)
- **Wide vs deep (Nguyen et al.).** Block structure appears in high-capacity models, driven by a dominant principal component, and is unique per model. Representations outside the block are similar across widths and depths. Error patterns differ even at matched accuracy. — [Nguyen, Raghu, Kornblith, ICLR 2021](https://arxiv.org/abs/2010.15327v2)
- **Transformer vs Mamba (Wang et al.).**
  - Most SAE features are similar across the two.
  - Feature similarity correlates with universality.
  - Mamba's induction circuits are "structurally analogous" to Transformers'.
  - They report an "Off-by-One motif": token info is written into the SSM state at the following position, a nuanced architecture-specific difference.

  — [Wang et al., ICLR 2025](https://arxiv.org/abs/2410.06672v1)
- **Mamba-130m vs Pythia-70m (later preprint).** Using SAEs and Jaccard similarity on 10M tokens, the authors report 99.98% of Mamba features aligning closely with Pythia's. Divergent features involve rigid syntax, and Mamba packs anomalies into polysemantic "junk drawer" neurons. *Caution:* this is a recent preprint by a small team, not peer-reviewed. The 99.98% figure depends heavily on the SAE and the Jaccard threshold. — [Nagaraj et al., arXiv 2609.24440, Sept 2026](https://arxiv.org/abs/2609.24440)
- **Stitching across architectures.**
  - Bansal et al.: stitchable representations across training regimes; asymmetric "more is better." — [NeurIPS 2021](https://arxiv.org/abs/2106.07682)
  - Hernandez et al.: different-shape ResNets stitch across distant layers, a warning about naive stitching. — [arXiv 2303.11277](https://ar5iv.labs.arxiv.org/html/2303.11277)
  - Balogh & Jelasity: on ResNet-18 and ViT-Ti, task-loss stitching fails self-identification sanity checks. — [AAAI 2025](https://arxiv.org/html/2412.11299v1)
- **Many-architecture embeddings.** NAS-Bench-101 architectures were embedded in a metric space using generalized shape metrics. — [Williams et al., NeurIPS 2021](https://arxiv.org/abs/2110.14739)
- **ReSi** spans 14 architectures across graph, language, and vision. Measure rankings change by domain, so lessons about "the best similarity measure" do not transfer across architecture families. — [ReSi, ICLR 2025](https://arxiv.org/html/2408.00531v2)
- **System identification.** Similarity metrics did not reliably identify the generating architecture. — [Han et al., ICML 2023](https://arxiv.org/abs/2302.06677v1)
- **Cross-model convergence claims** (Platonic representation hypothesis) largely vanish after calibrating for width and depth confounders, while local neighbourhood structure persists. — [Gröger et al., arXiv 2602.14486, 2026](https://arxiv.org/html/2602.14486)

### Inferences
- **Lesson 1.** Similarity heatmaps alone are descriptive. Causal claims ("residual connections cause uniform representations") came from *interventions on topology* (removing skips) within one architecture family. This is exactly the design the research program needs: matched-parameter topology ablations with function-space and representational read-outs.
- **Lesson 2.** Cross-architecture "universality" percentages from SAEs are instrument-laden (Leask 2025, known; Bereska 2025; Minder 2025). Confirm with causal tests of matched features, e.g., patching a matched feature direction in each model gives matched downstream effects.
- **Lesson 3.** Fine-grained motif differences (Off-by-One; Clock vs Pizza) can coexist with high aggregate similarity. Report both aggregate and motif-level measures.
- **Lesson 4.** Width and depth differences confound global similarity measures (Gröger 2026; Nguyen 2021). Matched parameter count does not imply matched width at a cut point, so calibration is essential.

### Gaps
- I did not find a published study comparing *same-parameter-count* networks that differ only in parameter-free connectivity (residual vs plain vs dense) with a multi-measure validated stack. Raghu 2021's skip-removal ablation is the closest. This is an evident gap the project would fill.

---

## Q8 (Synthesis). Recommended measurement stack, ranked by validity for cross-topology comparison

### Takeaway
Rank measures by how few correspondence or parameterization assumptions they need:
1. function-space read-outs;
2. causal abstraction to a fixed high-level algorithm with restricted maps;
3. Bayesian and geometric developmental measures (LLC, susceptibilities);
4. calibrated multi-family representational metrics at symmetry-matched cut points;
5. activation- and gradient-based structural statistics with topology-matched nulls;
6. instrument-heavy measures (SAE- and crosscoder-based counts, TDA).

Weight-space measures (raw sharpness, weight-graph clusterability, LMC) should only be used *within* a topology.

### Cited Findings
(The ranking below is a synthesis. The supporting citations are those given in Q1–Q7.)
- Function-level measures are invariant to parameterization but sensitive to metric choice and input distribution — [Schaeffer 2023](https://arxiv.org/abs/2304.15004v1); [Lengerich 2020](https://arxiv.org/abs/1911.04974v2)
- Causal abstraction is formally grounded but vacuous without map restrictions — [Geiger JMLR 2025](https://jmlr.org/papers/v26/23-0058.html); [Sutter NeurIPS 2025](https://arxiv.org/pdf/2507.08802); [Makelov ICLR 2024](https://arxiv.org/abs/2311.17030)
- Susceptibilities are reparameterization-robust and localizable but costly and probe-scale dependent — [Baker ICLR 2026](https://arxiv.org/pdf/2504.18274)
- No representational measure dominates; calibration is required — [ReSi ICLR 2025](https://arxiv.org/html/2408.00531v2); [Gröger 2026](https://arxiv.org/html/2602.14486); [Ding 2021](https://ar5iv.labs.arxiv.org/html/2108.01661); [Cloos ICLR 2025](https://arxiv.org/html/2407.07059v3)
- Raw sharpness is not invariant — [Dinh 2017](https://arxiv.org/abs/1703.04933); [Kristiadi 2023](https://arxiv.org/abs/2302.07384v3); [Andriushchenko 2023](https://arxiv.org/abs/2302.07011)

### Inferences
**Recommended stack (most to least valid for "topology X produced structure Y"):**

1. **Function-space read-outs (Tier 1, always on).**
   - Measures:
     - soft and hard output agreement vs within-topology seed baselines;
     - Walsh/Fourier degree profile or purified functional-ANOVA interaction order under a stated input measure;
     - targeted behavioural probes that discriminate candidate algorithms (Clock vs Pizza-style signatures);
     - mechanistic-mode-connectivity-style invariance tests (shared input invariances).
   - Why: parameterization-free, and the graph change cannot create artifacts here.
   - Risks: metric discontinuity and input-measure dependence. Use continuous metrics and fixed input measures.

2. **Causal abstraction against a shared high-level model (Tier 1 for synthetic tasks with known algorithms).**
   - Method: DAS or interchange interventions with an identical restricted map class (linear or orthogonal, fixed rank).
   - Required controls:
     - held-out IIA;
     - random-init controls of each topology;
     - map-complexity reporting;
     - natural-distribution projection ablation to rule out dormant pathways;
     - multiple-comparison correction for the different number of candidate sites per topology.
   - Why: compares each network to the same reference, not to each other, so no unit correspondence is needed.

3. **Developmental Bayesian and geometric measures (Tier 2).**
   - LLC trajectories and refined (per-component) LLC, with estimator hyperparameters tuned and reported per topology.
   - Susceptibility PCA for component-level structure, compared within a single probe setup.
   - Phase-transition timing from LLC or loss plateaus.
   - Why: invariant to smooth reparameterization; captures when structure forms.
   - Risks: SGLD sensitivity and cost.

4. **Calibrated representational similarity at symmetry-matched sites (Tier 2–3).**
   - Choice of site: compare residual stream with residual stream, and post-nonlinearity with post-nonlinearity.
   - Report at least three families:
     - an orthogonal-invariant metric (angular Procrustes / generalized shape metric);
     - debiased CKA;
     - mutual k-NN.
   - Calibration: permutation nulls with layer-search correction, and seed baselines.
   - Pair every score with linear decodability of task variables (Harvey 2024; Cloos 2025).
   - Add direct-matching stitching with in-distribution checks, plus A→B vs B→A asymmetry.
   - Why below Tier 2: choosing cut points and invariance groups is an instrument decision, and scores are confounded by width and PC weighting.

5. **Structure statistics on activations and gradients (Tier 3).**
   - Bias-corrected participation ratio or effective rank at matched sample sizes.
   - Upstream and downstream functional modularity (reported separately).
   - Importance and coherence of clusters.
   - NTK kernel change relative to initialization, plus target alignment.
   - Compositionality metrics (TRE) on synthetic compositional tasks.
   - Weight-graph clusterability only against topology-mask-preserving nulls.
   - Why: single-network descriptors avoid correspondence, but definitions disagree (upstream vs downstream) and baselines must respect the fixed connectivity.

6. **Instrument-heavy measures (Tier 4, corroborative only).**
   - SAE- or crosscoder-based feature universality and superposition ψ: use identical recipes and dictionary sweeps, BatchTopK over L1.
   - Training-data attribution profiles (TRAK/EK-FAC), with seed baselines and LDS validation.
   - Why: results depend on the SAE, crosscoder, or curvature approximation, all of which can interact with architecture.

7. **Within-topology only.** Do not use these to compare across graphs:
   - linear mode connectivity / Git Re-Basin barriers (use as "solution uniqueness per topology" statistics instead);
   - raw Hessian sharpness;
   - weight-graph modularity without topology-matched nulls.

**Cross-cutting validity rules:**
- (a) Every between-topology difference must exceed the within-topology seed spread.
- (b) Use the same input distribution, sample size, and site-selection rules for both topologies.
- (c) Report a null for every score: permutation for representations, random-init for causal measures, topology-mask-preserving weight shuffles for weight-graph measures.
- (d) Demand convergent evidence across at least two tiers before claiming "topology X produced structure Y."
- (e) Prefer topology-intervention designs: matched parameter count, with skips or branches toggled, as in Raghu 2021's skip removal. Correlational comparisons across unrelated architectures are weaker.

### Gaps
- The ranking is a reasoned synthesis. No published paper benchmarks these families against each other on same-parameter, different-topology networks.
- Compute budgets are approximate. Only susceptibilities had concrete published cost figures (Baker 2026: about $2k and $5k per checkpoint for Pythia-410M and 1.4B).
