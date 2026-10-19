# Annotated literature map: evidence, methods, and limitations

**Checked against primary paper/author or conference pages in October 2026.** This is a focused reading list, not a claim of exhaustive scholarship or endorsement. Distinguish reported results in a paper from our proposed applications. Full model, code and metric behavior can change across implementations.

## Reading route for a new research agent

Start with (1) residual path behavior, (2) gradient starvation + grokking dynamics, (3) mechanism discovery + causal patching, (4) superposition/SAE critique, and only then (5) architecture search. Papers in the NAS group mostly optimize benchmark performance and should **not** define the project's objective.

## I. Residual paths and numerical optimization

**He et al. (2016), Deep Residual Learning for Image Recognition.** [arXiv](https://arxiv.org/abs/1512.03385). Canonical residual formulation and empirical depth/optimization motivation. **Our use:** conceptual baseline $h_{l+1}=h_l+F_l(h_l)$. **Limit:** a design success does not establish which learning circuits are formed or that the objective is convex.

**Veit, Wilber & Belongie (2016), Residual Networks Behave Like Ensembles of Relatively Shallow Networks.** [NeurIPS](https://papers.nips.cc/paper_files/paper/2016/hash/37bc2f75bf1bcfe8450a1a41c200364c-Abstract.html). Explores effective path lengths, gradient contributions and lesion behavior. **Our use:** quantify path inventories and discriminate reachable pathways from actually contributing routes. **Limit:** linearized path-expansion intuition cannot be directly equated with unique nonlinear circuits or semantic complexity.

**Balduzzi et al. (2017), The Shattered Gradients Problem.** [PMLR](https://proceedings.mlr.press/v70/balduzzi17b.html). Shows gradient correlations can deteriorate with depth and skip connections can preserve correlation; special initialization can mitigate without skips. **Our use:** a control on gradient structure distinct from gradient magnitude. **Limit:** not an account of memorization vs generalization.

**Li et al. (2018), Visualizing the Loss Landscape of Neural Nets.** [NeurIPS](https://mlanthology.org/neurips/2018/li2018neurips-visualizing/). Filter-normalized landscape visualization and relation to architecture. **Our use:** why landscape comparisons demand parameterization/normalization caution. **Limit:** loss visualizations do not prove global convexity or mechanistic circuit differences.

**Tarnowski et al. (2019), Dynamical Isometry Is Achieved in Residual Networks in a Universal Way for Any Activation Function.** [AISTATS/PMLR](https://proceedings.mlr.press/v89/tarnowski19a.html). Studies Jacobian singular spectra and emphasizes initialization as confounder. **Our use:** calibrated-signal controls in topology comparisons. **Limit:** well-conditioned propagation is not sufficient for reusable circuit emergence.

## II. Example efficiency, CE competition and changing representations

**Jacot, Gabriel & Hongler (2018), Neural Tangent Kernel: Convergence and Generalization in Neural Networks.** [arXiv](https://arxiv.org/abs/1806.07572). Describes function-space local learning geometry, with limiting constant NTK at infinite width under its assumptions. **Our use:** single-example transfer formula and diagnostic contrast of lazy vs feature-learning dynamics. **Limit:** NTK drift alone is not a useful circuit or its causal role.

**Pezeshki et al. (2021), Gradient Starvation: A Learning Proclivity in Neural Networks.** [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html). Formalizes how CE gradient dynamics can favor some predictive features over others. **Our use:** shared residual and shallow-feature saturation motivation. **Limit:** feature competition depends on data/statistics/optimizer and is not proof that architecture should isolate branches.

**Power et al. (2022), Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.** [arXiv](https://arxiv.org/abs/2201.02177). Small algorithmic tasks and delayed generalization after memorization. **Our use:** experimental task/scale regimes, checkpointing through long training tails. **Limit:** a specific grokking setup is not a universal training dynamic.

**Nanda et al. (2023), Progress Measures for Grokking via Mechanistic Interpretability.** [arXiv](https://arxiv.org/abs/2301.05217), [paper/code site](https://www.neelnanda.io/grokking-paper). Reconstructs a Fourier-based modular-addition mechanism and separates memorization, circuit formation, and cleanup stages. **Our use:** mechanistic progress measures ahead of visible test accuracy; training-time circuit histories. **Limit:** this is a hand-understood mechanism; don't force every discovery into the same Fourier template.

**Kumar et al. (2024), Grokking as the Transition from Lazy to Rich Training Dynamics.** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/63ed15a46a143ff57484b38cd6b85d91-Abstract-Conference.html). Studies delayed shift from approximately kernel-like fitting to feature learning, and sensitivity to initial feature alignment, data and scaling. **Our use:** separate fast fitting from circuit development. **Limit:** some conclusions derive from simplified teacher-student/polynomial settings.

**Olsson et al. (2022), In-context Learning and Induction Heads.** [Transformer Circuits](https://www.transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). Links induction-head emergence and a phase change in in-context learning, including architectural interventions. **Our use:** precedent for tracking a developing mechanism and changing structure to test causality. **Limit:** attention-specific learned routing differs from fixed skip edges.

## III. Mechanistic circuit discovery and causal validity

**Conmy et al. (2023), Towards Automated Circuit Discovery for Mechanistic Interpretability.** [arXiv](https://arxiv.org/abs/2304.14997), [code](https://github.com/ArthurConmy/Automatic-Circuit-Discovery). ACDC automates selecting important graph edges for specified behaviors using activation patching. **Our use:** inspiration for graph interventions and subgraph search. **Limit:** still needs an external behavior/metric and interventions can be baseline-dependent.

**Marks et al. (2025), Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ba4d47a83e498c2b1a0868cba20f6de-Abstract-Conference.html), [code](https://github.com/saprmarks/feature-circuits). Shows automatic identification of numerous circuits for automatically discovered behaviors, using sparse features. **Our use:** closest precedent for unanticipated behavior → circuit discovery. **Limit:** fidelity and feature definitions depend on sparse decomposition.

**Bricken et al. (2025), Circuit Tracing: Revealing Computational Graphs in Language Models.** [Methods article](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html), [open tools](https://www.anthropic.com/research/open-source-circuit-tracing). Cross-layer feature attribution graphs, including discussion of approximation errors, inactive/suppressive features and unmodeled attention-query/key circuitry. **Our use:** attribution candidate generation and empirical coverage accounting. **Limit:** an approximate replacement model's graph is not automatically a complete causal decomposition of the original.

**Geiger et al. (2023), Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability.** [arXiv](https://arxiv.org/abs/2301.04709). Formalizes high-to-low level causal correspondence, faithful abstractions and intervention tests. **Our use:** operational circuit definitions and graded faithfulness. **Limit:** no single canonical abstraction or guaranteed discovery method.

**Zhang & Nanda (2024), Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html). Shows that corruption/intervention choice and metrics can change circuit-localization results. **Our use:** mandatory sensitivity checks across patch baselines. **Limit:** measured component importance depends on chosen prompts and counterfactuals.

## IV. Superposition and measurement robustness

**Elhage et al. (2022), Toy Models of Superposition.** [Transformer Circuits](https://transformer-circuits.pub/2022/toy_model/index.html). Analyzes feature packing, sparsity, importance and geometry in toy networks, and provides code/replication resources. **Our use:** vary latent feature demand, activity and width jointly; test whether topology affects interference. **Limit:** toy features aren't a complete ontology for LLM representations.

**Kornblith et al. (2019), Similarity of Neural Network Representations Revisited.** [ICML/PMLR](https://proceedings.mlr.press/v97/kornblith19a). Introduces/characterizes CKA and invariances/limitations of representation similarity. **Our use:** basis-aware comparison across seeds/checkpoints. **Limit:** similarity of representations does not prove identical computation or causal role.

**Leask et al. (2025), Sparse Autoencoders Do Not Find Canonical Units of Analysis.** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/84ca3f2d9d9bfca13f69b48ea63eb4a5-Abstract-Conference.html). SAE stitching/meta-SAEs demonstrate incompleteness and non-atomicity. **Our use:** no SAE feature counts as primary 'number of circuits'; sensitivity across dictionaries. **Limit:** SAEs can still provide useful pragmatic analyses.

**Chanin et al. (2025), A Is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders.** [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/764ff7477b8e24dbe01531f6791e8bdf-Abstract-Conference.html). Studies features failing to fire because information is absorbed by subfeatures, even when apparently monosemantic. **Our use:** test detector sensitivity to under- and overcomplete dictionaries, inactive features. **Limit:** not a general proof all feature dictionaries are unusable.

**Chanin (2026), Are Sparse Autoencoder Benchmarks Reliable?** [arXiv preprint](https://arxiv.org/abs/2605.18229), [code](https://github.com/decoderesearch/saebench-reliability-audit). Audits SAE benchmark noise, ground-truth correlations and discrimination; several common settings unreliable. **Our use:** validate interpretability metrics with planted ground truth and reseeding. **Status/limit:** preprint; specific audited metrics and versions, not a blanket rejection of SAE analysis.

## V. Neural architecture search: methodology, not objective

**Li & Talwalkar (2020), Random Search and Reproducibility for Neural Architecture Search.** [PMLR](https://proceedings.mlr.press/v115/li20c.html). Random search is a competitive baseline; emphasizes reproducibility. **Our use:** a later unbiased comparator for graph search. **Limit:** optimizes benchmark accuracy, not causal circuit properties.

**Mellor et al. (2021), Neural Architecture Search without Training.** [ICML/PMLR](https://proceedings.mlr.press/v139/mellor21a.html). Activation overlap at initialization can predict architecture benchmark performance in some NAS spaces. **Our use:** possibility of cheap topology descriptors. **Limit:** initial activation proxy may favor short-term fit rather than long-term mechanism formation.

**White, Neiswanger & Savani (2021), BANANAS: Bayesian Optimization with Neural Architectures for Neural Architecture Search.** [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17233), [code](https://github.com/naszilla/naszilla). Neural surrogate and path-based architecture encodings. **Our use:** later mechanism-aware surrogate modeling after scientific metrics become predictive. **Limit:** surrogate objectives and sampling strategy can optimize the wrong quantity.

**Dong & Yang (2020), NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search.** [arXiv](https://arxiv.org/abs/2001.00326). A standardized small DAG search space with reproducible evaluations. **Our use:** inspiration for exhaustively defined graph spaces and transparent resource accounting. **Limit:** architecture-level image accuracy doesn't measure circuit formation.

## What literature *doesn't* establish for our project

No listed work proves that long paths intrinsically generalize, that shallow paths intrinsically memorize, that maximizing branch disagreement helps learning, that topology alone explains human one-shot learning, or that an SAE/attribution graph exposes every computational circuit. These are open, falsifiable questions.

## Reading-to-implementation map

- Exact scalar math/path oracle → Veit + Tarnowski; our docs/03.
- Softmax CE and residual learning dynamics → Jacot + Pezeshki + Kumar; our docs/04.
- Time-resolved mechanism progress → Power + Nanda + Olsson.
- Circuit extraction and causal validation → Conmy + Marks + Bricken + Geiger + Zhang/Nanda.
- Representational packing and measurement bias → Elhage + Kornblith + Leask + Chanin (2025/2026).
- Future controlled topology search → Li/Talwalkar + Mellor + BANANAS + NAS-Bench-201.

Never let a method's convenience determine which scientific phenomenon is declared to exist.
