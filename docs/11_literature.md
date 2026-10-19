# Annotated literature map: evidence, methods, and limitations

**Expanded after critical review, 19 October 2026.** The [critical review](14_critical_review_and_revised_scope.md) identified closer prior art in multiple-pathway deep-linear learning, spurious/core feature complexity, training-data origins of circuits and formal limitations of circuit fidelity. Read these **before** asserting our broad conceptual motivation is novel. References link to primary conference/author/preprint sources when available.

**Checked against primary paper/author or conference pages in October 2026.** This is a focused reading list, not a claim of exhaustive scholarship or endorsement. Distinguish reported results in a paper from our proposed applications. Full model, code and metric behavior can change across implementations.

## Reading route for a new research agent

**Revised priority:** (1) Saxe/Shi/Xu on exact and parallel-path learning dynamics, (2) Qiu/Pezeshki on feature and shortcut competition, (3) empirical functional change and training-data origins of circuits, (4) limitations of mechanistic intervention, (5) representational crowding/superposition, and only then (6) NAS. See [first matched study](15_matched_linear_topology_study.md) and [decision gates](17_decision_gates_and_novelty.md). Papers in the NAS group mostly optimize benchmark performance and should **not** define the project's objective.

## 0. Critical closest prior art: pathway competition, finite-width modes and learning origins

**Saxe, McClelland & Ganguli (2014), Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks.** [arXiv](https://arxiv.org/abs/1312.6120). Mathematically shows nontrivial plateau/learning-time and mode dynamics despite input-output linearity; depth and initialization alter optimization. **For this project:** essential analytic baseline and justification for a linear first experiment. **Limit:** their exact results rest on particular linear models/objectives and do not imply generalization circuits in a language model.

**Shi, Shea-Brown & Buice (NeurIPS 2022), Learning Dynamics of Deep Linear Networks with Multiple Pathways.** [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html). Analyzes shared-objective parallel pathways, finding that data correlation modes can specialize into competing pathways, with depth and finite-width initialization influencing symmetry breaking. **For this project:** closest baseline for shared-loss pathway specialization. Replicate relevant phenomena before declaring competition itself novel. **Limit:** their exact theory uses specific wide deep-linear assumptions; our edge-only pair and softmax CE require distinct analysis.

**Qiu, Kuang & Goel (ICML 2024), Complexity Matters: Feature Learning in the Presence of Spurious Correlations.** [PMLR](https://proceedings.mlr.press/v235/qiu24e.html). Controls Boolean spurious/core feature complexity and correlation; shows stronger/simpler shortcuts slow core-feature learning, subnetworks can specialize, phases can overlap and spurious features persist. **For this project:** direct prior art on the 'productive frustration' intuition and an experimental generator/calibration precedent. **Limit:** doesn't establish a general fixed-routing design law; do not reproduce the phenomenon then claim it as new.

**Xu & Ziyin (ICLR 2025), Three Mechanisms of Feature Learning in a Linear Network.** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9941833e8327910ef25daeb9005e4748-Abstract-Conference.html). Exact finite-width linear learning trajectories and distinct alignment, disalignment and rescaling regimes across width/learning-rate/initialization. **For this project:** warns that width and optimizer regime can explain a supposed topology effect; motivates phase maps. **Limit:** tightly specified solvable one-hidden-layer setting.

**Chen, Luo & Pan (ICML 2026), Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units.** [PMLR](https://proceedings.mlr.press/v306/chen26cv.html). Uses influence-based training-data attribution and targeted example interventions to change emergence of interpretable heads in Pythia models. **For this project:** concrete methodology/novelty baseline for connecting *acquisition* of mechanisms to training-time causes. **Limit:** interpreting influence functions and head-based units does not directly prove our parameter-free edge hypothesis.

## 0b. Negative evidence / formal tools for circuit claims

**Miller, Chughtai & Saunders (2024), Transformer Circuit Faithfulness Metrics are not Robust.** [arXiv](https://arxiv.org/abs/2407.08734). Shows circuit faithfulness scores are highly sensitive to choices of ablation method; an extracted circuit is relative to the task and intervention. **For this project:** reason to make $D_k,U_k$ secondary, use multiple baselines and never call one retained subgraph *the* algorithm.

**Anani et al. (ICML 2026), Certified Circuits: Stability Guarantees for Mechanistic Circuits.** [PMLR](https://proceedings.mlr.press/v306/anani26a.html), [code](https://github.com/AlaaAnani/certified-circuits). Certifies stability of circuit component inclusion under bounded perturbations of the concept dataset using randomized subsampling. **For this project:** inspire an abstention/uncertainty protocol and sensitivity checks. **Limit:** guarantees concern a specified discovery procedure and dataset edits, not universal semantic completeness.

**Hadad, Katz & Bassan (ICLR 2026), Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees.** [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html). Uses neural verification to provide input-domain robustness, patching and minimality guarantees in selected continuous-input circuit settings. **For this project:** rigorous calibration for small tractable graphs and a warning against casual 'minimal circuit' language. **Limit:** verifying a specified property is not unrestricted source-agnostic mechanism discovery.

**Dinh, Pascanu, Bengio & Bengio (2017), Sharp Minima Can Generalize For Deep Nets.** [arXiv](https://arxiv.org/abs/1703.04933). Parameter-space sharpness can change dramatically under equivalent reparameterizations. **For this project:** do not mistake a sharper/flatter Hessian for an invariant causal property of learned functions or circuit reuse.

**Anthropic (2023), Superposition, Memorization, and Double Descent.** [Transformer Circuits](https://transformer-circuits.pub/2023/toy-double-descent/index.html). In toy models with finite data, a small-data regime packs instance-specific points in superposition, whereas larger-data regimes pack generalizing features. **For this project:** memorization and generalization are not necessarily neuron/path labels; feature load, sample support and sparsity are hidden scaling covariates. **Limit:** toy reconstruction setting, not universal claim about transformers or CE learning.


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

No listed work proves that long paths intrinsically generalize, that shallow paths intrinsically memorize, that maximizing branch disagreement helps learning, that topology alone explains human one-shot learning, or that an SAE/attribution graph exposes every computational circuit. **Several do demonstrate specific forms of pathway competition, specialized subnetworks, delayed circuit learning and sensitivity to intervention**—these general concepts are prior art, not our novelty claim. These are open, falsifiable questions.

## Reading-to-implementation map

- Initial matched linear-pair experiments and prior-art checks → Saxe + Shi + Xu; [study](15_matched_linear_topology_study.md), [math](03_first_principles_math.md).
- Softmax CE and shortcut learning dynamics → Jacot + Pezeshki + Qiu + Kumar; [dynamics](04_gradient_dynamics.md), [function/transfer](16_functional_learning_and_transfer_measurement.md).
- Time-resolved mechanism progress → Power + Nanda + Olsson.
- Later causal extraction, origin and validation → Conmy + Marks + Bricken + Geiger + Zhang/Nanda + Miller + Chen + Anani + Hadad.
- Representational packing and measurement bias → Elhage + Anthropic toy-double-descent + Kornblith + Leask + Chanin (2025/2026).
- Future controlled topology search → Li/Talwalkar + Mellor + BANANAS + NAS-Bench-201.

Never let a method's convenience determine which scientific phenomenon is declared to exist.
