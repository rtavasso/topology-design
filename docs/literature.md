# Literature map

Grouped by the link of the causal chain ([SPEC §2](SPEC.md)) each work informs, with what we use it for. Links marked † were checked against the publisher or arXiv page on 19 Oct 2026. Unlinked entries are standard references cited by title and venue; verify details before quoting them.

The v0 literature map ([archive](../archive/v0/docs/11_literature.md)) is retained. Its entries are reused here, with one correction: the *Circuit Tracing* methods paper is **Ameisen et al. (2025)**, not Bricken et al.

## 1. Parameterization → dynamics → implicit bias (Phase 0)

These answer most of the v0 Study 1 in closed form. The v0 review did not cite any of them.

- **Arora, Cohen & Hazan (ICML 2018), *On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization*.** Balanced deep linear chains follow a closed-form preconditioned flow on the end-to-end matrix. **Use:** Prop. 3 and Prop. 4.
- **Du, Hu & Lee (NeurIPS 2018), *Algorithmic Regularization in Learning Deep Homogeneous Models*.** Layer balancedness is conserved under gradient flow.
- **Arora, Cohen, Hu & Luo (NeurIPS 2019), *Implicit Regularization in Deep Matrix Factorization*.** Depth strengthens low-rank bias. **Use:** expected implicit-bias direction for chains.
- **Hardt & Ma (ICLR 2017), *Identity Matters in Deep Learning*; Bartlett, Helmbold & Long (ICML 2018), identity-initialized deep linear residual networks.** **Use:** the identity-initialized regime of our residual family; explains why Saxe-style plateaus are absent at identity init.
- **Kawaguchi (NeurIPS 2016); Laurent & von Brecht (ICML 2018).** Deep linear networks have no bad local minima. **Use:** Prop. 6 endpoint equivalence.
- **Soudry, Hoffer, Nacson, Gunasekar & Srebro (JMLR 2018); Gunasekar, Lee, Soudry & Srebro (NeurIPS 2018, linear convolutional networks); Ji & Telgarsky (ICLR 2019).** Directional implicit bias under separable cross-entropy, and how architecture changes it. **Use:** Prop. 6 separable case (open).
- **Woodworth et al. (COLT 2020), *Kernel and Rich Regimes in Overparametrized Models*.** Parameterization plus init scale interpolate between kernel and rich implicit bias. **Use:** the richness axis.
- [**Kunin et al. (NeurIPS 2024), *Get Rich Quick*.**](https://arxiv.org/abs/2406.06158)† Unbalanced initializations promote rapid feature learning; exact solutions. **Use:** Phase 0 imbalanced-init item.
- [**Dominé et al. (ICLR 2025), *From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks*.**](https://mlanthology.org/iclr/2025/domine2025iclr-lazy)† Lazy-to-rich transitions controlled by λ-balancedness.
- [**Marcotte, Gribonval & Peyré (NeurIPS 2023), *Abide by the Law and Follow the Flow*.**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c7bee9b76be21146fd592fc2b46614d5-Abstract.html)† An algorithm for finding all conservation laws of a parameterization. **Use:** Prop. 7, mixed graphs.
- **Kunin et al. (ICLR 2021), *Neural Mechanics*; Zhao et al. (ICLR 2023), *Symmetries, Flat Minima, and the Conserved Quantities of Gradient Flow*.** Symmetry → conserved quantity → dynamics.
- [**Lippl, Abbott & Chung (2022), *The Implicit Bias of Gradient Descent on Generalized Gated Linear Networks*.**](https://www.researchgate.net/publication/358423122_The_Implicit_Bias_of_Gradient_Descent_on_Generalized_Gated_Linear_Networks)† Architecture-dependent implicit bias with fixed gates.
- **Saxe, McClelland & Ganguli (ICLR 2014)** [arXiv](https://arxiv.org/abs/1312.6120); **Xu & Ziyin (ICLR 2025)**; **Shi, Shea-Brown & Buice (NeurIPS 2022)**. Retained from v0. Note that Saxe 2014's staged mode learning depends on *small* initialization.

## 2. Routing → emergent structure (Phase 1, 3)

- [**Saxe, Sodhani & Lewallen (ICML 2022), *The Neural Race Reduction: Dynamics of Abstraction in Gated Networks*.**](https://arxiv.org/abs/2207.10430)† Fixed gating defines pathways; learning is a race won by shared pathways, producing abstraction. **Use:** the core theory for Phase 1.
- [**Jarvis, Klein, Rosman & Saxe (ICLR 2025), *Make Haste Slowly*.**](https://arxiv.org/abs/2503.06181)† Feature-learning ReLU networks map onto equivalent gated deep linear networks. **Use:** linear-to-nonlinear bridge.
- [**Jarvis, Klein, Rosman & Saxe (ICLR 2023), *On the Specialization of Neural Modules*.**](https://mlanthology.org/iclr/2023/jarvis2023iclr-specialization)† When modular architectures specialize and support systematic generalization. **Use:** Phase 3 shared-subcomputation task.
- **Lampinen & Ganguli (ICLR 2019), *An Analytic Theory of Generalization Dynamics and Transfer Learning in Deep Linear Networks*.** **Use:** Phase 3 transfer baseline.
- **Csordás, van Steenkiste & Schmidhuber (ICLR 2021), *Are Neural Nets Modular?*; Mittal et al. (NeurIPS 2022), *Is a Modular Architecture Enough?*** Modular structure does not automatically yield specialization or reuse. **Use:** a null to beat.
- **Qiu, Kuang & Goel (ICML 2024); Pezeshki et al. (NeurIPS 2021).** Retained from v0: shortcut versus core-feature competition.

## 3. Construction, plateaus, hidden progress (Phase 2: the formal version of "frustration")

- [**Abbe, Boix-Adserà & Misiakiewicz (COLT 2023), *SGD Learning on Neural Networks: Leap Complexity and Saddle-to-Saddle Dynamics*.**](https://proceedings.mlr.press/v195/abbe23a.html)† Which Boolean functions SGD learns, and in what order. **Use:** designing Phase 2 targets; leap relative to learned intermediates is our hypothesis.
- [**Barak et al. (NeurIPS 2022), *Hidden Progress in Deep Learning*.**](https://arxiv.org/abs/2207.08799)† Measurable progress during loss plateaus. **Use:** Phase 2 plateau measurements; this is the v0 "hidden learning" question.
- **Jacot et al. (2021), saddle-to-saddle dynamics in deep linear networks; Pesme & Flammarion (NeurIPS 2023), diagonal linear networks.** Plateau and escape theory.
- **Poggio et al. (2017), *Why and When Can Deep — but Not Shallow — Networks Avoid the Curse of Dimensionality*.** Compositional targets and when architecture matches task structure.
- **Nanda et al. (ICLR 2023); Power et al. (2022); Kumar et al. (ICLR 2024).** Retained from v0: grokking and lazy-to-rich transitions.

## 4. Topology as a variable in practice (Phase 4, 5)

- [**You, Leskovec, He & Xie (ICML 2020), *Graph Structure of Neural Networks*.**](https://proceedings.mlr.press/v119/you20b.html)† Relational-graph statistics (clustering, path length) predict performance. **Use:** the closest precedent for "graph descriptors predict outcomes."
- **Huang et al. (CVPR 2017), DenseNet** (explicit feature-reuse claim); **Larsson et al. (ICLR 2017), FractalNet**; **Xie et al. (ICCV 2019), randomly wired networks**; **Veit et al. (NeurIPS 2016)**, retained from v0.
- [**Chowdhery et al. (2022), PaLM**](https://arxiv.org/abs/2204.02311)†, and GPT-J: parallel attention/MLP blocks, with a reported scale-dependent quality cost relative to sequential blocks. **Use:** Phase 4 transformer contrast.
- **Bachlechner et al. (2021), ReZero; De & Smith (NeurIPS 2020), SkipInit; Zhang et al. (ICLR 2019), Fixup.** Zero- or small-scaled residual branches. **Use:** the branch multiplier is our richness knob.
- [**Bordelon et al. (ICLR 2024), *Depthwise Hyperparameter Transfer in Residual Networks*.**](https://mlanthology.org/iclr/2024/bordelon2024iclr-depthwise)† **Yang et al. (ICLR 2024), *Tensor Programs VI*.** **Use:** Phase 5 fair scaling.

## 5. Reuse, interference, memorization (Phase 3)

- **Chatterjee (ICLR 2020), *Coherent Gradients*.** Generalization explained by gradient agreement across examples; this is the v0 $T(i,j)$ idea, already developed.
- **Fort et al. (2019), *Stiffness: A New Perspective on Generalization in Neural Networks*.**
- **Pruthi et al. (NeurIPS 2020), TracIn; Koh & Liang (ICML 2017), influence functions.**
- **Ren & Sutherland (ICLR 2025), *Learning Dynamics of LLM Finetuning*.** Single-step influence decomposition under softmax cross-entropy.
- **Maini et al. (ICML 2023), *Can Neural Network Memorization Be Localized?*** **Use:** testing the v0 H07 (memorization and generalization need not occupy disjoint subgraphs).
- **Arpit et al. (ICML 2017); Feldman (STOC 2020).**
- **Chen, Luo & Pan (ICML 2026)**, retained from v0: training-data origins of interpretable units.

## 6. Instruments

- **Lau et al. (2023), local learning coefficient estimation; Hoogland et al. (2024), *The Developmental Landscape of In-Context Learning*;** [**Wang et al. (ICLR 2025), refined LLC**](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fad7c708dda11f3e72cc1629bb130379-Abstract-Conference.html)†. Source-agnostic developmental stage detection. Singular learning theory also predicts that parameterization degeneracy shapes which solutions are preferred.
- **Lindner et al. (NeurIPS 2023), Tracr;** [**Gupta et al. (NeurIPS 2024), InterpBench**](https://arxiv.org/abs/2407.14494)†. Models with known ground-truth circuits for validating causal tools.
- **Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*.** The residual stream as a shared read/write channel; path expansion.
- Retained from v0 with their caveats: Miller et al. 2024; Zhang & Nanda 2024; Conmy et al. 2023; Geiger et al. 2023; Ameisen et al. 2025 (Circuit Tracing); Anani et al. 2026; Hadad et al. 2026; Leask et al. 2025; Chanin 2025/2026; Elhage et al. 2022; Kornblith et al. 2019; Dinh et al. 2017.

## Novelty boundary (current best understanding)

**Known (do not claim):**
- deep linear flows and implicit bias for chains, diagonal, convolutional and gated-linear parameterizations;
- pathway races in gated networks;
- shortcut/core competition;
- gradient coherence as an account of generalization.

**Candidate contributions (check further before claiming):**
1. implicit bias and conserved quantities for **general linear residual DAGs** (Phase 0, Prop. 7);
2. **richness-gated topology effects** across a residual family;
3. **composition lowering effective leap** relative to learned intermediates (Phase 2);
4. **topology-dependent erasure cost** (Phase 4).
