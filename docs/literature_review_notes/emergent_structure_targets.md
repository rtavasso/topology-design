# Emergent Structures Networks Optimize Toward, and the Architectural/Connectivity Factors That Change Them

Scope note: the papers the program already has (Elhage 2022 TMS; Anthropic 2023 superposition/memorization/double descent; Nanda 2023; Maini 2023; Csordás 2021 and Mittal 2022 modularity; Qiu 2024; Saxe 2022 neural race) are only referenced, not re-summarized. Each item gives its year, venue, and URL. "[venue unverified]" means the page I fetched did not state a venue and I could not confirm one from it. A tag marks what was manipulated in each item: **[CONNECTIVITY]** = residual/skip/dense/parallel routing manipulated; **[NORM]** = normalization placement; **[ATTN-vs-MLP]** = block type; **[DEPTH/WIDTH ONLY]** = only size or shape varied; **[NONE]** = descriptive. Search budget ran out near the end of the session, so a few well-known items rely on prior knowledge plus their canonical URL. Those are marked "(not re-fetched)".

---

## 1. Low-rank / simplicity / spectral bias: definition, drivers, and architecture dependence

### Takeaway
Simplicity bias is strongly architecture-dependent, and connectivity is among the few factors with direct causal evidence. Residual connections push random-init networks to the lowest functional complexity regardless of depth (Neural Redshift). Pure attention without skips collapses to rank 1 doubly-exponentially. *Decaying* the identity shortcut lowers feature effective rank and improves abstraction in generative SSL. Plain depth lowers effective rank in linear/plain nets, but ResNets do not show monotonic rank contraction. Kernel analyses suggest skips leave the frequency-decay rate largely unchanged and mainly change locality and conditioning.

### Cited Findings
**Definitions / operationalizations**
- Low-rank simplicity bias (Huh et al., TMLR 2023): deeper networks are biased toward embeddings with lower *effective rank* (entropy of the normalized singular-value distribution of the embedding Gram matrix). The proposed cause is that "the volume of functions that maps to low effective rank embedding increases with depth." The bias appears at init and after training — [Huh et al. 2023, arXiv 2103.10427](https://arxiv.org/abs/2103.10427); venue: [TMLR 2023 (mlanthology)](https://mlanthology.org/tmlr/2023/huh2023tmlr-lowrank)
- In the same paper, linear over-parameterization (expanding each layer into a product of linear layers, with the same capacity) induces low rank and helps CIFAR. On CIFAR-100: 57.0 → 63.5 (4x FC+Conv); on ImageNet the gain shrinks with depth (AlexNet +1.8, ResNet10 +0.9, ResNet18 +0.4), and "starting from ResNet34... linearly over-parameterized models perform worse" — [Huh et al. PDF](https://arxiv.org/pdf/2103.10427v4)
- **[CONNECTIVITY] Huh et al. appendix:** "ResNets, unlike linear networks, *do not* exhibit a monotonic rank contracting behavior." Effective rank "plateaus after 8 layers, regardless of using batch-normalization or not" (analyzed in linear residual nets). A full residual study is left to future work — [Huh et al. PDF](https://arxiv.org/pdf/2103.10427v4)
- Rank diminishing (Feng et al., 2022): claims a "universal monotonic decreasing property of network rank" with depth from composition rules, checked per layer on ResNets, deep MLPs, and Transformers on ImageNet. Skips are not analyzed separately — [arXiv 2206.06072](https://arxiv.org/abs/2206.06072) [venue unverified; I believe NeurIPS 2022]
- Spectral bias (Rahaman et al., ICML 2019): networks fit low-frequency components first. This is the canonical frequency operationalization (Fourier spectrum of the learned function over training) — [arXiv 1806.08734](https://arxiv.org/abs/1806.08734) (not re-fetched)
- Parameter-function-map simplicity (Valle-Pérez et al., 2019): the parameter→function map is "exponentially biased toward simple functions" (via an algorithmic-information-theory bound). Shown for FC and CNN nets on Boolean functions, MNIST, and CIFAR10, with PAC-Bayes bounds through GP marginal likelihood — [arXiv 1805.08522](https://arxiv.org/abs/1805.08522) [venue: ICLR 2019, from prior knowledge]

**Neural Redshift (Teney et al., CVPR 2024): the most direct architecture-ablation study of simplicity at initialization**
- They measure the complexity of functions implemented by random-weight networks (Fourier frequency content / compressibility) while varying activations, weight magnitude, depth, width, bias, LayerNorm, residuals, multiplicative gating, and untrained GPT-2 — [arXiv 2403.02241](https://arxiv.org/html/2403.02241v3); [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Teney_Neural_Redshift_Random_Networks_are_not_Random_Functions_CVPR_2024_paper.html)
- **[CONNECTIVITY] Residual connections** force "the preferred complexity to some of the lowest levels for all activations," regardless of depth, and reduce sensitivity to weight magnitude. The attributed reason is that skips bypass the stacked nonlinearities that add complexity with depth — [Teney et al.](https://arxiv.org/html/2403.02241v3)
- **[NORM] LayerNorm** removes "variations in complexity with the weights' magnitude for all activations." With GELU, the learnable post-LN scale acts as a one-number complexity knob — [Teney et al.](https://arxiv.org/html/2403.02241v3)
- **Multiplicative interactions** (gating, as in attention/highway) give a clear depth- and magnitude-dependent *increase* in complexity, even with ReLU — [Teney et al.](https://arxiv.org/html/2403.02241v3)
- **[DEPTH/WIDTH]** Width and bias magnitude have no meaningful effect. ReLU networks "maintain their simplicity bias for any depth and weight magnitudes." TanH, Gaussian, and sine grow sharply more complex with depth — [Teney et al.](https://arxiv.org/html/2403.02241v3)
- Untrained GPT-2 produces fairly simple sequences. Scaling activation magnitude via the LN scale raises complexity monotonically — [Teney et al.](https://arxiv.org/html/2403.02241v3)

**Rank collapse and skip connections in attention**
- **[CONNECTIVITY][ATTN-vs-MLP]** Without skip connections or MLPs, pure self-attention output "converges doubly exponentially to a rank-1 matrix" (token uniformity). Skips and MLPs "stop the output from degeneration" — [Dong, Cordonnier, Loukas 2021, arXiv 2103.03404](https://arxiv.org/abs/2103.03404) [venue: ICML 2021, from prior knowledge]
- **[CONNECTIVITY] Decayed identity shortcuts** (Zhang, Jiang, Gao, Willett, Maire, 2024): x_{l+1} = α_l·x_l + f(x_l), with α decaying linearly with depth to α_min. Lower α_min gives lower feature effective rank and better abstraction. On MAE ViT-B/16 ImageNet-1K, linear probe goes 67.8 → 72.7% and k-NN 27.4 → 63.9%, while fine-tuning drops slightly (83.6 → 82.9). SiT-XL/2 diffusion FID goes 17.2 → 11.8. The authors present the rank↔probe link as correlational. A supervised ResNet-18 also shows lower rank with lower α_min — [arXiv 2404.10947, "Residual Connections Harm Generative Representation Learning"](https://arxiv.org/pdf/2404.10947) [venue unverified]

**Kernel (NTK/GP) view of skips and frequency bias**
- **[CONNECTIVITY]** With ReLU, residual convolutional kernels' eigenvalues decay polynomially at about the same rate as non-skip kernels, so the frequency bias is similar. Residual kernels are *more locally biased* and better conditioned at finite depth (faster GD convergence) — [Barzilai, Geifman, Galun, Basri 2022, arXiv 2211.14810](https://arxiv.org/abs/2211.14810) [venue unverified]
- **[CONNECTIVITY]** ResNTK on the sphere: eigenvalues decay as k^{-d}, the same as FC-NTK, so the RKHS is the same. Without biases, deep ResNTK is biased toward even frequencies. Unlike deep FC-NTK, which becomes "spiky," ResNTK stays stable. Its properties depend on a skip/residual balance hyperparameter — [Belfer et al., JMLR 25, 2024](https://jmlr.org/papers/v25/22-0597.html)

**Depth-dependent "dead depth" in Pre-LN**
- **[NORM][CONNECTIVITY]** "Curse of Depth": in Llama/Mistral/DeepSeek/Qwen, "nearly half of the layers are less effective than expected." The attributed cause is Pre-LN output variance growing with depth, so deep blocks act near-identity. The fix, LayerNorm Scaling (∝1/√depth), improves 130M–7B pretraining — [arXiv 2502.05795](https://arxiv.org/abs/2502.05795), NeurIPS 2025 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html))
- **[CONNECTIVITY]** Depthwise μP: scaling the residual branch by 1/√depth gives a well-defined feature-learning limit as width and depth grow (DMFT), and makes hyperparameters transfer across depth for ResNets and ViTs — [Bordelon et al. 2023, arXiv 2309.16620](https://arxiv.org/abs/2309.16620) [venue: ICLR 2024, from prior knowledge]

### Inferences
- Branch scaling (α on the identity, 1/√L on the residual branch, LN scale) behaves like a continuous control parameter for simplicity and effective rank. It is a parameter-free, connectivity-level knob, well suited to a topology study's design matrix.
- Evidence conflicts on residuals and rank. Neural Redshift says residuals lower *functional* complexity at init, while Huh et al. and Zhang et al. say identity shortcuts keep *representation* rank high (no monotone contraction; carrying high-frequency detail forward). These are different quantities (input→output function complexity vs. embedding rank), so a topology study should measure both.
- Kernel results suggest skips change locality and conditioning more than the asymptotic frequency-decay rate. So in the lazy regime the spectral-bias exponent may be a weak dependent variable, and differences are more likely to appear in feature-learning regimes.

### Gaps
- I found no study that holds parameters fixed and compares *parallel vs composed* branch arrangements, or dense (DenseNet-style) vs residual connectivity, on effective rank or spectral bias.
- Rahaman 2019 and Valle-Pérez 2019 do not examine residual/dense topologies as far as I know (not verified this session).
- I could not confirm the exact complexity metric Neural Redshift uses for the transformer experiments (sequence compressibility is likely).

---

## 2. Modularity, specialization, and compositional generalization

### Takeaway
Trained networks become more clusterable than chance, and modular "functional experts" emerge early in pretraining. However, *structural* modularity does not guarantee *functional* specialization: it appears only when the task is separable and resources are constrained. Connectivity-like manipulations with real effects on compositional generalization are weight sharing across layers (Universal Transformer), relative position encodings, copy mechanisms, and depth at fixed parameters. Multi-stream residuals (hyper-connections) tend to collapse onto one dominant stream instead of specializing. Lake & Baroni show that a standard transformer reaches human-like systematicity through the training distribution (meta-learning), not through architecture.

### Cited Findings
**Measuring emergent modularity**
- Clusterability (Filan et al. 2021): partition the weight graph into groups of neurons with strong internal and weak external connections (spectral partitioning, n-cut, in the paper body). Trained nets are more clusterable than random inits and than random nets with the same weight distribution. Regularizers can raise clusterability in MLPs at small accuracy cost — [arXiv 2103.03386](https://arxiv.org/abs/2103.03386) [venue unverified]
- BIMT ("Seeing is Believing," Liu, Gan, Tegmark 2023): neurons are embedded in 2D/3D space, with a distance-weighted L1 wiring cost plus function-preserving neuron swaps. Independent sub-formulas split into parallel modules, and compositional computations route through a single intermediate neuron. Modular addition becomes a tree with 3 voting modules (knocking out one drops accuracy from 100% to about 15–34%). S4 leaves 9 active embedding neurons. MNIST accuracy goes 98.5 → 98.0% — [arXiv 2305.08746](https://ar5iv.labs.arxiv.org/html/2305.08746); published in Entropy (Dec 2023) per [DOAJ](https://doaj.org/article/2d52897fe9054cbdac32e40e03e5980d)
- Emergent modularity in pretrained transformers (Zhang et al., ACL Findings 2023): neurons specialize, and MoE-style partitions yield "functional experts" whose perturbation hurts the matching function. **Modular structure stabilizes early in pretraining, before individual neuron functions** (coarse-to-fine) — [arXiv 2305.18390](https://arxiv.org/abs/2305.18390v2); [ACL Findings](https://preview.aclanthology.org/setup/2023.findings-acl.250)
- **Structural ≠ functional modularity** (Béna & Goodman, Nature Communications 2025): sparse inter-module connectivity alone does not guarantee specialization. Specialization emerges only when environment features are separable and the network is strongly resource-constrained, and it varies over time with the timing and bandwidth of inter-module communication. These results held across several architecture variants — [Nat. Commun. 16, 2025, doi 10.1038/s41467-024-55188-9](https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-024-55188-9.html)
- **[CONNECTIVITY]** Modular hypernetwork students recover the teacher's modules (up to a linear transform) when support is compositional and connected and the student is not over-parameterized. They then generalize to held-out module combinations, while monolithic ANIL/MAML do not — [Schug et al. 2023, arXiv 2312.15001](https://arxiv.org/abs/2312.15001) [venue unverified; I believe ICLR 2024]
- **[CONNECTIVITY]** On modular tasks, theory says modular networks' sample complexity is "independent of task dimensionality," while nonmodular nets need samples exponential in input dimension — [Boopathy et al. 2024, arXiv 2409.05780](https://arxiv.org/abs/2409.05780) [venue unverified]
- (Already known) end-to-end modular systems are sub-optimal at specializing — [Mittal, Bengio, Lajoie 2022, arXiv 2206.02713](https://arxiv.org/abs/2206.02713)

**Multi-stream residuals: specialization vs collapse (directly relevant to fixed routing)**
- **[CONNECTIVITY]** Hyper-connections (HC) replace the single residual stream with n learnable-mixed streams. They are motivated by the "seesaw effect between gradient vanishing and representation collapse" and report gains over residuals in dense and MoE LLM pretraining and in vision — [Zhu et al. 2024, arXiv 2409.19606](https://arxiv.org/abs/2409.19606) [venue unverified; possibly ICLR 2025]
- **[CONNECTIVITY] Stream collapse:** in mHC-lite nanoGPT (n=4), one stream comes to dominate read/write routing and norm growth and holds most crosscoder-recovered features from layer 4 on. Deep-layer mixing matrices are near-identity, and freezing most of them to identity barely hurts perplexity. The proposed cause is rich-get-richer amplification of a small early asymmetry, since identical-copy init gives streams no prior role. A learned per-stream scale (LSS) reduces imbalance (perplexity 26.22 → 25.65) — [Alimaskina, Molodtsov, Beznosikov, ICML 2026 Workshop on Weight-Space Symmetries, arXiv 2606.03483](https://arxiv.org/abs/2606.03483)
- **[CONNECTIVITY]** Causal ablate-and-rescue on a 781M HC model: mid-layer streams form two CKA groups (a checkerboard pattern). Some streams are functionally redundant (each can rescue a joint ablation), others are asymmetric despite similar representations ("high representational similarity does not imply functional interchangeability"), and complementarity is rare. The study is small (65 prompts, one model) — [Peng et al. 2026, arXiv 2603.14833](https://arxiv.org/abs/2603.14833)

**Compositional generalization: architecture dependence**
- Hupkes et al. "Compositionality Decomposed" defines five tests (systematicity, productivity/length, substitutivity, localism, overgeneralization) on PCFG SET and compares recurrent, convolutional, and transformer seq2seq models — [arXiv 1908.08351](https://arxiv.org/abs/1908.08351) [venue: JAIR 2020, from prior knowledge]
- **[CONNECTIVITY: weight sharing; position routing]** Ontañón et al. 2022, averaged over 12 compositional datasets: relative position encodings with decoder-to-encoder relative positions raise sequence accuracy from 0.137 (absolute) to 0.354 (Addition 0.005 → 0.988). A copy decoder raises the average from 0.346 to 0.403. **Weight sharing across layers** gives the best results (small-2s 0.468 vs small-2 0.403; best overall large-4s 0.527) but hurts COGS. Depth 4 beats depth 2, and 6 hurts. Intermediate representations raise COGS from 0.278 to 0.784 — [arXiv 2108.04378](https://arxiv.org/abs/2108.04378) [venue: ACL 2022, from prior knowledge]
- **[CONNECTIVITY: weight sharing][NORM-like: embedding scaling]** Csordás, Irie, Schmidhuber (EMNLP 2021). Universal Transformer (shared layers): PCFG productivity 65 → 78% (85% with relative PE), PCFG systematicity 87 → 93%, CFQ output-length 57 → 77%, COGS slightly worse. Relative PE on SCAN length (cutoff 26): 30 → 72%. Position-embedding downscaling: COGS 62 → 80%. Dropping IID early stopping: COGS 35 → 81% — [arXiv 2108.12284](https://ar5iv.labs.arxiv.org/html/2108.12284); [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.49)
- **[DEPTH/WIDTH ONLY, parameter-matched]** At about 41M/134M/374M parameters, trading width for depth improves compositional generalization after fine-tuning, with quickly diminishing returns. The gain is not explained by LM loss — [Petty et al. 2023, arXiv 2310.19956](https://arxiv.org/abs/2310.19956) [venue unverified; I believe NAACL 2024]
- **[NONE: training-distribution control]** Lake & Baroni (Nature 2023): a standard 3+3-layer transformer of about 1.4M parameters with meta-learning (MLC) reaches 100% systematic answers in the best run (humans 80.7%), and SCAN add-jump error goes 99.27% → 0.22%. The same architecture without MLC scores 0%. Systematicity is "not inherent properties of the neural network architecture but, instead, are induced from data." MLC still fails the SCAN length and COGS structural splits — [Lake & Baroni, Nature 2023 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072)
- **[DEPTH]** Transformers learn shallow "shortcut" solutions to automata (O(log T), often O(1) depth) instead of recurrent ones, and these shortcuts are brittle OOD — [Liu et al. 2022, arXiv 2210.10749](https://arxiv.org/abs/2210.10749) [venue: ICLR 2023, from prior knowledge]
- **[CONNECTIVITY: cross-layer sharing]** Grokked transformers learn implicit reasoning over parametric knowledge, but OOD generalization fails for *composition* and succeeds for *comparison*. The authors suggest "encouraging cross-layer knowledge sharing" — [Wang et al. 2024, arXiv 2405.15071](https://arxiv.org/abs/2405.15071) [venue: NeurIPS 2024, from prior knowledge]

### Inferences
- Weight sharing across depth (a fixed recurrent routing) is the most repeatable architecture-level positive manipulation for compositional generalization (Ontañón; Csordás). It is not parameter-free in the program's sense, since it reduces parameters, so a parameter-matched contrast is needed.
- The HC stream-collapse result suggests that adding parallel routes does not by itself produce specialization. Symmetry-breaking at init and inter-route bandwidth seem decisive, consistent with Béna & Goodman. This bears directly on "parallel vs composed branches."
- Clusterability-style weight-graph metrics are cheap but confounded by wiring constraints. Functional metrics (ablation selectivity, task-conditional activation) are needed alongside them.

### Gaps
- I found no controlled study comparing residual vs dense vs plain topologies on clusterability or functional modularity at matched parameters.
- I did not verify whether Filan et al. tested CNNs or ResNets, or how clusterability changes with skips. Their abstract names only MLPs.

---

## 3. Feature reuse vs interference, and architecture dependence of catastrophic forgetting

### Takeaway
For forgetting, width matters a lot. Depth tends to hurt. Skip connections have no measurable effect in the one controlled ablation. BatchNorm and global average pooling matter (GAP raises forgetting by narrowing the classifier). Dense and depth-weighted connectivity produce structured, measurable reuse of distant-layer features, so reuse patterns are directly observable for a topology study.

### Cited Findings
- **[CONNECTIVITY][NORM][DEPTH/WIDTH]** Mirzadeh et al. "Architecture Matters in Continual Learning" (2022), Split CIFAR-100. Width lowers forgetting (CNN×4/×8/×16: 8.7/8.0/4.7). Depth raises it (MLP 2 → 8 layers: forgetting 31.5 → 35.4; CNN 3 → 6 layers: 8.7 → 12.4; ResNet-34 > ResNet-18: 39.9 vs 36.8). **Skip connections have little effect** (CNN×4 with skip: accuracy 68.2 vs 68.1, forgetting 8.9 vs 8.7, within std). BN mainly raises learning accuracy (76.4 → 81.7) and can *increase* forgetting when input statistics shift (Permuted MNIST). GAP raises forgetting (8.7 → 14.3), which a wider pre-GAP layer fixes (5.2). ResNets/WRNs learn better, while simple CNNs forget less — [arXiv 2202.00275](https://ar5iv.labs.arxiv.org/html/2202.00275) [venue unverified]
- **[DEPTH/WIDTH ONLY]** Wide networks forget less. Explanations involve gradient orthogonality, sparsity, and lazy training. Only width is studied — [Mirzadeh et al., ICML 2022](https://proceedings.mlr.press/v162/mirzadeh22a.html); [arXiv 2110.11526](https://arxiv.org/abs/2110.11526)
- **[CONNECTIVITY]** Lu et al. study width/depth, skip connections, global pooling, and down-sampling for CL, and build ArchCraft (AlexAC/ResAC) with 86%/61%/97% fewer parameters at SOTA CL performance — [arXiv 2404.14829](https://arxiv.org/abs/2404.14829); [IJCAI 2024](https://www.ijcai.org/proceedings/2024/514)
- **[NORM]** Plasticity loss splits into independent mechanisms. LayerNorm plus weight decay together preserve plasticity across synthetic nonstationary tasks and Atari — [Lyle et al. 2024, arXiv 2402.18762](https://arxiv.org/abs/2402.18762); [CoLLAs 2024](https://mlanthology.org/collas/2024/lyle2024collas-disentangling)
- **[CONNECTIVITY: dense windowing]** In DenseNets, local windows of about 7–13 previous layers come within about 0.3–0.5 points of full DenseNet-40 on CIFAR-10 (92.3 vs 92.65%) with about 37% of the parameters. Window 1 (plain chain) reaches about 68%. Reuse patterns shift with window size: small windows rely on the earliest accessible features, large windows favor nearby layers. Single dataset, no repeated runs — [Hess 2018, arXiv 1806.01935](https://ar5iv.labs.arxiv.org/html/1806.01935)
- **[CONNECTIVITY: dense depth-weighted averaging]** DenseFormer's learned DWA weights show "strong and structured reuse of activations from distant layers," and it matches deeper transformers' perplexity — [Pagliardini et al. 2024, arXiv 2402.02622](https://arxiv.org/abs/2402.02622) [venue unverified; I believe NeurIPS 2024]
- **[CONNECTIVITY: dynamic dense, per-stream]** MUDDFormer, with dense cross-layer connections decoupled into Q/K/V/residual streams: Pythia shows ">0.97 similarity in later layers" (progressive collapse), while MUDD keeps representations more distinct and has about 2.4× more active heads in deep layers. Removing the value-stream dense connections hurts most (perplexity 10.77 → 11.05). MUDDPythia-2.8B matches Pythia-6.9B — [Xiao et al. 2025, arXiv 2502.12170](https://arxiv.org/html/2502.12170v1) [venue unverified; I believe ICML 2025]
- **[CONNECTIVITY: value residual]** A value shortcut from layer 1 to every layer (ResFormer) reduces attention sinks and concentration (start-token importance under 10× other tokens, hidden-state norm peak under 3× average) and smooths the "concentration–dispersion–concentration" depth pattern. It reaches equal loss with about 10.4% fewer parameters (82M–468M) — [Zhou et al. 2024, arXiv 2410.17897](https://arxiv.org/html/2410.17897v3) [venue unverified]
- Cross-layer superposition: per-layer SAEs duplicate features that persist across layers, and crosscoders capture shared structure across layers. Features peak in one layer but often decay slowly with drifting directions, so they are "not simply passed along unchanged through residual connections." Early and late layers are shared across model sizes, while new representations appear in middle layers with scale — [Lindsey et al., Transformer Circuits, Oct 2024](https://transformer-circuits.pub/2024/crosscoders/index.html)
- Residual stream as communication bus: heads write to and read from low-rank subspaces ("communication channels") that can be predicted from weight SVDs. A 3D subspace in GPT-2 indexes list positions, and crowding it causes copy errors — [Merullo, Eickhoff, Pavlick 2024, arXiv 2406.09519](https://arxiv.org/abs/2406.09519) [venue unverified]

### Inferences
- In the one direct test, the residual vs plain contrast is a **null for forgetting**. That is an informative baseline: a topology study should check whether denser routing (DenseNet- or DWA-style), not just identity skips, changes interference.
- MUDD, ResFormer, and DenseFormer all report that adding cross-layer routes reduces late-layer representation collapse and dead heads. "Fraction of effective/active layers and heads" is a tractable dependent variable that responds to connectivity.
- Feature reuse can be operationalized directly through learned aggregation weights (DWA/HC matrices) or crosscoder cross-layer feature persistence.

### Gaps
- I found no study measuring gradient interference or forgetting under dense vs residual vs plain routing at matched parameters.
- Mirzadeh's skip ablation was in small CNNs only. ViTs with and without skips were not tested for forgetting.

---

## 4. Memorization vs generalization, grokking, and phase transitions

### Takeaway
Memorization in natural-text LMs is entangled with general features (forced localization fails), and isolating it requires routing-style interventions (MemSinks). Grokking is governed by laziness and output scale, weight norm, and circuit efficiency. Architectural effects come mainly through normalization placement and attention-vs-linear balance. Clock vs Pizza is switched by attention rate and width. The LN-on-MLP-input vs LN-on-QK choice changes grokking speed. Residual + pre-LN stabilization recovers grokking at depth 8 where plain depth-4 MLPs fail. Evidence on phase transitions and "emergence" warns that metric discontinuity can manufacture apparent transitions.

### Cited Findings
**Where memorization lives**
- Paragraph memorization (GPT-Neo 125M in the paper body) is spread across layers, but memorized paragraphs have larger gradients in *lower* layers, and a low-layer attention head attends to rare, distinctive tokens. Fine-tuning only high-gradient weights unlearns them — [Stoehr et al. 2024, arXiv 2403.19851](https://arxiv.org/abs/2403.19851) [venue unverified]
- "Early layers generalize while later layers memorize." Prediction depth (the first layer at which a k-NN probe's prediction matches the final output) tracks per-example difficulty — [Baldock, Maennel, Neyshabur 2021, arXiv 2106.09647](https://arxiv.org/abs/2106.09647) [venue: NeurIPS 2021, from prior knowledge]
- **[CONNECTIVITY/routing] Memorization Sinks** (Ghosal et al., ICML 2025): MLP neurons split 70/30 into shared and "sink" neurons, and a sequence-ID-seeded mask activates a subset of sinks per sequence. Removing sinks at eval raises loss on repeated sequences to about 66% of a non-memorizing model while matching validation loss. At SmolLM 360M/1.7B, the train–val gap closes by ≥50%. **Gradient-masked forced localization fails** because of co-adaptation, and min-norm bias favors reusing general features for memorization — [arXiv 2507.09937](https://arxiv.org/html/2507.09937v2); [PMLR v267](https://proceedings.mlr.press/v267/ghosal25a.html)
- Knowledge capacity is about 2 bits/parameter (also under int8). GPT-2 with rotary embeddings matches or beats LLaMA/Mistral gated-MLP models at knowledge storage, especially at short training, because gated MLPs are "less stable and harder to train" — [Allen-Zhu & Li 2024, arXiv 2404.05405](https://arxiv.org/abs/2404.05405) [venue unverified; I believe ICLR 2025]
- Classic background: Arpit et al. 2017 (networks fit real patterns before noise) — [arXiv 1706.05394](https://arxiv.org/abs/1706.05394); Feldman 2020 (memorizing long-tail examples is needed for near-optimal generalization) — [arXiv 1906.05271](https://arxiv.org/abs/1906.05271) (both not re-fetched)

**Grokking drivers, and architecture**
- Lazy→rich: grokking is delayed when the initial NTK's top eigenvectors are misaligned with labels. The rate of feature learning is set by output-scale parameters — [Kumar, Bordelon, Gershman, Pehlevan, ICLR 2024](https://iclr.cc/virtual/2024/poster/17515)
- Circuit efficiency: generalizing circuits are slower to learn but produce larger logits per unit parameter norm. Memorizing circuits lose efficiency as data grows, giving a critical dataset size. This predicts "ungrokking" and "semi-grokking" — [Varma et al. 2023, arXiv 2309.02390](https://arxiv.org/abs/2309.02390)
- Omnigrok: L-shaped train loss and U-shaped test loss as functions of weight norm explain weight-decay dependence. Grokking was induced on image, language, and molecule tasks — [Liu, Michaud, Tegmark, arXiv 2210.01117](https://arxiv.org/abs/2210.01117) [venue: ICLR 2023, from prior knowledge]
- **[ATTN-vs-MLP]** Clock vs Pizza (Zhong et al. 2023): interpolating attention from full (α=1) to constant (α=0) causes a phase change. Clock dominates when attention is prominent, Pizza when linear layers dominate. The boundary moves roughly linearly with width (wider favors Pizza). Deeper (2–4 layer) nets were circular in only 6–12% of runs vs 34% for 1-layer — [arXiv 2306.17844](https://arxiv.org/pdf/2306.17844) [venue: NeurIPS 2023, from prior knowledge]
- **[NORM]** LayerNorm position in 1-layer transformers on modular addition: LN on the MLP input groks fastest. LN on QK reduces attention entropy and did not generalize within 10k epochs. LN on V helps. All configurations converge to Fourier solutions, and the onset of compressibility (effective rank) follows the same order as generalization. Similar with RMSNorm and in 3-layer models — [Singh, Misra, Orvieto, Feb 2026, arXiv 2602.06702](https://arxiv.org/html/2602.06702v1) (ICLR 2026 listing per [iclr.cc](https://iclr.cc/virtual/2026/10019118), unverified)
- **[CONNECTIVITY+NORM, confounded]** Systematic grokking study (mod 97): depth-2 MLPs grok in 4/5 seeds, flat depth-4 MLPs in 0/5 within 400k steps, and **depth-8 pre-LN residual MLPs** (residual scale 0.5, gradient clipping) in 3/5 with a shorter delay. Residual, LN, width, and clipping were not separated. Transformer vs MLP is about 1.1× at matched configurations and 1.9× at each one's optimal weight decay, with confounded optimizers. Weight decay dominates — [Manir & Rupa 2026, arXiv 2603.25009](https://arxiv.org/html/2603.25009) [venue unverified; small-seed study]

**Phase transitions and emergence**
- Hidden progress: SGD on k-sparse parities shows sudden jumps after about n^O(k) steps, driven by gradual amplification that the loss does not show — [Barak et al. 2022, arXiv 2207.08799](https://arxiv.org/abs/2207.08799) [venue: NeurIPS 2022, from prior knowledge]
- Apparent emergent abilities can come from nonlinear or discontinuous metrics — [Schaeffer, Miranda, Koyejo 2023, arXiv 2304.15004](https://arxiv.org/abs/2304.15004) [venue: NeurIPS 2023, from prior knowledge]

### Inferences
- The cleanest connectivity-relevant grokking manipulations are normalization placement and attention-vs-MLP ratio. No paper I found isolates residual vs no-residual or parallel vs serial blocks for grokking time. This is an open, cheap experiment: modular arithmetic trains in minutes.
- MemSinks shows that fixed, input-conditioned routing *can* separate memorization from generalization, while gradient-only localization cannot. Static topology alone probably will not localize memorization without a routing signal.

### Gaps
- No controlled residual/skip ablation on grokking delay at matched parameters (the 2026 study is confounded).
- I did not find work on where memorization localizes as a function of skip topology (e.g., do dense connections push memorization toward specific layers?).

---

## 5. Iterative inference, layerwise separability, neural collapse across depth, and intermediate-layer quality

### Takeaway
Residual connectivity is the clearest causal driver of *iterative refinement*. Residual blocks align with the negative gradient, later blocks only refine, and single-block deletion is benign in ResNets but catastrophic in VGG. Short effective paths carry the gradient, and middle transformer layers share a representation space that tolerates reordering and parallelization. Removing skips in a ViT block splits the CKA structure. Neural collapse increases progressively with depth, which is proven optimal for deep regularized ResNets and transformers with LN. Mid-depth layers are often the best general-purpose representations, especially in autoregressive models.

### Cited Findings
**Iterative refinement**
- **[CONNECTIVITY]** Residual blocks push outputs toward −∂L/∂h. The cosine between F_i(h_i) and the gradient is negative, most so in higher blocks. Lower blocks learn representations, higher blocks refine (dropping most of them barely hurts) and mainly fix borderline examples. Unrolling the last block 20 extra steps raises borderline test accuracy from about 43% to 53%. Naive weight sharing fails unless BN statistics are unshared — [Jastrzębski et al., ICLR 2018, arXiv 1710.04773](https://ar5iv.labs.arxiv.org/html/1710.04773)
- **[CONNECTIVITY]** ResNets behave like ensembles of shallow paths. Deleting one block barely hurts (except downsampling blocks), while deleting a VGG layer drops to chance. Most gradient comes from paths of about 5–17 modules out of 54 (about 0.45% of paths), and training only those gives 5.96% vs 6.10% error — [Veit, Wilber, Belongie, NeurIPS 2016](https://ar5iv.arxiv.org/html/1605.06431)
- **[CONNECTIVITY]** In ViTs, skips matter more than in ResNets. The CLS token is carried mainly by the skip path in early layers. **Removing the skip in one block "partitions the layer representations on either side"** in CKA and costs about 4% accuracy in middle blocks — [Raghu et al., NeurIPS 2021, arXiv 2108.08810](https://ar5iv.labs.arxiv.org/html/2108.08810)
- Stages of inference: deleting or swapping adjacent layers without fine-tuning keeps 72–95% of top-1 accuracy, with middle layers most robust. The proposed stages are detokenization → feature engineering → prediction ensembling → residual sharpening — [Lad, Lee, Gurnee, Tegmark 2024, arXiv 2406.19384](https://arxiv.org/abs/2406.19384); [ICML 2024 workshop](https://mlanthology.org/icmlw/2024/lad2024icmlw-remarkable)
- **[CONNECTIVITY: post-hoc parallelization]** "Transformer Layers as Painters": middle layers of Llama2-7B/13B and BERT-Large share a representation space. Skipping, shuffling, and running middle layers *in parallel* (averaging outputs) degrade gracefully, except on math (GSM8K), where order matters. Looping the parallel block helps. Repeating one layer is catastrophic — [Sun et al. 2024, arXiv 2407.09298](https://arxiv.org/html/2407.09298v1); AAAI 2025 per [archive listing](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers/AAAI/2025/transformer_layers_as_painters__3a155e35.pdf)
- **[CONNECTIVITY]** Pythia adjacent-layer cosine similarity exceeds 0.97 in later layers (collapse), and dense multiway connections reduce it — [MUDDFormer](https://arxiv.org/html/2502.12170v1)

**Layerwise separability and the tunnel**
- Linear probes: linear separability increases monotonically with depth in Inception v3 and ResNet-50 — [Alain & Bengio 2016, arXiv 1610.01644](https://arxiv.org/abs/1610.01644)
- Law of equi-separation: each layer improves class separation by a roughly constant geometric factor, across architectures and datasets — [He & Su, PNAS 2023, arXiv 2210.17020](https://arxiv.org/abs/2210.17020v2)
- **[DEPTH/WIDTH]** Tunnel effect: in deep enough supervised image nets, early layers create linearly separable representations and later "tunnel" layers compress them with little effect on performance. The tunnel forms early in training, its length depends on capacity vs task complexity, and it hurts OOD transfer — [Masarczyk et al., NeurIPS 2023, arXiv 2305.19753](https://arxiv.org/abs/2305.19753)
- **[DEPTH/WIDTH]** Block structure: high capacity relative to data gives a CKA "block structure" of layers that propagate a dominant principal component. Wide and deep models make different per-class errors at similar accuracy — [Nguyen, Raghu, Kornblith 2020, arXiv 2010.15327](https://arxiv.org/abs/2010.15327v2) [venue: ICLR 2021, from prior knowledge]

**Neural collapse across depth**
- Original NC (variability collapse, simplex ETF, self-duality, NCC) at the last layer — [Papyan, Han, Donoho, PNAS 2020, arXiv 2008.08186](https://arxiv.org/abs/2008.08186) (not re-fetched)
- Intermediate NC: within-class/between-class covariance decreases with depth, and top-layer weight singular vectors align with class means — [Rangamani et al., ICML 2023](https://proceedings.mlr.press/v202/rangamani23a.html)
- Effective depth (first layer that is NCC-separable) is small under SGD and grows with label noise, which gives a generalization bound — [Galanti, Galanti, Ben-Shaul 2022, arXiv 2202.09028](https://arxiv.org/abs/2202.09028)
- **[CONNECTIVITY: theory relies on residual structure]** Progressive feedforward collapse: in ResNets, NC1 (Tr Σ_W / Tr Σ_B), ETF distance, and NCC accuracy all improve monotonically through layers. A proof uses the view of a weight-decayed ResNet as a geodesic in Wasserstein space, which relies on the residual dynamical-system structure. Non-residual nets were not tested — [Wang, Gai, Zhang 2024, arXiv 2405.00985](https://arxiv.org/html/2405.00985v1) [venue unverified]
- **[CONNECTIVITY+NORM]** Global optima of deep *regularized ResNets and transformers with LayerNorm* are approximately collapsed, increasingly so with depth, and end-to-end training reduces to the unconstrained features model. Earlier data-aware theory covered only MLPs — [Súkeník, Lampert, Mondelli, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/3e18f7152b5e1696e73722dc5121b6f2-Abstract-Conference.html) (an ICML 2025 virtual page with the same title also exists: [icml.cc](https://icml.cc/virtual/2025/47667); venue listings conflict)
- Not all layers are equal from an NC perspective — [Zhou et al., CPAL 2025](https://proceedings.mlr.press/v280/zhou25a.html) (not fetched in detail)

**Intermediate-layer representation quality**
- Mid-depth layers beat the last layer by up to 16% on 32 MTEB tasks in transformers and SSMs. Autoregressive objectives produce a pronounced mid-layer "compression valley" (matrix-entropy based), and bidirectional or masked models show milder changes. Autoregressive image transformers show the same pattern, so the objective, not the modality, drives it — [Skean et al., ICML 2025 (oral), arXiv 2502.02013](https://arxiv.org/html/2502.02013v2)
- **[CONNECTIVITY]** Decayed shortcuts improve linear-probe and k-NN quality of MAE features — [Zhang et al. 2024](https://arxiv.org/pdf/2404.10947) (see §1)

### Inferences
- Layer-lesion curves (accuracy vs deleted, swapped, or parallelized layers), the gradient-alignment cosine, and adjacent-layer similarity are inexpensive, well-precedented readouts that respond strongly to connectivity (residual vs plain vs dense). They are the most direct way to show that topology changes "how computation is distributed over depth."
- Progressive NC gives a scalar per-layer curve (NC1 vs depth), and its theory currently *depends on residual structure*. Testing PFC in plain, dense, and parallel-branch nets would be novel.

### Gaps
- No paper found that measures progressive NC in non-residual or dense architectures at matched accuracy.
- Lad et al. do not name the role of residual connections in the robustness they report. The link is inferred.

---

## 6. Universality / convergent representations, and persistent architecture-specific differences

### Takeaway
There is real but partial convergence. About 1–5% of GPT-2 neurons are universal across seeds. Stitching works across training methods within an architecture. Larger models align more (the Platonic hypothesis). Robust architecture-specific differences persist: ViT vs CNN layer-similarity structure is governed by skip connections, wide vs deep nets make different errors, Clock vs Pizza solutions depend on attention rate, Adam (not LN) causes privileged residual bases, and transformers vs SSMs differ on copying.

### Cited Findings
- Convergent learning: different-seed nets learn some features one-to-one and others only up to subspaces — [Li, Yosinski, Clune, Lipson, Hopcroft, ICLR 2016, arXiv 1511.07543](https://arxiv.org/abs/1511.07543) (not re-fetched)
- Platonic Representation Hypothesis: representations in vision and language models converge with scale toward a shared statistical model of reality. Alignment is measured with a mutual k-nearest-neighbor kernel alignment metric (from prior knowledge; the fetched abstract does not name it) — [Huh, Cheung, Wang, Isola 2024, arXiv 2405.07987](https://arxiv.org/abs/2405.07987) [venue: ICML 2024, from prior knowledge]
- Universal neurons: 1.23% (GPT2-medium), 1.26% (Pythia-160M), and 4.16% (GPT2-small) of neurons are universal across 5 seeds (excess max Pearson correlation > 0.5 over a rotated baseline, on 100M Pile tokens) — [Gurnee et al., TMLR 2024, arXiv 2401.12181](https://arxiv.org/abs/2401.12181)
- Model stitching: same-architecture nets trained very differently (supervised vs self-supervised) stitch without loss, and "more is better" (representations from more data or width improve weaker models). Typical SGD minima are "stitching-connected" — [Bansal, Nakkiran, Barak, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html)
- **[CONNECTIVITY]** ViT vs CNN: ViTs have more uniform layer-similarity structure, driven by strong skip connections. Removing one skip splits it — [Raghu et al. 2021](https://ar5iv.labs.arxiv.org/html/2108.08810)
- **[DEPTH/WIDTH]** Wide vs deep: similar accuracy but different per-class errors, and block structure arises at high capacity — [Nguyen et al.](https://arxiv.org/abs/2010.15327v2)
- **[ATTN-vs-MLP]** Algorithm non-universality: Clock vs Pizza depends on attention rate and width — [Zhong et al.](https://arxiv.org/pdf/2306.17844)
- Privileged basis: residual-stream outlier dimensions are basis-aligned and attributed to Adam's per-dimension normalizers. LayerNorm and floating-point precision are "confidently" ruled out — [Elhage, Lasenby, Olah, Transformer Circuits, Mar 2023](https://transformer-circuits.pub/2023/privileged-basis/index.html)
- Crosscoder scaling: early and late layers are shared across model sizes, and new representations emerge in middle layers with scale — [Lindsey et al. 2024](https://transformer-circuits.pub/2024/crosscoders/index.html)
- **[ARCH: attention vs SSM]** Transformers beat state-space models at copying (theory plus experiments) — [Jelassi et al., ICML 2024](https://proceedings.mlr.press/v235/jelassi24a.html) (details not re-fetched)
- Representation strength is biased by feature complexity, learning order, and output position, independent of computational role. This distorts similarity and intervention analyses, and it varies with architecture and optimizer — [Lampinen, Chan, Hermann 2024, arXiv 2405.05847](https://arxiv.org/abs/2405.05847) [venue unverified]

### Inferences
- For a topology study, cross-topology stitching or CKA at matched layers is a natural universality readout: do residual, dense, and parallel nets converge to the same mid-layer representations? Raghu's skip ablation suggests no at the level of layer-wise structure. Mutual-kNN alignment of final features may still converge (PRH).
- Lampinen et al. warn that similarity metrics can be dominated by "simple, early-learned" features. Topology-induced differences in learning order (cf. Saxe neural race) could show up as similarity differences without different computation.

### Gaps
- No study found computing PRH-style alignment specifically between residual, dense, and plain topologies at matched size and data.

---

## 7. Induction heads and other circuits whose emergence depends on architecture

### Takeaway
Induction heads need cross-layer *composition*: a provable exponential separation between 1-layer and 2-layer transformers. Small wiring changes that give a single layer access to previous-token information ("smeared keys," the 2026 symplectic lift) remove the need for depth. Formation is a phase change driven by three interacting subcircuits (previous-token, QK match, copy) with many-to-many redundancy. Their timing can be tuned by data and, through clamping, by architecture, which makes them an ideal connectivity probe.

### Cited Findings
- Olsson et al. 2022: induction heads form during a phase change that coincides with the in-context-learning loss bump. They require K-composition across two layers, and a "smeared key" modification (keys mix current and previous token) lets 1-layer models form them (from prior knowledge) — [Transformer Circuits / arXiv 2209.11895](https://arxiv.org/pdf/2209.11895) (not re-fetched)
- **[DEPTH = composition]** Lower bound: a 1-layer transformer solving induction heads needs h·m·p = Ω(n) (heads × dim × precision), while 2 layers need only O(1) heads and dim with O(log n) precision, an exponential separation (proved via one-way communication) — [Sanford, Hsu, Telgarsky 2024, arXiv 2408.14332](https://arxiv.org/html/2408.14332v1)
- **[CONNECTIVITY within attention]** A symplectic shear of post-RoPE Q/K streams (a "Hairer lift") lets a *single layer* reach 74.2% induction accuracy, with a closed-form transition at γ*c = (d_k log(T−1))^{1/4}. The paper frames the obstruction as a filter-order gap (a standard bilinear score has order (0,0); induction needs (1,1)) — [Maitra et al., NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/149119)
- Singh et al. 2024 (2-layer attention-only, 8 heads/layer, Omniglot few-shot): induction heads form after a plateau at about 50% accuracy. Several L2 heads become induction heads that act additively (ablating the strongest barely hurts). Previous-token heads wire many-to-many with them. There are three interacting subcircuits (A: previous-token copy; B: QK match; C: copy, which prior work overlooked). Clamping L1 sharpens the phase change. More classes delays B, more labels delays C. V-composition matters, unlike Olsson. Layer count and MLPs were not varied — [arXiv 2404.07129](https://arxiv.org/abs/2404.07129) [venue: ICML 2024, from prior knowledge]
- Birth of a Transformer: in a simplified 2-layer model, global bigrams are learned quickly and the induction mechanism slowly. Weight matrices act as associative memories — [Bietti et al. 2023, arXiv 2306.00802](https://arxiv.org/abs/2306.00802) [venue: NeurIPS 2023, from prior knowledge]
- Residual stream as the composition substrate: low-rank inter-layer channels, predictable from weights — [Merullo et al. 2024](https://arxiv.org/abs/2406.09519). Background framework: [Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) (not re-fetched)
- Related composition-dependent circuits: shortcut automata solutions — [Liu et al. 2022](https://arxiv.org/abs/2210.10749); copying advantage of attention over SSMs — [Jelassi et al. 2024](https://proceedings.mlr.press/v235/jelassi24a.html)

### Inferences
- Induction-head formation is the cleanest testbed for "composed vs parallel" topology. In a 2-block network, putting the attention blocks in *parallel* (both reading the embedding) should block K-composition. The Sanford bound predicts an exponential size cost, and Olsson/Maitra show which skip-like wiring changes undo it. Dependent variables: phase-change step, prefix-matching score, and ICL-score curve.
- Dense or cross-layer routes (e.g., value residuals, MUDD Q/K/V streams) should change which layer pairs can compose, which could shift formation time or head redundancy.

### Gaps
- I found no published experiment that directly compares sequential vs parallel attention arrangements (or dense vs residual routing) on induction-head emergence time.
- Singh et al.'s venue and Olsson's smeared-key details were not re-verified this session.

---

## 8. Ranked emergent-structure targets for a topology (parameter-free connectivity) study

### Takeaway
The best targets have three properties: (a) cheap, reproducible scalar or curve operationalizations; (b) existing evidence that connectivity, not just size, moves them; (c) theory predicting direction. Ranked:

1. **Layerwise computation profile / iterative refinement.** This is the most connectivity-sensitive structure on record (ResNet vs VGG lesions; the ViT skip-removal CKA split; parallel or shuffled middle layers). Best operationalization: block-lesion and swap curves (Veit 2016; Lad 2024; Sun 2024), plus the gradient-alignment cosine cos(F_i(h_i), ∂L/∂h_i) (Jastrzębski 2018), plus adjacent-layer CKA or cosine.
2. **Effective rank / representation collapse and function complexity.** Residuals, decayed shortcuts, skip removal in attention, and dense routing all move it causally, and branch scale offers a continuous knob. Best operationalization: per-layer effective rank of embeddings (Huh 2023; Zhang 2024), plus Neural-Redshift complexity of the random-init and trained function (Teney 2024), plus token-uniformity or rank collapse in attention (Dong 2021).
3. **Induction-head / composition-circuit emergence.** It needs cross-layer composition by proof, so composed vs parallel wiring makes a sharp, falsifiable prediction, and it trains in minutes. Best operationalization: phase-change step and prefix-matching score in 2-layer attention-only models with subcircuit clamping (Singh 2024; Olsson 2022; Sanford 2024 bound).
4. **Progressive neural collapse across depth.** It gives a clean per-layer curve, and its theory currently *assumes* residual structure (Wang 2024; Súkeník 2025), so plain, dense, and parallel nets test it directly. Best operationalization: NC1 = Tr(Σ_W)/Tr(Σ_B) and NCC accuracy per layer, plus "effective depth" (Galanti 2022), plus tunnel length (Masarczyk 2023).
5. **Grokking / memorization–generalization timing (modular arithmetic).** It is cheap, has a known end-state (Fourier circuits; Clock vs Pizza), and is already shown to depend on LN placement and attention ratio, while skip topology is untested. Best operationalization: grokking delay and the effective-rank compressibility onset (Singh/Misra/Orvieto 2026), restricted and excluded loss (Nanda 2023), and Clock/Pizza classification by gradient symmetricity (Zhong 2023).
6. **Functional modularity / route specialization.** It is highly relevant to parallel branches, but metrics are noisier, and the evidence shows parallel routes tend to *collapse* onto one stream rather than specialize. That makes it an informative but riskier target. Best operationalization: per-branch read/write share and ablate-and-rescue (Alimaskina 2026; Peng 2026), plus clusterability n-cut (Filan 2021) and task-conditional ablation selectivity (Béna & Goodman 2025).

### Cited Findings
- Connectivity-causal evidence for target 1 — [Veit 2016](https://ar5iv.arxiv.org/html/1605.06431); [Raghu 2021](https://ar5iv.labs.arxiv.org/html/2108.08810); [Jastrzębski 2018](https://ar5iv.labs.arxiv.org/html/1710.04773); [Sun 2024](https://arxiv.org/html/2407.09298v1); [Lad 2024](https://arxiv.org/abs/2406.19384)
- For target 2 — [Teney 2024](https://arxiv.org/html/2403.02241v3); [Zhang 2024](https://arxiv.org/pdf/2404.10947); [Dong 2021](https://arxiv.org/abs/2103.03404); [Huh 2023](https://arxiv.org/pdf/2103.10427v4); [MUDDFormer 2025](https://arxiv.org/html/2502.12170v1); [Curse of Depth 2025](https://arxiv.org/abs/2502.05795)
- For target 3 — [Sanford 2024](https://arxiv.org/html/2408.14332v1); [Singh 2024](https://arxiv.org/abs/2404.07129); [Maitra 2026](https://neurips.cc/virtual/2026/poster/149119)
- For target 4 — [Wang/Gai/Zhang 2024](https://arxiv.org/html/2405.00985v1); [Súkeník 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/3e18f7152b5e1696e73722dc5121b6f2-Abstract-Conference.html); [Rangamani 2023](https://proceedings.mlr.press/v202/rangamani23a.html); [Masarczyk 2023](https://arxiv.org/abs/2305.19753)
- For target 5 — [Singh/Misra/Orvieto 2026](https://arxiv.org/html/2602.06702v1); [Zhong 2023](https://arxiv.org/pdf/2306.17844); [Kumar 2024](https://iclr.cc/virtual/2024/poster/17515); [Manir & Rupa 2026](https://arxiv.org/html/2603.25009)
- For target 6 — [Alimaskina 2026](https://arxiv.org/abs/2606.03483); [Peng 2026](https://arxiv.org/abs/2603.14833); [Béna & Goodman 2025](https://ideas.repec.org/a/nat/natcom/v16y2025i1d10.1038_s41467-024-55188-9.html); [Filan 2021](https://arxiv.org/abs/2103.03386)
- Informative null to include as a control: skip connections do not change forgetting in small CNNs — [Mirzadeh 2022](https://ar5iv.labs.arxiv.org/html/2202.00275)

### Inferences
- Targets 1–4 can share one experimental grid (plain / residual / decayed-residual / dense / parallel-branch, at matched parameters and FLOPs) and one set of per-layer readouts. Target 3 needs its own small transformer suite.
- The biggest open gaps, and so the most publishable contrasts, are: parallel vs composed arrangement at matched parameters (on any target), dense vs residual on neural collapse and effective rank, and skip topology on grokking delay.

### Gaps
- Few studies hold parameters and FLOPs fixed while changing only connectivity. Many "connectivity" papers (HC, DenseFormer, MUDD, ResFormer) also add learned parameters, however few, and report mainly perplexity, with structure analyses secondary.
- The web-search budget was exhausted before I could check: Elhage 2021 framework details, Olsson's smeared-key experiment, the PRH alignment-metric specifics, and venue confirmation for several arXiv items (marked "[venue unverified]" or "from prior knowledge").
