# Empirical Studies: Network Connectivity/Topology → Learned Representations, Emergent Structure, Learning Behavior

Scope note: this catalogue covers studies beyond those listed as "already known" (DenseNet, FractalNet, Xie 2019 random wiring, You et al. 2020, Veit 2016, PaLM/GPT-J parallel blocks, ReZero/SkipInit/Fixup, DenseFormer, ResFormer value residual, Elhage 2021). Each entry gives: **Manipulation | Measured | Finding | Controls | Relevance**. Venue/year tags marked "(venue unverified)" come from my background knowledge and were not confirmed against a primary page in this session. All numbers come from the fetched primary pages (arXiv HTML/abstract, proceedings, or the authors' blog) unless flagged. The web-search budget ran out near the end of the session, so a few planned lookups were not done (listed under Gaps).

---

## 1. Transformer topology variants and what they reveal about residual-stream structure

### Takeaway
A wave of 2024–2026 "learned residual routing" papers (Hyper-Connections, mHC, MUDDFormer, DeepCrossAttention, LAuReL) all report the same diagnostic in the plain pre-LN residual stream: adjacent-layer representations become very similar in deep layers ("representation collapse"). When these papers let the model learn its own cross-layer routing, the patterns it learns look structured: Λ-shaped (recent layers plus embeddings), partly parallel blocks, and value streams that reach back to early layers. Separately, frozen-model intervention studies (layer skip, swap, reorder, parallelize, prune) show that pre-LN LLMs have a robust, interchangeable "middle" with fragile first and last layers. Pre-LN itself, not residual connections per se, is implicated in making deep layers near-identity.

### Cited Findings

**Hyper-Connections (HC). Zhu et al., ByteDance Seed; ICLR 2025; arXiv 2409.19606**
- Manipulation: the residual stream is widened into n copies (expansion rate n). Learnable "depth-connections" weight each layer's output against earlier hidden states, and "width-connections" mix the n copies. Static HC uses fixed learned weights; dynamic HC (DHC) predicts the weights per token. Residual, Pre-Norm and Post-Norm can all be written as special fixed HC matrices. n=1 is worse than baseline, n=4 is best, n=8 adds little. — [arXiv 2409.19606v3](https://arxiv.org/html/2409.19606v3)
- Measured, beyond loss: cosine similarity between adjacent layers' inputs, and visualizations of the learned cross-layer connection matrices. — [arXiv](https://arxiv.org/html/2409.19606v3)
- Finding: the Pre-Norm baseline has high adjacent-layer similarity (collapse), while HC has lower and more varied similarity. The learned connection matrices show a "Λ-shaped" pattern: recent layers are favored and bottom layers stay accessible. Local jagged patterns suggest some adjacent layers learned to run in parallel, resembling parallel transformer blocks. — [arXiv](https://arxiv.org/html/2409.19606v3)
- Results: OLMoE-1B-7B with DHC×4 converges about 1.8× faster, with about +6 points on ARC-Challenge at 500B tokens. OLMo-7B average accuracy goes from 0.701 to 0.710 with no loss spikes. ViT-L with DHC×2 reaches 79.94% vs 77.25% on ImageNet. — [arXiv](https://arxiv.org/html/2409.19606v3)
- Controls: about +0.03% parameters and +0.2% FLOPs (OLMo-1B, n=4), but about **+26% activation memory**. The paper frames the problem as a "seesaw": Pre-Norm causes collapse, Post-Norm causes gradient vanishing. — [arXiv](https://arxiv.org/html/2409.19606v3)
- Relevance: **direct precedent.** Parallel-vs-sequential arrangement *emerges* from learned routing. Visualizing the learned connection matrix is a reusable measurement. The memory overhead is a confound that FLOP/parameter matching hides.

**mHC: Manifold-Constrained Hyper-Connections. Xie, Wei, Cao et al., DeepSeek-AI; arXiv 2512.24880 (v2 dated 5 Jan 2026; venue not stated)**
- Manipulation: HC's residual-mixing matrix H^res (n×n) is projected onto the Birkhoff polytope (doubly stochastic matrices) with Sinkhorn-Knopp (20 iterations), to "restore the identity mapping property." Uses n=4. — [arXiv 2512.24880v2](https://arxiv.org/html/2512.24880v2)
- Measured: gradient norm; "Amax gain magnitude" (maximum absolute row sum for forward, column sum for backward) of single-layer and **composite cross-layer** H^res products; visualizations of learned H^res; memory I/O. — [arXiv PDF](https://arxiv.org/pdf/2512.24880v2)
- Finding: unconstrained HC shows a loss surge near 12k steps in a 27B model, tied to gradient-norm instability. Composite HC mappings reach Amax gain of about **3000**, and the learned HC matrices have large mixed-sign entries. mHC's composite gain stays at most about **1.6**, its matrices are non-negative with row and column sums near 1, and its final loss is 0.021 below baseline. An ablation finds H^res (stream mixing) contributes the largest share of HC's gain. — [arXiv PDF](https://arxiv.org/pdf/2512.24880v2)
- Results (27B MoE, DeepSeek-V3-style; Baseline / HC / mHC): BBH 43.8 / 48.9 / 51.0; DROP 47.0 / 51.6 / 53.9; MMLU 59.0 / 63.0 / 63.4; MATH 22.0 / 26.4 / 26.0. The advantage persists across 3B, 9B and 27B compute scaling with "only marginal attenuation." — [arXiv PDF](https://arxiv.org/pdf/2512.24880v2)
- Controls: FLOPs roughly unchanged; 6.7% wall-clock overhead after kernel fusion and recomputation. A parameter-matched baseline was not clearly described in the text I could see.
- Relevance: **key precedent and measurement.** Products of routing matrices across depth (composite gain) are a cheap, reusable stability and topology diagnostic. It shows that *how* routing is constrained (here, norm-preserving doubly stochastic mixing) matters as much as connectivity itself.

**MUDDFormer (Multiway Dynamic Dense connections). Xiao et al.; ICML 2025; arXiv 2502.12170**
- Manipulation: each block's input is a per-token, dynamically weighted aggregate of all earlier layer outputs. The aggregation is decoupled into four streams (Q, K, V, residual), each with its own weights. FFN widths are also reallocated toward depth so total parameters stay fixed. — [arXiv 2502.12170v2](https://arxiv.org/html/2502.12170v2)
- Measured: adjacent-layer cosine similarity (collapse); **attention-head activity** (a head counts as active if its strongest attention falls outside the first or special tokens, i.e. not an attention sink); learned stream weight patterns. — [arXiv](https://arxiv.org/html/2502.12170v2)
- Finding: Pythia's later layers exceed **0.97** adjacent cosine similarity, while MUDDPythia stays more distinct, especially in the V stream. MUDDPythia has about **2.4× more active heads**, mostly in deep layers. The V stream learns salient dynamic weight on the **first layer's output**, a long-range channel for upper-layer attention. Removing dense connections from any single stream hurts, and removing them from V hurts most. — [arXiv](https://arxiv.org/html/2502.12170v2)
- Results: MUDDPythia-2.8B matches Pythia-6.9B on Pile perplexity and downstream tasks. 834M MUDDFormer matches Transformer++ at about 1.89× compute. Gains persist to 42 layers in DeepNarrow models, while Transformer++ plateaus beyond 24. — [arXiv](https://arxiv.org/html/2502.12170v2)
- Controls: +0.23% parameters and +0.4% FLOPs (theoretical), but measured training throughput is only 84–96% of baseline. **Confound:** FFN reallocation is bundled into the main model; it helps MUDDFormer but slightly hurts Transformer++. — [arXiv](https://arxiv.org/html/2502.12170v2)
- Relevance: **reusable measurements** (attention-sink / head-activity rate; per-stream routing). It shows that dense connectivity alters attention *behavior*, not only loss. It also warns that bundled width reallocation confounds the topology effect.

**DeepCrossAttention (DCA). Heddes et al.; ICML 2025 (PMLR v267); arXiv 2502.06785**
- Manipulation: residual sums are replaced by learned, input-dependent weightings (GRN) of earlier layer outputs, with separate GRNs producing Q, K and V. The authors describe this as letting the model move continuously between purely sequential and fully parallel block arrangements. Only the first and last-k layer outputs are kept explicit for memory. — [arXiv 2502.06785v2](https://arxiv.org/html/2502.06785v2)
- Measured: perplexity vs training time; learned weight distribution (Appendix H, not seen); a theory of low-rank target functions ("collective rank" r* < d). — [arXiv](https://arxiv.org/html/2502.06785v2)
- Finding: same quality "up to 3x faster"; fewer loss spikes. The learned weights put most mass on the model **inputs and the last few layers**, consistent with HC's Λ pattern. Theory: dense routing helps when the ratio of collective rank to dimension falls below a task-dependent threshold. — [arXiv](https://arxiv.org/html/2502.06785v2)
- Relevance: an independent replication of the "embeddings plus recent layers" routing pattern, and a **theoretical hook**: the benefit of dense connectivity depends on the target function's rank structure. That gives a testable prediction for synthetic tasks.

**LAuReL (Learned Augmented Residual Layer). Menghani et al., Google; arXiv 2411.07501 (venue unverified; an ICML-hosted page exists)**
- Manipulation: x_{i+1} = α·f(x_i) + g(·), where the skip path is (RW) learned scalar weights, (LR) identity plus a low-rank update, or (PA) a mix of k previous activations. — [arXiv 2411.07501v3](https://arxiv.org/html/2411.07501v3)
- Measured: accuracy and downstream benchmarks only, plus parameter, latency and memory cost. No representational analysis.
- Finding: on ResNet-50/ImageNet, RW+LR matches the gain from adding one extra layer (75.20%) at about 2.6× fewer added parameters. A 1B LLM at +0.012% parameters improves most benchmarks (BoolQ 58.07 → 65.66). — [arXiv](https://arxiv.org/html/2411.07501v3)
- Controls: weak. The ResNet runs were **one epoch only**; gains are sensitive to rank and initialization. — [arXiv](https://arxiv.org/html/2411.07501v3)
- Relevance: **confound warning.** Tiny parametric modifications to the skip path break the "parameter-free topology" premise. The project should treat learned skip scalars as a separate condition.

**Transformer Layers as Painters. Sun, Pickett, Nain, Jones; AAAI 2025; arXiv 2407.09298**
- Manipulation, applied to frozen Llama2-7B/13B/70B and BERT-Large: skip layers; reverse or randomize the order of middle layers; run middle layers in **parallel** and average them; loop the parallel block; repeat the center layer. — [arXiv](https://arxiv.org/html/2407.09298v1)
- Measured: normalized benchmark scores (ARC, HellaSwag, GSM8K, WinoGrande, LAMBADA; GLUE); cosine similarity between hidden states. — [arXiv](https://arxiv.org/html/2407.09298v1)
- Finding: first and last layers are critical. Middle layers share a representation space; cosine similarity forms about 4–5 groups in Llama2-13B. Skipping, reversing, randomizing and parallelizing middle layers all degrade gracefully, but **sharing weights among middle layers is catastrophic** (they do different things). Looped-parallel (3×) is the least harmful variant. **Math and reasoning (GSM8K, ARC) are the most order-sensitive**; semantic tasks are not. — [arXiv](https://arxiv.org/html/2407.09298v1)
- Controls: no retraining, so these are post-hoc perturbations of models trained sequentially.
- Relevance: **reusable intervention battery** (skip, swap, reverse, parallelize, loop). It shows that sequential composition matters chiefly for compositional or arithmetic tasks. It is a precedent for testing "parallel vs composed" on trained nets.

**The Remarkable Robustness of LLMs: Stages of Inference? Lad, Gurnee, Tegmark; NeurIPS 2025 (per mlanthology); arXiv 2406.19384**
- Manipulation: zero-ablate (delete) single layers, or swap adjacent layers, in Pythia, GPT-2 and Phi models without fine-tuning. — [arXiv](https://arxiv.org/html/2406.19384v1)
- Measured: KL to the original output; top-1 agreement; prediction entropy; attention to the previous 5 tokens; logit-lens KL; counts of prediction and suppression neurons. — [arXiv](https://arxiv.org/html/2406.19384v1)
- Finding: deletion and swapping retain **72–95%** of prediction accuracy, and deeper models are more robust. First-layer ablation is catastrophic. The authors propose four stages: detokenization; feature engineering; prediction ensembling (prediction neurons peak at about 85% depth); residual sharpening (suppression neurons dominate). GPT-2 (trained with dropout) is more robust than Pythia. — [arXiv](https://arxiv.org/html/2406.19384v1); [NeurIPS 2025 listing](https://mlanthology.org/neurips/2025/lad2025neurips-remarkable)
- Relevance: **reusable measurements** (prediction/suppression neuron census, local-attention fraction by depth). The dropout-vs-robustness difference is a training-regime confound in cross-model comparisons.

**The Unreasonable Ineffectiveness of the Deeper Layers. Gromov et al.; ICLR 2025; arXiv 2403.17887**
- Manipulation: drop the contiguous block of n layers with the smallest angular distance between its input and the input n layers later, then "heal" with QLoRA. — [arXiv](https://arxiv.org/html/2403.17887v1)
- Finding: MMLU stays flat and then collapses once about **45–55%** of layers are removed (Llama-2), 35% (Mistral-7B), 25% (Phi-2) or 20% (Qwen). Deeper layers are very similar to their neighbors; the final layer is not. Simply dropping the deepest layers (excluding the last) works nearly as well. — [arXiv](https://arxiv.org/html/2403.17887v1); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html)
- Relevance: angular distance between block input and output is a cheap per-layer "effective contribution" metric.

**The Curse of Depth in LLMs. Sun, Song, Li, Yin, Zheng, Liu; NeurIPS 2025; arXiv 2502.05795**
- Manipulation: Pre-LN vs Post-LN (open models plus an in-house LLaMA-130M trained both ways), and LayerNorm Scaling (LNS, multiply the LN output by 1/√layer). — [arXiv](https://arxiv.org/html/2502.05795v3)
- Measured: per-layer pruning drop (MMLU/SQuAD) and angular distance. — [arXiv](https://arxiv.org/html/2502.05795v3)
- Finding: in Pre-LN, output variance grows with depth, so deep blocks' Jacobians approach identity and deep layers are prunable and redundant. **In Post-LN (BERT-Large, and Post-LN LLaMA-130M) the pattern inverts**: early layers are redundant and deep layers are critical. LNS helps from 130M to 7B. Scaled initialization interacts with LNS and reduces its benefit. — [arXiv](https://arxiv.org/html/2502.05795v3); [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html)
- Relevance: **major confound warning.** Normalization placement is part of the topology and determines *which* layers end up redundant. Any skip-connection study must fix or vary LN placement explicitly, and initialization scale interacts with it.

**What Matters in Transformers? Not All Attention is Needed. He et al.; arXiv 2406.15786 (venue unverified)**
- Manipulation: drop attention sublayers vs MLP sublayers vs whole blocks, ranked by input–output cosine similarity. — [arXiv](https://arxiv.org/html/2406.15786v6)
- Finding: deep attention layers are highly redundant. Dropping half the attention layers in Llama-2-70B costs about 2.4% performance for up to 48.4% speedup; MLP and block drops hurt more. On MAP-Neo-7B checkpoints, **attention redundancy is present throughout training**, while MLPs become more important over time. — [arXiv](https://arxiv.org/html/2406.15786v6)
- Relevance: redundancy differs by sublayer type. Topology experiments (parallel vs sequential attn/MLP) should measure attention and MLP contributions separately, including over training.

**Do Language Models Use Their Depth Efficiently? Csordás, Manning, Potts; NeurIPS 2025; arXiv 2505.13898**
- Measured, on Llama 3.1 and Qwen 3: each layer's contribution norm relative to the residual; skip a layer and measure the effect on later layers' computations for future tokens; logit lens; whether depth used grows with multihop/MATH difficulty; linear maps between a shallow (1.5B) and a deep (14B) Qwen 2.5. — [arXiv](https://arxiv.org/html/2505.13898v1)
- Finding: second-half layers change the residual much less (except the last few). There is a phase shift near the middle from "erasing" to "strengthening" features. Skipping a second-half layer barely affects future-token computation. Harder problems do not use more depth, and there is **no evidence of compositional processing across depth**. Shallow-to-deep linear maps fit best at the same *relative* depth, so a deeper model "stretches" the same computation. — [arXiv](https://arxiv.org/html/2505.13898v1)
- Controls: the main analysis uses only 10 GSM8K examples on Llama 3.1 70B, and one model pair for the linear maps (authors' own caveat). MoEUT may use depth better. — [arXiv](https://arxiv.org/html/2505.13898v1)
- Relevance: **key measurement precedent** for "composition vs refinement." Its relative-depth linear mapping is a reusable way to compare topologies.

**What Affects the Effective Depth of LLMs? Hu, Zhou, Zhang; arXiv 2512.14064 (Dec 2025)**
- Finding: across Qwen-2.5 1.5B–32B, the number of effective layers grows with size but the effective-depth *ratio* stays stable. Long-CoT distilled models and harder tasks do not increase effective depth. Effective depth is defined as the first layer where residual cosine turns positive, or where logit-lens KL falls below half its maximum. — [alphaXiv page](https://www.alphaxiv.org/abs/2512.14064). **Partly unverified:** model and benchmark details came from an AI-generated overview on that page, not the paper text.

**Residual-stream structure and iterative inference (lens and norm analyses)**
- Tuned lens (Belrose et al., 2023; arXiv 2303.08112): per-layer affine probes into vocabulary space are "more predictive, reliable and unbiased" than the logit lens, on models up to 20B. Prediction trajectories across layers can detect malicious inputs, and causal tests show the probe uses features similar to the model's. — [arXiv](https://arxiv.org/abs/2303.08112). Relevance: the standard reusable tool for measuring *iterative refinement* across depth in any residual topology.
- Residual stream norms grow exponentially (Heimersheim and Turner, LessWrong/Alignment Forum, 2023): GPT-2-XL norms grow about **1.045× per layer** (layers 5–41), and attention/MLP output norms and OV weight norms grow at the same rate. The hypothesis is that, under pre-LN, features are "hard to cancel but easy to overshadow," so later layers write larger outputs. EleutherAI models showed decreases. — [LessWrong](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward). Not peer-reviewed. Relevance: the per-layer norm growth rate is a cheap topology-sensitive metric, and it links to the Curse-of-Depth mechanism.
- Sparse crosscoders (Anthropic, Transformer Circuits, 2024): one global crosscoder beat 18 per-layer SAEs at equal feature count on eval loss, evidence of **cross-layer superposition** and redundancy. Most features peak in one layer and decay gradually, and decoder directions drift across layers, so persistent features are rewritten, not passively relayed. The authors call this preliminary. — [Transformer Circuits](https://transformer-circuits.pub/2024/crosscoders/index.html). Relevance: crosscoders could *measure* how skip topology changes feature persistence and drift across depth.

**Pre-LN vs Post-LN as topology**
- Xiong et al., ICML 2020: Post-LN has large expected gradients near the output at initialization, which makes warm-up necessary. Pre-LN gradients are well behaved, and warm-up can be removed. — [PMLR](https://proceedings.mlr.press/v119/xiong20b.html). Together with Curse of Depth and HC's seesaw framing, LN placement trades trainability against deep-layer collapse.

**Parallel vs sequential composition (beyond PaLM/GPT-J)**
- FFN Fusion (NVIDIA, arXiv 2503.18908, 2025): after attention layers are pruned from Llama-3.1-405B, consecutive FFN layers show low inter-dependency and can be run in parallel (Ultra-253B-Base: 1.71× latency speedup, 35× lower per-token cost). Some full blocks can also be parallelized, and it works better at larger scale. — [arXiv](https://arxiv.org/abs/2503.18908). Relevance: post-hoc evidence that trained sequential stacks contain large, nearly commutative or additive segments.
- Simplifying Transformer Blocks (He and Hofmann, arXiv 2311.01906; ICLR 2024, venue unverified): with signal-propagation-guided initialization, skip connections, value/projection parameters, *sequential* sub-blocks (i.e. switching to parallel) and normalization can all be removed while matching per-update training speed, with about 15% higher throughput and about 15% fewer parameters. — [arXiv](https://arxiv.org/abs/2311.01906). Relevance: **confound warning.** Skip connections can be substituted by initialization and signal propagation, so "skip vs no skip" comparisons must control initialization.
- Looped transformers (Saunshi et al., ICLR 2025; arXiv 2502.17416): a k-layer model looped L times nearly matches a kL-layer model on addition, p-hop induction and math, and is competitive on downstream reasoning. The authors describe a **reasoning-vs-memorization trade-off** (looping favors reasoning). — [arXiv](https://arxiv.org/abs/2502.17416v1). Relevance: the weight-tied composition axis, a parameter-matched comparison of depth topology against emergent capability type.
- Canon layers (Allen-Zhu, Physics of LMs Part 4.1; NeurIPS 2025; arXiv 2512.17351): lightweight horizontal (neighboring-token) mixing components, evaluated with **controlled synthetic pretraining tasks** that isolate reasoning depth and breadth and knowledge manipulation. The author argues academic-scale pretraining (about 1.3B parameters, 100B tokens) is too noisy for architecture comparisons. Reports up to 2× reasoning depth, NoPE lifted to RoPE level, and linear attention brought close to Mamba2/GDN. — [arXiv](https://arxiv.org/abs/2512.17351v1). Relevance: **methodology precedent.** Synthetic, capability-isolating tasks are a cleaner testbed for topology effects than perplexity.

### Inferences
- Several independent learned-routing papers (HC, MUDDFormer, DCA) converge on the same learned structure: strong weight on embeddings or early layers plus the most recent layers, and some parallelization of neighbors. That suggests the optimizer "wants" a topology between pure sequential and fully dense, and that the default pre-LN residual over-weights the cumulative sum. This convergence is my synthesis, not a claim any single paper tested.
- "Representation collapse" (high adjacent-layer cosine) is measured by HC, MUDDFormer, Curse of Depth, Gromov and He et al. It is the most widely used shared metric and an obvious common outcome variable for the project.
- Post-hoc reorder and parallelize studies show that *trained* sequential models end up with near-commutative middles, except for compositional or arithmetic tasks. This frames a clean hypothesis: composed topology matters mainly where the task requires serial composition.

### Gaps
- I could not see a figure or analysis in mHC of *what* the doubly stochastic routing learns (beyond its magnitude), nor a parameter-matched baseline description.
- I did not verify cross-layer attention / KV-sharing papers (e.g. cross-layer KV sharing, skip-layer attention) in this session.
- LayerDrop (Fan et al.) abstract: structured layer dropout lets sub-networks of any depth be extracted without fine-tuning — [arXiv 1909.11556](https://arxiv.org/abs/1909.11556). Its venue (ICLR 2020) is unverified, and I did not check whether it reports representation effects.
- I found no study that trains parallel vs sequential blocks at matched parameters and FLOPs and then compares *internal representations* (CKA, lens trajectories, feature persistence). Only loss and benchmarks are reported for PaLM-style blocks. This looks like an open niche.

---

## 2. CNN/MLP topology studies that measure representations, landscapes, or graph descriptors

### Takeaway
In CNNs and MLPs, skip connections have measurable representational signatures: Residual Alignment, iterative-refinement cosine with the negative gradient, a checkerboard CKA pattern, ViT CLS/spatial token phase transitions, and preserved gradient correlations. They also change landscape geometry (convex to chaotic without skips as depth grows). In NAS, *topology* (wide/shallow vs deep/narrow cells) mattered more than operation type for convergence. Graph-level descriptors are less robustly predictive than you might hope, and in pruning, layer-wise density mattered more than specific connections.

### Cited Findings

**Residual Alignment (RA). Li, Papyan et al.; NeurIPS 2023; arXiv 2401.09018**
- Manipulation: ResNets with vs without skip connections (fully connected and convolutional); stochastic depth; weight decay; number of classes. — [arXiv](https://arxiv.org/html/2401.09018v1)
- Measured: SVD of each residual branch's Jacobian, checking whether the top singular vectors align across depth (randomized SVD for convolutional nets). — [arXiv](https://arxiv.org/html/2401.09018v1)
- Finding: in well-generalizing ResNets, (RA1) intermediate representations are **equally spaced along a line**; (RA2) top singular vectors align across depths; (RA3) Jacobian rank ≤ number of classes; (RA4) top singular values scale as about 1/depth. **RA disappears when skip connections are removed.** Stochastic depth and larger weight decay strengthen it. It co-occurs with layer-wise neural collapse (no causal claim). — [arXiv](https://arxiv.org/html/2401.09018v1); [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/69897)
- Controls: the plain 16-block net failed to converge, so the no-skip comparison used 8 blocks (**a depth confound**). Measurements were on training data only. — [arXiv](https://arxiv.org/html/2401.09018v1)
- Relevance: **among the strongest precedents.** A parameter-free topology change (removing skips) switches an emergent geometric structure on or off, and Jacobian SVD alignment is a reusable measurement.

**Residual connections encourage iterative inference. Jastrzębski, Arpit, Ballas, Verma, Che, Bengio; arXiv 1710.04773 (ICLR 2018, venue unverified)**
- Measured: cosine between each block's output F(h) and the gradient ∂L/∂h; the ratio ‖F(h)‖/‖h‖; block-dropping accuracy; per-block classifier accuracy; unrolling the last block; sharing residual weights. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1710.04773)
- Finding: the cosine is negative in all blocks (each block moves features along the negative gradient), and more negative in higher blocks (about -0.1 for the last block of ResNet-110). The first blocks change the representation strongly (up to about 2× the norm) and dropping them is catastrophic; higher blocks make small refinements and dropping them is mild. Unrolling the last block 20 extra steps (with rescaling) raised accuracy on borderline examples from 43% to 53%. **Naive weight sharing causes activation explosion, plus both overfitting and underfitting**; unsharing BN statistics is the key fix. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1710.04773); [arXiv abstract](https://arxiv.org/abs/1710.04773)
- Relevance: the reusable metric cos(F(h), −∇_h L) directly operationalizes "iterative refinement" and can be compared across topologies. The weight-sharing result warns that normalization statistics confound weight-tying experiments.

**Highway and Residual Networks learn Unrolled Iterative Estimation. Greff, Srivastava, Schmidhuber; ICLR 2017; arXiv 1612.07771**
- Finding: they argue that successive layers within a stage refine estimates of the *same* features. In a 50-layer ResNet, mean estimation error stays near zero in stages 1–3 and its spread shrinks across blocks. Highway transform gates are sparse per sample but not on average (most units copy their input). On ImageNet with BN: Highway 7.53% top-5, ResNet 7.17%. Without BN, Highway fits training data better but overfits more. **On Penn Treebank language modeling, gated-identity variants reach about 79 perplexity vs 91.3 for Residual (Plain 92.6)**, so a learned gate on the identity path mattered there. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1612.07771); [ICLR 2017](https://mlanthology.org/iclr/2017/greff2017iclr-highway)
- Relevance: a gated vs fixed skip comparison whose result is **task-dependent** (vision: fixed skip is fine; that LM setting: gating is much better). Also, removing one layer from a 15-layer VGG pushes CIFAR-10 error to about 90%, vs a minor drop for residual nets (cited therein).

**Visualizing the Loss Landscape of Neural Nets. Li, Xu, Taylor, Studer, Goldstein; NeurIPS 2018; arXiv 1712.09913**
- Manipulation: ResNet-20/56/110 with vs without shortcuts; Wide-ResNet width k=1–8. Measured filter-normalized 2D loss surfaces and test error. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1712.09913)
- Finding: without skips, landscapes go from nearly convex (depth 20) to **chaotic** (depth 56, 110), and a 156-layer no-skip net could not be trained. CIFAR-10 test error: ResNet-56 5.89% vs no-shortcut 13.31%; ResNet-110 5.79% vs 16.44%; ResNet-20 7.37% vs 8.18% (small difference when shallow). Width makes landscapes flatter. Flatness correlates with test error only qualitatively. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1712.09913); [NeurIPS](https://papers.nips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html)
- Relevance: the effect of skips **interacts with depth**. At shallow depth topology barely matters, so experiments need to span depth.

**The Shattered Gradients Problem. Balduzzi et al.; ICML 2017**
- Finding: in plain feedforward nets, gradient correlations between inputs decay **exponentially** with depth (white-noise-like gradients); in ResNets and highway nets they decay **sublinearly**. A "looks-linear" initialization can train deep nets without skips. — [PMLR](https://proceedings.mlr.press/v70/balduzzi17b.html)
- Relevance: gradient-correlation decay is a cheap, topology-sensitive metric to measure at initialization. The looks-linear result again shows initialization can stand in for topology (a confound).

**Attention is not all you need: pure attention loses rank doubly exponentially. Dong, Cordonnier, Loukas; arXiv 2103.03404 (ICML 2021, venue unverified)**
- Finding: decomposing self-attention networks into paths through heads shows that, **without skip connections or MLPs**, outputs collapse doubly exponentially to rank 1 (token uniformity). Skips and MLPs prevent this. — [arXiv](https://arxiv.org/abs/2103.03404)
- Relevance: a mechanistic reason the residual topology matters for token-level representation diversity. Rank of the token matrix by depth is a reusable metric.

**Do Wide and Deep Networks Learn the Same Things? Nguyen, Raghu, Kornblith; arXiv 2010.15327 (ICLR 2021, venue unverified)**
- Finding: high-capacity models (relative to dataset size) show a CKA "block structure" in which contiguous layers propagate the dominant principal component. The block is model-specific, while representations outside it are shared across models. Wide and deep models make different per-class errors. **Pruning blocks inside the block structure barely hurts accuracy.** Block structure also appears **without residual connections**. The CKA checkerboard is attributed to residual connections (post-residual representations resemble each other more), and linear probes suggest residuals help preserve representations within the block. — [arXiv](https://arxiv.org/abs/2010.15327); [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.15327)
- Follow-up, On the Origins of the Block Structure Phenomenon (Nguyen, Raghu, Kornblith; arXiv 2202.07184, venue unverified): the block comes from a few "dominant datapoints" with simple shared statistics (e.g. background color) and large activation norms. It varies by seed and disappears if those points are excluded. Color augmentation, first-PC regularization, Shake-Shake, transfer learning and small batches reduce or eliminate it. — [arXiv](https://arxiv.org/abs/2202.07184)
- Relevance: **confound warning for CKA-based topology comparisons.** Block structure can be driven by a handful of datapoints and by seed, so CKA differences between topologies need multiple seeds and outlier checks.

**Do Vision Transformers See Like CNNs? Raghu et al.; arXiv 2108.08810 (NeurIPS 2021, venue unverified)**
- Measured: the ratio ‖z‖/‖f(z)‖ of skip-branch to residual-branch norm, per token type; CKA after **removing the skip at a single block i** (retrained). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.08810)
- Finding: the CLS token is carried mostly by the skip path in the first half and mostly by the residual branch in the second half, with the reverse for spatial tokens (a "phase transition"). ViT relies on skips more than ResNet-50. Removing one skip **splits the CKA map into two dissimilar groups** (before and after block i) and costs about 4% accuracy for middle blocks. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.08810)
- Relevance: **direct precedent** for a single-edge topology ablation with a representational readout. The skip/branch norm ratio by token type is a reusable metric.

**NAS topology analyses**
- Understanding Architectures Learnt by Cell-based NAS (Shu, Wang, Cai; arXiv 1909.09569; ICLR 2020, venue unverified): DARTS, ENAS and others select **wide and shallow** cells (e.g. NASNet width 5c, depth 2). Randomizing connections to deeper, narrower variants slows convergence (C3, depth 4, fails at lr 0.25). **Randomizing operations while keeping the topology barely changes convergence, so topology matters more than operation type for convergence.** Wide/shallow cells have smoother landscapes and lower gradient variance but "may not generalize better." — [ar5iv](https://ar5iv.labs.arxiv.org/html/1909.09569); [arXiv](https://arxiv.org/abs/1909.09569)
  - Relevance: **confound warning.** Search methods that use early training speed as a proxy are biased toward topologies that converge fast, which is not the same as better solutions. Measure at convergence.
- NAS-Bench-101 (Ying et al., ICML 2019): average accuracy peaks at cell depth 3 and rises with width up to 5. Swapping 3×3 conv for 1×1 costs 1.16% and for max-pool 1.99%. Accuracy is locally correlated within about edit distance 6. Parameters and training time correlate positively with accuracy, but the best cell is not the most expensive. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1902.09635)
- Are NAS Benchmarks Well Designed? (arXiv 2303.16938): across NAS-Bench-101/201 and TransNAS-101, convolution operations dominate performance and a small subset of the operation pool is enough. Fixed outer skeletons and small operation pools limit what these benchmarks can say. — [arXiv](https://arxiv.org/abs/2303.16938). Relevance: NAS benchmarks conflate topology with operation choice. Graph-descriptor conclusions from them are bounded by the fixed macro skeleton.

**Small-world and sparse topologies**
- SWNet (Javaheripi et al., arXiv 1904.04862): transforming a topology toward the small-world boundary gives about 2.1× faster convergence to target accuracy on CIFAR-10/100/ImageNet, and DenseNet-level accuracy with fewer parameters. The abstract does not describe controls. — [arXiv](https://arxiv.org/abs/1904.04862)
- Pruning at Initialization: Missing the Mark (Frankle et al.; arXiv 2009.08576; ICLR 2021, venue unverified): for SNIP, GraSP and SynFlow, **randomly shuffling the kept weights within each layer, or reinitializing them, does not hurt accuracy**, so per-layer sparsity, not which specific connections, drives performance. — [arXiv](https://arxiv.org/abs/2009.08576)
  - Relevance: **strong confound warning** for connectome or sparse-topology claims. The proper null is a layer-density-matched random rewiring.

### Inferences
- The most rigorous "topology → emergent structure" evidence in CNNs comes from *single-factor* ablations with representational readouts (RA, Raghu's single-skip removal, Jastrzębski's cosine). Graph-statistic correlational studies are weaker.
- Skip-connection effects grow with depth (Li 2018, Balduzzi). Small-depth experiments may show null effects that do not generalize.

### Gaps
- I did not retrieve a primary-source study that tests whether small-world coefficient, clustering or path length *predicts* performance across many architectures with proper controls beyond You et al. 2020 (excluded as already known) and SWNet.
- I did not verify DenseNet feature-reuse ablation follow-ups (e.g. analyses of which dense connections carry weight) beyond the original DenseNet heatmap.

---

## 3. Connectivity changes that measurably altered emergent structure (modularity, specialization, composition, memorization, superposition, interference)

### Takeaway
Explicit spatial or wiring-cost constraints reliably produce modularity (Clune 2013; BIMT; seRNNs), and joint-task training alone can produce specialization (Dobs 2022). But learned routing in modular architectures often fails to specialize (Mittal 2022), and standard networks show weight-level specialization without reuse (Csordás 2021). Residual topology itself can *hurt* abstraction in generative self-supervised learning (Zhang et al., CVPR 2026). For continual learning, architecture choices (width, BN, pooling) matter much more than skip connections.

### Cited Findings

**Residual Connections Harm Generative Representation Learning. Zhang, Jiang, Gao, Willett, Maire; CVPR 2026; arXiv 2404.10947**
- Manipulation: x_{l+1} = α_l·x_l + f(x_l), with α_l decaying linearly with depth to α_min. This is a fixed schedule with no learnable parameters, applied to MAE ViT encoders and U-ViT diffusion. — [arXiv v4](https://arxiv.org/html/2404.10947v4)
- Measured: linear probe, kNN, fine-tuning, unsupervised segmentation mIoU, and effective feature rank. — [arXiv](https://arxiv.org/html/2404.10947v4)
- Finding (MAE ViT-B/16, IN-1K): kNN 27.4 → **63.9**; linear probe 67.8 → **72.7**; fine-tuning 83.6 → 82.9 (slightly worse); COCO mIoU 4.10 → 10.41. The authors argue identity shortcuts inject "an echo of shallower representations into deeper layers," and they find that low effective feature rank correlates with better downstream performance (the CVPR abstract words it as a correlation between low effective rank and downstream performance). A learnable α does worse than the fixed schedule (79.5 vs 83.6 on IN-100). α_min ≤ 0.4 is unstable. Deeper encoders need a smaller effective α. — [arXiv](https://arxiv.org/html/2404.10947v4); [CVPR 2026](https://cvpr.thecvf.com/virtual/2026/poster/36560)
- Note: the paper reports a baseline linear probe of 67.3% in one place and 67.8% in Table 1 (internal inconsistency). — [arXiv](https://arxiv.org/html/2404.10947v4)
- Relevance: **among the most direct precedents.** A parameter-free change in skip strength changes the *abstraction level and rank* of learned features, and the effect depends on the objective. Its finding that learned skip weights underperform fixed ones is notable.

**Are Neural Nets Modular? Csordás, van Steenkiste, Schmidhuber; ICLR 2021; arXiv 2010.02066**
- Method: learn differentiable binary weight masks (Gumbel-sigmoid) on frozen networks per subtask; measure IoU of the masks and the accuracy drop when a mask is removed. — [arXiv](https://arxiv.org/abs/2010.02066)
- Finding: some specialization (addition vs multiplication use partly separate weights), but **weak reuse**: the same operation on different inputs uses different weights, and "weight sharing across tasks is mostly driven by shared I/O rather than task similarity." On SCAN, masks show non-compositional, pattern-matching solutions (about half the I_JUMP output weights are removed in the add-jump split). In CIFAR-10 CNNs, removing class-exclusive weights drops that class's accuracy by 40–60% (simple CNN) and nearly 100% (ResNet-110). — [arXiv](https://arxiv.org/abs/2010.02066)
- Relevance: **reusable measurement** (mask-based functional modularity and reuse). It is a baseline expectation that standard topologies specialize without composing.

**Is a Modular Architecture Enough? Mittal, Bengio, Lajoie; NeurIPS 2022; arXiv 2206.02713**
- Manipulation: monolithic vs modular (learned soft routing) vs modular-op (routing by rule context only) vs GT-Modular (oracle routing), on synthetic multi-rule data. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.02713)
- Measured: Collapse-Avg and Collapse-Worst (module under-use), Alignment (distance of the module-activation matrix to the nearest permutation matrix), Adaptation, Inverse Mutual Information. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.02713)
- Finding: modular beats monolithic slightly, both in-distribution and OOD. **End-to-end learned routing does not reach oracle specialization**, and collapse worsens with more rules. Specialization helps a lot when there are many rules. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.02713)
- Relevance: **reusable specialization metrics**, and a warning that providing parallel modules does not guarantee they specialize.

**Pruned Neural Networks Are Surprisingly Modular / Clusterability. Filan, Hod, Wild, Critch, Russell; arXiv 2003.04881 (venue unverified)**
- Method: spectral clustering of the |weight| graph of MLPs (k=4), scored by n-cut against 320 within-layer shuffles. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2003.04881)
- Finding: pruned MLPs are often more modular than all shuffles (all Fashion-MNIST networks, about 2/3 of MNIST networks, 1/10 of CIFAR-10 networks), but still fairly monolithic in absolute terms. **Dropout increases modularity.** Memorizing networks (no dropout) became modular. On random labels, SGD alone did not raise modularity before pruning. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2003.04881)
- Relevance: a reusable graph-modularity metric with a proper shuffle null. Its results show training regularizers (dropout, pruning) confound modularity measurements.

**Brain-Inspired Modular Training (BIMT). Liu, Gan, Tegmark; Entropy 2023; arXiv 2305.08746**
- Manipulation: neurons embedded in geometric space, with a length-weighted connection penalty. Finding: visible modules emerge that reveal compositional structure in symbolic formulas, interpretable classification features, and structure in algorithmic tasks. — [arXiv](https://arxiv.org/abs/2305.08746v1); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10814460/)
- Relevance: a precedent for a connectivity *cost* (not fixed topology) producing modularity. This differs from parameter-free routing and should be kept separate.

**Spatially embedded RNNs. Achterberg, Akarca, Strouse, Duncan, Astle; Nature Machine Intelligence 2023**
- Finding: RNNs in 3D space with sparse, wiring-cost-constrained connectomes converge on **modular small-world** structure, spatial clustering of functionally similar units, and an energy-efficient **mixed-selective** code, with these features emerging together. — [Cambridge repository](https://www.repository.cam.ac.uk/items/6a19fa95-1ab1-4ef1-beb1-4782f4cd137b). A follow-up (arXiv 2409.17693) reports that spatial embedding promotes a specific form of modularity with low entropy and heterogeneous spectral dynamics. — [arXiv](https://arxiv.org/pdf/2409.17693) (abstract only seen via title)

**The Evolutionary Origins of Modularity. Clune, Mouret, Lipson; arXiv 1207.2743 (Proc. R. Soc. B 2013, venue unverified)**
- Finding: networks evolved under performance plus connection cost became significantly more modular **and more evolvable** than performance-only networks. — [arXiv](https://arxiv.org/abs/1207.2743)

**Brain-like functional specialization emerges spontaneously. Dobs, Martinez, Kell, Kanwisher; Science Advances 2022**
- Finding: CNNs trained jointly on face and object recognition spontaneously segregate into separate face and object subsystems; single-task networks do not. Segregation varies for other categories. — [CBMM](https://cbmm.mit.edu/node/5085)
- Relevance: specialization can come from *task mixture* with no topology change. It is a confound when attributing modularity to topology.

**Connectome-constrained networks predict neural activity across the fly visual system. Lappalainen et al.; Nature 2024**
- Finding: an optic-lobe connectome (64 cell types) with unknown single-neuron and synapse parameters, trained on motion detection, predicts neural activity matching 26 experimental studies. It works better when connectivity is sparse. — [RePEc/Nature](https://ideas.repec.org/a/nat/nature/v634y2024i8036d10.1038_s41586-024-07939-3.html)
- Relevance: strong evidence that fixed connectivity plus task training constrains emergent function. This is the biological analogue of the project's premise.

**Architecture Matters in Continual Learning. Mirzadeh et al.; arXiv 2202.00275 (ICML 2022, venue unverified)**
- Finding (Split CIFAR-100): **skip connections had a negligible effect** on forgetting (CNN×4: 68.1 → 68.2 average accuracy, 8.7 → 8.9 forgetting, within the standard deviation). Width helps; depth hurts even with more parameters (CNN×4 at depth 6: 62.9 average accuracy, 12.4 forgetting vs 68.1 and 8.7 at depth 3). BN helps on Split CIFAR but hurts on Permuted MNIST. Global average pooling increases forgetting (60.1 average accuracy, 14.3 forgetting). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00275)
- Controls: partial. The MLP width/depth comparison is roughly parameter-matched; the GAP control confounds width with parameters (32.3M). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00275)
- Relevance: a **null result** for skips on interference, at shallow depth. Worth replicating with richer topologies.

**Generative skip models (VAEs). Dieng, Kim, Rush, Blei; AISTATS 2019**
- Finding: skip connections from the latent to the likelihood provably increase mutual information between observations and latents, and reduce posterior collapse at similar predictive performance (MNIST, Omniglot, Yahoo). — [PMLR](https://proceedings.mlr.press/v89/dieng19a.html)
- Relevance: here connectivity changes *which information* a representation must carry, a clean information-routing precedent.

### Inferences
- Pairing Zhang et al. (skips hurt abstraction in MAE and diffusion) with RA and iterative-inference results (skips produce refinement-like, low-rank, aligned updates) suggests a coherent hypothesis: identity skips bias networks toward *refinement of a persistent representation* rather than *re-representation*. That is useful for gradient flow but can limit abstraction when the objective rewards reconstructing low-level detail. This is my synthesis, not a tested claim.
- Modularity results split into two kinds: those driven by explicit costs or regularizers (BIMT, seRNN, Clune, dropout, pruning) and those driven by task mixture (Dobs). Clean evidence that a *parameter-free fixed topology* (e.g. parallel branches) by itself causes functional specialization is scarce.

### Gaps
- I found no primary study measuring **superposition** (e.g. feature-to-neuron ratio, interference) as a function of skip or branch topology at fixed parameters. Crosscoders give cross-layer superposition evidence but did not vary topology.
- I found no study that varies parallel vs composed branches and measures **compositional generalization** at matched parameters, beyond looped transformers (weight tying) and Painters (post-hoc only).
- I did not retrieve memorization-vs-topology studies beyond Filan (memorizing networks become modular) and Saunshi (looping trades memorization for reasoning).

---

## 4. Highway/gated vs fixed skips; U-Net skip connections; skips in generative models

### Takeaway
Gated identity paths beat fixed residuals in some sequence-modeling settings (Greff PTB) but not in BN-equipped vision nets. In U-Nets, skips carry high-frequency detail and can work against backbone semantics (FreeU), and some skips are useless or harmful depending on dataset (UCTransNet). Learned skip weights (LAuReL, HC) help LLMs, but in MAE a learned α underperformed a fixed decay schedule (Zhang et al.).

### Cited Findings
- **Highway vs ResNet** (Greff et al., ICLR 2017): ImageNet top-5 with BN: Highway 7.53%, Highway-Full 7.29%, ResNet 7.17%. Without BN, Highway 10.03% vs ResNet 9.40%, with Highway fitting training data better but overfitting more. On PTB, gated variants reach about 79.1 perplexity vs Residual 91.3. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1612.07771)
- **FreeU** (Si et al., CVPR 2024; arXiv 2309.11497): in a diffusion U-Net, the backbone mainly drives denoising, while **skip connections mainly feed high-frequency detail** to the decoder, which can override backbone semantics. Rescaling at inference (amplifying the backbone, damping the low-frequency skip components in the Fourier domain) improves samples with no training. — [arXiv](https://arxiv.org/abs/2309.11497); [CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Si_FreeU_Free_Lunch_in_Diffusion_U-Net_CVPR_2024_paper.html)
  - Relevance: **reusable measurement**, a frequency-domain decomposition of what flows through each path. It is evidence that the information routed through a skip is frequency-selective.
- **UCTransNet** (Wang et al., AAAI 2022; arXiv 2109.04335): in medical segmentation U-Nets, some skip connections are ineffective or *degrade* accuracy because of an encoder–decoder "semantic gap." On some datasets, a U-Net **without any skips beats the original U-Net**. — [arXiv](https://arxiv.org/abs/2109.04335v3); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20144)
- **Zhang et al. (CVPR 2026)** also add encoder–decoder skips to MAE to preserve reconstruction detail while decaying the identity shortcuts. That is a deliberate re-routing of low-level information *out of* the encoder's residual stream. — [arXiv](https://arxiv.org/html/2404.10947v4)
- **Dieng et al. 2019**: latent-to-decoder skips in VAEs reduce posterior collapse. — [PMLR](https://proceedings.mlr.press/v89/dieng19a.html)

### Inferences
- Across U-Net, MAE and VAE, the common theme is that skips decide *which information reaches the decoder or loss without passing through the bottleneck or deep layers*. That shapes what the deep path is forced to represent. This makes "information bypass" a measurable topology property (e.g. frequency content or mutual information along each path).

### Gaps
- I did not verify, from a primary source, analyses of skip removal in diffusion U-Nets beyond FreeU (search budget exhausted).
- I found no controlled study comparing gated vs fixed skips in modern pre-LN transformers on representational metrics.

---

## 5. Confounds: what these studies controlled or failed to control

### Takeaway
The recurring confounds are: (1) normalization placement, which flips which layers become redundant; (2) initialization and signal-propagation scaling, which can substitute for skip connections; (3) hidden costs (activation memory, throughput) that parameter/FLOP matching ignores; (4) bundled changes (FFN reallocation, extra learned scalars); (5) depth mismatches when the no-skip baseline won't train; (6) seed and outlier-driven CKA structure; (7) per-layer density vs specific wiring; (8) short-training or convergence-speed proxies; (9) codebase and implementation dependence.

### Cited Findings
- **Normalization placement:** Pre-LN makes deep layers near-identity and prunable; Post-LN makes early layers redundant instead. — [Curse of Depth](https://arxiv.org/html/2502.05795v3). HC frames Pre/Post-Norm as a seesaw and shows both are special cases of fixed connection matrices. — [HC](https://arxiv.org/html/2409.19606v3). Post-LN needs warm-up because of large gradients near the output. — [Xiong 2020](https://proceedings.mlr.press/v119/xiong20b.html)
- **Initialization substitutes for topology:** looks-linear initialization trains deep plain nets. — [Balduzzi 2017](https://proceedings.mlr.press/v70/balduzzi17b.html). With signal-propagation-guided choices, skips and normalization can be removed at matched training speed. — [He and Hofmann](https://arxiv.org/abs/2311.01906). Scaled initialization interacts with LayerNorm Scaling. — [Curse of Depth](https://arxiv.org/html/2502.05795v3)
- **Hidden costs:** HC's +26% activation memory and mHC's 6.7% time overhead despite unchanged FLOPs — [HC](https://arxiv.org/html/2409.19606v3), [mHC](https://arxiv.org/pdf/2512.24880v2). MUDDFormer's 84–96% training throughput despite +0.4% FLOPs. — [MUDDFormer](https://arxiv.org/html/2502.12170v2)
- **Bundled changes:** MUDDFormer's FFN width reallocation helps it but hurts the baseline. — [MUDDFormer](https://arxiv.org/html/2502.12170v2). LAuReL and HC add learned scalars, a low-rank map or dynamic weights, so they are not parameter-free. — [LAuReL](https://arxiv.org/html/2411.07501v3)
- **Depth mismatch:** RA's no-skip comparison used 8 blocks because 16 failed to converge. — [RA](https://arxiv.org/html/2401.09018v1). Li 2018 could not train a 156-layer no-skip net. — [Li 2018](https://ar5iv.labs.arxiv.org/html/1712.09913)
- **CKA artifacts:** block structure is driven by a few dominant datapoints, varies by seed, and is changed by augmentation and batch size. — [Nguyen 2022](https://arxiv.org/abs/2202.07184)
- **Density vs wiring:** shuffling kept weights within a layer preserves pruned-at-initialization accuracy. — [Frankle 2020](https://arxiv.org/abs/2009.08576)
- **Training-regime confounds in robustness and modularity:** GPT-2's dropout correlates with greater layer-deletion robustness. — [Lad et al.](https://arxiv.org/html/2406.19384v1). Dropout increases measured modularity. — [Filan et al.](https://ar5iv.labs.arxiv.org/html/2003.04881)
- **Proxy metrics:** NAS favors fast-converging wide/shallow cells that may not generalize better. — [Shu et al.](https://ar5iv.labs.arxiv.org/html/1909.09569). LAuReL's ResNet runs lasted one epoch. — [LAuReL](https://arxiv.org/html/2411.07501v3)
- **Implementation dependence:** most Transformer modifications did not transfer across a shared codebase, and the ones that helped were mostly minor or developed in that codebase. — [Narang et al., arXiv 2102.11972](https://arxiv.org/abs/2102.11972) (EMNLP 2021, venue unverified)
- **Noise at academic scale:** Allen-Zhu argues that 1.3B/100B-token pretraining is too noisy for architecture comparisons and recommends controlled synthetic tasks. — [Canon layers](https://arxiv.org/abs/2512.17351v1)
- **Small samples in interpretability claims:** Csordás et al.'s main depth analysis uses 10 GSM8K examples and one model pair for the linear maps. — [arXiv](https://arxiv.org/html/2505.13898v1)

### Inferences
- A rigorous topology study in this program should, at minimum: fix LN placement (or cross it with topology as a factor); match parameters, FLOPs, *and* report activation memory and throughput; tune learning rate per topology (or use muP-style transfer); use several seeds with CKA outlier checks; include density-matched random-rewiring nulls; train to convergence; and include a depth sweep, because skip effects scale with depth.
- Several high-profile topology gains (HC, MUDDFormer, DCA) are reported at a single tuning point without per-architecture LR sweeps in the material I saw. Their effect sizes should be treated as upper bounds until replicated under independent tuning. I did not find explicit per-arm LR sweeps in the fetched text, but I may have missed appendix details.

### Gaps
- I could not confirm, for HC, MUDDFormer or DCA, whether hyperparameters were re-tuned per architecture; this is likely in appendices I did not see.
- Several venues are marked unverified, and the web-search budget ran out before I could check them or look up additional U-Net, diffusion and cross-layer-attention studies. A follow-up search could verify venues for Jastrzębski 2018, Nguyen 2021/2022, Raghu 2021, Shu 2020, Frankle 2021, Dong 2021, Narang 2021, He and Hofmann 2024, Mirzadeh 2022, Clune 2013 and LayerDrop 2020.
