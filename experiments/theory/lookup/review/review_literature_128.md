# Review of docs/literature.md, section "The deeper and wider search" (Entry 128)

Independent review, 23 Nov 2026. Nothing in the repository was edited. 34 web searches, about 45 page reads.

**Read depth convention.** Every page was read through the fetch tool, which returns a summarising reader's answer over a character range of the page. "Complete" means the whole page fell in the range read; otherwise the share is given. Quotes are as that reader returned them; I did not see raw page text. Two pages could not be read: Hummos, *Thalamus* (ICLR 2023, ar5iv.labs.arxiv.org/html/2205.11713; permission timed out) and pmc.ncbi.nlm.nih.gov/articles/PMC11774440 (bot check).

---

## Task 1. The "not found" lines

Summary: none is overturned outright. All four are narrower than the table states, and (b) and (d) are wrong as worded.

### (a) Unseen configurations of single units, selected by context: narrowed, stands only in a restricted form

| Candidate | Read | What it does | Effect |
|---|---|---|---|
| Barin Pacela, Joshi, Camacho, Lacoste-Julien & Klindt, *Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation* (ICML 2026). https://arxiv.org/html/2603.28744v1 | first 93 % | Sparse latents; test supports are **sets of co-active units never co-active in training** ("the novel combination (z1, z3) is held out for OOD evaluation"), scaled to 10,000 latents. Compared: ReLU, JumpReLU, TopK and matching-pursuit autoencoders (one hidden layer, each unit an encoder row plus a decoder column, top-k selection by the input) against per-sample iterative inference (FISTA). Autoencoders plateau at 0.2–0.5 correlation with the true latents; FISTA on the true dictionary reaches ≥ 0.95; FISTA on the autoencoder's own dictionary only 0.15–0.4: "the SAE-learned dictionary itself is the binding constraint". Theory: restricted isometry gives recovery for every k-sparse support, seen or not. | **Narrows strongly.** Held-out unit configurations of a learned one-hidden-layer block, tested as such, exist, with a negative result for one-shot selection and a theory. Differences from the line: selection is by the input itself and not by a separate context; the block is trained to reconstruct; no loop in the learned model. |
| Pokel & Grewe, *Learning Compositional Latent Structure with Vector Networks* (arXiv 2605.28007, May 2026). https://arxiv.org/html/2605.28007 | first 41 % (Appendix E.1 "Sparse recovery of novel compositions" and all result tables unread) | Each layer's weight matrix is replaced by "a library of reusable rank-1 weight atoms" A = u sᵀ (a read row with a write column). Per input, a top-k sparse set of atoms and coefficients is inferred by minimising a layer energy that includes a top-down term, giving an input-specific weight matrix. Tested on recombined factors (cross-family function sums, mixed forces, digit compositions); claims "out-of-distribution error about an order of magnitude lower", but "both Transformer and Mamba outperform VN" on two reconstruction settings. Libraries are per layer; learning is local, not backpropagation. No measurement of whether test-time atom sets were new. | **Narrows.** Unit-grain, input-dependent weights composed from a learned bank with recombination tests is published. Held-out is at the level of data factors, not atom sets. |
| Hou et al., *Instruction-Following Pruning for Large Language Models* (ICML 2025). https://arxiv.org/html/2501.02086v3 | complete | A predictor reads the prompt and selects which feed-forward units run (9B to 3B active), trained jointly with the model; beats a dense 3B model (e.g. MMLU 65.5 against 61.8). Reports mask overlap across domains; no test of new unit combinations. | Narrows the "selected by context" half: done at language-model scale at unit grain. Leaves the held-out half. |
| Shaier et al., *More Experts Than Galaxies* (COMET; arXiv 2410.08003, venue not confirmed). https://arxiv.org/html/2410.08003v6 | complete | Per-unit masks from a fixed random routing network plus k-winners; C(N, k) implicit experts, "every input will have its own expert". Standard held-out test sets only. | Leaves standing; it is the dropout row of the section with input-dependent masks. |
| Schug, *Sparsely gated tiny linear experts* (arXiv 2606.07414, June 2026). https://arxiv.org/pdf/2606.07414 | complete | Single-unit linear experts, top-8 per token by product keys; the selection is the only nonlinearity (18.19 perplexity against 18.30 with PEER's router). No held-out combinations. | Leaves standing; a 2026 neighbour the section lacks. |
| Cheng et al., *Mixture of Neuron Experts* (arXiv 2510.05781). https://arxiv.org/abs/2510.05781 | abstract | Top-k units inside each selected expert. | Leaves standing. |
| Shan, Minni & Duncker, *Separating the what and how of compositional computation* (arXiv 2510.20709). https://arxiv.org/pdf/2510.20709v1 | first 94 % | Recurrent weights are a belief-weighted sum of per-context low-rank components; one unseen task reaches about 83 % in 40 trials with the network frozen. Component grain, one held-out task. | Leaves standing. |
| Shi, Ouyang & Qian, *Sparsity-Driven Emergence of Functional Modularity* (arXiv 2610.32153). https://arxiv.org/pdf/2610.32153 | first 85 % | Post hoc weight masks per rule, recombined to rebuild a composite rule that was in training. | Leaves standing. |

**Required wording.** "Not found: unit configurations held out as such, where a context separate from the input selects the units, in a block trained on a task." The autoencoding case is published, with a failure mode (one-shot selectors) that bears directly on the design.

### (b) Single-unit bank shared across passes: wrong as worded; a narrower line stands

| Candidate | Read | What it does | Effect |
|---|---|---|---|
| Gregor & LeCun, LISTA (ICML 2010), via Monga, Li & Eldar, *Algorithm Unrolling* survey. https://arxiv.org/pdf/1912.10557 | first 71 % of the survey; original not read | Unrolled sparse coding; in the original "the same parameters are shared across different layers", so "the resulting network resembles an RNN"; later variants untie. | **Overturns the literal line.** A weight-tied network whose one bank of units is reused every pass, with a sparse active set chosen by the state, dates from 2010. |
| Rolfe & LeCun, *Discriminative Recurrent Sparse Auto-Encoders* (ICLR 2013). https://ar5iv.arxiv.org/html/1301.3775 | first 68 % | z^{t+1} = max(0, Ex + Sz^t − b), 11 passes, 400 units, "a deep network, with parameters shared between the hidden layers". Units take different roles by pass: part-units fire on the first pass, categorical-units "activate late in the dynamics". No untied comparison in the part read. | Same. Different units active on different passes of one tied bank, with task training, in 2013. |
| *Evaluating Sparse Autoencoders: From Shallow Design to Matching Pursuit* (arXiv 2506.05239; authors not checked). https://arxiv.org/html/2506.05239v1 | first 87 % | One unit picked per pass from one tied dictionary by the current residual; "each selection is driven by what remains unexplained". | Same, with explicit top-1 selection per pass. |
| Berges et al., *Memory Layers at Scale* (ICML 2025). https://arxiv.org/html/2412.09764v2 | complete | "a shared pool of memory parameters across all memory layers" (three layers, untied network); says PEER "is conceptually similar to a memory layer". | Already in Entry 126 but **missing from this table's "nearest"**, which names only 128-unit experts. It is the nearest at unit grain. |
| Jiao et al., *Improving Parameter Utilization by Sharing Neural Experts Across Layers* (CS-MoE; EMNLP 2026). https://arxiv.org/pdf/2609.22199 | complete | A central pool of block experts "accessible by every layer" plus one private expert per layer. Per-layer router 10.08 perplexity against one shared router 10.29; no private experts 10.32; static tying 12.74; top-10 % expert overlap 68.4 % between adjacent layers, 42.1 % five layers apart. Block grain, untied attention. | Not unit grain. A new data point for the state-only-selector question. Not in the notes. |
| Chen Weiguo, *DeepThink: A Conceptual Model for Knowledge-Addressable Recurrent Language Modeling* (arXiv 2610.16171, 3 Nov 2026). https://arxiv.org/pdf/2610.16171 | complete | A bank of "Knowledge Units", an activator that queries by state, context and step index with top-K, an assembly step, one shared operator reused across steps; grains down to low-rank increments; "A large N may coexist with small K". "We have not conducted experiments." | The architecture of the vision is now in print as a proposal, twenty days old. No evidence either way. Not in the notes. |
| *Ultra-Sparse Memory Network* (UltraMem; ICLR 2025; authors not checked). https://arxiv.org/html/2411.12364v2 | first 79 % | Memory split into "multiple smaller units" across layers; no shared table found. | Leaves standing. |
| Monea et al., *The Surprising Effectiveness of Shared Memory in Looped Transformers* (arXiv 2610.02383); Oncescu et al., *The Recurrent Transformer* (arXiv 2604.21215); Sapunov, *Universal Transformers Need Memory* (arXiv 2604.21999) | all complete | Memory is activations (caches, register tokens), not stored parameters. | Not relevant. |

Searches for single-unit experts retrieved by key inside a weight-tied loop (PEER or product-key memory in a universal or looped transformer) returned nothing.

**Required wording.** "Not found: unit-grain experts *retrieved by key*, so that compute is below bank size, from one bank reused on every pass of a weight-tied model. A tied bank of units with a state-chosen sparse active set is old (LISTA 2010; Rolfe & LeCun 2013); a unit-grain bank shared across depth exists in an untied network (Memory Layers)."

### (c) Bank size against the layers' sharing, with exact references: stands

| Candidate | Read | What it does | Effect |
|---|---|---|---|
| Tian & Shen, *Statistical Gains from Looped Estimation under Parameter Budgets* (arXiv 2609.25778). https://arxiv.org/pdf/2609.25778 | first 50 % | Minimax rates for a looped against an untied estimator under one parameter budget; the ratio of looped to untied worst-case risk tends to zero. Target is a smooth function; no parameter for how much layers share; no partial-sharing estimator. | Leaves standing. The nearest theory; should be cited. |
| *Neural Parameter Allocation Search* (ICLR 2022; authors, Plummer et al., from memory). https://ar5iv.arxiv.org/html/2006.10598 | first 64 % | Layers draw from a shared bank of any size, assigned by learned layer similarity; results at 6.9–100 % budgets. No ground truth. Static. | Leaves standing. |
| *Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning* (MASA; arXiv 2508.04581; authors not checked). https://arxiv.org/html/2508.04581v2 | complete | Shared dictionary atoms with per-layer coefficients; 2, 4, 6, 8 atoms for 12 layers; attention matrices only. No teacher. | Leaves standing. |
| Savarese & Maire (in the notes), synthetic section re-read. https://arxiv.org/pdf/1902.09701 | complete | Shortest-path task with "a natural recurrent algorithm": only one template per layer is tried; the number of templates is not varied against the task's structure. | Leaves standing. |
| Wang et al., *Loop-μP* (arXiv 2606.18524) | first 40 % | Residual scaling by loop count. | Not relevant to (c); relevant to stability. |

Not fetched, from memory only, so unverified: the multitask representation-learning theory (Tripuraneni et al. 2020; Du et al. 2020) answers the same question with tasks in place of layers.

### (d) Selection conditioned on past selections, for error correction or search: over-claimed

| Candidate | Read | What it does | Effect |
|---|---|---|---|
| Bakermans et al. (2025), already in Entry 118. https://arxiv.org/pdf/2510.01858v1 | complete | A gating network conditions each module choice on the previous one, g_t = G(x_t, g_{t−1}, z_{t−1}); a particle filter must "track different hypotheses on all the possible sequences of module selections until new feedback arrives", then "the posterior continues only from the hypothesis confirmed by this feedback"; no parameter updates. | **Narrows.** Search over module sequences with selection conditioned on past selections, discarding wrong branches on evidence, at test time. The section's own notes hold it. It is sampling-based inference, not one trained trajectory. |
| Sabour, Frosst & Hinton, *Dynamic Routing Between Capsules* (NeurIPS 2017). https://ar5iv.labs.arxiv.org/html/1710.09829 | complete | "The initial coupling coefficients are then iteratively refined by measuring the agreement"; logits accumulate over three iterations inside the forward pass; presented as "explaining away". | **Narrows.** Routing revised inside the forward pass from the outcome of the previous routing. One layer pair, not a sequence of blocks. Capsules appear nowhere in the notes. |
| Matching pursuit as a network (2506.05239, above) | first 87 % | Each pick conditioned on the residual left by earlier picks; residual norm falls monotonically. | Narrows at unit grain. |
| Qiu et al., *HERO-MoE* (arXiv 2609.32581). https://arxiv.org/pdf/2609.32581 | complete | Earlier layers' routing distributions added to the current router's logits; a prior, not a correction (loss 1.6393 to 1.6184). | Leaves standing; a second instance of the 2025 mechanism. |
| Frey et al., *Some Tokens Think, Others Remember* (arXiv 2605.30202) | complete | Dense gates over a looped path and a wide path. | Not relevant. |

**Required wording.** "Not found: a trained looped network that reads its own record of selections and returns to an earlier choice among stored blocks. Evidence-driven revision of selection exists as particle filtering over module sequences (Bakermans), as iterative routing (capsules) and as residual-driven selection (matching pursuit)."

---

## Task 2. Spot-checks

| # | Claim | Result |
|---|---|---|
| 1 | MoEUT | **Holds.** Selection is "s_t = σ(x_t W_S)" on the layer input only. Both sentences are verbatim, the first returned by the reader in two pieces ("We can conclude that MoEUT is capable of specializing in a specific layer if necessary and sharing weights between them when advantageous."; "This indicates that experts usually change dynamically in a single column, performing different functionality in different layers."). Groups "between 2 and 4" of "non-shared" layers. Add: layers inside a group have separate weights, so position within the group is fixed by construction; "no pass index" is true of repetitions only. |
| 2 | Looping Beyond Twice | **Holds with correction.** The paper says routers "repeatedly select the same experts across loops" and assignments are "nearly invariant", not identical on every loop. Nine loops unsegmented: "severe degradation, with evaluation loss reaching approximately 4.4", perplexity 81.04, marked "an unrecoverable loss spike". A fourth remedy, a moving average of earlier attention outputs, is omitted. |
| 3 | Chain-of-Experts | **Wrong as stated.** The ablation is not one gate network re-applied to the updated state. It "removes iteration-specific gating and instead reuses the same expert selection across all steps", with gating weights "fixed across iterations" (Sec. 5.3, Eq. 8). The result ("plateaus around 1.5, worse than both standard CoE and the MoE baseline") shows that repeating one selection is bad, not that a shared selector produces one selection. "Two iterations match three times the width" holds (abstract). |
| 4 | SCOFF | **Holds.** Query q̃_k = h_{t−1,k} W̃^q; key κ̃_{k,j} = (h̃_{t,k,j} W̃^e)ᵀ with h̃_{t,k,j} = GRU_{θ_j}(z_k, h_{t−1,k}); hard Gumbel choice. Add: the parameter-matched GRU scores 74 %, not 56 %. |
| 5 | Arrow | **Holds.** "decomposing the outer product A_iB_i^T with SVD and taking the right first singular vector"; "doesn't require access to training data"; logits = abs(W h) because "both v and −v are valid singular vectors"; A and B averaged separately; 66.6 against 64.8; 63.9 against 64.6. |
| 6 | Compositional Attention | **Holds.** Keys K̄_ij = Retrieval_ij W̄_k with Retrieval_ij = Search_i V_j. 0.17 / 0.18 against 0.24 / 0.67 is the best row (4 searches, 8 retrievals, width 512, L1 loss); at width 128 the same model scores 0.47 and 0.88 out of distribution. Table text was garbled in extraction; moderate confidence in the row labels. |
| 7 | Kobayashi et al. | **Holds.** Six modules; the transformer "fails to compositionally generalize to the held-out OOD tasks" although "it is possible to linearly decode the task latent code on OOD tasks from the residual stream"; the hypernetwork model generalises and "severely degrades for disconnected task support". |
| 8 | Saxe et al. | **Holds with correction.** The 40 % figure is from simulation at M = 10 (Fig. 5c: "roughly 40% of input-output routes"). M = 7, K = 4 are the parameters of Fig. 4, and nothing is "exact" about the threshold. 25 % is leaky-ReLU MNIST only (Fig. 9). The small-initialisation sentence and "does not provide a mechanism to select the gating structure" are verbatim. |
| 9 | Bahdanau et al. | **Holds with correction.** The 32 % (31.89 ± 0.75, p(tree) 0.08) and "stays at 0.5" (0.56, error 1.64 %) are Residual modules at 1 rhs/lhs. With Find modules a start at 0.5 converges to the tree (0.94), and at 18 rhs/lhs three of five Residual runs converge from 0.1. "Only from a lucky start" is too strong; it depends on module type and data variety. The blurred-parametrisation result is confirmed from the introduction and the Fig. 5 caption only; its numbers lie beyond the 59 % I read. |
| 10 | Lory | **Holds.** θ̄ = Σ e_i θ_i; routing from "the average of the hidden representations of its preceding segment", T = 256; 13.9 % is Books, 0.3B with 32 experts; "about 900 experts among 1536" is the 1.5B/32-expert model; experts start as duplicates of a warm-up dense model. Soften "needs": the 7B runs were trained without similarity batching. |
| 11 | Riveland & Pouget | **Holds.** 83 %, 39 %, 88 %; 50 tasks, five held out. |
| 12 | Jayakumar et al. | **Holds with correction.** "W′ = zᵀW + V and b′ = zᵀU + b"; Theorem 1 is strict inclusion; "summation is trivially easier for an MLP". The paper says "we do not specifically consider attention" and names mixtures of experts only as an earlier example; neither is derived as a restricted form. |

---

## Task 3. The conclusions

**Over-claims of remaining novelty.**
1. Table lines (a), (b), (d) as worded; see Task 1.
2. Design item 4, "the information is where these do not reach: learned selection, and unit grain." Unit grain is reached: sparse-recovery theory covers unseen supports, and Barin Pacela et al. measure the failure of one-shot selectors there. That result also gives the loop a job the section does not mention: iterative, residual-driven selection is what works on unseen supports, given a good dictionary.
3. "Two 2026 groups work one step away." At least six do: add Pokel & Grewe, Barin Pacela et al., Schug, and the DeepThink proposal.

**Under-claims (too much conceded).**
4. "Claim (3) of the vision at this grain is this paper [MoEUT]." Fair for the mechanism: one pool across repetitions, a state-only selector, different experts at different depths, shown by routing statistics. Not fair for the claim itself. By the paper's hyperparameter table as the reader returned it, the 243M model stores about 99,000 pooled expert units against about 74,000 feed-forward units in the 18-layer baseline, at equal depth and equal total parameters (13.24 against 13.40). A bank smaller than the layers it replaces is not shown, and there is no reference network of distinct layers. MoRE, in the same section, reports MoEUT clearly worse in its setting.
5. "In a transformer the token's state already carries the context, so 'chosen by context items and state' is not an extra ingredient." Not correct as stated, and contradicted inside the section. Kobayashi et al.: the combination is linearly decodable from the state and the transformer still fails to act on it. Section E: a pooled address loses which item called which block. Lory routes from a separate context summary in a transformer because the token state is not used for it. A state that contains the context is not a selector that binds units to context items.
6. "Selection by the stored function's own response has at least seven published forms", used to say contribution B has prior art. The seven are three different things: keys from the weights with no data (Arrow only); keys from each module's output on the input, which costs a run of every module (SCOFF, Compositional Attention, Supermasks, modular meta-learning); a separate scorer trained beside each module (Expert Gate, Local Module Composition, MOSAIC). Entry 118's line "keys computed from the stored weights themselves" is overturned by Arrow alone.
7. Design item 1, "selection without a pass index has a known failure and a known cure." With Chain-of-Experts removed (check 3), the evidence for collapse of a state-only selector is one paper (Looping Beyond Twice), plus CS-MoE's small gap (10.29 against 10.08). Against it: MoEUT, How to Loop MoE, and Sparse Layers are Critical (Entry 126). The table row "no without a per-pass selector" must drop Chain-of-Experts. Keeping the per-pass arm is still right.

**Bodies of prior work missing altogether.**
- **Sparse coding and unrolled iterative inference.** A dictionary is a bank of units; inference selects a sparse subset per input; the same dictionary is reused every iteration. Fetched: LISTA via the unrolling survey, Rolfe & LeCun 2013, matching-pursuit autoencoders 2025, Barin Pacela et al. 2026, Vector Networks 2026. The notes contain no mention of LISTA, sparse coding, unrolling or compressed sensing. This is the most important gap; it touches (a), (b) and (d).
- **Capsule routing** (fetched: Sabour et al. 2017): routing revised inside the forward pass.
- **Context-conditioned pruning of units** (fetched: Hou et al. 2025; COMET).
- **Statistical theory of looped against untied estimators** (fetched: Tian & Shen 2026).
- Not fetched: associative memories (Hopfield networks, sparse distributed memory), tensor-product binding, and Hummos 2023 (blocked).

**"How much weight."** The paragraph's direction is right and this review supports it: a fifth pass narrowed every remaining line within a few hours. It should say that.

---

## Verdict

**Sound with corrections.**

1. Reword line (a) as given in Task 1 and cite Barin Pacela et al. 2026, Vector Networks 2026, Hou et al. 2025.
2. Replace line (b): the literal claim is false (LISTA 2010; Rolfe & LeCun 2013). State the narrower claim, and name Memory Layers at Scale as the nearest at unit grain.
3. Keep line (c); add Tian & Shen 2026, Neural Parameter Allocation Search and MASA as nearest.
4. Narrow line (d) and item 6 of "overturns or narrows": cite Bakermans et al. (already in the notes), capsule routing and matching pursuit.
5. Correct the Chain-of-Experts entry: the ablation reuses one expert selection; it does not share a gate network. Remove it from the table row on state-only selectors, from section G's list, and from the evidence behind design item 1.
6. Correct Saxe et al.: about 40 % is a simulation at M = 10; delete "M = 7, K = 4 exactly"; 25 % is one leaky-ReLU case.
7. Correct Bahdanau et al.: the quoted numbers are Residual modules at 1 rhs/lhs; Find modules converge from 0.5; drop "only from a lucky start".
8. Qualify "Claim (3) … is this paper": mechanism yes; a bank smaller than the replaced layers, and a reference of distinct layers, no.
9. Withdraw "the token's state already carries the context, so … not an extra ingredient", or restrict it to "a router reading the state is weakly context-conditioned".
10. Split "seven published forms" into the three kinds; say Arrow alone computes keys from weights.
11. Add a subsection on sparse coding and unrolled inference, and add capsules, CS-MoE and DeepThink to the entries; rewrite design item 4 to write predictions against the sparse-recovery result (one-shot selector fails on unseen supports; iterative selection needs a good dictionary).
12. Minor: SCOFF's parameter-matched baseline is 74 %; Looping Beyond Twice says "nearly invariant" and reports perplexity 81.04; Lory does not strictly need similarity batching; Jayakumar et al. do not treat attention; the Compositional Attention figures are the best row.
