# Rung 0: learned depth allocation on pointer chasing with hidden chain length

**Status:** design, 27 Oct 2026. Not yet a preregistration. This is the first toy of the ladder in history Entries 45–51: one shared block, learned halting, no selection among primitives yet. It is written to be reviewed for soundness *and* for experimental value before any code (lesson L22), then implemented, piloted and preregistered.

**Ledger:** I25/I26 retired as thesis tests (Entry 44); the thesis itself is Entry 48 (amortized EM over per-token routes) and the motivation Entry 49.

**Scope note (AGENTS hard rule 1).** This round compares static depth with learned dynamic depth, at the researcher's direction. Every claim is labeled expressivity, optimization or mechanism.

---

## 1. The question, and why this toy

**The thrust (Entry 49).** A static transformer charges every token its full depth. The depth a token needs is dynamic and heavy-tailed: most tokens settle early, a few need many steps. Looping a shared block fixes the parameter cost, not the compute cost. The object to learn is a *depth policy*.

**What exists.** Per-token recursion over a shared block (Mixture-of-Recursions, depths ≤ 4, no ground truth for demand); learned halting (ACT, PonderNet, Universal Transformer; ACT collapses toward constant depth); looped transformers needing input injection (Yang 2024) and known step counts for length generalization (Fan 2025).

**The minimal unanswered question.** Whether a depth policy learned from next-token loss plus a compute cost, with no step-count supervision:
- **Q1** allocates depth to match true per-token demand when demand is heavy-tailed, so deep tokens are rare and carry rare gradient;
- **Q2** extrapolates to demands deeper than any seen in training;
- **Q3** can be driven by an internal signal (the reducible residual, Entry 46) rather than a ponder cost;
- **Q4** trains stably through many loops.

Plus the learning-signal claim of Entry 50 (**Q5**): one shared block pools gradient across depths, so its accuracy on rare deep tokens should not depend on how rare they are, while a static model's does.

**Why pointer chasing.** The required depth per token is exact, serial by construction and hidden until executed; the static ceiling is derivable; exact references survive; and it is a real transformer circuit (iterated induction-head composition).

## 2. The task

**Symbols.** Keys K (|K| = 64), answers A (|A| = 16), markers {Q}. Keys and answers have **frozen random orthogonal embeddings** (80 symbols ≤ d); the marker and position embeddings are learned.
- Rationale: content must live in the context, not the weights, which is the thesis's assumption; it makes held-out symbols meaningful at test (P-transfer); and it makes the planted construction exact (§5).
- A control with learned symbol embeddings is run in one cell on in-vocabulary symbols.

**An episode.** A forest of chains over keys, each chain ending at an answer symbol. A chain of length k is k pairs: key₁→key₂, …, key_k→answer.
- Qn = 8 queried chains with lengths k_i ~ Zipf(α) truncated at k_max; distractor chains fill the pair budget P = 48. Keys are unique across all pairs (a forest), drawn fresh per episode.
- **Tokens.** Each pair is **one token** whose embedding is e_key(key) + e_val(value), with separate tables for the key slot and the value slot (so the slots occupy different subspaces). After the P shuffled pair tokens come the queries: for each, a query token e_Q + e_start(start key), then the answer token e_ans(answer). Sequence length P + 2·Qn = 64.
- **Loss.** Standard next-token loss at every position. Only the answer predictions (at the query positions) are informative; pair tokens and starts are uniform random. Accuracy and all per-k metrics are measured at query positions.
- **Hidden k.** The chain's length is not visible from the start key, the pair order or anything positional: it is learned only by following the chain.
- **Compute demand per token.** A pair token needs no computation (its own embedding is its final state up to a linear map); a query token needs k hops. So per-token demand is heavy-tailed within every sequence: 48 tokens at 0–1 steps, 8 tokens at k+1.

**No shortcuts.** Pair order is shuffled; keys and answers are fresh per episode; starts are chain heads of distinct chains, so earlier queries' answers never help later queries; answers are uniform. The only static shortcut is doubling (a layer that turns i→j and j→l into i→l for every pair at once), which is exactly the ceiling we derive.

**Distributions.**
- Training: α ∈ {1.0, 1.5, 2.0} (three cells), k_max = 8.
- Evaluation: k = 1 … 32 stratified (uniform over k, 1,000 queries per k), on three symbol sets: in-vocabulary, held-out keys, held-out answers.

## 3. Arms

Block = pre-norm causal attention + MLP + a linear map on the stream. The linear map lets a block *overwrite* a subspace (register replacement), which looped execution needs and which static stacks get for free by rotating into new subspaces each layer. All arms use the same block type.

| arm | depth | halting | injection | role |
|---|---|---|---|---|
| **S4** | 4 unique blocks | — | — | static; doubling ceiling k ≤ 4 |
| **S8** | 8 unique blocks | — | — | static; ceiling k ≤ 64 by doubling, k ≤ 7 by serial hops |
| **LF** | 1 shared block × T_train = 9 loops (test up to 33) | none (fixed) | yes | reuse without allocation |
| **LO** | shared, oracle | loops k+1 on queries, 1 on pairs (uses ground truth) | yes | privileged upper bound on allocation |
| **LH-s** | shared, learned | halting head reads the **state only** | yes | the thesis arm |
| **LH-i** | shared, learned | halting head reads state **and step index** | yes | P-signal control |
| **LF-ni**, **LH-s-ni** | as above | | **no** injection | Q4 |
| **LR** | LF's checkpoint | residual-triggered at test: ‖Δh‖ or step-KL below a threshold | yes | Q3, no extra training |
| **LV** | shared | a value head trained on remaining hops (privileged) decides halting | yes | labeled privileged; Entry 46 |

**Learned halting (LH).** PonderNet: λ_t = σ(w·LN(h_t) + b) (plus a step-index embedding for LH-i); p_t = λ_t Π_{s<t}(1 − λ_s); training loss Σ_t p_t · CE(head(h_t), y) + β · KL(p ‖ Geometric(λ_p)); at test, halt at the first t with cumulative p ≥ 0.5, and **freeze** the token's state thereafter (other tokens attend to the frozen state). Compute is counted as loops used per token. β and λ_p are tuned in the pilot; λ_p sets the compute prior and is reported.

**Input injection.** h_{t+1} = h_t + Block(h_t + e₀), with e₀ the token's embedding; the no-injection arms drop e₀.

**Parameter matching.** Looped blocks are widened so that each looped arm has the parameter count of S8 within 5% (d ≈ 360 against d = 128). A width-matched LF (d = 128) is run in one cell as a secondary comparison. Compute is reported as steps per token (average and maximum), so S4 (4 per token) is compared with looped arms at their *average* cost (≈ 1.3 at oracle allocation, §4).

## 4. Derived before measuring (`derive/`)

**4.1 The static ceiling and the compute ratio** (`ceiling.py`). k ~ Zipf(α) truncated at 32:

| α | E[k] | P(k > 8) | static L = 3 | L = 4 | L = 5 | L = 6 | oracle query cost E[k]+1 |
|---|---|---|---|---|---|---|---|
| 1.0 | 7.9 | 0.33 | 0.37 | 0.51 | 0.67 | 0.83 | 8.9 |
| 1.5 | 4.4 | 0.15 | 0.60 | 0.74 | 0.85 | 0.94 | 5.4 |
| 2.0 | 2.5 | 0.05 | 0.77 | 0.88 | 0.95 | 0.98 | 3.5 |

(Static entries are the fraction of queries within the doubling ceiling 2^(L−2).) Over a whole sequence at α = 1.5, oracle allocation costs **1.29 steps per token**; S4 costs 4 (3.1×), S8 costs 8 (6.2×). The distinguishing quantity is nonzero and exact: S4 loses 26% of queries at α = 1.5 by construction, and any looped arm saves ≥ 3× compute at equal accuracy.

**4.2 Examples per depth** (`ceiling.py` C). With 2M training queries and k_max = 8: at α = 1.5, k = 8 has 46k examples and k = 1 has 1.04M; at α = 2.0, k = 8 has 20k. The α cells therefore vary the count of deep examples by 2–3× at fixed total, which is what P-pool needs.

**4.3 The basis frontier** (`basis_frontier.py`, Entry 51 level 1). All maps of an n-symbol alphabet to itself (n^n of them) are generated by 3 primitives {cycle, transposition, collapse}, at the price of long words; many generators give short words and huge redundancy:

| n | n^n | 3 gens: mean length / diameter / minimal words per map | all transpositions + collapses: gens / mean length / diameter / words per map |
|---|---|---|---|
| 4 | 256 | 6.0 / 9 / 1.5 | 18 / 2.5 / 3 / 8.5 |
| 5 | 3,125 | 9.9 / 16 / 2.1 | 30 / 3.4 / 4 / 51 |
| 6 | 46,656 | 14.9 / 24 / 4.4 | 45 / 4.3 / 5 / 404 |

Permutations alone (2 gens) cover 1.5% at n = 6. This is the level-1 fact behind the thesis: a small basis exists for the token-map part of any layer, and basis size trades against serial length and against identifiability (words per map). It is not used by rung 0's arms; it calibrates what "a language of primitives" costs.

## 5. Exact references and the planted construction

- **Answers are deterministic**; chance is 1/16.
- **Branch value:** remaining hops from any state, from the symbolic executor.
- **Reducible residual per step:** 1 if the register holds a key, 0 if it holds an answer. State-only halting is therefore linearly decodable from the register subspace by construction.
- **Planted block (gate G1).** With orthogonal symbol embeddings, one block implements a hop exactly: attention queries the register subspace against the key slot (dot product 1 on the unique match, 0 elsewhere; softmax scale 30), copies the value slot into a scratch subspace; the linear map writes scratch → register and clears scratch. Applied k times from the start key it reaches the answer with accuracy 1.0 on k ≤ 32. This proves the looped architecture can express the algorithm, validates the data pipeline, and gives the oracle executor's neural twin.

## 6. Endpoints

All at query positions, by k, by α cell, by symbol set.

1. **Accuracy and code length** by k, in range (k ≤ 8) and out of range (9–32).
2. **Steps used vs k** for halting arms: slope, intercept and excess over k+1; steps on pair tokens; the full compute distribution.
3. **P-pool:** for k ∈ {6, 7, 8}, accuracy against the number of training examples at that k across the three α cells, per arm; the fitted slope.
4. **Extrapolation:** accuracy at k ∈ {12, 16, 24, 32}.
5. **Transfer:** accuracy and steps on held-out keys and held-out answers, against in-vocabulary.
6. **Residual signals** (LF/LR): per step, ‖Δh‖, step-KL of the readout, readout entropy, against the exact reducible residual and remaining hops; precision–recall of each threshold rule for "done".
7. **Trainability:** gradient norms through loops, loss curves by k over training, time to 99% at k ≤ 3.
8. **Instrumentation** (L6): loss by k, attention entropy, the halting head's λ_t by step and by k, the share of halted tokens per step.

## 7. Predictions (to be fixed with thresholds at preregistration)

Verdicts by run-level paired bootstrap over 10 seeds; "P > 0.9" means P(A better than B) > 0.9.

| # | prediction | falsifier | type |
|---|---|---|---|
| H1 | **Ceiling.** S4 is at chance for k ≥ 6 in every cell; S8 and all looped arms exceed 0.95 for k ≤ 8 in the α = 1.5 cell. | S4 above 0.5 at k = 8 (then the doubling ceiling is wrong or there is a shortcut); or any looped arm below 0.95 in range (then the block is not trainable as a hop). | expressivity (sanity) |
| H2 | **Extrapolation.** LH-s ≥ 0.9 at k = 16 and ≥ 0.8 at k = 32 (trained on k ≤ 8); S8 < 0.5 at k = 16. | LH-s < 0.6 at k = 16, or S8 ≥ LH-s at k = 16 (P > 0.9). | mechanism (dynamic depth buys what static cannot) |
| H3 | **P-signal.** LH-i extrapolates worse than LH-s: at k = 16, LH-s − LH-i ≥ 0.3, and LH-i's steps stop growing beyond T_train. | LH-i within 0.1 of LH-s at k = 16 (P > 0.9). | mechanism (state-based halting pools across depth) |
| H4 | **Calibration.** LH-s steps on queries: slope ≥ 0.9 against k, excess ≤ 1.5 steps; steps on pair tokens ≤ 2. | Steps constant in k (collapse), or excess > 3, or pair tokens > 4. | mechanism (compute allocation) |
| H5 | **P-pool.** Across α cells, S8's accuracy at k = 8 rises with the example count (slope > 0 in log count, P > 0.9); LH-s's and LO's do not (slope within the null). | LO's or LH-s's slope ≥ half of S8's. | mechanism (signal pooling) |
| H6 | **Injection.** LF-ni and LH-s-ni lose ≥ 0.2 accuracy against their injected twins at k ≥ 16. | Difference < 0.05. | optimization |
| H7 | **Residual triggers.** On LF's checkpoint, a step-KL threshold reaches ≥ 0.9 of LO's accuracy at ≤ 1.5× LO's compute at k ≤ 16; the exact reducible residual is linearly decodable from LF's state with ≥ 0.99 accuracy. | Needs ≥ 3× compute, or loses > 0.1 accuracy, or decodability < 0.9. | mechanism (Entry 46) |
| H8 | **Transfer.** Every arm within 0.05 of its in-vocabulary accuracy on held-out keys and answers; LH-s's steps within 0.5 of in-vocabulary. | Any drop > 0.1 (then content leaked into the weights). | mechanism |

**Planted models give both verdicts (L17):** the planted block with oracle halting (passes H1, H2, H4, H8); a constant-depth halter at T_train (fails H2, H4); a memorizer of (start → answer) within an episode is impossible by construction, so a "copy the previous answer" model stands in (fails H1); a doubling construction of depth L (passes H1 up to its ceiling, fails H2).

## 8. Cells, grid and seeds

- **Cells:** α ∈ {1.0, 1.5, 2.0}, k_max = 8, P = 48, Qn = 8.
- **Arms:** S4, S8, LF, LO, LH-s, LH-i, LF-ni, LH-s-ni, LV (9 trained arms; LR is derived from LF).
- **Grid:** 9 arms × 3 cells × 10 seeds = **270 runs**, plus a width-matched LF and a learned-embedding control in the α = 1.5 cell (20 runs).
- **Pilot first:** learning rate per arm (3 values × 2 seeds, α = 1.5); PonderNet β and λ_p for LH; the null spread between seeds; gates G1–G3.
- **Size:** d = 128 for static (≈ 1.6M parameters for S8), d ≈ 360 for looped (matched); 4 heads; batch 64 episodes; 30k steps (≈ 15M query tokens); AdamW; cosine schedule.

## 9. Cost

Sequences are 64 tokens; models are ≤ 2M parameters. A looped arm runs ≤ 33 block applications per sequence at test and 9 in training. On a T4, one run is minutes; the grid is well under 20 GPU-hours even without stacked batching. Evaluation is 32 k-strata × 1,000 queries × 3 symbol sets per checkpoint, cheap.

## 10. Gates before the main runs

- **G1 planted.** The hand-built hop block reaches 1.0 on k ≤ 32; the symbolic executor and the planted block agree on every state; the generator's forest and length checks pass (tests first, AGENTS rule 7).
- **G2 trainability.** LO reaches ≥ 0.99 on k ≤ 8 in the pilot budget; if not, the block type or the injection is revised and recorded before any other arm runs.
- **G3 nulls.** The seed spread of identical arms sets minimum detectable effects for H2–H5.
- **G4 learning rates** tuned per arm; PonderNet β chosen so that LH's in-range accuracy matches LO's within 0.02 (otherwise a halting arm fails for the wrong reason).
- **An independent review** of the code against this document before the grid.

## 11. What this rung does not answer

- Selection among several primitives (rung 1), hence nothing yet about composition of distinct circuits or route identifiability.
- Learned primitives, the interface between them, and library formation.
- Search and learned values beyond LV's privileged halting.
- Whether the depth claim survives data with parallel shortcuts other than doubling.

## 12. Open for the preregistration

Thresholds in H1–H8 after the null pilot; β, λ_p and learning rates; whether LV runs in all cells; the second literature pass (Mixture-of-Recursions' halting variants; PonderNet's parity extrapolation setup) to confirm nothing here is already measured.
