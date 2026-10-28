# Rung 0: learned depth allocation on pointer chasing with hidden chain length (revision 2)

**Status:** design, 27 Oct 2026, revised after the independent review of revision 1 (kept at the end, with how each finding is handled). Implemented in `src/topology_design/ptr/` with tests (`tests/test_ptr.py`); the pilot and gates (§10) decide the preregistration.

**Ledger:** the thesis is history Entry 48 (amortized EM over per-token routes); the motivation Entry 49; the predictions' origins Entries 50–51. The Z_7 toys are retired as thesis tests (Entry 44).

**Scope note (AGENTS hard rule 1).** This round compares static depth with learned dynamic depth, at the researcher's direction. Every claim is labeled expressivity, optimization or mechanism.

---

## 0. What changed from revision 1

| finding (review 1) | revision 2 |
|---|---|
| The static ceiling 2^(L−2) does not apply to pairs-as-tokens; pair tokens can double in parallel | **Context tokens are read-only** in the primary regime: pairs attend only to BOS and themselves, so a static depth-L model reaches exactly k ≤ L (serial hops only). Stated as a restriction by construction. The "doubling allowed" regime is a secondary cell with the reviewer's simulated causal-doubling reach as the static reference. |
| The §5 planted block destroyed the answer after reaching it | The implemented block (`planted.py`) uses a BOS sink, a marker-gated MLP overwrite and keeps the answer part of the register: exact, idempotent after termination through 40 loops, context tokens fixed points (tested). |
| Halting semantics differed between training and test | **Context tokens run exactly one step by construction** in every looped arm, in training and at test. Only query tokens have a learned depth. Train and test agree. |
| H8 (symbol transfer) was empty either way | Dropped. No held-out symbols. "Content lives in the context" is enforced by frozen embeddings and fresh permutations per episode, not tested as a hypothesis. |
| H3 (state vs state+index) confounded | LH-i replaced by **LH-x, an index-only halter** (the constant-depth null). A linear probe for the step index from the state is an endpoint. H3 demoted to a diagnostic. |
| H5 (P-pool) would saturate; the α axis changes the curriculum | α fixed at 1.5; **the deep tail (k ≥ 6) is subsampled directly** by a factor over four decades. |
| Width confound (d 360 vs 128) | **Equal width is primary** (d = 264 for every arm; S8 then has 8× the parameters, and the same per-step cost as one loop). Parameter-matched static (d = 96) is secondary. |
| Compute accounting off; T_train = k_max + 1; cost underestimated | Token count 61; T_train = 10 = k_max + 2; cost from the pilot, with concurrent runs per GPU. |
| No chance baseline beyond 1/16 | The no-hop baseline (uniform over the terminals present in the context) is computed per evaluation. |
| Q3 barely tested | Stated: rung 0's halting *detects* completion (a 0/1 register flag), it does not predict demand. |

## 1. The question

**Thrust (Entry 49).** A static transformer charges every token its full depth; demand is dynamic and heavy-tailed; looping shares parameters but not compute; the object to learn is a depth policy.

**Known.** Per-token recursion over a shared block (Mixture-of-Recursions, depths ≤ 4, no ground truth); learned halting (ACT, PonderNet, Universal Transformer, Sparse Universal Transformer); recurrent-depth extrapolation by iterating longer with recall/injection (Bansal 2022; Geiping 2025); adaptive loops with *known* step counts (Fan 2025); pointer chasing with recurrent depth is fragile out of distribution (Kohli 2026; a public recurrent-depth run reaches 0.27–0.44 beyond its training range).

**Open, and asked here.**
- **Q1/Q2 (headline).** With hidden per-token demand and no step supervision, does a halting policy learned from next-token loss plus a step prior extrapolate its "continue" decisions beyond the training depth, scored against exact demand? **LO isolates block extrapolation from halting extrapolation:** the gap LH-s − LO is the cost of learning *when to stop*.
- **Q5 (P-pool, Entry 50).** Does one shared block's accuracy on deep examples depend on how rare they are, less than a static stack's does?
- **Q3** only as a diagnostic: which internal signals track completion (LR, the step probe).
- **Q4** as a one-cell ablation: input injection on/off.

## 2. The task

**Symbols.** 48 keys, 16 answers; frozen orthonormal embeddings in four disjoint subspaces of the stream (key slot, value slot, register, scratch), all orthogonal to the ones direction so LayerNorm is a pure rescaling on them. The BOS token, the query marker, the answer-token marker and the positions are learned. Every episode uses all 48 keys in a fresh permutation, so content is in the context.

**An episode.** A forest of chains over the 48 keys, each ending at an answer. Qn = 6 queried chains with lengths k ~ Zipf(1.5) truncated at k_max = 8 (Qn·k_max = 48 = P, so no length is ever rejected); distractor chains fill the P = 48 pairs. Tokens: BOS, then the 48 pair tokens in shuffled order (one token per pair: key-slot(key) + value-slot(value)), then for each query a query token (marker + register(start)) and its answer token (marker + register(answer)). 61 tokens.

**Loss.** Cross-entropy over the 16 answers at the query positions, read out from the register subspace of the final state through the frozen answer embeddings. (Pair tokens and starts are uniform random, so no other position is informative.)

**Hidden k.** Nothing but following the chain reveals its length.

**Read-only context (primary regime).** Pair tokens attend only to BOS and themselves; query tokens attend to everything before them. A static depth-L model can then make at most L hops: each layer lets the query read one pair, and the pair the query needs next is determined only by the register it already holds. This is a restriction by construction, chosen so that the static reach is exact. The **doubling-allowed regime** (pairs attend to earlier pairs and, in looped arms, may loop) is a secondary cell; there the static reference is the simulated greedy causal-doubling reach (`review/static_reach_causal.py`): S4 reaches k = 5/6/7/8 with probability 0.96/0.85/0.68/0.51; S8 reaches k = 16 with 0.98 and k = 32 with 0.68.

**Evaluation.** k ∈ {1,…,8, 10, 12, 16, 20, 24, 32} stratified: one forced query per episode (at a random query position) plus two Zipf fillers, 250 episodes per k, T_eval = 40 loops. The no-hop baseline (uniform over the terminals present) is reported per k; it is ≈ 0.08 here.

## 3. Arms

Block = pre-norm causal attention + MLP + a linear map on the stream (identity at init). All arms share it. d = 264, 2 heads, MLP 4×.

| arm | depth | halting (queries) | role |
|---|---|---|---|
| **S2** | 2 unique blocks | — | static, compute-matched to allocation (≈ 2 steps/token); reach k ≤ 2 |
| **S8** | 8 unique blocks | — | static, equal width (8× parameters, same per-step cost as a loop); reach k ≤ 8 |
| **S8-pm** | 8 unique blocks, d = 96 | — | static, parameter-matched (secondary) |
| **LF** | 1 shared block, T_train = 10 | none; evaluated at T = 10 and T = 40 | reuse without allocation |
| **LO** | shared | oracle: exactly k loops (uses ground truth) | privileged upper bound; block extrapolation alone |
| **LH-s** | shared | PonderNet, head reads the state | **the thesis arm** |
| **LH-x** | shared | PonderNet, head reads the step index only | constant-depth null (what a policy that cannot see the state does) |
| **LR** | LF's checkpoint | residual-triggered at test: halt at the first step with readout step-KL < 10⁻² | **primary arm** (promoted after pilot 1, Entry 56): the zero-parameter stopping rule the learned policy must beat, in accuracy and in compute |
| **LF-ni, LH-s-ni** | as LF, LH-s | | injection off (one cell) |

Context tokens (BOS, pairs, answers) run **exactly one step** in every looped arm, in training and at test, so their states are fixed and the semantics are coherent. Only queries loop.

**Learned halting.** λ_t = σ(w·LN(h_t) + b) (LH-s) or λ_t = σ(b_t) (LH-x; the last logit repeats beyond T_train); p_t = λ_t Π_{s<t}(1 − λ_s); loss = Σ_t p_t·CE(readout(h_t), y) + β·KL(p ‖ Geometric(λ_p)) over query tokens. At test: halt at the first t with cumulative p ≥ 0.5, then freeze. β, λ_p from the pilot (G4). Compute is counted as loops per token.

**Input injection.** h_{t+1} = h_t + Block(h_t + e₀); the no-injection arms drop e₀.

## 4. Derived before measuring (`derive/`)

**Compute** (`ceiling.py` B). Over a 61-token training sequence, oracle allocation costs **1.12 steps per token** (1.33 at evaluation with k up to 32); S2 costs 2 (1.8×), S8 costs 8 (7.1×).

**Reach.** Read-only regime: exactly k ≤ L for a static depth-L model; so S2 fails k ≥ 3 and S8 fails k ≥ 9 by construction (the latter is the point: a static model cannot extrapolate in k at all here). Doubling regime: the table in §2.

**Examples per depth** (`ceiling.py` C). Budget 20k steps × 64 episodes × 6 queries = 7.68M queries. At factor 1, k = 8 has 176k examples; factors 10⁻¹…10⁻⁴ give 19k / 1.9k / 192 / 19 at k = 8.

**Basis frontier** (`basis_frontier.py`, Entry 51 level 1): all n^n token maps are generated by 3 primitives with mean word length ≈ 2.5n (diameter 24 at n = 6), or by all transpositions and collapses with mean length ≈ n − 2 and ~400 minimal words per map at n = 6. Not used by the arms; it calibrates what a language of primitives costs.

## 5. Exact references and the planted block

- Answers are deterministic; the no-hop baseline is the chance reference.
- Branch value: remaining hops from any state (`data.remaining_hops`).
- Reducible residual per step: 1 while the register holds a key, 0 once it holds an answer. State-based halting is therefore linearly decodable from the register subspace.
- **Planted block (G1, implemented and tested).** Attention head 0 matches the register against the key slots (score β = 30 on the unique match) with a BOS sink (score κ = 15, value 0) for tokens with no match, and copies the matched value slot into the query's scratch subspace. The MLP is a marker-gated linear map on query tokens: clear the key part of the register, move scratch into the register, clear scratch; the answer part of the register is kept, so a finished query is a fixed point. One loop = one hop, exactly: accuracy 1.0 at k ≤ 32 under oracle halting and under 40 fixed loops; context tokens are fixed points; T = 4 fails k = 16 as it should.

## 6. Cells and grid

| cell | regime | arms | seeds |
|---|---|---|---|
| **main** | read-only, α = 1.5, factor 1 | S2, S8, LF, LO, LH-s, LH-x (+ LR derived) | 29 for S8, LO, LH-s; 10 for the rest |
| **P-pool** | read-only, factors 10⁻¹, 10⁻², 10⁻³, 10⁻⁴ | S8, LO, LH-s | 10 |
| **injection** | read-only, factor 1 | LF-ni, LH-s-ni | 10 |
| **doubling** | pairs attend to earlier pairs and loop; hard-freeze halting semantics | S8, LF, LO, LH-s | 10 |
| **param-matched** | read-only | S8-pm | 10 |

Runs: main 29·3 + 10·3 = 117; P-pool 120; injection 20; doubling 40; param-matched 10: **307**, plus the pilot (60 runs). **Run order:** pilot → main → P-pool; the other three cells only with budget left over. Cut order if short: doubling cell; P-pool to factors 10⁻¹, 10⁻², 10⁻³; LH-x. Never fewer than 10 seeds; the headline's 29 are kept.

**Batching.** Runs of one arm are stacked into one model with a leading run dimension (`stacked.py`, as in the worlds round): all 29 seeds of LH-s train in one process, one kernel per layer, per-run learning rates and gradient clipping, fp16 autocast. Episodes are generated on the GPU (`gen_torch.py`), paired by seed across arms. Stacked and separate training agree to float precision (`tests/test_ptr.py`). One shard per T4; Kaggle's two T4s take two shards.

```
.venv/bin/python experiments/fleet/kaggle/launch.py --user riley9 --ptr-grid experiments/ptr/grids/pilot.json --no-wait
.venv/bin/python experiments/fleet/kaggle/launch.py --user riley9 --ptr-grid experiments/ptr/grids/pilot.json --fetch
PYTHONPATH=src python experiments/ptr/analyze.py results/kaggle_ptr-pilot
```

**Training.** 20k steps, batch 64, AdamW (0.9, 0.95), weight decay 0.01, cosine schedule with 500 warm-up steps, gradient clip 1.0; learning rate per arm from the pilot. Evaluation every 2,500 steps.

## 7. Endpoints

All at query positions, by k.
1. Accuracy and code length by k, against the no-hop baseline; in range (k ≤ 8) and out of range (9–32).
2. Steps used vs k for LO, LH-s, LH-x: slope, excess over k, the full distribution.
3. **P-pool:** for k ∈ {6, 7, 8}, accuracy against log example count across the five factors, per arm; the fitted slope and the count at which each arm leaves ceiling.
4. **Extrapolation:** accuracy at k ∈ {10, 12, 16, 24, 32} for LF (T = 40), LO, LH-s, LH-x.
5. **Residual signals** (traces on every looped arm): per step, ‖Δh‖, step-KL of the readout, readout entropy, correctness, against remaining hops; precision–recall of threshold rules for "done"; accuracy and compute of LR against LO.
6. **Step-index probe:** linear decodability of t from LN(h_t) on chasing states (how far "state-only" can count).
7. **Random-halting control** (free): from LO/LF traces, the accuracy of halting at a random step drawn from LH-s's empirical step distribution, independent of the token: matched average compute, no information.
8. Trainability: gradient norms, loss curves by k, time to 99% at k ≤ 3; comparisons at matched training loss from the curves.
9. Instrumentation (L6): loss by k, halting λ_t by step and k, attention entropy.

## 8. Predictions (thresholds fixed at preregistration; run-level paired bootstrap)

| # | prediction | falsifier | type |
|---|---|---|---|
| H1 | **Sanity.** S2 at the no-hop baseline for k ≥ 3; S8 ≥ 0.95 for k ≤ 8 and at baseline for k ≥ 9; LO ≥ 0.99 for k ≤ 8. | Any violation: the static reach is wrong (expressivity) or the block is not trainable as a hop (G2). | expressivity / optimization |
| **H2** | **Headline: halting extrapolates.** LH-s ≥ 0.9 at k = 16 and ≥ 0.8 at k = 32, trained on k ≤ 8; LH-s within 0.1 of LO at every k ≤ 32 (the halting-extrapolation gap). **H2b (after pilot 1):** LH-s matches LR's accuracy within 0.02 at every k and uses fewer steps (≤ k + 0.5 against LR's ≈ k + 1.5). | LH-s < 0.6 at k = 16 while LO ≥ 0.9 (the block extrapolates, the policy does not); or LO itself < 0.6 at k = 16. **H2b:** LH-s no more accurate than LR and no cheaper: then the free stability exit suffices on this task. | mechanism (optimization of the policy) |
| H4 | **Calibration.** LH-s steps on queries: slope ≥ 0.9 against k, excess ≤ 1.5 for k ≤ 32. LH-x's steps are constant. | Steps constant in k (collapse to LH-x), or excess > 3. | mechanism (compute allocation) |
| **H5** | **P-pool.** S8's accuracy at k = 8 leaves ceiling by factor 10⁻² (≈ 2k examples) and falls with log count (slope > 0, P > 0.9); LO's and LH-s's stay at ceiling down to 10⁻³ (≈ 200). | LO/LH-s leave ceiling at the same count as S8, or S8 stays at ceiling at 10⁻⁴. | mechanism (signal pooling) |
| H7 | **Residual triggers.** On LF, a step-KL threshold reaches ≥ 0.9 of LO's accuracy at ≤ 1.5× LO's compute at k ≤ 16. | Needs ≥ 3× compute or loses > 0.1. | mechanism (diagnostic) |

Dropped from revision 1: H3 (confounded; replaced by the probe and LH-x), H6 (established; kept as an ablation), H8 (empty).

**Planted models give both verdicts (L17):** the planted block with oracle halting (passes H1, H2, H4); the planted block with a constant halter at T_train (fails H2, H4); a no-hop guesser (fails H1); LF at T = 10 (fails H2).

## 9. Fairness (AGENTS rule 4)

- Learning rate per arm (pilot: 3 values × 2 seeds); β and λ_p for LH chosen so that LH-s's in-range accuracy matches LO's within 0.02 (G4), otherwise a halting arm fails for the wrong reason.
- Equal width primary (every arm d = 264; the same per-step cost); parameter-matched secondary (S8-pm).
- Compute reported as steps per token, average and maximum; S2 is the compute-matched static arm; the random-halting control matches LH-s's average compute.
- 29 paired seeds for the headline trio; 10 elsewhere; a null pilot sets the minimum detectable effect.
- Matched-training-loss comparisons from the curves.
- Not done: an initialization-scale sweep (no closed-form richness parameter here); stated.

## 10. Gates before the main runs

- **G1 planted** (done, `tests/test_ptr.py`): exact hop, idempotent after termination, context fixed points, generator forest and length checks.
- **G2 trainability.** LO reaches ≥ 0.99 at k ≤ 8 within the pilot budget. If not, the block or the injection is revised and recorded before any other arm runs.
- **G3 nulls.** Seed spread of identical arms.
- **G4 learning rates and PonderNet constants.**
- **G5 cost.** The pilot's wall time sets the step budget and the number of concurrent runs per GPU; the grid must fit about two Kaggle weeks.
- **An independent review of the code** against this document before the grid.

## 11. What this rung does not answer

- Selection among several primitives (rung 1); nothing about composing distinct circuits.
- Learned primitives, their interface, library formation; search and learned values.
- Predicting demand (rung 0's halting only detects completion).
- Whether the result survives data with parallel shortcuts (the doubling cell is a first look only).

---

## Independent review of revision 1 (27 Oct 2026)

The reviewer's scripts are in `review/`.

**Verdict.** The toy is a legitimate, cheap place to ask the one thing not in the literature: whether a halting policy trained from next-token loss plus a step prior, with no step supervision and hidden per-token demand, extrapolates its "continue" decisions beyond T_train, measured against exact demand (LH-s vs LO), plus the P-pool claim if redesigned. Much of the 290-run grid was foregone, confounded or known. Recommended a lean rung 0 plus a redesigned P-pool.

**Blocking findings:**
1. The static ceiling 2^(L−2) was wrong for pairs-as-tokens (Sanford's +2 includes a previous-token layer this tokenization does not need); simulated causal greedy doubling: S4 reaches k = 5/6/7/8 with 0.96/0.85/0.68/0.51; S8 reaches k = 16 with 0.98. "Chance = 1/16" was also wrong: a no-hop guesser with elimination scores 0.17–0.20.
2. The §5 planted block, as described (linear overwrite), destroyed the answer one step after reaching it (answer mass 1.00 → 0.02); a gated overwrite holds it. (The implementation already used the gated construction; the text was wrong.)
3. Halting semantics were incoherent between training (all tokens run all steps) and test (pairs freeze early).
4. H8 was empty: held-out frozen symbol directions get no gradient, so transfer fails by construction; symbols in the training span transfer trivially.
5. H3 was confounded: the step index is decodable from the state, so "state-only" can count and LH-i can ignore its index.
6. H5 would saturate at the budget (k = 8 had 352k examples); the α axis also changed the curriculum.

**Non-blocking:** the compute accounting used the wrong token count; evaluation episodes were unspecified; T_train = k_max + 1 forced the halt at k = 8; width confound (d 360 vs 128); AGENTS gaps (29 seeds for the headline, matched-loss comparison, a random-halting control, a compute-matched static arm); cost ≈ 60–170 GPU-hours, not < 20; rung 0's halting detects completion rather than predicting demand; H6 is established and should be an ablation.

**Prior work found:** PonderNet (per-sequence halting; parity extrapolation), Universal Transformer (per-position ACT, halted state attended), NDR (Csordás 2022), Sparse Universal Transformer (expected halted state as K/V), Bansal 2022 (recall + progressive loss; prefix sums and mazes extrapolate), Geiping 2025 (recurrent-depth LM with a zero-shot KL exit), Fan 2025, Mixture-of-Recursions, Sanford 2024, Kohli 2026 (recurrent-depth pointer chasing is fragile), a public recurrent-depth pointer-chasing run (0.27–0.44 out of range), Ouro (learned per-token exit at LM scale).

**How revision 2 handles each finding:** the table in §0.
