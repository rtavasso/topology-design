# Rung 0: learned depth allocation on pointer chasing with hidden chain length (revision 3)

**Status:** preregistered design for the confirmatory grid, 27 Oct 2026, revised after pilots 1–2 (history Entries 55–57, `PILOT1.md`, `PILOT2.md`; decision D88) and after two independent reviews of this revision (`review/review3_design.md`, `review/review3_code.md`; how each finding is handled is in §12). Revision 2 and the review of revision 1 are kept at the end for provenance. Implemented in `src/topology_design/ptr/` with tests (`tests/test_ptr.py`); the gate grid is `grids/pilot3.json`, the confirmatory grid `grids/rev3.json`.

**Ledger:** the thesis is history Entry 48 (amortized EM over per-token routes); the motivation Entry 49; the predictions' origins Entries 50–51 and 56–57. The Z_7 toys are retired as thesis tests (Entry 44).

**Scope note (AGENTS hard rule 1).** This round compares static depth with learned dynamic depth, at the researcher's direction. Every claim is labeled expressivity, optimization or mechanism.

---

## 0a. What changed from revision 1 (review 1, 27 Oct)

Condensed; the full table is in the revision-2 text at the end. Context tokens made read-only so the static reach is exactly k ≤ L; the planted block made gated and idempotent; halting semantics made coherent (context tokens run one step, train and test); H8 dropped; LH-i replaced by the index-only null LH-x; the P-pool manipulation changed to subsampling the deep tail; equal width made primary; the no-hop baseline added.

## 0b. What changed from revision 2 (pilots 1–2, Entries 55–57)

| pilot finding | revision 3 |
|---|---|
| LH-s trains only at lr 3e-4 with β = 0.003 (at 1e-3, or β = 0.01/λ_p = 0.2, it collapses to immediate halting); revision 2's main grid had it at lr 1e-3 | **LH constants fixed** for every learned-halting arm: lr 3e-4, β = 0.003, λ_p = 0.1, halting-logit init −2 (L23). |
| H2 falsified for the state-reading head, with a mechanism: λ stays 0 through the training horizon and ramps after it while the register still holds a key; the step index is decodable from the state (0.35); the change-based exit on the same block type reaches 0.97 at k = 32 | **The question becomes what the policy reads.** Two new arms: **LH-Δ**, a learned head on change features only (log readout step-KL, log relative ‖Δh‖), and **LH-s24**, the state head trained with T = 24 (k_max stays 8) to test the horizon/drift explanation directly. |
| The zero-parameter stability exit (LR) reaches 0.97–1.00 at every k ≤ 32 at ≈ k + 1.5 steps (exploratory, Entry 56) | **LR is a primary arm**, computed on the full evaluation set of every LF run (`kl_exit`), with the preregistered threshold 10⁻². The bar a learned policy must beat. |
| S8 learns hops one layer at a time (k = 3 at ~10k steps at lr 1e-4; not k = 4 by 15k); at 20k steps and lr 3e-4 it is not a working baseline, and the P-pool cell at k = 6–8 would measure "unfinished", not "data-starved" | **Static arms at lr 1e-4 with a 25k-step budget, measured as steps to criterion by k** (time to learn each depth), not accuracy at a fixed budget. **Pooling as time:** steps to criterion at k = 3 against the k ≥ 3 share (k_deep = 3, factors 1, 10⁻¹, 10⁻²), arms LO, LH-s, S8. |
| Effects are 0.97 vs 0.16, not close calls | **10 seeds** for the primary arms, 5 for nulls and secondary arms; the 29-seed headline is dropped; the grid is 83 runs ≈ 13 T4-hours, one Kaggle session on two T4s. |
| Injection: without it the block never learns hop 2 (LO-ni at three learning rates) | Injection on everywhere; the no-injection cell is **not rerun** (a pilot-1 result, 6 runs). |
| The doubling cell and S8-pm were never reached | Deferred, with the reasons stated (§11). |
| **Review 3** (of this revision): the LH-Δ head as first parameterized (raw log features) could not reach a working policy within what AdamW moves in the step budget, and would fail in a way that mimics the horizon signature; P1's falsifier compared LH-Δ with LR on a different block; P7's LR line contradicted pilot 1; dead zones in P1/P2; censoring and window statistics in P6 | Change features **standardized** (§3); a head learning-rate multiplier and a **gate (pilot 3)** for the two untrained arms (§10); LR computed on **every arm's own block** (§7); thresholds restated (§8); windows by example count; three cheap cells added (LH-Δ in the pooling cell, LH-s with k_max = 4, S8 at a lower learning rate). Full handling in §12. |

## 1. The question

**Thrust (Entry 49).** A static transformer charges every token its full depth; demand is dynamic and heavy-tailed; looping shares parameters but not compute; the object to learn is a depth policy.

**Known.** Per-token recursion over a shared block (Mixture-of-Recursions; its routers are never evaluated beyond their training depth); learned halting (ACT, PonderNet — which reported extrapolation on parity with a state-reading head, so "state heads fail beyond the horizon" is not general — Universal Transformer, SUT); recurrent-depth extrapolation by iterating longer with recall/injection (Bansal 2022, Schwarzschild 2021: injection necessity and over-running are their results, and P7's LF line reproduces them); Geiping 2025's zero-shot KL exit **is** LR, so P7's LR line and Q2 are a reproduction at exact demand; CALM (Schuster et al. 2022) exits on softmax response and consecutive-hidden-state saturation with a calibrated threshold — LH-Δ's two features, in distribution and at LM scale; adaptive loops with known step counts (Fan 2025); recurrent-depth pointer chasing is fragile out of distribution (Kohli 2026). What none of these test is the extrapolation of a *learned* change-reading head beyond its training depth (Q1). **From the pilots (Entries 55–57):** on this task the shared block extrapolates 4× under oracle halting, a fixed loop over-runs short chains, a state-reading learned halter is calibrated in range and fails past ≈ 1.5× its training horizon, and a change-based rule with no parameters extrapolates.

**Open, and asked here.**
- **Q1 (headline, revised).** Is the failure of learned halting beyond the training horizon a property of *what the head reads*? A head on change features (LH-Δ) should extrapolate as the free rule does; a state head trained with a longer horizon (LH-s24) should extrapolate up to its own horizon. If both hold, the mechanism is confirmed and a depth policy must be built from change (or trained past the depths it must serve).
- **Q2.** Does any learned policy beat the stability exit (LR) in accuracy *and* compute? If not, on crisp-completion tasks the free rule suffices, and the case for a learned depth policy moves to tasks whose completion is not a fixed point.
- **Q5 (pooling, Entry 50, revised).** Does a shared block learn a given depth in a time that is insensitive to how rare examples of that depth are, while a static stack's time grows with rarity? And does the learned *policy* (not the block) depend on those examples?
- **Q3** as a diagnostic: which signals track completion (LR, the step probe, λ by loop).

## 2. The task

**Symbols.** 48 keys, 16 answers; frozen orthonormal embeddings in four disjoint subspaces of the stream (key slot, value slot, register, scratch), all orthogonal to the ones direction so LayerNorm is a pure rescaling on them. The BOS token, the query marker, the answer-token marker and the positions are learned. Every episode uses all 48 keys in a fresh permutation, so content is in the context.

**An episode.** A forest of chains over the 48 keys, each ending at an answer. Qn = 6 queried chains with lengths k ~ Zipf(1.5) truncated at k_max = 8 (Qn·k_max = 48 = P, so no length is ever rejected); distractor chains fill the P = 48 pairs. Tokens: BOS, then the 48 pair tokens in shuffled order (one token per pair: key-slot(key) + value-slot(value)), then for each query a query token (marker + register(start)) and its answer token (marker + register(answer)). 61 tokens.

**Loss.** Cross-entropy over the 16 answers at the query positions, read out from the register subspace of the final state through the frozen answer embeddings.

**Hidden k.** Nothing but following the chain reveals its length.

**Read-only context.** Pair tokens attend only to BOS and themselves; query tokens attend to everything before them. A static depth-L model can then make at most L hops. This is a restriction by construction, chosen so that the static reach is exact (the doubling-allowed regime is deferred, §11).

**Tail manipulation (pooling cells).** The pmf of k ≥ k_deep = 3 is multiplied by the tail factor and renormalized. At factor 1 the share of k ≥ 3 is 0.297 and of k = 3 is 0.100; at 10⁻¹, 0.041 and 0.014; at 10⁻², 0.0042 and 0.0014 (≈ 54 k = 3 examples per 100-step window of 38,400 queries). k_deep = 3 is set in every revision-3 run (inert at factor 1).

**Evaluation.** k ∈ {1,…,8, 10, 12, 16, 20, 24, 32} stratified: one forced query per episode plus two Zipf fillers, 250 episodes per k, T_eval = 40 loops. The no-hop baseline (uniform over the terminals present) is ≈ 0.08.

## 3. Arms

Block = pre-norm causal attention + MLP + a linear map on the stream (identity at init). All arms share it. d = 264, 2 heads, MLP 4×. One static layer costs the same as one loop.

| arm | depth | halting (queries) | role | seeds |
|---|---|---|---|---|
| **S2** | 2 unique blocks | — | static, compute-matched to allocation; reach k ≤ 2 | 5 |
| **S8** | 8 unique blocks | — | static, equal width (8× parameters); reach k ≤ 8; lr 1e-4, 25k steps | 5 (+3 at factor 10⁻¹; +2 at lr 3e-5 to bracket the pilot's best rate from below) |
| **LF** | 1 shared block, T_train = 10 | none; evaluated at T = 10 and T = 40 | reuse without allocation | 10 |
| **LR** | LF's checkpoint, and every looped arm's own block run as a fixed loop | halt at the first loop t ≥ 2 with KL(r_t ‖ r_{t−1}) < 10⁻² (the frontier 10⁻¹ … 10⁻⁴ reported) | **primary:** the one-parameter stopping rule (threshold fixed on two pilot-1 runs, Entry 56) the learned policies must beat; Geiping 2025's exit at exact demand | from LF and from each arm |
| **LO** | shared | oracle: exactly k loops (uses ground truth) | privileged upper bound; block extrapolation alone | 10 (+5, +5 at factors 10⁻¹, 10⁻²) |
| **LH-s** | shared, T = 10 | PonderNet, head reads LN(h_t) | the state-reading policy (pilot 2: fails past ≈ 15 loops) | 10 (+5, +5) |
| **LH-Δ** | shared, T = 10 | PonderNet, head reads **standardized change features only** | **the thesis arm, revised:** the learned counterpart of LR | 10 (+5 at factor 10⁻²) |
| **LH-s24** | shared, **T = 24** | PonderNet, head reads LN(h_t) | the horizon test: same policy, trained on depths up to 24 for chains ≤ 8 | 5 |
| **LH-s-k4** | shared, T = 10, **k_max = 4** | PonderNet, head reads LN(h_t) | does the state head's failure track the loop horizon T (≈ 15 loops, pilot 2) or the chain horizon 1.5·k_max (≈ 6)? | 3 |
| **LH-x** | shared, T = 10 | PonderNet, head reads the loop index only (the last trained logit, loop T − 1, repeats beyond T) | constant-depth null | 5 |

Context tokens (BOS, pairs, answers) run **exactly one step** in every looped arm, in training and at test. Only queries loop.

**Learned halting.** λ_t = σ(w·LN(h_t) + b) (LH-s, LH-s24, LH-s-k4); λ_t = σ(b_t) (LH-x; b_{T−1} repeats beyond T_train, since b_T is overridden by λ_T = 1 and never trains); **LH-Δ:** λ_t = σ(w₁ z₁ + w₂ z₂ + b) with the standardized change features z₁ = clamp((log(KL(r_t ‖ r_{t−1}) + 10⁻⁶) + 3.3)/0.75, ±6) and z₂ = clamp((log(‖h_t − h_{t−1}‖/‖h_{t−1}‖ + 10⁻⁶) + 3.0)/1.0, ±6), h₀ = e₀, r_t the readout after loop t, computed in fp32; the head carries no information about the absolute state. The centres and scales are fixed from the pilot-1 fixed-loop traces (log step-KL median −1.07 while chasing, −5.6 once settled, +1.1 on the loop that lands the answer; §4), so that a chasing step reads z₁ ≈ +3 and a settled step ≈ −3: the analogue of the LayerNorm the state head reads through. All heads: b init −2 (λ ≈ 0.12 at every loop), w init N(0, 0.02²); the head's learning rate is lr × `halt_lr_mult`, chosen in pilot 3 (§10). p_t = λ_t Π_{s<t}(1 − λ_s), λ_T = 1; loss = Σ_t p_t·CE(r_t, y) + β·KL(p ‖ Geometric(λ_p)) over query tokens; β = 0.003, λ_p = 0.1, lr 3e-4. At test: halt at the first t with cumulative p ≥ 0.5, then freeze. Training accuracy is logged for the *halted* readout (the policy's prediction, on the all-T training trajectory, where earlier queries do not freeze) and for the T-loop readout (the block's). The head's parameters and λ by loop are recorded.

**Input injection.** Each loop applies the block to h_t + e₀ (the block's own residual stream carries the sum); on everywhere.

## 4. Derived before measuring (`derive/`, pilots)

**Compute** (`ceiling.py` B). Over a 61-token training sequence, oracle allocation costs **1.12 steps per token**; S2 costs 2, S8 costs 8. LR at threshold 10⁻² cost k + 1.6/1.1/1.3/1.9/2.1/2.3 at k = 1/4/8/16/24/32 on the pilot-1 traces (Entry 56): ≈ k + 1.3 in range, ≈ k + 2 out of range (one loop to see that the readout has settled, and a trained block's readout keeps moving a little after the answer); a learned policy that halts at the first post-answer loop costs k + 1.

**What the optimizer can reach** (review 3, B.i). With AdamW each parameter moves at most ≈ lr per step, so over 8k steps (500 warm-up, cosine) Σ lr ≈ 1.13 at lr 3e-4: no scalar of the halting head can move more than ≈ 1.1 units from its init. A change head must give a chasing step λ ≲ 0.01 (else the cumulative halting mass reaches 0.5 by loop ≈ 15–35 on its own, a premature halt that would mimic the horizon failure of LH-s) and a settled step λ ≳ 0.5. With raw log features (chasing ≈ −1, settled ≈ −6) that needs b ≈ −6: unreachable. With the standardized features (chasing ≈ +3, settled ≈ −3) it needs w₁ ≈ −1, b ≈ −2: one unit from the init. The planted LH-Δ head (§5) uses exactly such values. Pilot 3 also tries a 10× head learning rate, and the λ-by-loop trace at k = 32 separates a flat premature-halt floor from the post-horizon ramp.

**What else LH-s24 changes** (review 3, B.ii). Besides the horizon, T = 24 changes the truncated geometric prior's retained mass (0.65 at T = 10 → 0.92), the mass that the forced λ_T = 1 anchor carries at init (0.32 → 0.05, so the deep-step learning signal that anchored pilot 2's narrow basin is weaker), and the block's training: loops 9–24 are only ever post-answer states for k ≤ 8, so the block is trained to idle for 16 loops (LF's is not), and the head sees drift at loops 9–24 only together with register = answer. So ¬P2 is compatible with drift being the mechanism (an optimization failure or a confounded exposure), and P2 can pass through the block (drift suppressed) rather than the head. The endpoints that separate these: the in-range gate (G4b), LH-s24's step-probe decodability against LH-s's 0.35 (lower ⇒ the block changed), and λ by loop at k = 32 (ramp after 24 ⇒ the head reads depth). LH-s-k4 asks the complementary question at T = 10: whether the state head's failure follows T or k_max.

**Reach.** Read-only regime: exactly k ≤ L for a static depth-L model. So S8 cannot extrapolate in k at all, by construction; the comparison with S8 is about *learning*, not reach.

**Why time to criterion, not accuracy at a budget** (pilot 2). S8 learns k = 1/2/3 at 0.5k/3k/10k steps and the shared block all eight by 4k. At any fixed budget the shared block saturates and the stack is mid-curriculum, so accuracy differences confound "cannot" with "not yet". The endpoint t95(k) is the end of the first training window with accuracy ≥ 0.95 at k, confirmed by the next window, where a window is consecutive 100-step logging windows aggregated until they hold ≥ 250 examples of k (so the criterion has the same statistical weight at every tail factor: at factor 10⁻² a 100-step window holds ≈ 54 examples of k = 3, where "≥ 0.95" would be ≤ 2 errors and a 0.97 model would pass only 60% of the time — review 3, B.iv). Runs that never reach it are censored at the budget; medians are censoring-aware (a censored median prints as "> budget", a ratio as "≥"), and in the bootstrap a censored run enters at the budget (one-sided, conservative). Note the schedules differ across arms (cosine over 6k–25k steps), so cross-arm time comparisons are optimization observations.

**What LR measures** (`tests/test_ptr.py::test_kl_exit_rule_on_synthetic_readouts_and_its_planted_caveat`). The stability exit relies on the readout *moving while the chain is being followed*. For the exact planted block the readout is uniform while the register holds a key (the register is orthogonal to the answer directions), so LR sees "no change" at loop 2 and halts there for every k ≥ 3. LR therefore measures a property of *trained* blocks (their readouts change during chasing, as the pilot-1 traces show), not of the task. The relative-‖Δh‖ feature of LH-Δ does not have this blind spot: the register changes at every hop.

**Examples per depth** (pooling, §2). At 8k steps × 384 queries: k = 3 has 307k examples at factor 1, 42k at 10⁻¹, 4.3k at 10⁻². Under oracle halting the block needs no k ≥ 3 example to learn the hop (it is the same map at every loop), so LO's time to k = 3 should not move; S8's third layer gets gradient only from k ≥ 3 queries, so its time should scale roughly with the inverse share (which puts its factor-10⁻¹ time near or beyond the 25k budget: the S8 line is one-sided); LH-s's *policy* needs k ≥ 3 examples to learn to continue past loop 2, so its halted criterion should move. Its T-loop criterion is **not** a clean block measure at 10⁻²: when the policy halts by loop 3 on almost every query, p_t ≈ 0 at the deeper loops and the block's gradient there is policy-gated (L23), so a moved T-loop time can be the policy starving the block, not data hunger. The clean block test is LO's line; LH-Δ at 10⁻² asks whether a change-reading policy's data dependence differs from the state-reading one's. k = 2 (whose share barely changes with the factor) is the within-arm control in the same table.

## 5. Exact references and the planted models

- Answers are deterministic; the no-hop baseline is the chance reference.
- Branch value: remaining hops from any state (`data.remaining_hops`). Reducible residual per step: 1 while the register holds a key, 0 once it holds an answer.
- **Planted block (G1, tested).** One loop = one hop, exactly; answers absorbing; context tokens fixed points; accuracy 1.0 at k ≤ 32 under oracle halting and under 40 fixed loops; T = 4 fails k = 16.
- **Planted LH-Δ (tested).** The planted block with the change head set to w = (0, −1), b = −2 (one unit of weight from the init, §4) halts at exactly k + 1 loops with accuracy 1.0 at k ∈ {1, 3, 8, 32}: the change features are sufficient for exact allocation on this task, including far beyond any training horizon, and a sufficient head is reachable. The L17 pair: the planted block with a constant halter at T_train fails extrapolation; LH-Δ-planted passes it. This shows the *measure* detects the mechanism, not that the trained arm will.
- **Planted LR caveat:** §4.

## 6. Cells and grid (`grids/pilot3.json`, 6 runs; `grids/rev3.json`, 93 runs)

| grid / cell | arms × seeds | steps | purpose |
|---|---|---|---|
| **pilot 3** (gate G4b) | LH-Δ × head-lr multiplier {1, 10} × 2 seeds; LH-s24 × 2 seeds | 8k | do the two untrained arms train in range at the shared constants? pick LH-Δ's multiplier by the rule in §10 |
| **main** (factor 1) | LO 10, LF 10 (→ LR), LH-s 10, LH-Δ 10, LH-s24 5, LH-x 5, LH-s-k4 3, S8 5 (+2 at lr 3e-5), S2 5 | looped 6k (LO, LF, LH-x) / 8k (LH-s, LH-Δ, LH-s24, LH-s-k4); S8 25k; S2 8k | Q1, Q2, sanity, time to criterion by k |
| **pooling** (k_deep = 3) | factor 10⁻¹: LO 5, LH-s 5, S8 3; factor 10⁻²: LO 5, LH-s 5, LH-Δ 5 | as above | Q5: steps to criterion at k = 3 vs share |

Seeds are paired across arms (seed s draws the same episode stream in every arm). Evaluation every 2k steps (looped) / 5k (static) and at the end; at the end also traces, the step probe, and for every looped arm the block run as a fixed loop of 40 with the stability exit applied (LR on its own block) and its fixed-loop accuracy.

**Cost** (pilot timings: 0.065 s per run-step for a looped arm at T = 10, scaling with T; 0.052 for S8): pilot 3 ≈ 1.3 T4-hours (two shards of 0.7 / 0.6 h). rev3: LO 2.2 h, LF 1.1, LH-s 2.9, LH-Δ 2.2, LH-s24 1.7, LH-x 0.5, LH-s-k4 0.4, S8 3.6, S2 0.1 = **14.8 T4-hours**, assigned to two shards by estimated cost (7.4 / 7.3 h, plus a few minutes of final evaluation per stack): one Kaggle session on 2×T4, within the 12-hour limit with margin for a 50% overrun. Runs are resumable; a second session finishes whatever a first one did not.

**Batching.** Runs of one arm that differ only in seed, learning rate and tail factor are stacked into one model with a leading run dimension (`stacked.py`), up to 10 per stack, one kernel per layer, per-run learning rates and gradient clipping, fp16 autocast with per-run loss scaling; stacked and separate training agree to float precision (`tests/test_ptr.py`). Memory (review 3, code 4): a 10-stack at T = 10 holds ≈ 7 GB of saved activations, the 5-stack at T = 24 ≈ 8 GB, the S8 10-stack ≈ 5 GB; on an out-of-memory error a stack is split in two and retried. Shards are balanced by estimated cost (`assign_shards`).

```
# both stages in one unattended Kaggle session: the gate grid, then gate.py applies the G4b rule and writes the gated
# grid (gated_rev3.json, returned with the logs together with gate.json), then the gated grid runs
caffeinate -i .venv/bin/python experiments/fleet/kaggle/launch.py --user riley9 \
    --ptr-gate experiments/ptr/grids/pilot3.json --ptr-grid experiments/ptr/grids/rev3.json
# (with --no-wait instead: later `--fetch` with the same two grid arguments downloads into results/kaggle_ptr-pilot3-rev3)
PYTHONPATH=src python experiments/ptr/analyze.py results/kaggle_ptr-pilot3-rev3 --md experiments/ptr/REV3.md
```

The gate's records (`pilot3_*`) and the grid's (`rev3_*`) come back in one folder; `analyze.py` keeps them in separate rows (pilot names carry their learning rate and PonderNet constants in the row label). The rule in `experiments/ptr/gate.py` is the one in §10, applied by the kernel before the grid, with no one in the loop.

**Training.** Batch 64, AdamW (0.9, 0.95), weight decay 0.01, cosine schedule with 500 warm-up steps, gradient clip 1.0; learning rates: LO/LF 1e-3, learned halting 3e-4 (the head × its multiplier), S8 1e-4 and 3e-5, S2 3e-4 (pilots).

**Run order and cuts.** Pilot 3, then the grid as one session. If the grid must be cut: LO main 10 → 5 (LO is settled, spread 0.02), LH-x 5 → 3, then the S8 factor-10⁻¹ cell; never fewer than 5 seeds per primary arm; LH-Δ, LH-s24, LF/LR and the 10⁻² pooling cells at full count.

## 7. Endpoints

All at query positions, by k, run-level.
1. Accuracy and steps by k against the no-hop baseline, in range (k ≤ 8) and out (10–32), for every arm including LR on LF's block (threshold 10⁻² primary; the frontier 10⁻¹ … 10⁻⁴ reported), **LR on each arm's own block**, each block's fixed-loop accuracy at 40, and LF at T = 10 and 40.
2. **Extrapolation gaps:** each halting arm minus LO, minus LR(LF) and minus LR on its own block at k = 16 and 32; per-seed outcome counts (≥ 0.9 / 0.6–0.9 / < 0.6) at k = 16 and 32; run-level paired bootstrap P(A > B) over seeds for every comparison named in §8 (`analyze.py: preregistered`).
3. **Calibration:** steps vs k, slope and excess, for LH-s, LH-Δ, LH-s24, LH-x, LR.
4. **λ by loop** at k = 8, 16, 32 for every learned arm, with the mean chasing λ (loops < k) and the head's parameters: a post-horizon ramp (λ ≈ 0 through T, rising after) is the pilot-2 signature; a flat floor from loop 1 is a premature-halt policy, not a horizon effect.
5. **Steps to criterion by k** (§4) for every arm; for learned arms both the halted and the T-loop readout.
6. **Pooling by time:** steps to criterion at k = 3 (and 4) against the tail factor, per arm, with the ratio to factor 1.
7. **Step-index probe:** linear decodability of the loop index from LN(h_t) on chasing states, per arm (LH-Δ should not differ from LH-s: the probe is about the block, not the head).
8. Residual traces (‖Δh‖, step-KL, entropy, correctness vs remaining hops) and the random-halting control (LF traces halted at a random step from each learned arm's step distribution).
9. Trainability: loss and gradient-norm curves; per-run loss-scale events.

## 8. Predictions (thresholds fixed here; run-level paired bootstrap P(A > B) with 10 seeds, 5 for the secondary arms)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **What the policy reads (headline).** Precondition ("no test" otherwise): LH-Δ ≥ 0.95 at every k ≤ 8 (G4b). Then: ≥ 8/10 seeds of LH-Δ ≥ 0.9 at k = 16, ≥ 8/10 ≥ 0.8 at k = 32 (trained on k ≤ 8 with T = 10), run mean within 0.1 of LO at every k ≤ 32; and ≥ 8/10 seeds of LH-s < 0.6 at k = 16 (pilot 2 replicated). A mechanism claim also needs LH-Δ's λ by loop at k = 32 to stay < 0.05 through loop 24 (no flat floor). | ≥ 8/10 seeds of LH-Δ < 0.6 at k = 16 while LR *on LH-Δ's own block* ≥ 0.9: the change signal is in the block but the learned head does not transfer it (if LR on its own block also fails, the block was shaped by the head and the comparison is about the block). Seeds split between the two (the dead zone): reported per seed, no mechanism claim. LH-s ≥ 0.9 at k = 16 in ≥ 5/10 seeds: pilot 2's failure was not robust. | mechanism (what the policy conditions on), conditional on the precondition and the λ signature; otherwise optimization |
| **P2** | **Horizon.** Precondition: LH-s24 ≥ 0.95 at every k ≤ 8 (G4b). Then ≥ 4/5 seeds ≥ 0.9 at k = 16 and at k = 24; its λ at k = 32 stays < 0.1 through loop 24 and ramps after it (the pilot-2 signature displaced to the new horizon); its step-probe decodability within 0.1 of LH-s's (the block did not change). No claim at k = 32 (≈ 1.3× the horizon). **LH-s-k4:** its accuracy falls to chance between k = 12 and 16 (the failure follows T ≈ 15 loops, as in pilot 2), not between k = 5 and 8 (which would follow 1.5·k_max). | ≥ 4/5 seeds of LH-s24 < 0.6 at k = 16: the state head's failure is not displaced by the horizon it was trained on (drift may still be the mechanism, §4: the prior, the anchor and the block's idling changed too; then the step probe and λ by loop decide which). LH-s-k4 failing by k = 8: the failure tracks the chain lengths seen, not T. | mechanism (state drift beyond the trained depths), with the §4 caveats |
| **P3** | **The bar** (judged against the LR threshold frontier, and against LR on LH-Δ's own block). LH-Δ's run-mean accuracy within 0.03 of LR(10⁻²) at k ∈ {8, 16, 32} and cheaper by ≥ 0.3 loops at k = 8 and k = 16 (paired bootstrap P(LH-Δ cheaper) ≥ 0.9), with no LR threshold on the frontier matching LH-Δ's accuracy at a lower cost. Outcomes in between (cheaper but less accurate; more accurate but not cheaper) are reported as a frontier, not a verdict. | LH-Δ neither more accurate nor cheaper than the best point on LR's frontier: the one-parameter rule suffices on this task. | performance comparison (accuracy and compute), not a mechanism claim |
| **P4** | **Calibration.** LH-Δ and LH-s24 steps: slope ≥ 0.9 against k, mean excess ≤ 1.5 over k ≤ 32 (LH-Δ) / k ≤ 24 (LH-s24). (LH-x's constancy in k holds by construction and is reported, not predicted.) | Steps flat in k beyond some k' < 24 (collapse to a constant), or excess > 3. | mechanism (allocation tracks demand) |
| **P5** | **Sequential learning (optimization).** S8 (lr 1e-4, and no faster at 3e-5): steps to criterion at k = 1, 2, 3, 4 strictly increasing, each at least 2× the previous (k = 4 reached only if the budget allows; "> 25k" counts as increasing); LO's t95 at k = 8 < S8's t95 at k = 3 (paired P ≥ 0.9; a replication of pilot 2 at 5–10 seeds); LO's t95 grows by < 2× from k = 2 to k = 8. | S8 reaches k = 4 within 1.5× its time to k = 3; or LO's t95 at k = 8 ≥ 3× its t95 at k = 2; or S8 at 3e-5 learns k = 3 in less than half the time (the rate was not bracketed). | optimization |
| **P6** | **Pooling by time.** (a) *Block, mechanism:* LO's t95(k = 3) at factor 10⁻² within 2× of factor 1 (its t95(k = 2) is the control). (b) *Static, one-sided:* S8's t95(k = 3) at factor 10⁻¹ is censored at 25k or ≥ 2× factor 1 (the ratio is reported as "≥" when censored). (c) *Policy:* LH-s's halted t95(k = 3) at 10⁻² ≥ 2× factor 1 or censored at 8k; LH-Δ's halted t95(k = 3) at 10⁻² within 2× of factor 1 (a change-reading policy's continue decision does not depend on k ≥ 3 examples). The LH-s T-loop line is a diagnostic only (§4). | (a) LO's t95 at 10⁻² ≥ 3× factor 1: the shared block is as data-hungry as the stack at this depth. (b) S8's t95(3) at 10⁻¹ within 1.5× of factor 1. (c) LH-s's halted t95(3) at 10⁻² within 1.5× of factor 1: the state-reading policy learns to continue from shallower depths (a falsified prediction that favors the thesis); LH-Δ's ≥ 3×: the change head's continue decision is as example-hungry as the state head's. | (a) mechanism (signal pooling); (b) optimization, one-sided; (c) mechanism (what the policy's decision is learned from) |
| P7 | **Sanity (confirmatory; the LF lines reproduce Bansal 2022 / Schwarzschild 2021, the LR line Geiping 2025's exit at exact demand).** LO ≥ 0.99 for k ≤ 8 and ≥ 0.8 at k = 32; S2 at baseline for k ≥ 3; LF at T = 10 ≥ 0.95 for k ≤ 8 and at T = 40 < 0.5 at k ≤ 4 (over-running); LR(10⁻²) ≥ 0.95 at every k ≤ 32 at ≤ k + 1.6 steps for k ≤ 8 and ≤ k + 3 for k ≤ 32 (pilot 1: k + 1.1–2.3); LH-x: a constant depth, 1.0 for k ≤ that depth and baseline beyond. | Any violation reopens the corresponding pilot result. | expressivity / optimization |

**Planted models give both verdicts (L17):** the planted block with oracle halting (passes P7, calibration); the planted LH-Δ (passes P1, P3, P4 as a mechanism); the planted block with a constant halter at T_train (fails P1, P4); LR on the planted block (fails P7's LR line, §4).

**What the outcomes mean** (P1 and P2 are read only when their preconditions hold; a failed precondition is "no test": the arm is reported as not trainable at the shared constants, and nothing about the mechanism follows).

| outcome | reading |
|---|---|
| P1 ∧ P2 (∧ P3) | The failure of learned halting is a conditioning failure, not an optimization failure: a depth policy must read change, or be trained past the depths it serves. With P3, a learned change-reading policy also beats the one-parameter rule on compute. |
| P1 ∧ P2 ∧ ¬P3 | The mechanism holds but learning buys nothing over the free rule here; rung 1 needs a task whose completion is not a fixed point. |
| P1 ∧ ¬P2 | Drift lives in the state and a change head is immune to it; exposing the state head to deeper loops did not fix it (the §4 confounds of LH-s24 decide whether the head or the block is why). The drift explanation is *confirmed*, with T-exposure insufficient as a remedy. |
| ¬P1 ∧ P2 | Conditioning on change is not enough for a learned head: a longer horizon rescues the state head, the change head does not transfer. If LR on LH-Δ's own block passes, the signal exists and the head failed to learn its threshold under the PonderNet objective (the λ trace says whether by a flat floor or a late ramp); if LR on its own block fails too, the block shaped by the head lost the signal. |
| ¬P1 ∧ ¬P2 | Neither remedy works at these constants; the pilot-2 failure is robust to what the head reads and to the horizon. The next candidate is the objective (the geometric prior pulling toward halting at unfamiliar depths), tested by λ_p → 0 and by a prior-free halting loss. |
| dead zone in P1 or P2 (seeds split) | Reported per seed; no mechanism claim; the per-seed λ traces and head parameters are the next analysis. |
| P4–P7 | Enter the reading only where they fail: a P4 collapse voids the arm's P1/P2 line; P5/P6 are independent of P1–P3 and are read on their own; a P7 failure reopens the pilot result it contradicts. |

## 9. Fairness (AGENTS rule 4)

- Learning rates per arm from the pilots (3 values × 2 seeds for every arm that ran in pilot 1; 4 PonderNet settings × 2 learning rates for LH-s in pilot 2); the two new arms get their gate in pilot 3 (LH-Δ: two head multipliers; LH-s24: the shared constants); the same PonderNet constants and the same halting-logit init for every learned arm (L23).
- Equal width for every arm (the same per-step cost); S8 has 7.8× the parameters of a looped arm and is given 3× the step budget at its best learning rate, now bracketed from below (3e-5) as well as above (3e-4, pilot 2).
- Compute reported as steps per token, average and maximum; S2 is the compute-matched static arm; the random-halting control matches each learned arm's average compute.
- Paired seeds; 10 for the primary arms, 5 for nulls and secondary arms; run-level bootstrap.
- Time-to-criterion comparisons are made on the same training stream per seed; matched-training-loss comparisons from the curves.
- LR's threshold (10⁻²) was the first value tried on two pilot-1 runs and is fixed here for the new LF runs; it is a one-parameter rule, and the full threshold frontier is reported so that P3 is judged against the frontier rather than one point. LR is compared with LH-Δ on LF's block and on LH-Δ's own block.
- Not done: an initialization-scale sweep (no closed-form richness parameter here); parameter-matched static (deferred); stated.

## 10. Gates (status)

- **G1 planted** (done): exact hop, idempotent after termination, context fixed points, the planted LH-Δ, the LR caveat (`tests/test_ptr.py`).
- **G2 trainability** (passed, pilot 1): LO reaches 1.00 at k ≤ 24 and 0.89 at k = 32 at lr 1e-3; injection is essential.
- **G3 nulls** (pilot 1): seed spread of identical arms is ≤ 0.02 in range.
- **G4 learning rates and PonderNet constants** (passed narrowly, pilot 2): fixed as in §3.
- **G4b, pilot 3 (runs before the grid, in the same session; the rule is applied by `experiments/ptr/gate.py`, tested):** LH-Δ at head multipliers 1 and 10, LH-s24 at the shared constants, 2 seeds each, 8k steps. **Rule, fixed now:** an arm passes if both seeds reach ≥ 0.95 at every k ≤ 8 (the halted readout) and its steps track k in range (slope ≥ 0.8); for LH-Δ the multiplier is the *smaller* one that passes (the one closer to every other arm); out-of-range accuracy is not consulted. If no LH-Δ setting passes, LH-Δ enters the grid at multiplier 10 anyway, labeled "failed G4b", and P1 is "no test" for it; the same for LH-s24. The gate records are pilot results (not preregistered).
- **G5 cost** (pilots): 0.065 s per run-step at T = 10; pilot 3 is 1.3 and the grid 14.8 T4-hours.
- **Independent reviews of revision 3 and of the code** (done; §12).

## 11. What this rung does not answer

- Selection among several primitives (rung 1); nothing about composing distinct circuits.
- Learned primitives, their interface, library formation; search and learned values.
- Predicting demand: every halting rule here *detects* completion (a fixed point of the register). LH-Δ reads change; it cannot say how far there is to go.
- Whether the result survives data with parallel shortcuts (the doubling cell is deferred: it changes the static reference to a simulated causal-doubling reach and adds 40 runs; it is a different question from Q1–Q2 and should follow them).
- Parameter-matched static (S8-pm, d = 96): deferred; the equal-width S8 is already slower to learn each depth with 8× the parameters, which is the direction the parameter-matched arm would push further.
- The objective itself (λ_p → 0, a prior-free halting loss) as the explanation of a robust failure: the follow-up if ¬P1 ∧ ¬P2.

## 12. Independent reviews of revision 3 (27 Oct 2026) and how each finding is handled

Two fresh reviewers, one of the design (`review/review3_design.md`, verdict "run with the listed changes"), one of the code (`review/review3_code.md`, "no blocking defect; fix 1–4 before the run").

| finding | handling |
|---|---|
| **D-B1 (blocking).** LH-Δ had never been trained; with AdamW Σ lr ≈ 1.13 over the budget, so the raw-log-feature head could not reach a chasing λ small enough and would halt prematurely in a way that mimics the horizon failure; §9's "learning rates from the pilots" was false for LH-Δ and LH-s24. | Change features standardized (§3, §4); a head learning-rate multiplier; **pilot 3** as gate G4b with a fixed selection rule (§10); the λ-by-loop signature (flat floor vs post-horizon ramp) and the head's parameters recorded and named in P1; "no test" preconditions in P1/P2; §9 corrected. |
| **D-B2 (blocking).** P1's falsifier compared LH-Δ with LR computed on LF's block; the head shapes its own block (features not detached). | `kl_exit` is now also computed on every looped arm's own block, run as a fixed loop at the final evaluation (`kl_exit_own`, plus the block's fixed-loop accuracy); P1's falsifier and P3 use LR on LH-Δ's own block as well as on LF's. |
| **D-B3.** P7's LR line (≤ k + 2) was already violated by pilot 1 at k ≥ 24; P3's anchor "≈ k + 1.5" holds in range only. | §4 and P7 restated with the pilot numbers (k + 1.1–2.3); P3 restated as a cheaper-by-0.3-loops claim at k = 8 and 16 against the LR threshold frontier. |
| **D-B4.** P6's S8 line was unconfirmable as written (factor-1 t95 ≈ 10k vs a 25k budget); the LH-s halted line had no falsifier. | P6 split into (a) block, (b) static one-sided and censoring-aware, (c) policy with falsifiers for both LH-s and LH-Δ; censoring-aware medians and ratios in `analyze.py`. |
| **D-B5.** Dead zones 0.6–0.9 in P1/P2; the outcome table lacked ¬P1∧¬P2, P1∧¬P2 and no-test rows. | Per-seed fractions (≥ 8/10, ≥ 4/5) in P1/P2 with an explicit dead-zone rule; the outcome table extended; P4's LH-x line demoted to "by construction". |
| D (confounds of LH-s24: prior mass, anchor, idle training, drift co-occurring with answer). | Stated in §4; the step probe and λ by loop named as the separating endpoints; LH-s-k4 added as the complementary test (T vs k_max). |
| D (LR threshold tuned on two pilot-1 runs; "zero-parameter"). | Called a one-parameter rule; the threshold frontier (10⁻¹ … 10⁻⁴) reported; P3 judged against the frontier. |
| D (window statistics at factor 10⁻²; k = 2 control; policy-gated gradient confounds the LH-s T-loop line). | Windows aggregated to ≥ 250 examples of k; k_deep − 1 column in the pooling table; the LH-s T-loop line demoted to a diagnostic. |
| D (paired bootstrap named but not implemented; S8 rate not bracketed; literature: Geiping = LR, Bansal/Schwarzschild, CALM, PonderNet parity). | `analyze.py: preregistered` runs every §8 comparison as a paired seed bootstrap; S8 at 3e-5 × 2 added; §1 and P7 cite the reproductions. |
| D (cheap additions). | LH-Δ at factor 10⁻² × 5 and LH-s-k4 × 3 added; ‖h_t‖ added to traces; extra LR thresholds and head parameters recorded. Cut order revised (LO 10 → 5 first). |
| **C-1.** LH-x beyond T used untrained logits (`halt_bias[min(t−1, 63)]`), not "the last logit repeats". | Fixed: the logit of loop T − 1 repeats (tested); pilot LH-x numbers stand (it halted before T). |
| **C-2.** The pooling ratio used the median of uncensored runs only. | Censoring-aware medians (`cmedian`) everywhere; "≥" ratios when the median is censored. |
| **C-3.** The paired bootstrap was not implemented. | Implemented (above), with ties reported. |
| **C-4.** "10 fits a T4 at T = 24" was wrong (≈ 15 GB); the fp32 fallback needs more memory. | Help text and §6 corrected; `run_ptr.py` default 10; on failure a stack is split in two, fp32 only for a single run. The rev3 stacks are ≤ 10 at T = 10 and 5 at T = 24. |
| C-5..10 (minor). | The stacked-vs-single test now plants nontrivial delta-head weights; traces rounded to 6 decimals; calibration prints the excess by k range and omits LR's `max_steps`; the all-T-trajectory caveat of `acc_halt_by_k` labeled; the fp16 feature floor noted here (settled-state only; eval halts no later than training). |

---

## Revision 2 (27 Oct 2026, superseded by revision 3 before running; kept for provenance)

### What changed from revision 1

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

### Revision-2 cells (not run as designed)

| cell | regime | arms | seeds |
|---|---|---|---|
| main | read-only, α = 1.5, factor 1 | S2, S8, LF, LO, LH-s, LH-x (+ LR derived) | 29 for S8, LO, LH-s; 10 for the rest |
| P-pool | read-only, factors 10⁻¹ … 10⁻⁴ | S8, LO, LH-s | 10 |
| injection | read-only, factor 1 | LF-ni, LH-s-ni | 10 |
| doubling | pairs attend to earlier pairs and loop | S8, LF, LO, LH-s | 10 |
| param-matched | read-only | S8-pm | 10 |

Training as in revision 3 but 20k steps for every arm and lr per arm from pilot 1 (LH-s at 1e-3, S8 at 3e-4, which pilot 2 showed to be a collapsed and an unfinished arm respectively). The grid files `grids/{main,ppool,injection,doubling,pm}.json` are kept as generated.

### Revision-2 predictions (for the record)

| # | prediction | outcome in the pilots |
|---|---|---|
| H1 | S2 at baseline for k ≥ 3; S8 ≥ 0.95 for k ≤ 8; LO ≥ 0.99 for k ≤ 8 | S2 ✓, LO ✓; S8 reached only k ≤ 3 in 15k steps (budget/measure, revised) |
| H2 | LH-s ≥ 0.9 at k = 16, within 0.1 of LO everywhere; H2b: LH-s matches LR and is cheaper | falsified for the state head (0.28 at k = 16 vs LO 1.00); cheaper than LR in range only |
| H4 | LH-s steps: slope ≥ 0.9, excess ≤ 1.5 | holds for k ≤ 12; flat at ≈ 15 beyond |
| H5 | S8 leaves ceiling at k = 8 by factor 10⁻²; LO/LH-s stay to 10⁻³ | not run; replaced by pooling-by-time |
| H7 | a step-KL threshold reaches ≥ 0.9 of LO at ≤ 1.5× compute | 0.97–1.00 at ≈ 1.05× (exploratory) |

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

**How revision 2 handled each finding:** the table under "Revision 2" above.
