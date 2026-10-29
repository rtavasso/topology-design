# Rung 2: independent code review 1 (before the pilot)

**Scope.** Branch `claude/theory-first-restructure`, HEAD `e886574`. Reviewed: `src/topology_design/route/estimators.py`, the estimator switch and new records in `src/topology_design/route/train_route.py`, `tests/test_routefind.py`, `experiments/routefind/{make_grids,analyze}.py`, `experiments/routefind/derive/search.py`, the Kaggle wiring (`experiments/fleet/kaggle/launch.py`, `run_on_kaggle.py`) and `experiments/route/run_route.py`. The question was whether the code computes what DESIGN.md (§§3–8) says. A separate reviewer is assessing the design itself.

**How it was checked.** Small independent scripts, all CPU, `torch.set_num_threads(1)`. They are in `/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/review_rf_code/` (named below). No repository file was changed apart from this review.

## Verdict

The training code computes what §3 says.
- **VIMCO and RLOO** are unbiased for their stated objectives. I checked this against exact gradients obtained by enumerating every route, not against the repo's own tests.
- **The beam** returns the brute-force MAP route on 40 of 40 new cases, covering both deferrals, T up to 6 and ε-mixing on.
- **Runs in a stack** are independent.
- **The "exact" and "teacher" paths** produce byte-identical records to the rung-1 commit.
- **Cost:** the estimators are no more expensive than the exact marginal, so a T4 stack of 10 will not OOM or slow down.

There is **one blocking item**, and it sits in code that produces preregistration numbers rather than in the trainer. The §5 derived beam numbers at the "uniform" policy, which P3 relies on, are an artifact of deterministic tie-breaking (finding 1). There are also several should-fix items. Two of them matter for reading the results:
- the search statistics are not split by P-words, so they cannot be compared with the §5 predictions (finding 2);
- the router init depends on which runs share a stack (finding 6).

---

## Blocking

### 1. The §5 beam numbers at the uniform policy come from tie-breaking, not from the search. (They feed P3.)

**What I ran:** `check_ties.py` and `check_ties2.py`. These use the beam on the derive episodes (400 episodes, `rng(0)`, W = 16, ε = 0.1) under five policies:
- the exact uniform policy, as `derive/search.py` uses it;
- uniform plus 1e-4 logit noise, which only breaks ties;
- logit noise with std 0.37, about the router's init scale (head std 0.02 × √344);
- uniform with SET + 1e-4;
- the **actual `StackedRouter` at init**, seeds 0 and 1.

Fraction of k = 3 / 4 / 5 P-words where the beam's MAP correct route is the oracle's:

| policy | δ = 1 | δ = 0 |
|---|---|---|
| exact uniform (the DESIGN §5 numbers) | 0.03 / 0.25 / 0.05 | **0.94** / 0.43 / 0.09 |
| uniform + 1e-4 noise (ties random) | 0.53 / 0.25 / 0.11 | 0.43 / 0.34 / 0.09 |
| noise std 0.37 | 0.45 / 0.36 / 0.05 | 0.45 / 0.27 / 0.10 |
| real router init, seed 0 (W=16 / W=4 at k=3) | 0.23 / 0.06 | 0.74 / 0.39 |
| real router init, seed 1 (W=16 / W=4 at k=3) | 0.87 / 0.10 | 0.10 / 0.00 |

**Why this happens:**
- On a k = 3 P-word there are two equally long correct routes:
  - SET at s₀, then HOP, PREV, HOP. This is the δ = 1 oracle.
  - HOP, then SET at s₁, then PREV, HOP. This is the δ = 0 oracle.
- Under the δ = 1 tables the second route is also correct: SET at s₁ gives c = 1 at the exit of s₁.
- Both routes contain one action of each kind plus HOPs, so under an exactly uniform π̃ their scores tie exactly. SET + 1e-4 does not break the tie either.
- `key.max`, `argsort` and `topk` then resolve the tie by index order, which favours HOP-first. The kept route starts with HOP 94 % of the time under both deferrals.

**What this means:**
- The δ = 0 "window at init" (0.94) and the δ = 1 init value (0.03) are the same tie read through two different oracles.
- At the real init, which route EI imitates on short P-words depends on whether the seed's random router prefers SET at s₀ or at s₁. Seed 0 favours the δ = 0 oracle and seed 1 the δ = 1 oracle; the init does not depend on δ. So:
  - the EI-16 δ = 0 outcome is largely a coin flipped by the init seed;
  - EI at δ = 1 can start from the wrong route for some seeds;
  - EI-4 at init is not 0.00 on δ = 0: it is 0.39 for seed 0.
- The tiny trainer run below agrees. The 100-step window shows EI16d1 / EI16d0 kept routes equal to the oracle on 0.69 / 0.98 of all k = 3 words, which is not the 0.03 / 0.94 asymmetry.

**Fix (before P3's thresholds are fixed):**
- Re-derive §5 with the actual router at init, over the grid's seeds and with ties broken randomly. Report per-seed values.
- Restate P3, in particular "EI-4 in 0/3" and the "oracle MAP 94 % at init" rationale.
- Consider logging the init-time search stats per run (see finding 9). Seeds can then be read against their own init preference.

This is a code finding that changes a design premise. It should go to the design reviewer as well.

---

## Should-fix

### 2. The search statistics are not split by P-words, but the §5 predictions they are meant to check are P-word quantities.

- **What the code records:** `train_route.run_stack` records `search_hit_by_k` and `search_oracle_by_k` over all queries of length k. The analyze table prints them at k = 3 / 5 / 8.
- **What §5 and P1/P3 reason from:** fractions on P-words ("≤ 2 % of sample sets contain the oracle route at the stray policy", "oracle MAP 94 % at init").
- **Why it matters:** at k = 3 only about 30 % of words contain a P (p_P = 0.3, w₁ = w_k = H). All-H words therefore dominate `search_oracle_by_k`, which can sit near 0.7 while the P-word rate is 0.
- **Fix:** accumulate a second pair of counters masked by `(fb["q_word"] == 1).any(-1)`. The cost is negligible. Without this split, the "what the estimator sees" endpoint (§4.2) cannot confirm or refute the mechanism behind P1 and P3.

### 3. In the beam, float32 keys with a 1e6 offset round scores to 1/16 nat.

- **The code:** `keys()` computes `right * 1e6 + (lp - λt)` in float32. The float32 spacing at 1e6 is 0.0625, and `check_beam.py` confirms that `1e6 - 1.00 == 1e6 - 1.03`.
- **Consequence:** step-cost differences (0.02 per move) and near-ties between correct routes cannot be resolved.
- **Measured against a copy of `beam_search` with float64 keys and no offset** (`check_beam.py`, W = 16, T = 14, 64 episodes × 6 queries):
  - random policies: the quantized beam keeps a lower-scoring correct route on 0–1.0 % of found queries, losing at most 0.044 nats;
  - near-oracle policies: 0 %.
- **Severity:** rare and small, but the fix is one line. Use `torch.where(right, lp - λt, -inf)` in float64. Non-right completions are never used, because `found` masks them. Then define `found = isfinite(best_key)`.
- **Note:** under a float64 default dtype (the test suite), `lp` is float64 and the quantization disappears. Tests and production therefore differ here.

### 4. A Kaggle kernel and output-folder collision with rung 1.

- **The code:** `launch.py` names a route job `"route-" + basename(grid)`. `experiments/routefind/grids/{pilot,main}.json` therefore become kernels `topology-fleet-route-pilot` and `topology-fleet-route-main`, datasets `…-code`, and output folders `results/kaggle_route-{pilot,main}`. These are exactly rung 1's sessions (PILOT.md and RESULTS.md provenance).
- **What follows:**
  - A rung-2 push becomes a new version of the rung-1 kernel.
  - If `results/kaggle_route-main/results` still exists locally, its 51 rung-1 records are uploaded as "prior". `run_on_kaggle.py` copies them into the rung-2 output, and they come back mixed with the rung-2 records. Names do not overlap, so nothing is skipped, but `analyze.py` would list RM, LF, S8 and the rest as extra arms.
  - The analyze docstring expects `results/kaggle_routefind-main`, which the launcher never produces.
- **Fix:** either rename the grids (e.g. `routefind_pilot.json`) or derive the job name from the grid's parent folder. Alternatively, always pass `--out` and `--kernel`.

### 5. The gradient tests cannot tell IW and RF apart. (The code passes my stronger check.)

- **What I ran:** `test_margin.py` repeats the two gradient tests on seeds 0–5. The RLOO cosine is 0.981–0.993 against a threshold of 0.97. The VIMCO cosine is 0.996–0.999.
- **The weakness:** on the same problems, the cosine between the two exact objectives' gradients (∇E[log p] vs ∇log q) is 0.91–0.98. It is 0.981 at seed 4, so swapping the iw and rf objectives would still pass the VIMCO test on some seeds.
- **What I checked instead:** `check_grad.py` and `check_grad2.py`. They give exact gradients of E[L_K] (K = 2, 3, by enumerating K-tuples over all 40–364 routes, T = 3–5, including a k = 3 P-word) and of −E[log p] + λE[steps].
- **Results** (averages of 40k–160k independent estimator draws, per-entry z-scores):
  - VIMCO K = 2/3: cosine 0.998–0.9996, max |z| 1.9–2.6 over 36–112 entries, mean |z| 0.71–0.95. One run gave max |z| 3.55 over 72 entries; it was not reproduced at 4× N, where max |z| was 2.60.
  - RLOO K = 2/16: cosine 0.9994–0.9999, max |z| 1.8–2.9.
  - The K-sample objective estimates match the exact E[L_K] within about 2 s.e.
  - The marginal with a log readout equals the enumerated −E[log p] + λE[steps] to 1e-6.
- **Fix:** add the K = 2 enumeration test with per-entry z-scores to `tests/test_routefind.py`. It is about 1 s at T = 3.

### 6. The router init and the sampling stream depend on stack composition. EX s0/s1 does not reproduce rung-1 RM s0/s1.

- **The code:** `run_stack` seeds the model with `1000003·seeds[0] + sum(seeds)` and draws all R runs' parameters from it. It seeds the shared sampling generator the same way. So a run's init depends on which runs share its stack:
  - EX δ1 (seeds 0–1, stack of 2) gets different inits from rung-1 RM seeds 0–1 (stack of 10). DESIGN §6's "reproducibility check" can therefore only be distributional.
  - EI16 (stack of 10), EI4 (3) and EX (2) see different inits for the same seed index.
  - IW16 and RF16 (both stacks of seeds 0–4) do share inits, so the P2 pairing is clean.
  - If `run_route` splits a stack after an exception, inits and samples change silently.
- **Why it matters:** given finding 1, the init decides much of the δ = 0 EI outcome.
- **What is fine:** the episode stream is per seed (`EpisodePool` has its own generator), as §10 says.
- **Fix:** at minimum, record the stack's seed list in `train`. Better, seed each run's parameter slice from its own seed (new arms only, so rung 1 is untouched).

### 7. `beam_search` returns the MAP route over a pruned route space, not over all routes.

- **What I ran:** `check_brute.py` (both deferrals; T = 4–6; k = 2–4; logit scales 1–4; ε = 0.1; W = Σ3ᵗ; 40 queries).
- **Against the same pruning:** the beam matches brute force in 40/40 cases, and `found` matches 40/40. Merging by state is correct, because π is Markov and all routes merged at step t have the same remaining budget.
- **Against the unpruned space:** on 26/40 queries the unpruned MAP correct route scores higher, by 0.4–3.0 nats.
  - These are routes that loop at the answer, or make a no-op move, so that they hit the forced halt at T without paying the HALT term.
  - Under random policies π(HOP | answer) is often larger than π(HALT | answer).
- **Assessment:** the pruning is a sensible choice, because EI then always imitates HALT at the first arrival. But the docstring, DESIGN §3/§8 and the repo test all call it "the MAP correct route found by brute force". The repo test applies the same pruning to its brute force, so it cannot detect this.
- **Fix:** say "MAP over routes without no-op moves, halting at the first arrival" in the docstring and DESIGN.

### 8. Gaps between `analyze.py` and DESIGN §§4, 7.

Dry run: `mini_grid.json` holds all 12 arms × 2 seeds plus one pilot run, at 100 steps and batch 4. I ran it through `run_route.py` on CPU (25 runs, 12 stacks, no errors), then ran `analyze.py` on it with `--ref experiments/route/results_data/main`. It completes, and the rung-1 reference maps correctly: EXd1(rung1) n = 10, criterion 800 [600–1200], SET flip 600, learned 10/10. Remaining gaps:

- **P4** asks for len16 and len32 within 0.05 of in8, and for pp4 through pp32 at or below 0.05. The code prints len32 and pp8 only.
- **P5** asks for `crit_obj` within 0.02 of EX's. The code prints a range without comparing it to EX. The rung-1 reference records have no `crit_obj`, so the table prints `nan`, and `np.nanmin` raises an All-NaN warning. The EX reference for `crit_obj` is therefore just the 2 EX runs in this grid.
- **Rollouts to criterion** (§4.1: episodes × queries × rollouts) is not computed. The table shows rollouts per query only.
- The search-stat table repeats a step when the end equals one of the listed steps (cosmetic).
- `crit_step`, like rung-1 `t95_of`, accepts an unconfirmed crossing at the last curve point (lenient). Say so, or require confirmation.
- `learned(r, deferral)` ignores `deferral`. This is harmless.
- For EI, `rollouts_per_query` is set to W. The beam actually scores 3W expansions plus W HALT completions per step. This is fine as a label, but the "≤ 16 rollouts per query" success criterion compares unlike things.

---

## Minor

### 9. There is no search measurement at step 0.

`search_*` values are 100-step window averages, and the first window already includes learning. The P3 "race" is decided at init, as finding 1 shows. Log one step-0 measurement (before the first update), or use 10-step windows for the first 100 steps.

### 10. Things checked and found correct (no action)

- **`route_logprob` and `sample_routes` indexing** (`check_index.py`, B = 3, Q = 6, K = 5, T = 14):
  - start states map back to `q_start[b, q]`;
  - replaying the sampled actions reproduces the sampled states;
  - final symbols match;
  - log π matches a manual sum, including the HALT term, nothing after halting, and no HALT term on the 23 forced-halt routes;
  - steps equal the number of moves.
  - The enumeration checks confirm that forced halts are treated as in the marginal.
- **VIMCO construction:** the diagonal of `rep` is replaced by the leave-one-out log geometric mean, so L₋ₖ depends only on r₋ₖ.
  - The surrogate's sign gives −∇L_K.
  - The step term is RLOO with `.mean` (correct for ∇E[steps]), and the adv term uses `.sum` (correct for VIMCO).
  - The reported objective is −L_K + λ·mean steps.
  - Tiny p_k cannot occur: the readout floor is ε/15, about 1.3e-3, and logsumexp is stable.
- **Estimator variance** (`check_var.py`, per query): single-estimate relative variance is 42 (IW-4), 7.9 (IW-16), 24 (RF-4) and 4.8 (RF-16). That is about 1/K, as expected. With 384 queries per run-step this is workable.
- **Run independence:** `indep.py` backpropagates run 0's surrogate in an R = 2 stack. Gradient leaking into run 1's parameters is exactly 0 for exact, iw, rf and ei. EI's normalization by found queries is per run.
- **Rung-1 reproducibility:** `repro.py` trains with the `19be219` source tree and with HEAD, for teacher ∈ {F, T} × δ ∈ {1, 0}, 2 runs per stack, 20 steps. Once the new fields (`crit_obj`, `rollouts_per_query`, estimator keys) are dropped, the JSON records are byte-identical. The new sampling generator never touches the global RNG.
- **Beam edge cases:**
  - Empty slots (−inf) never complete. The `isfinite` mask is applied.
  - W larger than the number of distinct candidates is handled (W = Σ3ᵗ in `check_brute.py`).
  - History is gathered by parent with `parent_of = arange(W).repeat_interleave(3)`, which matches `nxt.reshape(B, Q, 3W)`.
  - The forced-halt completion uses λT.
  - `replay` agrees with the beam tables.
- **Cost at full scale** (`check_cost.py`, B' = 640, Q = 6, T = 14, CPU 1 thread, router excluded, background load present):
  - exact marginal forward and backward: 1.72 s;
  - IW-16 sampling, log-prob and backward: 0.57 s;
  - beam W = 16: 1.32 s.
  - The estimators cost no more than the marginal that rung 1 already ran at 0.09 s per run-step on T4.
  - Memory: the (B, Q, 3W, 3W) bool is 8.8 MB, `route_logprob`'s gather is about 14 MB, and the `rep` tensor is 4 MB. The router activations (R, B, 390, 344) dominate, as in rung 1.
  - `est_cost` (0.09 s for every estimator) is sensible.
  - In `indep.py` (R = 2, batch 8, router included), per-step times were 688 / 659 / 719 / 747 ms for exact / iw / rf / ei.
- **GPU and dtype:**
  - There is no autocast in `train_route`.
  - The CUDA generator is used with `multinomial`. `tril`, `argsort` and `topk` on bool or float tensors are supported on CUDA. No CPU-only operations were found. This was not run on a GPU, since none is available here.
  - `readout_logp` and the router logits are explicitly float32.
  - The float64 default in `test_fleet*.py` is restored in `finally`, so it does not leak.
- **Stacking and grids:**
  - The `run_route` group key, which excludes name, seed and lr, keeps estimator, K, W, deferral and arm separate.
  - The pilot lr sweeps stack together per estimator, and `arm_of` separates them by lr.
  - The arm names match `analyze.py`'s preregistered lookups.
  - Grid counts are 56 main runs and 12 pilot runs, as §6 says.
- **Kaggle include list:** `test_routefind.py` imports only `topology_design`, numpy, pytest and itertools, all of which are uploaded (`src` is in INCLUDE). `run_route.py` needs nothing from `experiments/routefind`.
- **Tests:** `tests/test_routefind.py` gives 9 passed in 24 s on CPU. The tests are deterministic, because they use the global RNG seeded per test, so they are not flaky.
