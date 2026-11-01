# Rung 3: code review 1 (before stage 1)

**Reviewer:** independent code review, 1 Nov 2026. **Scope:** `git diff a5d6622 3715b02 -- src tests experiments` (the rung-3 implementation) read against [DESIGN.md](../DESIGN.md) revision 2, §§2–5, 8, 10. Files: `src/topology_design/route/{train_route,estimators,data}.py`, `experiments/route/run_route.py`, `experiments/fleet/kaggle/{launch,run_on_kaggle}.py`, `experiments/commit/{gate,make_grids,analyze}.py`, `tests/test_commit.py`.

**Method.** My own checks, CPU, `torch.set_num_threads(1)`, not the repo's tests (which I also ran: `test_commit.py` 11 passed, `test_route.py` + `test_routefind.py` 48 passed). Scratch scripts are in the session scratchpad under `review_commit_code/`: `check_completions.py` (an executor written from the pair lists, a merged beam written from scratch, a DFS over the pruned route space), `check_completions_pruned.py`, `check_acb.py`, `mem_completions.py`, `check_clean_strata.py`, `check_pp_placement.py`, `check_gate.py`, `sees_cpu.py`, `sees_debug.py`, `merge_check.py`, `check_grids.py` (with `fake_gate.json`), `check_ckpt.py`, `kaggle_sim/` (the stage-1 tarball rebuilt from `launch.py`'s include list), `analyze_fixed.py`. No repository file was changed apart from this review.

## Verdict

**The training code computes what §3 says; stage 1 cannot start as the Kaggle job is wired.** Two bugs block or corrupt the pipeline around the trainer, one bug sits in a shift-B endpoint, and one finding changes what the gate must record before stage 2 is preregistered.

- **Verified correct, independently:** `beam_completions` (the completion set, validity, replay consistency, log π̃, agreement with `beam_search`'s route on 136/136 queries, 0 duplicates); `pfb_loss` (weights ∝ π₀ e^{−λ moves} L_y over valid completions, the HALT term at answers, zero gradient without a label-reaching completion); `support_weights` and `rlb_loss` (weights under the router π, detached, normalized over valid completions; the gradient equals −Σ w (R − b) ∇log π(r) from my own executor and log-probabilities to 4e-8); `ac_loss` with weights (invalid completions contribute nothing to actor, critic or gradient; TD(0.9) by hand on a completion matches); the entropy bonus sign; checkpoints (all parameters and nothing else, the value head included, a loud failure on a value-head mismatch, `init_from` overriding `per_run_init_`, `torch.load` under this torch's `weights_only` default); stage 1 on disk equals `stage1()` and matches §6 field by field; `stage2()` from a fake gate report gives the 115 runs of §6 with unique names and `init_from` = checkpoint file name, and refuses with too few passes.
- **Bugs:** (1) the stage-1 Kaggle job fails its test step, because `test_commit.py` loads `experiments/commit/gate.py`, which `ROUTE_INCLUDE` does not upload; (2) `analyze.py` crashes on import (`A.load` missing: `import analyze` resolves to `routefind/analyze.py`); (3) `pp_set_placement` classifies the oracle's own route as E-E on only 80 of 300 pp8 words, because it keys on the first two PREVs of the word rather than on the first PP run, and counts SETs for other P-steps.
- **Finding for the derive step (not a code error):** under δ = 0 the committed E router's beam finds a shortest-correct route on 93–98 % of P-words at both ε, but it is program E's own route (SET at the unflagged predecessor), which the merge rule keeps over the δ-0 oracle route on 54/54 P-words; the oracle route is absent from the beam's support entirely. `gate.py` reports this as coverage 0.00, which is right for EI16's target, but ε is then not a coverage knob for this start and the reason EI16 does not move would not be "the beam loses the PREV child". §5.3 must record both quantities before the open rows are set.

---

## Bugs

### 1. BUG — stage 1 does not run on Kaggle: `gate.py` is not in the uploaded code, and the job's tests load it

`tests/test_commit.py:172` loads `experiments/commit/gate.py` by path. `launch.py:45` `ROUTE_INCLUDE` uploads `tests/test_commit.py` but not `experiments/commit/gate.py`; `run_on_kaggle.py:164` runs `pytest -x` on `test_commit.py` and exits with "TESTS FAILED: grid not run" on any failure.

**Repro** (`kaggle_sim/`: the tarball rebuilt from `launch.py`'s include list, then `pytest -q -x tests/test_commit.py -k gate_logic` inside it):
```
FileNotFoundError: .../code/tests/../experiments/commit/gate.py
1 failed, 10 deselected
```
**Fix:** add `"experiments/commit/gate.py"` to `ROUTE_INCLUDE` (its imports are all under `topology_design`, so nothing else is needed). Alternatively move `gate()` into `src/` and import it; but the one-line include is enough.

### 2. BUG — `experiments/commit/analyze.py` crashes at import

`analyze.py:21–23` inserts `../route` and then `../routefind` at `sys.path[0]` and runs `import analyze as A`, expecting rung 1's helpers. The last insert wins, so `A` is `routefind/analyze.py`, which has no `load`, `final` or `fmt_t95`:
```
$ PYTHONPATH=src python experiments/commit/analyze.py <folder>
AttributeError: module 'analyze' has no attribute 'load'
```
**Fix:** drop the `routefind` `sys.path` insert (the `routefind` module is already loaded by file with `importlib`). With that one line removed (`analyze_fixed.py`) the script runs on the CPU sanity output and prints the phase-1 and gate tables. Note that `routefind/analyze.py` itself does `sys.path.insert(0, route); import analyze as A`, so `route` must be the one on the path when it executes; the fix satisfies that.

### 3. BUG — `pp_set_placement` misclassifies most pp6 / pp8 words (shift B's primary endpoint)

`train_route.py:234–252` takes `first, second` = the symbols of the word's **first two PREVs**, and counts every SET of the route. Two failure modes:

- pp strata words contain other P-steps (`forced_word("pp")` adds a run to a `sample_word` draw with p_P = 0.3). With an isolated P *before* the run, `first` is the isolated P's leaving symbol and `second` the run's first; the run's second P-leaving symbol counts as "elsewhere". With an isolated P *after* the run, its SET counts as "elsewhere".
- PPP runs (40 % of pp6 / pp8 words) put the third SET in "elsewhere", so L-L-L reads 0-1-1-1 and E-E-E 1-1-1-0, neither of which is one of the five classes of §4.

**Repro** (`check_pp_placement.py`): the five hand routes classify as the docstring says (E-E 1-1-0-0, E-L 1-0-1-0, L-L 0-1-1-0, L-double 0-2-0-0, E-double 2-0-0-0). But with the **oracle route itself** as the committed route (program E on every P, so E-E on the run) on 300 generated pp words (δ = 1, T = 64):
```
pp6: {'1-1-0-0': 152, '1-1-1-0': 107, '1-0-1-1': 27, '1-1-0-1': 14}    isolated P before the run 27, after 14, 3-runs 107
pp8: {'1-1-0-0': 80, '1-1-1-0': 71, '1-0-1-1': 56, '1-1-0-1': 46, '1-0-1-2': 24, '1-1-1-1': 19, ...}   before 83, after 68, 3-runs 112
```
So on pp8 the endpoint "pp8 shortest-correct by route type" is readable on 27 % of the words, and E-E / L-L counts would be contaminated by 3-runs and by SETs that belong to other P-steps.

**Fix:** (a) locate the first PP run in `q_word` (consecutive P positions i..j), map P-steps to PREVs by ordinal (the m-th P-step of the word is the m-th PREV of the oracle), and take `first, second` (and `third` for a 3-run) from that run; (b) count only SETs at the run's predecessor symbol (the oracle's symbol just before `first`), at `first`, `second` (`third`), and put in "elsewhere" only SETs not attributable to another P-step (a SET at the predecessor of, or at, another P-leaving symbol is that P's E or L placement and should be ignored); (c) key the histogram by run length (`"2:1-1-0-0"`, `"3:1-1-1-0"`) so 2-runs and 3-runs are not pooled. Test it with the oracle route on generated pp8 words: every 2-run must read E-E.

---

## Should fix

### 4. SHOULD-FIX (before stage 2's preregistration) — the gate must record *which* shortest-correct route the beam keeps under δ = 0

`gate.py:87–92` defines shift-A beam coverage as `found & route_ok(kept route, oracle)`. On the CPU sanity E checkpoint (`cpu1_EX_p1_s0.pt`, 1,500 steps; `sees_cpu.py`, `sees_debug.py`, `merge_check.py`) under δ = 0, n = 48 episodes, 54 P-words:

| ε | beam kept route = δ0 oracle | δ0 oracle among the beam's completions | beam found a label-reaching route | some completion is shortest-correct |
|---|---|---|---|---|
| 0.1 | 0.00 | 0.00 | 0.94 | 0.93 |
| 0.3 | 0.00 | 0.00 | 0.98 | 0.98 |

The kept route is program E's route: e.g. word H P H, oracle `H S P H !`, kept `S H P H !`. Mechanism (`merge_check.py`): the E segment (SET at the unflagged predecessor, HOP, PREV at (s, 1, 0)) and the oracle segment (HOP, SET at the flagged state, PREV at (s, 1, 1)) each contain exactly one ε-floored action and end at the same product state three steps later, so the beam **merges** them; under π̃ they score −3.845 vs −3.845 nats and the tie is decided by the router's residual probabilities, π(SET | pred, 0, 0) ≈ 2e-6 vs π(PREV | s, 1, 1) ≈ 1e-7, in E's favour on 54/54 words. Raising ε changes both segments equally, so **ε is not a coverage knob for the E start**, and EI16's teacher-forced target from the E start is the E route, which §2 says is inexpressible for the Markov router under δ = 0. If EI16-from-E fails to move, that is the reason, not "the beam loses the improbable PREV child"; PA3's coverage → movement reading needs this distinction. (For the L start the picture is likely the opposite: the L router already PREVs at (s, 1, 1), so the oracle route costs one ε-deviation (the SET) and the E segment needs off-support decisions; so the L start may be the *easier* one for EI16, against PA4's premise. To be measured at the derive step.)

**Fix:** in `what_it_sees`, for shift A record, per ε, (i) kept route = oracle (as now), (ii) oracle route among the valid completions (`beam_completions` makes this one line: `(comp["acts"] == oa.unsqueeze(2)).all(-1) & comp["valid"]`), (iii) kept route is shortest-correct (`found & (st_b == osteps)`), and (iv) the kept route's program label on P-words (`program_stats` on the replayed kept route, as `search_stats` already does for EI). Set the open rows of §7 from (i), and say in §5.3 that EI16-from-E's target is E's route when (iii) ≫ (i).

### 5. SHOULD-FIX — the movement curve has no P-word field; `move_step` reads an all-word proxy, and the margin trajectory is not recorded

§4: "shortest-correct committed accuracy on P-words at in4 / in8 under δ = 0, every 100 steps; time to ≥ 0.95 from the end of warm-up; the PREV logit at the relabeled class over time". The 100-step criterion record (`train_route.py:647–659`) has `min_acc_by_k`, `min_acc`, `set_late[_iso]` but no shortest-correct-on-P-words field and no margin; `analyze.move_step` (`analyze.py:38–45`) therefore uses `min(min_acc_by_k over k ≤ 8)` over all words. At p_P = 0.3 this is a workable proxy (the k = 8 entry is 88 % P-words), at p_P = 0.08 it means "≥ 87 % of the k = 8 P-words" on ≈ 15 queries of the criterion set, and k ≤ 4 P-words are unconstrained. `moved` (end state, in4 / in8 `min_acc_P` on the clean evaluation when present) is right but is measured only at steps 2,500 and 3,000.

**Fix:** in `criterion()`, add `min_acc_P` (shortest-correct over the criterion set's P-words, k ≤ 8; n ≈ 250 at p_P = 0.3 and ≈ 90 at 0.08 for `crit_n` = 256, so consider a larger `crit_n` for the dose cells) and the margin at the relabeled class (from `pi` on the criterion set: `log π(HOP) − log π(PREV)` at (own-flag = 1, c = 1, h = 1) on the P-leaving symbols of P-words, which is the logit difference; no logits needed), and have `move_step` use `min_acc_P`. Both are a few lines and cost nothing. For the low-dose cells also report `n_P` of in4 / in8 (at p_P = 0.08, in4 has ≈ 37 P-words of 250).

### 6. SHOULD-FIX — the gate's margin is undefined for the L start

`gate.py:117–124` restricts the class (own = 1, c = 1, h = 1) to states the **committed** route visits on P-words. The E router visits them (CPU checkpoint: n = 59, margin 16.0–17.7 nats). The L router under δ = 0 never SETs (§2), so it never reaches (s, 1, 1): `sel` is empty and the report gives `None` for every L checkpoint, although §3 wants the margins at "the classes phase 2 must flip" for both starts.

**Fix:** take the class states from the **δ-0 oracle route's** visits (defined for both starts) and, for the L start, also record the SET − HOP margin at (own = 1, c = 0, h = 0) (its first decision). Keep the committed-visit version as a second number if wanted.

### 7. SHOULD-FIX — `wait_and_fetch` does not put `*.pt` files where `gate.py` and `--ckpt-dir` expect them when the CLI flattens

`launch.py:210–215` moves loose `*.json.gz` into `<out>/results` "because the CLI may flatten subdirectories", but does nothing for `*.pt`. `gate.py:140` reads `<results>/ckpt/<name>.pt` and raises `FileNotFoundError` on the first passing router if the files are loose; `launch.py --ckpt-dir` then has to point at the loose directory. Mirror the `.json.gz` handling: move loose `*.pt` into `<out>/ckpt`. Checkpoints are 5.5 MB each (1.44 M parameters, float32), 22 for stage 1.

### 8. SHOULD-FIX — AC16-B's critic diagnostics include the invalid beam slots, unweighted

`ac_loss` (`estimators.py:521–527`) sums `n, sG, sG2, sR, sR2, sTD` over `valid = acts >= 0`, which is true for the garbage action sequences carried by dead beam slots (36 % of the 240 slots under a random policy; `check_acb.py`), and ignores the support weights. The loss and gradient are unaffected (checked by perturbing the invalid slots' actions: loss, objective and both gradients unchanged), but `ev` (explained variance) and `td_abs`, which §4 names as core-B endpoints, are computed over routes the critic never trains on:
```
diag with / without the invalid slots' garbage: n 22848 / 24468, sTD 2676 / 2499, sR2 1671 / 757
```
**Fix:** multiply `vm` by `(weights > 0)` when weights are given (or by the support weights × M, to make `ev` the weighted explained variance).

### 9. SHOULD-FIX — checks C are not implemented

§3 "Checks C" and §8 G1 ("the agreement statistic and per-stratum AUROC on hand-built cases") have no code in this commit: `evaluate` records per-stratum means (`samp_acc`, `p_correct_marg`, `entropy_d*`), not the per-query triples (committed correct, 16-sample agreement with the committed answer, exact marginal of the committed answer) that PC1 needs, and nothing computes per-stratum AUROC. Not needed for stage 1 (the PF / PF-IS16 routers are saved), but it belongs with the derive step, and §10's list should say so.

---

## Minor

10. **MINOR** — `beam_eps*` for shift B is documented as "any shortest correct completion" (`gate.py:9–10, 85`) but computed as "the kept route is shortest-correct" (`found & (st_b == osteps)`; `st_b` is the kept route's moves). The kept route is the MAP label-reaching completion, not the shortest; either compute "any" from `beam_completions` or fix the docstring. For consistency with shift A (item 4) record both.
11. **MINOR** — ESS is normalized by M = W (T + 1) = 240 slots, of which only the valid ones carry weight (≈ 150 under a random policy, fewer for a sharp router), so `ess` ≤ n_valid / 240 and is not comparable with PF-IS16's `ess / K`. Divide by the number of valid completions, or report raw ESS and `n_valid` (the beam arms' analogue of "distinct routes per query", which `analyze.py` prints as NaN for them).
12. **MINOR** — clean and mixture strata are not paired. `strata_episodes` uses one `rng` for all strata, and the `pp_frac` draw (`data.py:176`) consumes it, so the streams diverge inside the first episode of `in1`: in `check_clean_strata.py` (seed 12345, 20 episodes) in4 shares 0/20 start keys and 0/20 contexts with its clean twin. The comparison is between two 250-episode samples of the same stratum, which is fine for means, but the comment "forced queries match" would be wrong; pairing would need the pp draw from a separate stream.
13. **MINOR** — `gate()` formats `min_acc_P` / `samp_min_acc_P` with `:.2f`; a `None` (no P-word in a stratum) raises `TypeError`. Guard with `or 0`. The sampled-determinism test is applied at in8 only; §3 does not name a stratum, in4 could be added.
14. **MINOR** — `analyze.py`'s per-seed AUROC pools RT and PF at coverage 1.0 with the open rows; PA3 is about the sampled / searched signals. Report the AUROC over {EX, EI16, EI16e3, PF-IS16, IW16} and, separately, with PF.
15. **MINOR** — the record's `cfg.init_from` is the resolved absolute path (`/kaggle/working/ckpt/...`); `train.init_from` is the basename and is what `analyze.py` and the gate match on. Harmless; `cfg` could keep the basename too.
16. **MINOR** — `launch.py` could refuse a grid with `init_from` entries when `--ckpt-dir` is not given; today the kernel starts, every stack fails with `FileNotFoundError` in `resolve_init` (loud, no file written), and the launch is wasted.
17. **MINOR** — `test_constant_schedule_and_short_warmup` checks the field and the curve length, not the schedule's values; a direct assertion on `sched(s)` for `schedule="constant"` (1 after `warmup`, linear before) would cover the two lines that matter. The G1 item "PFB16's target equal to PF's when the beam is exhaustive" is not achievable as stated: the merge keeps one prefix per (state, step), so even an exhaustive beam's support is not the route space (two prefixes reach the same state after SET-HOP-SET vs HOP-SET-SET); the right statement, which the test and my check verify, is "PFB's weights are PF's posterior restricted to the support".
18. **MINOR** — the core-B beam arms use ε = 0.1 (`make_grids.py:32–36`); §3 does not say which ε the completions are taken at. State it in DESIGN §3.

---

## Answers to the review brief, with what I ran

1. **`beam_completions`.** On 48 queries of a 12-key / 4-answer config with W = 64 ≥ the number of reachable states at every step (verified by DFS), the kept states equal the DFS reachable sets at every t, and the valid completions equal exactly "every kept state completed by HALT at every t < T, plus the forced halt at T" (1,408 completions, 0 duplicates), with `moves`, `end` and `logp_mix` equal to mine. With pruning (W = 3, 4, 16; T = 8, 14; ε 0.1 / 0.3; both deferrals; the full 48-key config) 134/136 queries agree exactly; the 2 disagreements are one float32 merge tie (two prefixes at −17.37346 to the same state). Dead / invalid slots are `valid = False` (log π̃ = −∞) and carry garbage routes, which every consumer masks. The t = 0 HALT completion is present (moves 0, end = start). A route arriving at an answer at t = T is the forced-halt completion without a HALT term, as in `marginal` and PF. The best correct completion under log π̃ − λ·moves equals `beam_search`'s route on 136/136 found queries (and its step equals `best_steps`). The start is never an answer in this generator.
   **Memory** (`mem_completions.py`, CPU RSS at a flat batch of 64 episodes × 6 queries, M = 240, T = 14): beam + replay 42–47 MB, loss forward 34 (pfb) / 40 (rlb) / 148 (acb) MB, backward ≤ 11 MB; the (B, Q, 240, 14, 4) float32 gather in `route_logprob` is 20 MB per run, `acts` and the replayed states 10 MB each. At a stack of 5 that is ≈ 0.5–1 GB on top of the trunk, whose activations dominate (≈ 170 MB per (R·B, NS, d) tensor at R = 5, ≈ 2–3 GB with autograd). Rung 1 ran stacks of 10 of the exact marginal and rung 2 stacks of 5 of EI16 / AC16 at batch 64 on the T4, so **a T4 holds a stack of 5 at batch 64** for rlb / acb / pfb, and the stage-1 stacks of 7 (EI16_p1, EX_p1) are below rung 1's 10. `run_route` halves a stack on an exception in any case.
2. **`pfb_loss`.** Weights match an independent computation of π₀(r) e^{−λ moves} L_y(end), normalized over valid completions only, to 1.3e-7 relative in the loss, with π₀(HALT | answer) = 1 (log 0) and 1/4 at keys; completions halting at a key get L_y = 0; a query with no label-reaching completion returns loss 0 and `ok = 0` (checked on queries where that happened). PF's exact posterior restricted to the support is by definition this normalization, so they agree; the beam's merge rule is why the support is not the route space (item 17).
3. **`rlb_loss` / `support_weights` / acb.** Weights are under the router π (`route_logprob(pi, ...)`, not π̃), detached, −∞ on invalid slots, softmax-normalized; they equal mine to 1.7e-7; invalid slots have weight exactly 0. R = p(y | end) − λ·moves with the rung-1 readout (1/16 at keys). The surrogate's gradient equals −Σ w (R − b) ∇log π(r) with b = Σ w R (to 4.5e-8), which is also the exact gradient of J = Σ w R with w self-normalized over a fixed support. ESS = 1/Σw² / 240 (item 11). `ac_loss(weights=w)` multiplies actor, critic and return by w·M and takes the mean, i.e. Σ_r w_r (·); TD(λ) along one completion matches a hand computation (−0.461007 vs −0.461007); termination is at the last valid action (HALT or the forced halt at T, v_next = 0); `actor_scale = float(step >= critic_warmup)` applies; invalid completions contribute nothing to the loss, the objective or the gradient, but do enter the diagnostics (item 8).
4. **Entropy bonus.** `sur −= β · mean entropy`, so descending the surrogate raises entropy (confirmed by the sign of the gradient on a hand case); `route_entropy` averages over `acts >= 0` (the HALT step included, the forced halt's T moves); the diag `entropy` is `ent.reshape(R, −1).mean(1)`, per run.
5. **Checkpoints.** `router_state` = `named_parameters` without the stack axis (buffers are the frozen bases, rebuilt from `emb_seed`, which the checkpoint also stores); a value-head state has 4 extra keys and `load_router_state_` raises on them; `run_stack` asserts `st["router"] == rc.dict()` first. `per_run_init_` runs first and `init_from` overwrites that run's slice only (checked). `run_route` writes `<ckpt>/<name>.pt.tmp` then `os.replace`, and `rec.pop("state")` before the JSON. `PER_RUN_KEYS` agree between `run_route.py` and `train_route.py` (`seed, lr, init_from`), so runs with different checkpoints stack (stage 2: stacks of 5 and 3 with distinct `init_from`). Kaggle fetch: item 7.
6. **Clean strata.** `evaluate` runs twice when `pp_frac > 0` and `clean_eval`; the clean result is `evals[-1]["clean"]`; `gate()` and `analyze.moved` use it when present; `analyze`'s phase-1 table uses the mixture (phase 1 has no PP, so it is the same). Seeds are identical but the episodes are not paired (item 12).
7. **`isolated_P`, `pp_set_placement`.** The five hand routes classify correctly; the classifier fails on generated pp words (item 3). PP runs of length 3 and words with other P-steps are the failure cases; an oracle with < 2 PREVs leaves −1.
8. **`gate.py`.** Thresholds and labels follow §3 (min_acc_P ≥ 0.95 at in4 and in8 on the clean evaluation when present; E ≤ 0.25 / L ≥ 0.75 at in8; sampled ≥ 0.95 at in8; R = +1 step with acc ≥ 0.95 at in4 / in8 / none8). Shift A's beam coverage is "kept route = oracle" (item 4); shift B's is "kept route is shortest-correct" (item 10); 16-sample coverage is "any sample = oracle" (A) / "any sample correct at the oracle's length" (B). The own-posterior is π(r*) (1 − ε) / q(y) with q(y) from `marginal` under the same readout, i.e. p_π(r* | y) for the route that halts at its first arrival; `clamp(max=1)` only guards numerics. The margin class is (own-flag = 1, c = 1, h = 1) on committed-route visits (item 6). `policy()` on the 1-stack router with the `(1, B, ...)` layout works (planted router: coverage 1.0 / posterior 0.999 on its own data; the CPU checkpoint runs in 2 s at n = 48).
9. **`make_grids`.** Stage 1: 58 runs (22 phase 1 + 36 core B), every field as §6 (constant schedule, lr 1e-3, warm-up 500, phase 1 at 2,500 steps with `save_state`, core B at 5,000, `critic_warmup` 200, gae 1.0 / 0.9, β per run, `ans_pool` 4 on the dose cells). Stage 2 from a fake report (7 E seeds with 2 failing, 7 L with 1): 115 runs, unique names, `init_from` basenames, the cells and seed counts of §6 (30 + 30 + 17 + 4 + 12 + 22), warm-up 100 on continued runs and 500 on the from-scratch references, and an assertion with < 3 passes per start.
10. **`analyze.py`.** Crashes at import (item 2); with the one-line fix it runs on the partial CPU output. `moved` matches §4's end state; `move_step` uses an all-word proxy (item 5).
11. **Kaggle.** Item 1 blocks stage 1; items 7 and 16 concern stage 2. `run_route` handles a missing checkpoint loudly (`FileNotFoundError` in `resolve_init`, "FAILED stack", no file). The grid file, `test_commit.py` and `run_route.py` are uploaded; the job's tests then pass once `gate.py` is included (11 tests, 51 s on one CPU thread). Budget at rung 2's rate: stage 1 ≈ 235k run-steps ≈ 5.5 T4-h, stage 2 ≈ 345k ≈ 8 T4-h, both within one two-GPU session.
