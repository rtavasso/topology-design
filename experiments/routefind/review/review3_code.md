# Code review 3: rung 2, revision 3 (`43726e7..f34b73d`)

**Reviewer:** independent code review, 29 Oct 2026. **Scope:** `git diff 43726e7 f34b73d -- src tests experiments/routefind`, read against DESIGN.md revision 3, §§2–7.
**Method:** I wrote my own checks and did not rely on the repo tests. Scratch scripts are in the session scratchpad under `review3_code/` (`bf_posterior.py`, `pf_tb_targets.py`, `set_late_check.py`, `ac_tb_init.py`, `tb_prefix.py`, `ac_check.py`, `misc.py`, `decoy.py`, `smoke.py`). All were run on CPU with 2 threads.

## Verdict

**Not ready for the pilot as is. Two bugs, both small to fix, and four should-fix items.**

- The exact pieces are correct:
  - the fixed-prior posterior;
  - PF's loss;
  - the actor-critic's TD algebra;
  - `set_late`;
  - the grids.
- The bugs:
  - **TB16's loss is dominated by an artifact.** Moves out of an answer or dead state get log π₀ = log 10⁻¹² (issue 1).
  - **AC16's critic is not the zero critic DESIGN §5 assumes.** Its construction also shifts the RNG, so AC16's policy head init is not paired with RL16's (issue 2).
- The should-fix items concern:
  - whether AC16 vs RL16 isolates bootstrapping (issues 3–5);
  - the analysis not implementing §1's success conjunction, and PF/TB's target being stochastic by construction (issue 6).

## What I verified (passes)

1. **`posterior_counts` is the exact posterior** p₀(r|y) ∝ π₀(r) e^{−λ·moves} L_y(r). I checked it against an independent brute force: a DFS over all routes, with an executor written from the pair lists rather than from `exec.maps`. Five cases: δ = 1 and 0, 16 and 4 answers, words alt3, alt4, train2 and pp4, T = 6–8.

   | case | max \|Z/Z_bf − 1\| | max \|counts − bf\| |
   |---|---|---|
   | δ1, 16 ans., alt3, T 7 | 7.8e-16 | 6.7e-16 |
   | δ0, 16 ans., alt3, T 7 | 7.8e-16 | 6.7e-16 |
   | δ1, 4 ans., alt4, T 7 | 2.9e-15 | 1.8e-15 |
   | δ0, 4 ans., train2, T 6 | 2.2e-16 | 1.1e-16 |
   | δ1, 16 ans., pp4, T 8 | 2.4e-15 | 1.4e-15 |

   The posterior expected number of moves (Σ of the HOP/PREV/SET counts) also equals the brute-force E[moves] to 1e-10. The conventions agree with the rest of the code:
   - **ε:** ε/(A − 1) on a wrong answer, 0 on a key or dead.
   - **Dead and answer states:** π₀ = HALT with probability 1.
   - **HALT** carries no step cost.
   - **Forced halt at T:** `beta[T] = L_y` with no HALT factor. This is `route_logprob`'s convention and the marginal's (`b @ R` at T).
   - **Step cost:** e^{−λ} on HOP, PREV and SET, which counts the same moves as `committed`'s `steps`, the marginal's `esteps` and the beam's `−λ·moves`.
2. **The PF target ignores the flag, as it should.** Z is identical under δ = 1 and δ = 0 on the same episodes, because the flat prior and the likelihood never read the flag. So PF's δ = 0 target is the δ = 1 target, and the router has to express it through flag features. That is consistent with §2 and P2.
3. **No Z = 0 and no underflow.** The smallest Z over 1,800 training queries (300 episodes × 6) is 1.9e-4 at k = 8 (`misc.py`: 1.91e-4 at batch 64). The Z > 0 guard and the division by `clamp_min(1e-300)` cannot produce NaN.
4. **`tb_loss`'s algebra is correct.**
   - The residual is d_k − d̄ with d = log π − log R, and d̄ is the VarGrad estimate of −log Z per query.
   - log R is detached, and log π₀ carries no gradient.
   - The gradient is 2/K Σ(d_k − d̄) ∇log π(r_k), and the sign is right.
5. **The 10⁻⁶ reward floor itself is harmless.** Its share of the floored target, F/(Z+F), where F is the floor mass on routes ending at a key or dead, has mean ≤ 0.003 and max 0.008 (k = 8 P-words, δ = 1, 16 answers). At 4 answers it is ≤ 0.002. (Issue 1 is a different floor, the 10⁻¹² inside `route_logprob`.)
6. **`ac_loss` matches hand computation.** I built two routes by hand (`ac_check.py`): one forced to halt at T with no HALT, and one that halts and then has padding. With gae ∈ {0, 0.5, 1}:
   - every actor advantage and every critic gradient (V − λ-return) matches;
   - there is no bootstrapping past termination, and padding is masked;
   - V receives no gradient through the actor term;
   - gae = 0 is TD(0), and gae = 1 telescopes to the Monte-Carlo return minus V(s_t);
   - the reward equals RL16's: readout p(y|r) at the end, −λ per move including SET and no-ops at an answer.
7. **No label leakage.** The router and the value head read only the pair tokens (key, value, prev, flag) and the product-state tokens. `q_start` and `q_ans` never enter `_trunk`.
8. **`set_late` is correct.** I planted E and L policies per episode, using the oracle's P-leaving states and their on-route predecessors (`set_late_check.py`, 200 episodes per stratum). On P-words of in4, in5, in8, alt8 and deepP8:

   | policy | δ = 1 | δ = 0 |
   |---|---|---|
   | E | min_ok = 1, set_late = 0, route_ok = 1 (it is the oracle) | min_ok = 1, set_late = 0, route_ok = 0 |
   | L | min_ok = 1, set_late = 1, route_ok = 0 | min_ok = 1, set_late = 1, route_ok = 1 (it is the oracle) |

   Pooling P-leaving symbols over an episode's queries is safe: each key belongs to one chain, and with `max_prun = 1` no predecessor is also a P-leaving state.
9. **The grids match DESIGN §6.** Regenerating them reproduces the committed JSON exactly: pilot 21 runs, main 76 runs, about 7.6 T4-hours.
   - Per-cell counts match the §6 table, seeds run 0…n−1, and names are unique.
   - `per_run_init` is on everywhere.
   - AC16 has `value_head = True, gae = 0`, and EI16B has `warm_steps = 1000` on δ0.
   - `run_route.group_key` separates the stacks by arm.
   - n_ans is plumbed all the way through. I ran tiny CPU stacks through `run_stack` for PFd1a4, AC16d1a4, TB16d0 and EI16Bd0 (2 runs each): `data.n_ans = 4` and `router.value_head = True` are recorded, `min_acc_P` and `set_late_P` are emitted, and EI16B switches from `exact` to `ei` after the warm start. The ε/(A − 1) = ε/3 likelihood at 4 answers is covered by check 1.
10. **The repo tests pass:** 21 passed in 3 min 49 s.

## Issues

### 1. BUG: TB16 scores moves out of an answer or dead state at log π₀ = log 10⁻¹² ≈ −27.6 each, so its loss and gradient are dominated by an artifact that penalizes correct routes

`estimators.py:296` together with `route_logprob` (`estimators.py:50`, `log(pi + 1e-12)`) and `flat_prior` (`:231`, π₀ = HALT with probability 1 at answers and dead).

**What goes wrong.** The router's sampled routes are not masked at terminal states. An untrained router keeps "moving" at an answer or at dead about 3/4 of the time. The executor treats these moves as no-ops, so RL16 charges only λ for them. TB charges −27.6 to log R for each one. As a result:

| at init (seed 0, 8 episodes, K = 16) | value |
|---|---|
| sampled routes with such a move | 41 % |
| median log π₀ of those routes | −85.7 |
| TB loss per query, median / max | 5,562 / 13,461 |
| per-run gradient norm, TB16 | **2,682** |
| per-run gradient norm, other arms (RL, IW, EX, AC, PF) | 3.1 / 13.5 / 13.2 / 23.6 / 23.7 |

With `grad_clip = 1`, clipping binds at every step by a factor of about 2,700.

**Why it matters.** VarGrad lowers log π of the *whole* route whenever d_k > d̄. Over 3 seeds × 32 episodes at init (`tb_prefix.py`):

| routes that reach the label | n | pushed down (d_k > d̄) |
|---|---|---|
| with one or more no-op moves after arriving | 1,134 | **63 %** |
| halting on arrival | 370 | 0 % |

So the routing decisions of correct routes get pushed down because of a no-op after arrival. This is not trajectory balance toward the fixed-prior posterior. In standard GFlowNet practice the forward policy is masked to the valid actions, which here means only HALT at terminal states. While it lasts, this confounds the readings "TB16 ≈ PF" and "TB16 ≪ PF".

**Fix.** Recommended:
- Truncate each sampled route at its first arrival at an answer or dead.
- Compute both log π(r) and log π₀(r) over the truncated prefix only, without a HALT term at the arrival state.
- Train the terminal decision locally with + Σ −log π(HALT | x) over the answer and dead states the samples reach. The posterior's conditional there is exactly HALT = 1, so this is PF's own target at those states.

The alternative is to mask the router's policy to HALT at terminal states for every arm, but that changes rung 1's model, so I do not recommend it.

**Test to add.** A route that reaches the label and then loops once must give the same TB residual as the same route halting on arrival.

### 2. BUG: AC16's value head is randomly initialized, not the zero critic DESIGN §5 assumes, and building it shifts the RNG so AC16's policy head is not paired with RL16's

`router.py:68–71`. `self.vhead = SLinear(R, d, 1)` uses SLinear's uniform(±1/√d) init, and it is built **before** `self.head.weight.normal_(std=0.02)`.

**(a) The critic is not zero.** At init (seed 0), V has std 0.41 across states and range [−1.44, +0.82]. The reward range is [−0.28, 0.98]. The TD(0) errors are therefore noise of the same size as the signal. The critic's MSE also dominates the shared trunk's update (`ac_tb_init.py`):

| at init | trunk gradient norm |
|---|---|
| critic only (π detached) | 19.4 |
| actor only (V detached) | 6.3 |
| total, AC16 | 23.5 (vs 3.1 for RL16) |

Under per-run clipping at 1, the policy's share of the step is small, and the critic is regressing toward random targets.

**(b) The RNG shift breaks the pairing with RL16.** With `per_run_init` at the same seed, AC16 and RL16 have identical parameters **except `head.weight`**. The vhead's construction consumes RNG before the head is drawn. The analysis's paired comparison AC16d1 − RL16d1 assumes the two start from the same policy.

**Fix.** Create the vhead after the `with torch.no_grad()` head-init block and zero its weight and bias. Construction then still consumes RNG, but after everything else is drawn, and `per_run_init_` copies the zeros. With the vhead zeroed (`misc.py`):

| at step 0, zero-init vhead | gradient norm |
|---|---|
| critic, trunk | 0.000 |
| critic, vhead | 3.0 |
| actor, trunk | 4.8 |

**Test to add.** At the same seed, the AC and RL routers have identical policy parameters and V ≡ 0 at init.

### 3. SHOULD-FIX: even with a zero critic, AC16's first updates are not RL16's; TD(0) starts with a push toward halting

DESIGN.md:113 says "AC16 starts from a zero critic, so its first updates are RL16's." With V ≡ 0, the TD(0) advantage of every non-final action is r_t = −λ, which ignores the label. Only the final action gets the readout reward. At init:
- 76 % of routes have two or more actions, so their first action's advantage is −0.02 regardless of the outcome.
- Across routes, the correlation between the first action's TD(0) advantage and RL16's RLOO advantage is −0.26.

HALT at a key is a final action with reward 1/16 and no baseline beyond V. So until V(start) learns about 1/16, the actor raises P(HALT) at keys and lowers every move by λ. That is the K = 1 bound's halting attractor, which rung 2 already observed in ELL16. This is not a bug, but the design's statement is false, and P4 is to be fixed after the CPU sanity run on the basis of this reasoning.

**Fix.**
- Correct §5: AC16's early updates favour halting until V reaches the scale of the readout.
- Log the critic's quality every 100 steps, for example the mean squared TD error and the explained variance of V on the held-out set's sampled routes. Otherwise "AC16 ≈ RL16" cannot separate "the value adds nothing" from "the value never fit". §4 currently records V only through its effect on the policy.

This is a few scalars in `search_stats`, not a new arm.

### 4. SHOULD-FIX: the critic's gradient reaches the shared trunk, so RL16 vs AC16 does not isolate bootstrapping

`router.py:113–116` computes the head and the vhead from one `_trunk(x)`. `ac_loss` returns actor + critic, and both are backpropagated into it. AC16 therefore differs from RL16 in three ways:
- the bootstrapped advantage;
- an auxiliary value-regression loss that shapes the routing features;
- how hard clipping binds (per-run gradient norm 23.5 vs 3.1 at init; issue 2).

The `ac_loss` docstring claims "RL16 vs AC16 isolates bootstrapping", and §7's reading table attributes any AC16–RL16 difference to bootstrapped value.

**Fix.** Pick one option and write it in §3:
- (a) Compute `vhead(x.detach())` so that only the advantage differs. This matches the stated isolation, and I recommend it.
- (b) Keep the shared trunk with an explicit value coefficient (for example 0.5 → 0.25) and say that AC16 includes a representation-shaping auxiliary loss.

### 5. SHOULD-FIX (reading caveat; no new arm): V(x) ignores the query while the reward depends on it, and TD(0) bootstraps across that aliasing exactly at the decoys

V is (episode, product state), shared by the episode's 6 queries. A stray route that HOPs from a P-leaving state into a decoy with c = 0 lands on a state that, in 25 % of cases (115/465 P-steps, `decoy.py`), lies on *another queried query's* oracle route at c = 0. There the owner's correct route also trains V. TD(0) then sets V(s_P, 0, 0) ≈ −λ + V(decoy, 0, 0), partly the owner's high value. So the error "HOP instead of SET" at the decisive state is under-penalized. Monte-Carlo returns (RL16, and AC16 with gae = 1) are unaffected.

This is not label leakage. It is a partially observed MDP inherent to a label-blind V, so it is a confound for P4. "AC16 ≈ RL16" or "AC16 < RL16" could come from this bias rather than meaning "at this horizon the value function adds nothing".

**Fix.** State the caveat in §3 or §7 ("Reading the outcomes") before P4 is set. No new arm is needed. If P4 comes out ambiguous, this is the first explanation to rule out (§11 already defers longer horizons).

### 6. SHOULD-FIX: the analysis does not implement §1's success conjunction, and PF/TB cannot be deterministic at their own optimum

**(a) The success test.** §1 requires shortest-correct accuracy ≥ 0.95 on P-words at in4 and in8, **deterministic**, **within 4× EX's episodes**. `analyze.py` reports these separately:
- `learned` (committed `min_acc_P` only, :46);
- "criterion within 4× EX" (:160);
- `samp_acc_P` only as a range in the checks section (:177).

The conjunction is never counted, so P1's "≥ 3/5 … within 4× EX" cannot be read off directly.

**(b) PF and TB.** Both fit a fixed posterior that is soft by construction. At E's SET states the target is SET 0.60 / HOP 0.40, and the flat prior puts 1/4 on each redundant SET. I took the PF-optimal per-episode policy, π*(a|x) ∝ Σ_q counts (`pf_tb_targets.py`, δ = 1, 16 answers, 300 episodes) and measured it on P-words:

| k | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| sampled p(correct) | 0.96 | 0.93 | 0.89 | 0.86 | 0.72 |
| committed shortest-correct accuracy | 0.91 | 0.91 | 0.75 | 0.59 | 0.27 |

E[steps] − oracle is about +1. At 4 answers the sampled p(correct) is 0.95 / 0.89 / 0.86 / 0.84 / 0.73 for the same k.

So a PF or TB router that fits its target well fails §1's "deterministic" clause and §7's check "learned runs are deterministic: sampled accuracy on P-words ≥ 0.95" by derivation. P2's falsifier text anticipates this, but §1 and §7 do not.

**Fix.**
- Add `success(r) = learned(r) and samp_acc_P ≥ 0.95 at in4 and in8 and crit_step(r) ≤ 4 × EX` and report its counts per arm beside `learned`.
- In DESIGN, before the pilot, state that PF and TB16 are judged on the committed route, and predict that their sampled accuracy stays below 0.95 (the values above).
- Otherwise, state that the PFN signal fails §1 by construction unless the router pools the soft target into a sharp policy.

### 7. MINOR: the program-label boundary disagrees with P5's falsifier

`analyze.py:61`: `prog_label` returns L when `set_late ≥ 0.5`, but P5's falsifier is "`set_late` ≤ 0.5 on δ = 0". A route that SETs at both places (0.5) is counted L. Use `> 0.5` for L, or add a "mixed" label for 0.25–0.75. `program()` also reads in8 only. That is fine for learned runs, but say so in the table header.

### 8. MINOR: `set_late` on PP strata is not a program measure

In a PP run the oracle's second SET is taken at a P-leaving state, so under δ = 1 the oracle (E) scores `set_late` = 0.55 on pp6 (`set_late_check.py`). Only in8 is used for identity, so nothing is affected today. Add a line to the `program_stats` docstring so nobody reads `set_late_pp*`.

### 9. MINOR: the curve's "loss" for TB and AC is not their loss

`train_route.py:333–340`: for TB the reported objective is the mean log-ratio (≈ −log Ẑ), and for AC it is the negative mean return. Name them (`tb_logZ`, `neg_return`) or log the surrogate as well, so that the curves are not misread.

### 10. MINOR: stale text

- `analyze.py:1` and `:146` say "revision 2".
- The `estimators.py` module docstring ("All three carry the step cost") predates PF, TB and AC.
- `derive/signals.py` still says programs "A/B" rather than E/L.
- DESIGN §3's "ε/15" is ε/3 in the 4-answer dose; the code uses ε/(A − 1), correctly.
- In `run_stack` the reshape `R * fb["pair_key"].shape[0] // R` equals `fb["pair_key"].shape[0]`. It is correct but hard to read.

### 11. MINOR (no change needed): memory and speed of `posterior_counts`

It keeps all T + 1 α and β tensors, each (B, Q, NS) in float64. At a stack of 10 × batch 64 (B = 640, Q = 6, NS = 390) that is about 30 × 12 MB = 0.36 GB, plus 48 MB for the counts and the transient (B, Q, NS, 4) float32 product in `pf_loss`. That is comfortable on a 16 GB T4. The kernels are scatter/gather, so they are bound by memory, not by the T4's FP64 rate; measure the s/step in the pilot as §6 plans. On CPU with 2 threads one batch-64 call takes 3.8 s, which matters only for CPU sanity runs.

## Summary of required changes before the pilot

| # | severity | change |
|---|---|---|
| 1 | BUG | TB: truncate routes at the first terminal arrival in log π and log π₀; local −log π(HALT) at terminal states; add the test. |
| 2 | BUG | Zero-init the vhead and build it after the head init; add the pairing test. |
| 3 | SHOULD-FIX | Correct §5's claim about zero-critic dynamics; log the critic's TD error and explained variance. |
| 4 | SHOULD-FIX | Detach the trunk for V (or document the shared-trunk auxiliary loss). |
| 5 | SHOULD-FIX | Write the aliasing caveat into the P4 reading. |
| 6 | SHOULD-FIX | `success()` conjunction in `analyze.py`; DESIGN statement on how PF and TB are judged, given their stochastic optimum. |
