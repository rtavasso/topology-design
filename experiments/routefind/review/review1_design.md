# Rung 2 design review 1 (independent, adversarial)

**Reviewed:** `experiments/routefind/DESIGN.md` revision 1 at `e886574`, with route/RESULTS.md (F28–F34), history Entries 61–63, AGENTS rules 3–5, L22/L25/L26, `route/estimators.py`, the estimator switch in `train_route.run_stack`, `derive/search.py`, `make_grids.py`. Code correctness is reviewed separately.

**Checks run** (CPU, one thread, under a minute each; scripts in `scratchpad/review_rf_design/`):
- `beam_jitter.py`: the §5 beam numbers at an exactly uniform policy and at near-uniform random policies (logit noise 0.05 and 0.3), for eps 0, 0.1 and 0.3.
- `init_grad.py`: the expected gradient at init for EX, IW-16 (VIMCO, averaged over 6 draws), RF (exact, by the recursion with the log readout) and RF-16 (RLOO). It uses a tabular policy over 150 episodes and is summed over six state categories.
- `reach_eps.py`: product states reachable per query within T = 14, and the beam at the stray policy for eps up to 0.5.
- I also read the finished CPU-sanity record `scratchpad/rf_cpu/EI16_d0.json.gz` (batch 16, 1,500 steps). The IW / EI / RF δ = 1 sanity runs had not finished at review time.

## Verdict: **run with changes.** Three blocking items. All are cheap and all change predictions before the pilot.

---

## Blocking

**B1. The §5 "init" column measures how the beam breaks ties, not the init.** At an exactly uniform π every prefix of a given length ties. Which states the beam keeps then depends on the argsort order of the candidates. I recomputed with near-uniform policies, the regime a real initialised router is in:

| W, quantity | exactly uniform (§5) | jitter 0.05 | jitter 0.3 |
|---|---|---|---|
| W = 16, MAP = oracle, δ = 0 P-words, k = 3 / 4 | 0.90 / 0.46 | **0.50 / 0.24** | 0.47 / 0.22 |
| W = 16, same, **δ = 1**, k = 3 / 4 | 0.03 / 0.32 | **0.45 / 0.32** | 0.45 / 0.34 |
| W = 4, MAP = oracle, all-H, k = 2 | 0.02 | **0.81** | 0.80 |
| W = 4, δ = 0 P-words, k = 3 | 0.00 | **0.17** | 0.22 |
| W = 16, all-H, k = 6 | 0.50 | 0.10 | 0.10 |

Consequences:
- The δ = 0 vs δ = 1 contrast at init that P3's "window" rests on (0.94 vs 0.03) does not exist once the policy is not exactly uniform (0.50 vs 0.45).
- EI-4's "derived null" (0.00) is also an artefact. EI-4 may well learn δ = 0, so the prediction "0/3" is unfounded.
- Fix: recompute §5 at the actual initialised routers (3–5 seeds) and restate P3 from those numbers. The code reviewer should also look at the tie order (unstable argsort over equal keys).

**B2. EI-16 on δ = 0 is already decided (L25), and the "race" in P3 does not happen.** The CPU sanity run (`EI16_d0`, batch 16, 1,500 steps, about 6 % of one run's episode budget) reaches:
- held-out route accuracy 0.97 (0.81 = the all-H share until step 600);
- SET 0.90 (flipping from step 600 to 1,100);
- PREV at the oracle's set-here state 1.00 **from step 100**. That is the opposite of EX-d0's basin, where PREV stayed at 0.00–0.10;
- `search_oracle` 0.95 from step 100, rising to 1.00.

So EI never builds the stray rule. The reason: at init the MAP correct route under a near-uniform π is the *shortest* correct route (λ = 0.02 is negligible against log π ≈ −1.39 per action). For the short words that carry the Zipf mass, the shortest correct route is the oracle's:
- the Zipf-weighted MAP = oracle fraction at init is 0.83 (W = 16, jittered; 0.86 at exact uniform);
- k ≤ 2 holds 70 % of the mass.

EI is therefore close to teacher forcing on oracle routes, and the stray rule needs averaging over stray-SET mass, which hard EM never does.

P3's EI line (≥ 5/10) and the reading "search supervises SET→PREV directly" are then expected by construction. They are the known result that hard EM with a short-program bias works when spurious programs are rare among the shortest correct ones (B2 is also the L22 answer).

Required:
- (a) Enter §9 with this run and the three δ = 1 sanity runs before preregistering.
- (b) Restate P3 so it can fail. Two options:
  - EI-16's speed against teacher forcing on δ = 0 (add RM-T-d0, 2 seeds);
  - **the arm that actually asks "can search do what the exact marginal could not": EI-16 started from an EX-d0 checkpoint inside the basin** (e.g. step 1,000), 5 seeds. My `reach_eps.py` predicts it fails: at the stray policy the W = 16 MAP correct route is the oracle's on 0.00 of P-words at every eps in {0, 0.1, 0.3, 0.5}.
- (c) Reduce EI-16 δ = 0 from 10 seeds to 5.

**B3. RF is not "the RL objective", and at init its expected gradient points into a different basin, which P2 does not anticipate.** With ε = 0.02 the per-route log reward is:
- −0.02 for the right answer;
- **−2.77** for halting or dying on a non-answer;
- **−6.62** for a wrong answer.

So E[log p] strongly prefers stopping to gambling. (The marginal is indifferent in expectation: a decoy that ends at a random answer and a halt both give ≈ 1/16.) Expected descent direction at near-uniform init, summed per category, per 1,000 queries:

| state | EX | IW-16 | RF (exact) |
|---|---|---|---|
| flagged (0,0), δ = 1 | SET +3.1, HALT −0.8 | SET +2.1, HALT 0.0 | **SET −5.4, HALT +5.7** |
| flagged (0,0), δ = 0 | SET +2.9 | SET +2.6 | **HALT +3.1, PREV(→⊥) +2.3, HOP −4.7** |
| unflagged (0,0) (the stray SETs) | SET +1.3 | SET +2.2 | **SET −62** |
| unflagged (1,0) | HOP +22, HALT −12 | HOP +24, HALT −12 | **HALT +10.8** |

Readings:
- IW-16 ≈ EX in every cell (see N1).
- RF suppresses the stray SET mass that built EX's rule (F29). It also learns HALT at the flag, which costs nothing on all-H words because flags occur only on P-words. That is a halt-at-uncertainty basin on **both** deferrals.
- P2 ("≥ 4/5 within 4×, slower than IW because coincidences earn the oracle's reward") would then fail for a reason it does not name. P3's "RF fails like EX" would be read as F29 when the mechanism is the readout.

Required:
- (a) Derive this and restate P2.
- (b) Relabel RF as what it is: the K = 1 IWAE bound / ELBO with posterior = prior, estimated with 16 samples. RF-16 vs IW-16 is then K = 1 vs K = 16 at equal samples (a clean framing).
- (c) Add the standard RL arm: REINFORCE with reward p(y | r) (or 1[y]). Its expectation is q(y), so it maximises Σ q, not Σ log q. That is Guu et al.'s RL-vs-MML contrast exactly, and it has no stop-vs-gamble asymmetry. Give it 5 + 5 seeds; keep E[log p] at 3 + 3.

## Non-blocking

**N1. IW-16 ≈ EX at init (table above), so P1 and P3-IW are close to "EX plus noise".**
- K = 16 already hits the label on 0.73 of the Zipf-weighted queries at init (1.00 / 0.82 at k = 1 / 2).
- The δ = 1 success, if it comes, holds in a regime where the bound is nearly tight on most of the mass. Claims should say so: report the init hit rate as the scope variable, and avoid "survives the loss of exact inference" without that qualifier.
- IW-4 adds little. Cut it, or move it to a stress setting (N3).

**N2. The state-space size is not making the beam exhaustive.** Reachable product states per query within 14 steps: 167 on average (136 at k = 1, 224 at k = 8). The frontier exceeds 16 for 94 % of queries, so W = 16 is a real restriction. What makes search easy is the answer and length distribution (N3), not the 390 states.

**N3. The toy's spurious-route density is low exactly where the mass is.** "W = 64 reaches the label spuriously on 100 %" (§3) is about *found* routes. Under MAP/shortest selection, spurious routes matter only on k ≥ 5 (9 % of Zipf mass at k ≥ 6). If the thesis question is search under spurious programs, add one dose of spurious density for EI-16 and IW-16 (3 seeds × δ ∈ {0, 1}): `n_ans` = 4, or `tail_factor` raising the k ≥ 5 mass.

**N4. Missing contrast: sample, then imitate (STaR / ReST / rejection-sampling fine-tuning).** Sample K = 16 and imitate the most probable correct sample. This separates "search" (beam with merging) from "hard EM" (imitation). §5 predicts it differs from EI at init: the oracle is among 16 samples on 0.01 of k = 3 P-words. It is the arm most relevant to the LLM-reasoning literature, and costs 3 + 3 runs (it could replace IW-4). EI with "shortest correct route" selection is *not* worth an arm: at init it equals MAP (B2).

**N5. eps = 0.1 does not decide P3.** At near-uniform policies eps 0 / 0.1 / 0.3 give identical beams. At the stray policy, eps up to 0.5 does not bring the oracle into the δ = 0 MAP (0.00), and leaves δ = 1 unchanged (0.95 → 0.98 at k = 3). Keep it; state these numbers.

**N6. Learning rate.** The rule selects on δ = 1, one seed per lr, at 4k steps. The δ = 0 failures that P3 predicts (IW, RF) would then be shown at a single lr, as EX-d0 was in rung 1. Run the δ = 0 pilot controls at two lrs for IW-16, or limit P3's IW/RF claim to "at the δ = 1-optimal lr". Also report, per arm, the fraction of steps on which `grad_clip` = 1.0 binds. The estimators' surrogate scales differ (IW sums over K, RF and EX average, EI sums route steps per found query), and clipping is the one place scale survives Adam.

**N7. P4 and P5 are mostly by construction; label them checks, not predictions.**
- The δ = 0 PP line is in-support by the design's own argument.
- F30's profile follows from the per-state decisions on the oracle's support.
- Only probe (b) can differ, and only through off-route visits: EI (oracle-route imitation) should match RM-T, while IW and RF may differ.
- `crit_obj` within 0.02 nats follows from deterministic oracle routes.
- Also, "the F30 profile" inside the success criterion would count a run that generalises *better* (pp > 0) as a failure. Define success on the rule and report the profile separately.

**N8. "Reading the outcomes" is missing rows:**
- EI learns δ = 0 but not δ = 1 (or more slowly than EX);
- EI-4 learns δ = 0 (likely after B1);
- RF fails on δ = 1 by halting at the flag (B3);
- IW learns δ = 1 more slowly than 4× because the stray rule forms later from sampled SETs.

**N9. Seeds and reference.**
- AGENTS rule 4 asks for ≥ 10 paired seeds and ≈ 29 for headlines. 5 / 3 / 2 is defensible only for count predictions on near-bimodal outcomes (rung 1: 10/10, 5/5). Say so explicitly.
- P2's paired P ≥ 0.8 at 5 seeds is weak.
- Rung-1 records lack `crit_obj` (checked: `main_RMd0_s0` curve keys), so P5's reference is the 2 new EX seeds per deferral only. Make it 3.

**N10. Budget.**
- Cut: IW-4 (6 runs), EI-16 δ = 0 from 10 to 5 seeds (5 runs).
- Add: RF-p 10 runs, EI-from-basin 5, RM-T-d0 2, sample-then-imitate 6, spurious-density stress 12.
- Net about +24 runs (≈ +3.6 T4-h at 0.09 s per step). If the budget is fixed, drop the stress first and keep EI-from-basin and RF-p.

## Prior work (F)

- **Guu et al. 2017** is mischaracterised. Their "RL" maximises E[R] with a binary reward (∝ q(y)), not E[log p(y | r)]. Their result (MML > RL under spurious programs; randomized beam plus the β-meritocratic update) maps onto IW vs RF-p, with EI as β → ∞.
- **Min et al. 2019** (hard EM beats MML when spurious solutions abound) predicts the EI vs IW ordering under N3's stress, not at this toy's density.
- Missing, and closest in spirit (answer-only supervision of a latent chain):
  - TRICE (Phan, Hoffman et al. 2023): MCMC-EM with control variates for latent rationales;
  - STaR (Zelikman et al. 2022) and ReST-EM (Singh et al. 2023): filtered self-imitation = EI with sampling (N4);
  - Self-Imitation Learning (Oh et al. 2018); Abolafia et al. 2018 (priority-queue training); MeRL (Agarwal et al. 2019);
  - Kool et al. 2019/2020 (stochastic beams; sampling without replacement), relevant once π sharpens and 16 samples duplicate;
  - Rainforth et al. 2018 (tighter bounds vs gradient SNR), for the K = 1 vs K = 16 framing of RF vs IW.
- The novelty sentence ("exact references for what each estimator sees") stands. However, the δ = 0 EI result as currently designed is the expected hard-EM result, not a new one (B2).
