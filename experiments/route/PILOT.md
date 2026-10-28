# Rung 1 pilot (G3): rates, λ, cost — and what fell out

**Provenance.** Commit `15f0f7e` (revision 2), Kaggle session `topology-fleet-route-pilot`, 28 Oct 2026, two T4s, 20 runs (`grids/pilot.json`), 1.2 h wall; the 14 route tests passed on the kernel first. Records: `results_data/pilot/*.json.gz`; the analysis `results_data/pilot_report.md` (`analyze.py`). Seeds 0–1 throughout; 4,000 steps × batch 64 for the routed arms, 6,000 for LF, 8,000 for S8. **Pilot results are pilot results** (D94): the preregistered P1–P9 are judged on the main grid.

## G3 decisions (the preregistered rule: in-range route accuracy at 4k steps; ties go to the default)

| arm | sweep | in-range route accuracy at 4k (seed 0 / 1) | choice |
|---|---|---|---|
| RM | lr 3e-4 / 1e-3 / 3e-3 | 1.00 / 1.00; 1.00 / 1.00; 1.00 / **0.00** (one seed in the "always-SET" basin, below) | **lr 1e-3** (default) |
| RM | λ 0.005 / 0.02 / 0.1 | **0.00 / 0.00** (never halts in time: 33 + k steps); 1.00 / 1.00; 1.00 / 1.00 | **λ 0.02** (default) |
| LF | lr 3e-4 / 1e-3 | answer accuracy at T = 10: identical (all-H 1.00, P-words at chance); k = 1 / 2 criterion at 300 / 600 vs 200 / 550 steps | **lr 1e-3** (default) |
| S8 | lr 3e-5 / 1e-4 | k ≤ 1 vs k ≤ 2 learned by 8k steps | **lr 1e-4** (default) |

So the main grid runs as written in `make_grids.py`. **Cost measured:** routed arms 0.089–0.093 s per run-step (a stack of 6 and of 2; no economy of stacking), LF 0.113, S8 0.095; main ≈ 7.9 T4-hours ≈ 4 h wall on two T4s (`SEC` updated in `make_grids.py` / `run_route.py`).

## What the pilot showed (2 seeds each; the main grid gives the counts)

**RM (lr 1e-3, λ 0.02), both seeds.** Committed route accuracy 1.00 on every in-range stratum (P-words included), 1.00 at every length to k = 32 (steps = the oracle's: 39.0 at len32), 1.00 on alternation and deep-P; p(HALT) 1.00 / 0.00 / 0.00 at remaining distance 0 / 1 / ≥ 2. Order of acquisition on the held-out set: HALT, HOP by step 100; PREV at 100; **SET at 600**; every k at criterion by 600–700. RM-T (traces): SET at **500**, criterion at 600 — the exact marginal credits the setup step ≈ 100 steps (6,400 episodes) later than teacher forcing. Head column norms 0.7 / 1.7 / 0.8 / 1.8 (HOP / PREV / SET / HALT), inside the AdamW displacement bound (L24).

**PP runs: 0.00 route accuracy for every RM configuration and for RM-T**, first deviation `SET>PREV` in 497–500 of 500 failures per run. The two counterfactual probes separate the reasons: probe (a) — the oracle PREV state with the count forced to 2 — **1.00 in all 12 routed runs** (the count is read ordinally); probe (b) — the flagged state entered with a pending count — **0.00–0.06** (PREV wins the untrained precedence). Exactly review 2's finding 1 and the CPU sanity run (DESIGN §11), now at the full batch.

**RM-s (symbol-only, the F4 null).** Committed: route accuracy 0 on every P-word, answer accuracy = the all-H share (in4 0.51 / 0.46, in8 0.13 / 0.11 ≈ the state-blind reference by k), all-H words exact; on P-words the committed route never halts (steps = T_eval: it hops into decoy cycles). Marginal: argmax accuracy **1.00** on in-range P-words and at len32, p(target) 0.75–0.76 in range and 0.60–0.62 at len32; sampled routes 0.75 / 0.60. Expected steps exceed the oracle's by 1.1–1.3 per P-word. P9's three lines, as restated after review 2.

**LF (both lrs, both seeds).** At its training depth (T = 10) and under the stability exit: all-H words 1.00 at every in-range k (none8 1.00), **P-words at chance (0.11–0.23) at every depth from the first evaluation to the last** (held-out accuracy on P-at-depth-2 words 0.09–0.23 across training, no trend); at the oracle depth and at any loop no better on P-words. Run to 40 loops the block diverges (0.01–0.15 everywhere: rung 0's overthinking beyond the horizon), so LR / LF@T10 are the readings. The block learns the hop chain and nothing of the P-rule in 6,000 steps — P5's prediction, in the strong form. S8 at 8k steps reaches k ≤ 2 (F26) and P-words at chance.

**Two basins for the router, selected by the step cost.** At λ = 0.005 (both seeds) and at lr 3e-3 (one seed) the router converges to **"SET at every key state, PREV where the predecessor's pair is flagged"**: SET flips at step 100 and PREV at 400–700, answer accuracy 1.00 everywhere in range and at length (marginal 1.00 too), but the committed route is never the oracle's (first deviation `HOP>SET` at every query) and costs k extra actions. This is the locally-readable solution review 2 described for the block — the deferred flag found by a value-slot match, with the pending count used as a free resource instead of a carried bit. At λ = 0.02 and 0.1 the oracle route wins in both seeds. So λ does more here than fix the halting time: it selects between two exact-accuracy programs of different length, which is B1's claim in its weakest form (recorded as an observation; not preregistered).

## Reading, before the main grid

- P1, P2, P3a, P4 (RM flat at 1.00 for every P depth), P8 and P9 are observed at 2 seeds; P3b's expected failure (precedence) is observed; P5's LF side is observed. The main grid adds the seed counts, RM-d0 (the local-trigger control), LF-d0 (does the block learn even a one-match gate?), RM-idx (the fitted state-blind reference at the full budget), and RM-T's flip-time reference at 5 seeds.
- Nothing in the pilot changes the design beyond the rates (none changed). The "always-SET" basin is added to the record as an observation and to the main grid's reading: the first-deviation table and the steps column distinguish it from the oracle route whenever route accuracy is 0 with answer accuracy 1.
