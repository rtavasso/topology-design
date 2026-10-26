# The "priors" round — results

Scored against [PREREGISTRATION.md](PREREGISTRATION.md). The preregistration and the analysis code were committed before any run.

**Run details**
- Kaggle, 2× T4, 25 Oct 2026.
- 634 of 634 runs: q2 170, q1u 176, q1s 288.
- The fleet tests and the round's instrument tests passed on the GPU before each job.

**Independent check:** a separate agent recomputed every number from the raw files with its own code before reading `analyze.py`. All preregistered numbers and verdicts matched. Its sensitivity analysis changes how some results are *stated*; see below.

**Data:** [results_data/](results_data/README.md) holds the summaries, a compact copy of every run, and the verification scripts.

---

## Q1 — is the data gap the prior or the search?

**Ungrokking.** Seeds that kept the rule (test ≥ 0.95 at every eval over the last 2,000 steps), out of 4, by subset size in pairs:

| graph, width | 44–435 | 522 | 696 | 870 | 1,044 | 1,218 |
|---|---|---|---|---|---|---|
| sym base | 0 | 0 | 1 | 4 | 3 | 4 |
| sym narrow | 0 | 2 | 4 | 4 | 4 | 4 |
| sym wide | 0 | 0 | 3 | 4 | 4 | 3 |
| asym base | 0 | 0 | 0 | 0 | 1 | 4 |

**Scratch.** n_floor, the smallest subset at which 2 of 3 seeds grok (sustained) at the best learning rate:
- sym: 870 pairs at every width (base at lr 4e-3, 8e-3 and 1.6e-2; narrow and wide only at 1.6e-2);
- asym: 1,218 (lr 1e-3).

No sym width groks from scratch at 696 or fewer (0 of 12 runs per width).

| graph, width | n_crit | n_floor | S | search share | flags |
|---|---|---|---|---|---|
| sym base | 870 | 870 | **1.00** | 0.00 | — |
| sym narrow | 522 | 870 | 1.67 | 0.09 | best lr at edge; budget-sensitive |
| sym wide | 696 | 870 | 1.25 | 0.04 | best lr at edge |
| asym base | 1,218 | 1,218 | 1.00 | 0.00 | budget-sensitive |

Search share = log S / log(n_floor / 3). The sym-base floor is reached at three learning rates; ties go to the lowest, so the edge flag isn't raised there.

### Verdicts (preregistered)

- **P1, ungrokking exists — supported.** 0/4 seeds keep the rule at 174 pairs; 4/4 at 1,218.
- **P2, search accounts for at most ~3× — supported.** S = 1.0 for sym base.
- **Replication check (Varma et al.: S ≈ 2) — not met** under the preregistered criterion. It is met only under the lenient criterion (S = 2.0, below).
- **P3, wiring acts on search — inconclusive (tie),** and in substance **uninformative.**
  - The ratios tie at 1.40.
  - The asym floor rests on 2 of 3 seeds at one learning rate.
  - Under first crossing, the asym floor is censored (≤ 870), which makes P3 untestable.
  - Under a criterion that ignores single-eval dips, asym n_crit is 1,044 against sym's 696. The ratio is then 1.5 against 1.4, which would falsify P3.
  - Read: asym needs more data on *both* sides, which places its extra cost in the prior at least as much as the search. That reading is exploratory.
- **P4, width ordering C (wider generalizes from less data) — not supported** under every variant tried. The Huang et al. direction never appears.
- **P4b, generic width can't close the gap — supported,** with a wide margin.

### Robustness (from the independent check)

S for sym base under alternative criteria:

| variant | S |
|---|---|
| preregistered | 1.00 (870/870) |
| grid resolution (both thresholds lie in (696, 870]) | 0.8–1.25 |
| ignore single-eval dips (mean or median of the last 2,000 steps ≥ 0.95, or ≤ 1 dip) | 1.25 |
| scratch at the ungrokking learning rate only (no min over lr) | 1.20 |
| dip-robust n_crit and lr 2e-3 together | 1.50 |
| n_crit needs ≥ 1/4 seeds | 1.25 |
| n_floor needs 3/3, ≥ 1/3, pooled over lrs, half budget, or held to the end of the run | 1.00 |
| ungrokking final accuracy ≥ 0.9 | 1.67 |
| lenient pair (median final ≥ 0.5; scratch first crossing) | 2.00 |

**Under any matched criterion, S is about 1, between 1.0 and 1.5.** S = 2 appears only when runs stuck at 50–85% test accuracy count as "keeping the rule". P2 (S ≤ 3) holds under every variant.

### What Q1 says

1. **The data gap is the prior, not the search.**
   - In the stable all-parallel network, the rule survives on a subset only about where training from scratch would find it anyway (S ≈ 1).
   - On a log scale, search accounts for 0–12% of the gap between gradient descent (870) and the program-prior learner (3); the prior accounts for the rest.
   - Below about 700–870 pairs, what AdamW, weight decay and this architecture prefer at equilibrium is memorization.
   - **Better search inside this model class can't close the gap.** That includes credit routing, competing members, growth, or better optimizers that settle at the same equilibrium.
   - Claim type: optimization, with an expressivity-like reading (the equilibrium preference).
2. **Our transformer reference had a bigger search gap.** Varma et al. found S ≈ 2 in a transformer. Here it is ≈ 1 under matched criteria, and 2 only under a criterion like theirs that counts semi-grokked runs.
3. **Width doesn't help, and the wiring's cost looks like prior.**
   - Narrowing by 14× or widening by 4× leaves the scratch floor at 870.
   - n_crit does not decrease with width. Narrow is lowest or tied (522, borderline: exactly 2/4, with both keepers drifting slowly down).
   - Base against wide (870 vs 696) is decided by single-eval dips: the window means of the "failing" base seeds are 0.985–0.996.
   - **The ordering narrow 522 < wide 696 < base 870 should not be read as real.**
4. **Semi-grokking near the threshold is a plateau, not a slow decay.**
   - Runs ending between 0.1 and 0.9 sit at 348–522 pairs (sym) and 348–870 (asym).
   - All but 3 change by less than 0.1 over 15,000 steps.
   - A longer post-shrink phase could move narrow n_crit up to 696. It would not move base or wide, which sit flat at about 0.999 at their n_crit.
5. **Exploratory:**
   - **Pairs seen in phase 1 and then dropped are held by the rule, not by memory.** Their accuracy tracks test accuracy to within 0.02–0.05, and collapses with it at small subsets.
   - **Nothing measured at the shrink predicts whether the rule is kept** (rule fraction 0.87–0.97, weight norm 36–59).
   - **Some scratch successes don't last.** 23 of 129 later dip below 0.95. Requiring the hold to the end of the run leaves the sym-base floor unchanged, raises wide's to 1,044, and censors asym's.

---

## Q2 — does the drift's direction matter? (I13)

Steps to sustained grokking, saved or lost against the paired base run. D = t(null) − t(drift).

| jump at | H | pairs | drift saves | radial null saves | D vs radial (+/−, Holm p) | random null saves | D vs random | rescale saves |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 250 | 10 | 87.5 | −25 | 112.5 (+9/−1, 0.021) | 25 | 75 (+9/−1) | −50 |
| 1,000 | 1,000 | 10 | 225 | −75 | 300 (+9/−1, 0.021) | 187.5 | **−25 (+3/−6)** | −175 |
| 2,000 | 250 | 9 | 125 | 25 | 100 (+9/−0, 0.008) | 0 | 100 (+9/−0) | 0 |
| 2,000 | 1,000 | 9 | 375 | −100 | 475 (+9/−0, 0.008) | −50 | 400 (+9/−0) | 25 |

**P5, the drift direction matters beyond per-tensor norm changes — supported.** All 4 cells are significant, and the pooled median D is 137.5 steps over 38 pairs.

**Replication:** the base and drift arms reproduce E2's runs to ≤ 5e-6 at every common eval. On E2's 100-step eval grid, the savings are exactly E2's (100 / 200 / 100 / 400).

**How to state it:**
- **The drift beats a jump matched in norm change in every cell, so its direction carries information.** But the effects are small: 100–475 steps on runs of about 2,900.
- **Part of the margin is the null hurting:** the radial part alone (the rescale arm) costs 50–175 steps at the earlier jump.
- **In one cell a plain random kick does as well as the drift** (jump at 1,000, H = 1,000).
- **The largest saving comes when the jump lands mid-transition** (jump at 2,000, H = 1,000: test accuracy 0.01 → 0.25 at the jump). There the saving is largely extrapolation along a trajectory that is already turning.
- **F16 is upheld in a narrower form:** a short-range, informative direction, worth a few percent of the walk.

Claim type: optimization.

---

## Verification

The agent recomputed all Q1 cell counts, thresholds, flags, S and verdicts, and the Q2 statistics: all match. Small definitional differences:
- 37 semi-grokked runs by the last eval, against 40 by the window mean;
- the tie-break that keeps the edge flag off for sym base.

The sensitivity analyses above are the agent's. Scripts are in `results_data/verification/`.
