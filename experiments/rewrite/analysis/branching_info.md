# Branching hypotheses along a proof, in bits (rewrite carrier, eval5)

Exact computations on the dataset and the executor; no model involved. Definitions in `branching_info.py`.

## Part A — the world is known

**A1 chain rule.** On 600 sampled instances, |Σ_u weight(u)·H(next state | u) − log2(#shortest state sequences)| ≤ 4.44e-15. The stored posteriors are consistent with the stored proof counts.

**A2 branching entropy by step** (bits; expected over the uniform-over-shortest-proofs walk; the action entropy, which includes several actions producing one state):

| proof length k | n | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | whole path |
|---|---|---|---|---|---|---|---|---|
| 1 | 117 | 0.02 |  |  |  |  |  | 0.02 |
| 2 | 71 | 0.47 | 0.20 |  |  |  |  | 0.67 |
| 3 | 163 | 0.99 | 0.77 | 0.22 |  |  |  | 1.98 |
| 4 | 131 | 1.22 | 1.24 | 0.85 | 0.33 |  |  | 3.65 |
| 5 | 69 | 1.34 | 1.15 | 1.06 | 0.68 | 0.22 |  | 4.45 |
| 6 | 49 | 1.17 | 1.23 | 1.07 | 0.85 | 0.54 | 0.10 | 4.96 |

**A3 reordering against different proofs.** Of 9000 instances, 6251 (69%) have more than one shortest proof. In 215 of those (3%) every shortest proof is the same proof up to the order of independent steps (one trace class).

Exact split on 400 sampled instances with several shortest proofs (uniform over shortest action sequences): path entropy 3.43 bits = 3.05 bits of *which proof* (trace class) + 0.38 bits of *which order*. Share that is order: 11%.

Of the same instances' path entropy, 0.69 bits come from several actions producing the same next state (log2 #action sequences − log2 #state sequences); the rest are proofs that pass through different terms.

| k | n | path entropy | which proof | which order |
|---|---|---|---|---|
| 1 | 5 | 1.12 | 1.12 | 0.00 |
| 2 | 44 | 1.51 | 1.38 | 0.13 |
| 3 | 102 | 2.27 | 1.98 | 0.29 |
| 4 | 123 | 3.87 | 3.46 | 0.41 |
| 5 | 76 | 4.61 | 4.15 | 0.47 |
| 6 | 50 | 4.84 | 4.25 | 0.59 |

**A4 proposition.** With the world known and a deterministic executor, the outcome of trying any action is a function of (state, goal, world): it carries 0 bits. Conditioning the posterior over next steps on "a is not taken" gives the posterior with a removed and renormalised. So on data like rung 4's, a kernel verdict fed back can be used only as a mask; anything more a model does with it is computation the model could have done without it.

## Part B — the world is hidden

600 instances (40 per true world, proof length ≥ 2). Prior: uniform over the worlds in which the goal is reachable within 8 steps.

- **Uncertainty about the world:** 7.1 consistent worlds on average, 2.68 bits.
- **Branching entropy at the first step:** 1.03 bits when the world is known, 1.44 bits when it is hidden.
- **One trial** (an action drawn from the current belief, verdict from the true kernel): rejected with probability 0.18; 0.34 bits about the world expected; world entropy after it 2.30 bits.
- **What the verdict says beyond a mask:** total variation between the exact update and the mask-only update 0.087; KL 0.132 bits.
- **Belief on the truly optimal first steps:** 0.774 before the trial, 0.817 after a mask-only update, 0.885 after the exact update.

| true world | n | consistent worlds | bits about the world per trial | P(reject) | TV(exact, mask) | mass on optimal: before | mask | exact |
|---|---|---|---|---|---|---|---|---|
| A | 40 | 9.5 | 0.40 | 0.22 | 0.090 | 0.77 | 0.84 | 0.92 |
| C | 40 | 8.5 | 0.18 | 0.11 | 0.045 | 0.89 | 0.92 | 0.95 |
| I | 40 | 9.3 | 0.54 | 0.23 | 0.105 | 0.75 | 0.79 | 0.88 |
| D | 40 | 8.6 | 0.17 | 0.11 | 0.050 | 0.89 | 0.90 | 0.95 |
| AC | 40 | 6.2 | 0.43 | 0.39 | 0.168 | 0.60 | 0.70 | 0.84 |
| AI | 40 | 8.6 | 0.58 | 0.25 | 0.132 | 0.69 | 0.75 | 0.85 |
| AD | 40 | 6.8 | 0.31 | 0.22 | 0.127 | 0.67 | 0.70 | 0.82 |
| CI | 40 | 8.2 | 0.48 | 0.20 | 0.104 | 0.75 | 0.79 | 0.86 |
| CD | 40 | 7.6 | 0.30 | 0.21 | 0.076 | 0.77 | 0.80 | 0.86 |
| ID | 40 | 6.9 | 0.45 | 0.20 | 0.081 | 0.73 | 0.80 | 0.86 |
| ACI | 40 | 7.2 | 0.37 | 0.12 | 0.070 | 0.82 | 0.86 | 0.90 |
| ACD | 40 | 4.3 | 0.23 | 0.24 | 0.101 | 0.73 | 0.80 | 0.88 |
| AID | 40 | 5.2 | 0.26 | 0.08 | 0.061 | 0.79 | 0.81 | 0.85 |
| CID | 40 | 5.3 | 0.30 | 0.12 | 0.066 | 0.81 | 0.84 | 0.87 |
| ACID | 40 | 4.0 | 0.16 | 0.00 | 0.037 | 0.95 | 0.95 | 0.97 |

Instances on which the exact update differs from the mask by TV > 0.05 in expectation: 47%.

