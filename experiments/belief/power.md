# Power of the belief experiment's scored questions (model-free)

Definitions in `power.py`. A stand-in model is a mixture of the two exact references plus noise; rung 4's routers sat 0.10–0.17 (TV) from their targets.

## Q1 (updating) on H8, the existing executor

600 instances, 1618 decisions after a verdict (2.7 per instance). Exact against mask-only update: TV > 0.05 on 32%, > 0.10 on 25%, > 0.20 on 19%; mean 0.090; mean on the informative ones 0.27. 0.14 s per instance.

Each cell: beta / vote share, for a model that makes the share lam of the exact update, at each noise level:

| model noise (TV to its own target) | lam = 0.0 | lam = 0.25 | lam = 0.5 | lam = 0.75 | lam = 1.0 | sd of beta, sd of vote at 2,000 instances |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 / 0.00 | 0.25 / 0.00 | 0.50 / 0.31 | 0.75 / 1.00 | 1.00 / 1.00 | 0.000, 0.000 |
| 0.05 | 0.00 / 0.08 | 0.24 / 0.13 | 0.47 / 0.42 | 0.73 / 0.88 | 0.98 / 0.94 | 0.002, 0.005 |
| 0.10 | -0.00 / 0.13 | 0.25 / 0.24 | 0.45 / 0.47 | 0.71 / 0.73 | 0.98 / 0.86 | 0.004, 0.008 |
| 0.15 | -0.01 / 0.22 | 0.26 / 0.34 | 0.44 / 0.46 | 0.65 / 0.65 | 0.98 / 0.86 | 0.005, 0.009 |
| 0.20 | -0.03 / 0.26 | 0.19 / 0.39 | 0.46 / 0.50 | 0.65 / 0.66 | 0.97 / 0.82 | 0.008, 0.010 |

## Q2 (coverage) on H8

500 instances whose true world is one of 64 held-out worlds; training sets nested, drawn from the other 191. Bootstrap over held-out worlds. 0.17 s per generated instance.

| training worlds | decisions | generalising against memorising: TV > 0.05 / mean TV | beta at gam = 0 / 0.5 / 1, noise 0.12 | vote at gam = 0 / 0.5 / 1 | sd of beta, sd of vote (64 worlds, these instances) |
|---|---|---|---|---|---|
| 16 | 1811 | 60% / 0.166 | 0.02 / 0.53 / 0.97 | 0.12 / 0.51 / 0.87 | 0.012, 0.016 |
| 64 | 1811 | 32% / 0.040 | -0.07 / 0.48 / 0.90 | 0.29 / 0.49 / 0.66 | 0.065, 0.024 |
| 160 | 1811 | 18% / 0.026 | 0.07 / 0.48 / 0.86 | 0.37 / 0.50 / 0.60 | 0.070, 0.027 |

## Signal against hidden bits (probe)

Same problems (seed 0); the carriers differ only in how the sixteen (rule, root / below-root) classes are grouped into hidden units.

| carrier | hidden units | world entropy given the problem, bits | decisions after a verdict per instance | TV(exact, mask) > 0.05 | mean TV | signal per instance, sum of squared differences | sd of beta at noise 0.12, 2,000 instances | s per instance |
|---|---|---|---|---|---|---|---|---|
| H4 | 4 | 2.03 | 2.3 | 13% | 0.037 | 0.056 | 0.008 | 0.20 |
| H6 | 6 | 4.10 | 2.4 | 16% | 0.049 | 0.077 | 0.006 | 0.24 |
| H8 | 8 | 6.22 | 2.8 | 33% | 0.094 | 0.138 | 0.004 | 0.24 |
| H10 | 10 | 8.26 | 3.0 | 41% | 0.121 | 0.200 | 0.003 | 0.16 |
| H12 | 12 | 10.36 | 3.1 | 48% | 0.134 | 0.232 | 0.004 | 0.20 |

## Q2 on a denser coverage grid (H8)

800 held-out-world instances (64 worlds), nested training sets; stand-in noise 0.12; bootstrap over worlds. The last columns use the first 200 and all 800 instances, to show how the error shrinks.

| training worlds | generalising against memorising: TV > 0.05 / mean TV | beta at gam = 0 / 0.5 / 1 | sd of beta, 200 instances | sd of beta, 800 instances |
|---|---|---|---|---|
| 8 | 41% / 0.160 | 0.02 / 0.51 / 1.00 | 0.017 | 0.009 |
| 16 | 58% / 0.167 | 0.02 / 0.53 / 0.99 | 0.019 | 0.011 |
| 32 | 60% / 0.148 | 0.02 / 0.52 / 1.00 | 0.021 | 0.012 |
| 64 | 32% / 0.042 | 0.04 / 0.49 / 0.92 | 0.082 | 0.028 |
| 128 | 35% / 0.041 | 0.07 / 0.46 / 1.01 | 0.075 | 0.038 |

## Q2 on a denser coverage grid (H10, 1023 worlds)

300 held-out-world instances (256 held-out worlds), nested training sets; stand-in noise 0.12; bootstrap over worlds. The last columns use the first 200 and all 300 instances, to show how the error shrinks.

| training worlds | generalising against memorising: TV > 0.05 / mean TV | beta at gam = 0 / 0.5 / 1 | sd of beta, 200 instances | sd of beta, all instances |
|---|---|---|---|---|
| 16 | 67% / 0.218 | 0.03 / 0.53 / 0.99 | 0.014 | 0.016 |
| 32 | 64% / 0.176 | 0.03 / 0.52 / 1.00 | 0.020 | 0.015 |
| 64 | 55% / 0.084 | 0.09 / 0.50 / 1.00 | 0.047 | 0.029 |
| 128 | 37% / 0.049 | 0.05 / 0.50 / 1.02 | 0.058 | 0.043 |
| 256 | 21% / 0.029 | 0.05 / 0.61 / 0.90 | 0.103 | 0.096 |

## Reading

1. **Sampling error is not the constraint on the existing executor.** For Q1 the projection coefficient is estimated to ±0.004–0.008 (one sd) from 2,000 instances, which cost about five minutes of one core; it tracks the true share of the exact update at every noise level tried (0.98 for a full updater at noise 0.10–0.15, 0.00 for a mask). For Q2 it is estimated to ±0.01 at 8–32 training worlds from 800 held-out instances.
2. **The vote count of design revision 0 would have sat on its own threshold.** A model making three quarters of the exact update scores 0.65–0.73 against a two-thirds bar at realistic noise. Replaced by the projection coefficient.
3. **Q2's informative range is short.** On H8 the generalising and memorising posteriors differ strongly with 8, 16 and 32 training worlds (mean TV 0.15–0.17) and weakly from 64 on (0.04). Three grid points, two octaves. If the model's crossover lies outside 8–32, the experiment returns a bound, not a location.
4. **More hidden units raise the signal per instance and lengthen that range a little.** From 4 to 12 units the mean separation of the exact update from a mask goes 0.037, 0.049, 0.094, 0.121, 0.134. On the ten-unit grouping the coverage signal at 64 training worlds is 0.084 against 0.042 on H8 — about one more usable grid point, at four times as many worlds to enumerate.
5. **So a larger algebra is not needed for power.** What limits the experiment is outside this analysis: seed-to-seed spread, structured model error (the stand-in's noise has no direction), and whether models of this size learn the task. The ten- and twelve-unit groupings run on the existing executor and are the fallback if the pilot shows the crossover beyond 32 training worlds.
