# Lens 3 — a gate that reads the state (EXPLORATORY)

Random tanh streams, d = 8, K = 4, step 0.5, 6 passes, 2000 inputs. Predictions in `lens3_partition.py`.

| temperature tau | sigma_max of the loop: median | 99th percentile | ratio | share of the gate-gradient norm from the tenth of inputs nearest a boundary | mean top gate |
|---|---|---|---|---|---|
| 4.0 | 3.02 | 5.44 | 1.80 | 8% | 0.31 |
| 2.0 | 3.21 | 5.75 | 1.79 | 8% | 0.36 |
| 1.0 | 3.76 | 7.87 | 2.09 | 8% | 0.47 |
| 0.5 | 4.71 | 14.82 | 3.14 | 7% | 0.63 |
| 0.25 | 5.41 | 30.44 | 5.63 | 9% | 0.79 |
| 0.1 | 6.06 | 46.72 | 7.70 | 14% | 0.91 |
| 0.05 | 7.61 | 101.65 | 13.36 | 20% | 0.95 |

**3.4** itineraries under an input perturbation of relative size 0.01 (tau = 0.25):

- inputs whose itinerary changed: 4.6%; of those, in the lowest-margin fifth: 88%
- distinct itineraries over 2000 inputs: 216 of 4096 possible
