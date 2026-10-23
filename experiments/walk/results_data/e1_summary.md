# E1 — information gap (measurement)

40 stream runs, 30 data-floor runs.

## Streams (32 new pairs per increment, after an initial 140)

| graph | lr | steps per increment | GD code length (nats) | program prior | uniform | GD / program | GD production point (examples; median) | censored |
|---|---|---|---|---|---|---|---|---|
| asym | 0.001 | 400 | 38146 | 537 | 5676 | 71.4 | 1392 | 5/5 |
| asym | 0.001 | 1600 | 39287 | 537 | 5676 | 73.0 | 1392 | 5/5 |
| asym | 0.002 | 400 | 35230 | 537 | 5676 | 66.1 | 1392 | 5/5 |
| asym | 0.002 | 1600 | 38165 | 537 | 5676 | 71.4 | 1356 | 2/5 |
| sym | 0.001 | 400 | 26064 | 537 | 5676 | 48.8 | 1392 | 3/5 |
| sym | 0.001 | 1600 | 38104 | 537 | 5676 | 70.8 | 1196 | 0/5 |
| sym | 0.002 | 400 | 25831 | 537 | 5676 | 48.4 | 1324 | 0/5 |
| sym | 0.002 | 1600 | 29773 | 537 | 5676 | 54.8 | 1100 | 0/5 |

## Data floor (fixed dataset, up to 60,000 steps)

| graph | train fraction | examples | grokked |
|---|---|---|---|
| asym | 0.15 | 522 | 0/3 |
| asym | 0.2 | 696 | 0/3 |
| asym | 0.25 | 870 | 0/3 |
| asym | 0.3 | 1044 | 0/3 |
| asym | 0.4 | 1392 | 3/3 |
| sym | 0.15 | 522 | 0/3 |
| sym | 0.2 | 696 | 0/3 |
| sym | 0.25 | 870 | 1/3 |
| sym | 0.3 | 1044 | 3/3 |
| sym | 0.4 | 1392 | 3/3 |

## Measurements
- Program prior: the rule is pinned down (posterior ≥ 0.99) after k* = [3] examples (calibration check, expected ≤ 16: passed).
- Gradient descent's code length is 44.9–87.9× the program prior's.
- Data floor (smallest training fraction where ≥ 2 of 3 seeds grok): asym: 0.4 (1392 examples, 464× k*), sym: 0.3 (1044 examples, 348× k*)
- Walk versus data: compare the production point at 400 vs 1,600 steps per increment (a drop means more walking per example helps) and both against the data floor.
