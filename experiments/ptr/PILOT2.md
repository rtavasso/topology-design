# Rung-0 pilot 2 (27 Oct 2026): 22 runs, Kaggle 2×T4, 1.6 h

Raw results: `results_data/pilot2_*.json.gz`. Reading: history Entry 57.

**Summary.**
- **Gate G4 passes in a narrow region.** LH-s with the spread halting init, lr 3e-4, β = 0.003 (λ_p 0.1 or 0.2) or
  β = 0.01/λ_p = 0.1: 1.00 for every k ≤ 10 (trained on k ≤ 8), steps ≈ k + 1 (slope ≈ 1). At lr 1e-3, and at
  β = 0.01/λ_p = 0.2, it collapses to immediate halting again: the basin is narrow (ACT/PonderNet's known fragility).
- **The learned policy does not extrapolate past ≈ 1.5× its training horizon.** Accuracy 0.95/0.56/0.62 at k = 12,
  0.13–0.28 at k = 16, chance beyond; steps saturate at ≈ 15 for every k ≥ 16. Mechanism: λ is 0 through step 10
  (the training horizon, T = 10) and then ramps 0.03 → 0.63 over steps 11–16 *while the register still holds a key*.
  The head reads the state in absolute terms, the state's composition drifts beyond the depths it was trained at
  (the step index is linearly decodable from the state at 0.35 against 0.03 chance), and the head's output drifts
  with it. The change-based exit (LR, Entry 56) is immune because it reads the readout's *change*, and reaches 0.97
  at k = 32 on the same block type.
- **Unique-layer depth learns hops one at a time.** S8 at lr 1e-4: k = 1 at 500 steps, k = 2 at 2–4k, k = 3 at
  9–11k, k = 4 not reached by 15k; at lr 3e-4 only k = 2 partially. S2 at lr 3e-4 never learns k = 2. The shared
  block (pilot 1) learned all eight by 4k. Each static layer must rediscover the hop, and only after the layer below
  has learned its own: a sequential curriculum with a cost that grows with depth.
- **Cost:** 0.065 s per run-step for looped arms (the sync removal changed nothing: compute-bound), 0.052 for S8.

# Rung-0 results: 22 runs

## Accuracy by k (run mean ± s.e.; n runs)

| arm | n | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=10 | k=12 | k=16 | k=20 | k=24 | k=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LHs[lr=0.0003,b=0.003,l=0.1] | 2 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 0.95±0.04 | 0.28±0.14 | 0.15±0.01 | 0.13±0.01 | 0.16±0.01 |
| LHs[lr=0.0003,b=0.003,l=0.2] | 2 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 0.99±0.01 | 0.56±0.36 | 0.13±0.02 | 0.14±0.02 | 0.14±0.00 | 0.15±0.03 |
| LHs[lr=0.0003,b=0.01,l=0.1] | 2 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 0.62±0.15 | 0.16±0.02 | 0.12±0.02 | 0.15±0.01 | 0.17±0.02 |
| LHs[lr=0.0003,b=0.01,l=0.2] | 2 | 1.00±0.00 | 0.18±0.02 | 0.18±0.01 | 0.19±0.02 | 0.19±0.00 | 0.16±0.02 | 0.19±0.01 | 0.19±0.02 | 0.21±0.00 | 0.16±0.00 | 0.20±0.02 | 0.25±0.01 | 0.22±0.01 | 0.26±0.03 |
| LHs[lr=0.001,b=0.003,l=0.1] | 2 | 1.00±0.00 | 0.13±0.00 | 0.14±0.03 | 0.16±0.01 | 0.16±0.01 | 0.12±0.02 | 0.18±0.00 | 0.16±0.04 | 0.17±0.01 | 0.13±0.01 | 0.14±0.01 | 0.16±0.04 | 0.16±0.01 | 0.23±0.01 |
| LHs[lr=0.001,b=0.003,l=0.2] | 2 | 1.00±0.00 | 0.14±0.01 | 0.14±0.01 | 0.15±0.01 | 0.18±0.00 | 0.14±0.01 | 0.19±0.01 | 0.15±0.03 | 0.16±0.00 | 0.15±0.02 | 0.15±0.01 | 0.18±0.00 | 0.19±0.02 | 0.23±0.02 |
| LHs[lr=0.001,b=0.01,l=0.1] | 2 | 1.00±0.00 | 0.15±0.01 | 0.16±0.02 | 0.16±0.01 | 0.14±0.00 | 0.16±0.02 | 0.18±0.02 | 0.16±0.02 | 0.18±0.00 | 0.14±0.03 | 0.17±0.01 | 0.19±0.01 | 0.19±0.01 | 0.20±0.01 |
| LHs[lr=0.001,b=0.01,l=0.2] | 2 | 1.00±0.00 | 0.14±0.00 | 0.15±0.01 | 0.14±0.00 | 0.17±0.01 | 0.14±0.00 | 0.18±0.02 | 0.16±0.02 | 0.16±0.01 | 0.15±0.03 | 0.18±0.01 | 0.20±0.01 | 0.18±0.00 | 0.24±0.01 |
| S2[lr=0.0003] | 2 | 1.00±0.00 | 0.18±0.00 | 0.18±0.03 | 0.17±0.04 | 0.18±0.01 | 0.18±0.03 | 0.17±0.01 | 0.17±0.02 | 0.17±0.01 | 0.19±0.04 | 0.21±0.00 | 0.24±0.02 | 0.22±0.02 | 0.26±0.00 |
| S8[lr=0.0001] | 2 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 0.17±0.03 | 0.16±0.00 | 0.17±0.01 | 0.16±0.01 | 0.18±0.00 | 0.17±0.00 | 0.20±0.01 | 0.18±0.01 | 0.22±0.02 | 0.22±0.01 | 0.22±0.01 |
| S8[lr=0.0003] | 2 | 1.00±0.00 | 0.31±0.05 | 0.17±0.01 | 0.18±0.05 | 0.18±0.01 | 0.18±0.02 | 0.15±0.02 | 0.18±0.01 | 0.15±0.03 | 0.20±0.03 | 0.21±0.00 | 0.23±0.00 | 0.22±0.02 | 0.23±0.00 |

## Query steps by k (run mean)

| arm | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=10 | k=12 | k=16 | k=20 | k=24 | k=32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LHs[lr=0.0003,b=0.003,l=0.1] | 3.9 | 4.8 | 5.5 | 6.1 | 7.0 | 7.8 | 8.0 | 9.0 | 10.2 | 12.0 | 14.9 | 14.9 | 15.1 | 15.4 |
| LHs[lr=0.0003,b=0.003,l=0.2] | 2.5 | 3.5 | 4.5 | 5.5 | 6.3 | 7.1 | 8.0 | 9.0 | 10.7 | 13.2 | 14.9 | 15.3 | 15.6 | 16.5 |
| LHs[lr=0.0003,b=0.01,l=0.1] | 4.0 | 5.0 | 5.8 | 6.2 | 7.0 | 8.0 | 8.0 | 9.0 | 10.4 | 12.0 | 12.7 | 12.8 | 13.0 | 13.5 |
| LHs[lr=0.0003,b=0.01,l=0.2] | 2.9 | 2.7 | 2.6 | 2.5 | 2.6 | 2.6 | 2.6 | 2.6 | 2.6 | 2.5 | 2.5 | 2.5 | 2.4 | 2.3 |
| LHs[lr=0.001,b=0.003,l=0.1] | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LHs[lr=0.001,b=0.003,l=0.2] | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LHs[lr=0.001,b=0.01,l=0.1] | 2.1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| LHs[lr=0.001,b=0.01,l=0.2] | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| S2[lr=0.0003] | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| S8[lr=0.0001] | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 |
| S8[lr=0.0003] | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 |

No-hop baseline accuracy ≈ 0.084

## H2: halting extrapolation gap (LHs − LO) by k


## H4 calibration: query steps vs k (least-squares slope, mean excess over k)

- LHs[lr=0.0003,b=0.003,l=0.1]: slope 0.42, mean excess -1.09 steps, max steps 20
- LHs[lr=0.0003,b=0.003,l=0.2]: slope 0.49, mean excess -1.24 steps, max steps 22
- LHs[lr=0.0003,b=0.01,l=0.1]: slope 0.32, mean excess -1.62 steps, max steps 17
- LHs[lr=0.0003,b=0.01,l=0.2]: slope -0.01, mean excess -8.15 steps, max steps 4
- LHs[lr=0.001,b=0.003,l=0.1]: slope -0.00, mean excess -9.71 steps, max steps 40
- LHs[lr=0.001,b=0.003,l=0.2]: slope -0.00, mean excess -9.71 steps, max steps 2
- LHs[lr=0.001,b=0.01,l=0.1]: slope -0.01, mean excess -9.64 steps, max steps 6
- LHs[lr=0.001,b=0.01,l=0.2]: slope -0.00, mean excess -9.68 steps, max steps 4

## Step-index probe (linear decodability of t from the normalized chasing state)

- LHs[lr=0.0003,b=0.003,l=0.1]: 0.32 (chance ≈ 0.03)
- LHs[lr=0.0003,b=0.003,l=0.2]: 0.37 (chance ≈ 0.03)
- LHs[lr=0.0003,b=0.01,l=0.1]: 0.36 (chance ≈ 0.03)
- LHs[lr=0.0003,b=0.01,l=0.2]: 0.10 (chance ≈ 0.03)
- LHs[lr=0.001,b=0.003,l=0.1]: 0.01 (chance ≈ 0.03)
- LHs[lr=0.001,b=0.003,l=0.2]: 0.01 (chance ≈ 0.03)
- LHs[lr=0.001,b=0.01,l=0.1]: 0.01 (chance ≈ 0.03)
- LHs[lr=0.001,b=0.01,l=0.2]: 0.01 (chance ≈ 0.03)
