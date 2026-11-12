# Hidden-world carrier: preliminary exact numbers

300 instances per row, seed 0; k ∈ {2, 3, 4}, accepted-step budget k + 2 (a run fails when an accepted non-optimal step uses the budget up). No model. Definitions in `prelim_info.py`.

| carrier | world entropy given (u, s), bits | consistent worlds | true world size | first-step entropy known / hidden, bits | P(first proposal rejected) | oracle: solved / kernel calls / rejections | q-greedy | mask-greedy | decisions after a verdict where exact and mask differ by TV > 0.05 | ball nodes; time per instance |
|---|---|---|---|---|---|---|---|---|---|---|
| H4, ρ = 0.5 | 2.03 | 5 of 15 | 3.1 of 4 | 0.99 / 1.20 | 0.05 | 1.00 / 3.2 / 0.0 | 0.99 / 3.3 / 0.1 | 0.98 / 3.4 / 0.2 | 12% of 695; mean TV 0.04 | 1497; 0.13s |
| H8, ρ = 0.35 | 5.88 | 82 of 255 | 4.3 of 8 | 0.99 / 1.58 | 0.15 | 1.00 / 3.6 / 0.0 | 0.90 / 3.8 / 0.5 | 0.85 / 4.1 / 0.8 | 35% of 849; mean TV 0.11 | 1497; 0.14s |
| H8, ρ = 0.5 | 6.22 | 82 of 255 | 5.1 of 8 | 0.95 / 1.57 | 0.13 | 1.00 / 3.5 / 0.0 | 0.93 / 3.8 / 0.4 | 0.90 / 4.0 / 0.7 | 33% of 832; mean TV 0.09 | 1497; 0.14s |
| H8, ρ = 0.75 | 5.16 | 82 of 255 | 6.4 of 8 | 1.01 / 1.46 | 0.09 | 1.00 / 3.3 / 0.0 | 0.95 / 3.4 / 0.3 | 0.93 / 3.6 / 0.4 | 22% of 734; mean TV 0.07 | 1497; 0.15s |

Longer proofs, k ∈ {4, 5} (75 instances per row):

| carrier | world entropy given (u, s), bits | consistent worlds | true world size | first-step entropy known / hidden, bits | P(first proposal rejected) | oracle: solved / kernel calls / rejections | q-greedy | mask-greedy | decisions after a verdict where exact and mask differ by TV > 0.05 | ball nodes; time per instance |
|---|---|---|---|---|---|---|---|---|---|---|
| H8, ρ = 0.35 | 5.39 | 57 of 255 | 4.7 of 8 | 1.13 / 1.82 | 0.12 | 1.00 / 5.1 / 0.0 | 0.93 / 5.7 / 0.6 | 0.87 / 6.0 / 1.0 | 34% of 349; mean TV 0.09 | 2295; 0.30s |
| H8, ρ = 0.5 | 5.69 | 57 of 255 | 5.3 of 8 | 1.21 / 1.83 | 0.10 | 1.00 / 5.0 / 0.0 | 0.93 / 5.4 / 0.4 | 0.89 / 5.7 / 0.9 | 34% of 332; mean TV 0.09 | 2295; 0.33s |

Held-out worlds (H8, ρ = 0.5, k ∈ {2, 3, 4}, 150 instances per row whose true world is not a training world). The generalising posterior puts its prior on all 255 worlds; the memorising posterior on the training worlds only:

| training worlds | training worlds consistent with (u, s) | generalising: solved / kernel calls / rejections | memorising | decisions where the two differ by TV > 0.05 |
|---|---|---|---|---|
| 8 | 4 | 0.92 / 3.8 / 0.4 | 0.89 / 4.2 / 0.7 | 47% of 996; mean TV 0.14 |
| 16 | 4 | 0.92 / 3.8 / 0.4 | 0.88 / 4.0 / 0.6 | 46% of 978; mean TV 0.12 |
| 32 | 8 | 0.93 / 3.8 / 0.4 | 0.91 / 4.2 / 0.7 | 46% of 978; mean TV 0.10 |
| 64 | 20 | 0.92 / 3.9 / 0.5 | 0.91 / 4.1 / 0.6 | 42% of 1010; mean TV 0.08 |
| 128 | 40 | 0.93 / 3.7 / 0.4 | 0.93 / 3.9 / 0.5 | 22% of 958; mean TV 0.06 |
