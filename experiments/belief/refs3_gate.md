<!-- draw 7 done -->
<!-- draw 8 done -->
<!-- draw 9 done -->
# Reference numbers for design revision 3 — the gate, training worlds (model-free)

H8. 60 instances per cell, three draws. Rejection probes that stay in support, on the memorising posterior's own greedy trajectory. The statistic is the sum of the three coefficients of the model's change on (dq_T, dq, do): updating by any of the three non-deleting mechanisms. Deleting models read 0 by construction.

| draw | training worlds | problems with a consistent training world | probes per instance | probes that would leave support (not used) | signal per instance, Σ\|dq_T\|² | memorising posterior | exact full posterior | planner | product-of-marginals prior | half-way updater | stand-in at 0 / 0.5 / 1 | sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 4 | 58% | 2.2 | 5.6 | 0.198 | 1.00 | 1.00 | 1.00 | 1.08 | 0.50 | -0.11 / 0.41 / 1.07 | 0.068 |
| 7 | 8 | 81% | 4.0 | 5.8 | 0.402 | 1.00 | 1.00 | 1.00 | 0.97 | 0.50 | -0.17 / 0.35 / 0.95 | 0.031 |
| 7 | 16 | 92% | 7.9 | 4.5 | 0.462 | 1.00 | 1.00 | 1.00 | 1.05 | 0.50 | -0.16 / 0.49 / 0.99 | 0.065 |
| 7 | 32 | 100% | 10.2 | 3.5 | 0.442 | 1.00 | 1.00 | 1.00 | 0.98 | 0.50 | 0.12 / 0.49 / 1.01 | 0.047 |
| 7 | 64 | 100% | 13.3 | 2.5 | 0.504 | 1.00 | 1.00 | 1.00 | 1.02 | 0.50 | 0.00 / 0.50 / 1.06 | 0.036 |
| 8 | 4 | 92% | 2.8 | 7.7 | 0.220 | 1.00 | 1.00 | 1.00 | 0.84 | 0.50 | -0.09 / 0.54 / 0.76 | 0.079 |
| 8 | 8 | 97% | 7.0 | 5.4 | 0.570 | 1.00 | 1.00 | 1.00 | 0.91 | 0.50 | -0.03 / 0.42 / 1.05 | 0.034 |
| 8 | 16 | 97% | 9.8 | 3.9 | 0.547 | 1.00 | 1.00 | 1.00 | 1.03 | 0.50 | 0.01 / 0.50 / 0.98 | 0.046 |
| 8 | 32 | 100% | 11.4 | 2.8 | 0.537 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.04 / 0.42 / 0.92 | 0.049 |
| 8 | 64 | 100% | 12.7 | 2.7 | 0.457 | 1.00 | 1.00 | 1.00 | 0.99 | 0.50 | -0.01 / 0.44 / 0.97 | 0.052 |
| 9 | 4 | 90% | 3.3 | 10.0 | 0.214 | 1.00 | 1.00 | 1.00 | 0.97 | 0.50 | -0.09 / 0.56 / 0.90 | 0.120 |
| 9 | 8 | 98% | 6.4 | 7.0 | 0.391 | 1.00 | 1.00 | 1.00 | 0.99 | 0.50 | 0.00 / 0.43 / 1.05 | 0.063 |
| 9 | 16 | 100% | 9.7 | 4.4 | 0.488 | 1.00 | 1.00 | 1.00 | 1.04 | 0.50 | 0.07 / 0.44 / 1.06 | 0.035 |
| 9 | 32 | 100% | 11.7 | 3.9 | 0.538 | 1.00 | 1.00 | 1.00 | 0.96 | 0.50 | 0.02 / 0.47 / 0.97 | 0.048 |
| 9 | 64 | 100% | 13.9 | 2.5 | 0.468 | 1.00 | 1.00 | 1.00 | 0.99 | 0.50 | -0.01 / 0.49 / 0.99 | 0.043 |
