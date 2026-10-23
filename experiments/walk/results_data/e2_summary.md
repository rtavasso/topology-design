# E2 — is the answer visible before the network produces it, and can a learner act on it?

20 base runs, 80 jump runs, 20 noise-floor runs.

| graph | runs | model 0.5 / 0.9 (median step) | refit readout 0.5 / 0.9 | linear jump-ahead H=1000 0.5 / 0.9 | sustained grok |
|---|---|---|---|---|---|
| asym | 10 | 18350.0 / 22700.0 | 18300.0 / 21550.0 | 19000.0 / 26800.0 | 10/10 |
| sym | 10 | 2550.0 / 2850.0 | 2500.0 / 2800.0 | 2100.0 / 2500.0 | 10/10 |

| graph | jump at | H | eligible | test acc before → right after | median steps saved | Holm p | median Δ collapses | noise floor (median abs Δ, H = 0) | cell supports |
|---|---|---|---|---|---|---|---|---|---|
| asym | 4000 | 250 | 10 | 0.01 → 0.01 | 1700.0 | 1.0000 | -0.5 | 0.0 | no |
| asym | 4000 | 1000 | 10 | 0.01 → 0.02 | 5900.0 | 0.3281 | 0.0 | 0.0 | no |
| asym | 8000 | 250 | 10 | 0.03 → 0.03 | 2150.0 | 1.0000 | -1.0 | 0.0 | no |
| asym | 8000 | 1000 | 10 | 0.03 → 0.03 | 5000.0 | 0.0859 | 0.5 | 0.0 | no |
| sym | 1000 | 250 | 10 | 0.00 → 0.00 | 100.0 | 0.0156 | 0.0 | 0.0 | yes |
| sym | 1000 | 1000 | 10 | 0.00 → 0.00 | 200.0 | 0.0156 | 0.0 | 0.0 | yes |
| sym | 2000 | 250 | 9 | 0.06 → 0.14 | 100 | 0.0234 | 0 | 0.0 | yes |
| sym | 2000 | 1000 | 9 | 0.06 → 0.44 | 400 | 0.0234 | 0 | 0.0 | yes |

## Verdicts (preregistered)
- **P1 (readout visibility):** refit reaches test accuracy 0.9 earlier than the model, median relative lead ≥ 0.2 with a sign test: median 0.01, +13/−1, p = 0.0018 → **NOT supported** (pilot expectation: not supported)
- **P1b (drift visibility):** the linear jump-ahead (H = 1,000) reaches 0.5 earlier than the model, median relative lead ≥ 0.1 with a sign test: median 0.08, +11/−6, p = 0.3323 → **NOT supported**
- **P2 (actionability):** at least one jump cell groks (sustained) earlier than its paired bases (Holm-adjusted sign test p < 0.05), by more than the H = 0 noise floor, without more collapses → **supported**
- *Exploratory:* the decay-aware operator (extrapd), longer horizons, and refit at 0.5 are in the json.
