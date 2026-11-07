# Rung 4 pilot (gate G3; L30: one seed, reads nothing about the predictions)

**Grid:** `grids/pilot.json` — 24 runs, seed 100, 3,000 steps, curves every 500 steps: every arm (RT-canon, RT-post, RT-all, RT-absolute, EI16, PF, RT-examples, RT-shuffled) at lr ∈ {3e-4, 1e-3, 3e-3} (review 2, S5: no arm inherits another's lr).

**What it sets (and the only things read from it):**
1. **lr per arm** — the lr whose *training-world* eval5 masked success at 3,000 steps is highest (ties: the lower loss).
2. **steps for the main grid** — 12,000 if the training-world curves are still rising at 3,000 for any arm; otherwise the step at which every arm's training-world curve plateaued, rounded up to a multiple of 2,000, with a floor of 8,000. Curves are inspected on the training-world rows only.
3. **throughput per arm** (seconds per step, eval time) — to confirm the main grid fits one Kaggle session on 2 × T4, and the EI16 beam's found-route count is non-trivial at init and rises (a beam that finds nothing cannot train).
4. **harness sanity** — the unmasked and masked readings differ in the expected direction (masked ≥ unmasked), the failure categories are populated, no NaN losses, the per-instance records and curves are complete, the data files are found on Kaggle.

**What it does not read:** held-out worlds, the binding stratum, sizes 6–7, k = 7–8, any arm-against-arm comparison. The pilot's result file is kept (`results/kaggle_rewrite-pilot/`) and is not part of the reported seeds.

**Launch (from the researcher's machine, after `git pull` and with the dataset at `data/rewrite/`):**

    caffeinate -i .venv/bin/python experiments/fleet/kaggle/launch.py --user riley9 --rewrite-grid experiments/rewrite/grids/pilot.json --data-dir data/rewrite

**After the pilot:** `make_grids.py`'s `LR` and `STEPS` are set from the readings above (recorded in history), `grids/main.json` regenerated, and the main grid launched the same way with `--rewrite-grid experiments/rewrite/grids/main.json`.
