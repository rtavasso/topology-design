# Running Phase 5

Commands are run from the repo root on a machine that can reach Kaggle, using the Python that has the `kaggle` CLI (for example `.venv/bin/python`). Every job:
- uploads the code and the grid;
- runs the fleet tests on the GPU (unless `--no-tests`), then the grid;
- downloads results into `results/kaggle_<grid>/results/*.json.gz`.

**Long jobs:** add `--no-wait` to push and exit. Later, the same command with `--fetch` waits for the job and downloads the results.

**Resuming:** if a job stops early (Kaggle's 12-hour limit, an error), relaunch the same command. Results already downloaded are uploaded with the code and skipped.

Times are for one T4, estimated from measured throughput (about 600 run-steps/s plain; Phase 5 features cost 1.1–1.6×).

## 1. Pilot (DESIGN §3, §5): about 50 min

```
python experiments/fleet/kaggle/launch.py --user NAME --grid experiments/fastslow/grids/pilot.json
python experiments/fastslow/analyze_pilot.py results/kaggle_pilot/results
```

The grid has 36 runs at a fixed 40k-step horizon:
- asym and sym at lr ∈ {1e-3, 2e-3, 4e-3}, 3 seeds each;
- the unscaled memory arms of asym, at the same lrs.

`analyze_pilot.py` applies the preregistered rules:
- the base lr is the largest lr at which both graphs grok in at least 2 of 3 seeds;
- the scale-matching multipliers are the branch output RMS ratios at step 20,000;
- it also reports the between-seed SDs of R.

It writes `pilot_summary.md` and the stage-1 grids `grids/stage1_5A.json` and `grids/stage1_5B.json`. If the lr rule fails, it stops and writes nothing. Commit the generated grids before launching them, so the stage-1 settings are on record.

## 2. Stage 1: 5A (about 1.5 h) and 5B (about 5.5 h)

```
python experiments/fleet/kaggle/launch.py --user NAME --grid experiments/fastslow/grids/stage1_5A.json --no-wait
python experiments/fleet/kaggle/launch.py --user NAME --grid experiments/fastslow/grids/stage1_5B.json --no-wait
```

- **5A:** 50 runs with signal measurement. asym and sym are measured per branch; pk0000, pk0120 and pk0123 per block. 10 seeds each.
- **5B:** 180 runs, 50k steps, no early stop: asym's 8 arms (20 seeds for the ★ arms, 10 for the rest) and sym's 5 arms (10 seeds).
- Each fits in one Kaggle session. Free accounts may be limited in how many GPU sessions run at once; if the second push waits or fails, launch it after the first finishes.

## 3. Later stages

- 5C's arms depend on 5A's gate.
- 5D's setting depends on 5B.
- 5E's arms depend on 5B and 5C.

Their grids are written after those analyses, as in DESIGN §9.
