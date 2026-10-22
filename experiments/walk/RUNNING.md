# Running "diagnose the walk"

Run from the repo root on your Mac, with the Python that has the `kaggle` CLI (e.g. `.venv/bin/python`). Pull first.

Each job:
- uploads the code;
- runs the fleet tests on the GPU;
- runs the grid, split across both T4s if Kaggle gives two;
- downloads results into `results/kaggle_<grid>/results/`.

`--no-wait` pushes and returns; the same command with `--fetch` downloads later. Rerunning the same command resumes an interrupted grid.

| exp | runs | estimated T4 time (one GPU) | command |
|---|---|---|---|
| E1 information gap | 60 | ~45 min | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e1.json --no-wait` |
| E2 visibility and jumps | 120 | ~1.5–2 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e2.json --no-wait` |
| E3 credit by speed | 100 | ~1.5 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e3.json --no-wait` |
| E4 stability regions | 648 | ~2.5–4 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e4.json --no-wait` |

Times are estimates from the pilot's measured throughput (441 run-steps/s for mixed batches). With two GPUs, roughly halve them.

The four jobs are independent; launch them in any order. A free account may queue a second GPU session until the first ends.

**After each finishes:**

```
python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/eN.json --fetch
python experiments/walk/analyze.py eN results/kaggle_eN/results
```

Or just tell me it has downloaded, and I'll run the analysis and update the ledger.
