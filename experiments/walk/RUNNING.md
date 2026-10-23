# Running "diagnose the walk"

Run from the repo root on your Mac, with the Python that has the `kaggle` CLI (e.g. `.venv/bin/python`). Pull first.

**Unattended (recommended).** Start the queue once and leave it running:

```
caffeinate -i .venv/bin/python experiments/fleet/kaggle/kaggle_queue.py --user riley9 \
    experiments/walk/grids/e1.json experiments/walk/grids/e2.json \
    experiments/walk/grids/e4.json experiments/walk/grids/e3.json
```

The queue:
- respects Kaggle's 2-GPU-session limit;
- waits for jobs already running, rather than relaunching them;
- downloads each finished job;
- relaunches a job whose results came back incomplete (it resumes, up to 2 times);
- starts the next job;
- runs each experiment's analysis.

You get a macOS notification as each job finishes. The log is in `results/kaggle_queue.log`. Stopping and restarting the queue is safe, because its state is in `results/kaggle_queue.json`. `caffeinate -i` keeps the Mac from idle-sleeping; closing the lid still sleeps it unless it's plugged into power and a display.

**By hand,** one job at a time:

Each job:
- uploads the code;
- runs the fleet tests on the GPU;
- runs the grid, split across both T4s if Kaggle gives two;
- downloads results into `results/kaggle_<grid>/results/`.

`--no-wait` pushes and returns; the same command with `--fetch` downloads later. Rerunning the same command resumes an interrupted grid.

| exp | runs | estimated T4 time (one GPU / two) | command |
|---|---|---|---|
| E1 information gap | 70 | ~1.7 h / ~1 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e1.json --no-wait` |
| E2 visibility and jumps | 120 | ~2.5 h / ~1.3 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e2.json --no-wait` |
| E3 credit by speed | 180 | ~4.5 h / ~2.3 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e3.json --no-wait` |
| E4 stability regions | 864 | ~6.5 h / ~3.3 h | `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/e4.json --no-wait` |

Times are estimates from the pilot's measured throughput (441 run-steps/s for mixed batches). With two GPUs, roughly halve them.

The four jobs are independent; launch them in any order. A free account may queue a second GPU session until the first ends.

**After each finishes:**

```
python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/walk/grids/eN.json --fetch
python experiments/walk/analyze.py eN results/kaggle_eN/results
```

Or just tell me it has downloaded, and I'll run the analysis and update the ledger.
