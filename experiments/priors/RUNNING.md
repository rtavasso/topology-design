# Running the "priors" round

Run from the repo root on your Mac, with the Python that has the `kaggle` CLI (`.venv/bin/python`). Pull first.

**Unattended (recommended).** Start the queue once and leave it running:

```
caffeinate -i .venv/bin/python experiments/fleet/kaggle/kaggle_queue.py --user riley9 \
    experiments/priors/grids/q2.json experiments/priors/grids/q1u.json experiments/priors/grids/q1s.json
```

The queue works as it did for E1–E4:
- it keeps within Kaggle's 2-GPU-session limit;
- it downloads each finished job, and relaunches incomplete ones (they resume);
- it runs the analysis after each job.

Q1's analysis reads both Q1 jobs, so its verdicts are only final once `q1u` and `q1s` have both downloaded. After `q1u` alone, the scratch-side numbers show as missing.

| job | runs | what it is |
|---|---|---|
| `q2` | 170 | drift jumps vs three null jumps (radial-matched, random, rescale-only), sym; the short one |
| `q1u` | 176 | ungrokking: grok on 40%, then train on a nested subset |
| `q1s` | 288 | scratch training on the same subsets, three or four learning rates |

**By hand,** if you prefer: `python experiments/fleet/kaggle/launch.py --user riley9 --grid experiments/priors/grids/<job>.json --no-wait`, then the same command with `--fetch` once it's done.

To run the analyses yourself:
- `python experiments/priors/analyze.py q1 results/kaggle_q1u/results results/kaggle_q1s/results`
- `python experiments/priors/analyze.py q2 results/kaggle_q2/results`

Or just tell me the jobs have downloaded.
