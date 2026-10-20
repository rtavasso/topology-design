"""Pick each arm's learning rate from the LR-tuning runs (PREREGISTRATION protocol step 1, amendment A2)
and write the main-study grid file.

Rule: per arm, choose the lr with the smallest median t_grok over the tuning seeds, with runs that never
grok counted as the step budget. Ties go to the lr closest to the shared default 1e-3. If every lr is
fully censored for an arm, use 1e-3 and flag it.

Usage:
    PYTHONPATH=src python experiments/grokking/select_lr.py --runs results/lr_tune --out experiments/grokking/grids/main_tuned.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ARMS = ["seq", "par", "sink_0", "sink_0.25", "sink_0.5", "sink_1", "sink_2", "sink_4"]
DEPTH_ARMS = ["seq", "par", "sink_1"]  # H_depth: weight decay 0
SHARED_LR = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--no-shared", action="store_true", help="omit the shared-lr (1e-3) grid")
    ap.add_argument("--no-depth", action="store_true", help="omit the weight-decay-0 H_depth grid")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    t = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(args.runs, "*.json")):
        r = json.load(open(f))
        if "summary" not in r:
            continue
        c, s = r["config"], r["summary"]
        t[c["arm"]][c["lr"]].append(s["t_grok"] if s["t_grok"] is not None else c["steps"])
    chosen, report = {}, {}
    for arm in ARMS:
        if arm not in t:
            raise SystemExit(f"no tuning runs for arm {arm}")
        med = {lr: float(np.median(v)) for lr, v in t[arm].items()}
        budget = max(max(v) for v in t[arm].values())
        best = min(med.values())
        cands = [lr for lr, m in med.items() if m == best]
        lr = min(cands, key=lambda x: abs(math.log(x) - math.log(SHARED_LR)))
        flag = all(m >= budget for m in med.values())
        if flag:
            lr = SHARED_LR
        chosen[arm] = lr
        grid_lrs = sorted(med)
        report[arm] = {"median_t_grok_by_lr": med, "n_by_lr": {k: len(v) for k, v in t[arm].items()},
                       "chosen_lr": lr, "all_censored": flag,
                       "at_grid_boundary": lr in (grid_lrs[0], grid_lrs[-1])}
    base = {"p": 59, "train_frac": 0.4, "d": 128, "hidden": 256, "steps": 20000, "eval_every": 250,
            "stop_after_grok": 1000, "tag": args.tag}
    grids = []
    for arm in ARMS:  # tuned lr, wd 1
        grids.append({"base": {**base, "weight_decay": 1.0, "arm": arm, "lr": chosen[arm]},
                      "tie_data_seed": True, "vary": {"model_seed": seeds}})
    if not args.no_shared:  # shared lr, wd 1 (dedups with tuned)
        grids.append({"base": {**base, "weight_decay": 1.0, "lr": SHARED_LR},
                      "tie_data_seed": True, "vary": {"arm": ARMS, "model_seed": seeds}})
    if not args.no_depth:  # H_depth: wd 0 at each arm's wd-1-tuned lr (amendment A2)
        for arm in DEPTH_ARMS:
            grids.append({"base": {**base, "weight_decay": 0.0, "arm": arm, "lr": chosen[arm]},
                          "tie_data_seed": True, "vary": {"model_seed": seeds}})
    with open(args.out, "w") as f:
        json.dump({"grids": grids, "lr_selection": report}, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
