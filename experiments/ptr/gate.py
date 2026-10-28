"""Gate G4b (experiments/ptr/DESIGN.md §10): decide the LH-Δ head learning-rate multiplier from the pilot-3 records
and write the gated confirmatory grid. Runs inside the Kaggle kernel between the two stages, or by hand.

    python experiments/ptr/gate.py --results results/kaggle_ptr-pilot3-rev3/results --grid experiments/ptr/grids/rev3.json \
        --out experiments/ptr/grids/rev3_gated.json --report gate.json

Rule, fixed before the runs: an arm passes if every seed reaches accuracy >= PASS_ACC at every k <= kmax (the halted
readout, final evaluation) and its query steps track k in range with a least-squares slope >= PASS_SLOPE. For LH-Δ the
multiplier chosen is the SMALLEST that passes (the one closest to every other arm); out-of-range accuracy is never
consulted. If no multiplier passes, LH-Δ enters the grid at the largest multiplier tried, labeled "failed G4b" (P1 is
then "no test"). LH-s24's pass/fail is recorded the same way; its configuration does not change.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

import numpy as np

PASS_ACC = 0.95
PASS_SLOPE = 0.8


def load(folder):
    recs = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            recs.append(json.load(f))
    return recs


def in_range_ok(rec):
    """(passes, in-range min accuracy, in-range step slope) for one record, from its final evaluation."""
    fin = rec["evals"][-1]["eval"]["in"]
    kmax = int(rec["data"]["kmax"])
    ks = [k for k in range(1, kmax + 1) if str(k) in fin or k in fin]
    get = lambda k, f: fin.get(str(k), fin.get(k, {})).get(f)
    accs = [get(k, "acc") for k in ks]
    steps = [get(k, "q_steps") for k in ks]
    if not ks or any(a is None for a in accs) or any(s is None for s in steps):
        return False, float("nan"), float("nan")
    slope = float(np.polyfit(np.array(ks, float), np.array(steps, float), 1)[0]) if len(ks) > 1 else float("nan")
    ok = min(accs) >= PASS_ACC and (np.isnan(slope) or slope >= PASS_SLOPE)
    return bool(ok), float(min(accs)), slope


def decide(records, gate_prefix="pilot3_"):
    """Returns the report dict: per arm and multiplier, the per-seed verdicts; the chosen LH-Δ multiplier."""
    report = {"rule": {"pass_acc": PASS_ACC, "pass_slope": PASS_SLOPE, "choice": "smallest passing multiplier"},
              "arms": {}}
    gate = [r for r in records if r.get("name", "").startswith(gate_prefix)]
    for r in gate:
        arm = r.get("arm", "?")
        mult = float(r.get("train", {}).get("halt_lr_mult", r.get("cfg", {}).get("halt_lr_mult", 1.0)))
        ok, acc, slope = in_range_ok(r)
        report["arms"].setdefault(arm, {}).setdefault(str(mult), []).append(
            {"name": r["name"], "seed": r.get("train", {}).get("seed"), "pass": ok, "min_acc_in_range": acc,
             "step_slope_in_range": slope})
    verdict = {}
    for arm, by_mult in report["arms"].items():
        verdict[arm] = {m: all(x["pass"] for x in runs) and len(runs) > 0 for m, runs in by_mult.items()}
    report["verdict"] = verdict
    lhd = verdict.get("LHd", {})
    passing = sorted((float(m) for m, ok in lhd.items() if ok))
    tried = sorted(float(m) for m in lhd)
    if passing:
        report["LHd_halt_lr_mult"], report["LHd_status"] = passing[0], "passed G4b"
    elif tried:
        report["LHd_halt_lr_mult"], report["LHd_status"] = tried[-1], "failed G4b (P1 is no test for LH-Δ)"
    else:
        report["LHd_halt_lr_mult"], report["LHd_status"] = None, "no gate records for LHd"
    s24 = verdict.get("LHs24", {})
    report["LHs24_status"] = ("passed G4b" if s24 and all(s24.values()) else
                              ("failed G4b (P2 is no test for LH-s24)" if s24 else "no gate records for LHs24"))
    return report


def apply(grid, report):
    """The gated grid: LH-Δ configs get the chosen multiplier; the gate outcome is recorded on every config."""
    out = []
    for c in grid:
        c = dict(c)
        if c.get("arm") == "LHd" and report.get("LHd_halt_lr_mult") is not None:
            c["halt_lr_mult"] = float(report["LHd_halt_lr_mult"])
        c["gate"] = {"LHd": report.get("LHd_status"), "LHs24": report.get("LHs24_status")}
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with the pilot-3 *.json.gz records")
    ap.add_argument("--grid", required=True, help="the confirmatory grid to gate")
    ap.add_argument("--out", required=True, help="where to write the gated grid")
    ap.add_argument("--report", default=None)
    ap.add_argument("--gate-prefix", default="pilot3_")
    args = ap.parse_args()
    report = decide(load(args.results), args.gate_prefix)
    grid = apply(json.load(open(args.grid)), report)
    with open(args.out, "w") as f:
        json.dump(grid, f, indent=0)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=1))
    for arm, by_mult in report["arms"].items():
        for m, runs in by_mult.items():
            print(f"{arm} mult {m}: " + "; ".join(f"seed {x['seed']}: {'pass' if x['pass'] else 'FAIL'} "
                                                 f"(min acc {x['min_acc_in_range']:.2f}, slope {x['step_slope_in_range']:.2f})" for x in runs))
    print(f"gated grid written to {args.out} ({len(grid)} runs); LHd halt_lr_mult = {report['LHd_halt_lr_mult']}")


if __name__ == "__main__":
    main()
