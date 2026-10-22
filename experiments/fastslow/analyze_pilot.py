"""Analyse the Phase 5 pilot (DESIGN §3, §5) and write the stage-1 grids for 5A and 5B.

    python experiments/fastslow/analyze_pilot.py results/kaggle_pilot/results

Rules, fixed in DESIGN.md before any run:
  - base lr: the largest pilot lr at which BOTH asym and sym grok (test acc >= 0.95) in >= 2 of 3 seeds within
    40,000 steps. If none qualifies the script stops: the pilot rule failed and the design has to be revisited.
  - scale-matching multipliers (5B scaled arms), at the base lr: alpha_X = RMS of branch X's output at step 20,000
    in base / in the unscaled memory arm, per seed; geometric mean over the 3 seeds.
  - between-seed SDs of the composition contrast R = c_L - c_S at t_end (mean over the last 2,000 steps), for power.
Writes pilot_summary.json and pilot_summary.md next to the results directory, and grids/stage1_5{A,B}.json.
"""

from __future__ import annotations

import json
import math
import os
import statistics as stt
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fleet"))

import make_grids as mg  # noqa: E402
from run_fleet import load_result  # noqa: E402

L_IDX, S_IDX = (1, 2, 3), (4, 5, 6)  # component indices of branch blocks (0 = embedding path)


def composition(rec_list: list[dict]) -> tuple[float, float]:
    """c_L, c_S from gen/mem averaged over the given records (DESIGN §3)."""
    def avg(key, idx):
        return stt.fmean(sum(r[key][i] for i in idx) for r in rec_list)
    out = []
    for idx in (L_IDX, S_IDX):
        g, m = avg("gen", idx), avg("mem", idx)
        out.append(g / (abs(g) + abs(m)) if (abs(g) + abs(m)) > 0 else float("nan"))
    return out[0], out[1]


def rec_at(traj: list[dict], step: int) -> dict | None:
    """The last record at or before `step` (exactly `step` for the real pilot, which evaluates every 100 steps)."""
    before = [r for r in traj if r["step"] <= step]
    return before[-1] if before else None


def main():
    rdir = sys.argv[1]
    runs = [load_result(os.path.join(rdir, f)) for f in sorted(os.listdir(rdir)) if f.endswith((".json", ".json.gz"))]
    runs = [r for r in runs if r["config"]["tag"].startswith("p5pilot")]
    if not runs:
        sys.exit(f"no pilot results in {rdir}")
    by = defaultdict(dict)  # (tag, arm, lr) -> seed -> run
    for r in runs:
        c = r["config"]
        by[(c["tag"], c["arm"], c["lr"])][c["model_seed"]] = r
    expected = len(mg.PILOT_LRS) * len(mg.PILOT_SEEDS) * 4
    lines = [f"# Phase 5 pilot\n", f"{len(runs)} of {expected} pilot runs found in `{rdir}`.\n"]
    # --- grokking table and base-lr rule
    lines.append("| lr | graph | grokked (of 3) | t_grok per seed | final test acc | collapses |")
    lines.append("|---|---|---|---|---|---|")
    ok_lr = []
    for lr in mg.PILOT_LRS:
        both = True
        for arm, name in (("pk012000", "asym"), ("pk000000", "sym")):
            seeds = by.get(("p5pilot-base", arm, lr), {})
            tg = [seeds[s]["summary"]["t_grok"] if s in seeds else "missing" for s in mg.PILOT_SEEDS]
            n = sum(1 for t in tg if isinstance(t, int))
            both &= n >= 2
            acc = [round(seeds[s]["summary"]["final_test_acc"], 3) for s in mg.PILOT_SEEDS if s in seeds]
            col = [seeds[s]["summary"]["n_collapses"] for s in mg.PILOT_SEEDS if s in seeds]
            lines.append(f"| {lr:g} | {name} | {n} | {tg} | {acc} | {col} |")
        if both:
            ok_lr.append(lr)
    if not ok_lr:
        lines.append("\n**Pilot rule failed:** no lr at which both graphs grok in >= 2 of 3 seeds. Stage 1 not written.")
        print("\n".join(lines))
        sys.exit(2)
    base_lr = max(ok_lr)
    lines.append(f"\n**Base lr = {base_lr:g}** (largest lr meeting the rule; qualifying: {ok_lr}).\n")
    # --- composition and SDs at t_end
    lines.append("| arm | lr | c_L per seed | c_S per seed | R per seed | SD(R) |")
    lines.append("|---|---|---|---|---|---|")
    sds = {}
    for (tag, arm, lr), seeds in sorted(by.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        cs = []
        for s in mg.PILOT_SEEDS:
            if s not in seeds:
                continue
            traj = seeds[s]["trajectory"]
            H = seeds[s]["config"]["steps"]
            tail = [r for r in traj if r["step"] > H - 2000]
            cs.append(composition(tail))
        R = [cl - cs_ for cl, cs_ in cs]
        sd = stt.stdev(R) if len(R) >= 2 else float("nan")
        sds[f"{tag}|{arm}|{lr:g}"] = sd
        f = lambda xs: "[" + ", ".join(f"{x:+.2f}" for x in xs) + "]"
        lines.append(f"| {tag.replace('p5pilot-', '')} {'asym' if arm == 'pk012000' else 'sym'} | {lr:g} | "
                     f"{f([c[0] for c in cs])} | {f([c[1] for c in cs])} | {f(R)} | {sd:.3f} |")
    # --- multipliers at the base lr
    alphas = {}
    lines.append("\n| memory arm | alpha_L per seed | alpha_S per seed | alpha (geometric mean) |")
    lines.append("|---|---|---|---|")
    for mem in ("mem4", "mem14"):
        base_runs = by[("p5pilot-base", "pk012000", base_lr)]
        mem_runs = by.get((f"p5pilot-{mem}", "pk012000", base_lr), {})
        per = []
        for s in mg.PILOT_SEEDS:
            if s not in base_runs or s not in mem_runs:
                continue
            t_m = min(20000, base_runs[s]["config"]["steps"] // 2)  # 20,000 at the pilot's 40k horizon
            rb, rm = rec_at(base_runs[s]["trajectory"], t_m), rec_at(mem_runs[s]["trajectory"], t_m)
            if rb is None or rm is None:
                continue
            per.append([rb["branch_rms"][q] / rm["branch_rms"][q] for q in range(2)])
        if len(per) < 2:
            sys.exit(f"multipliers for {mem}: fewer than 2 seeds with step-20,000 records")
        gm = [math.exp(stt.fmean(math.log(p[q]) for p in per)) for q in range(2)]
        alphas[mem] = gm
        lines.append(f"| {mem} | {[round(p[0], 3) for p in per]} | {[round(p[1], 3) for p in per]} | "
                     f"({gm[0]:.3f}, {gm[1]:.3f}) |")
        if any(not 0.25 <= a <= 4.0 for a in gm):
            lines.append(f"\n**Warning:** a {mem} multiplier lies outside [1/4, 4]. A fixed output multiplier that large "
                         "also changes the branch's effective step size; check the scaled arms' stability first.\n")
    # --- write grids and summary
    pa = mg.write("stage1_5A", mg.stage1_5A(base_lr))
    pb = mg.write("stage1_5B", mg.stage1_5B(base_lr, [round(x, 4) for x in alphas["mem4"]],
                                           [round(x, 4) for x in alphas["mem14"]]))
    lines.append(f"\nStage-1 grids written: `{os.path.relpath(pa)}`, `{os.path.relpath(pb)}`.")
    out_dir = os.path.dirname(os.path.abspath(rdir.rstrip("/")))
    summary = {"base_lr": base_lr, "qualifying_lrs": ok_lr, "alphas": alphas, "sd_R": sds, "n_runs": len(runs)}
    json.dump(summary, open(os.path.join(out_dir, "pilot_summary.json"), "w"), indent=1)
    open(os.path.join(out_dir, "pilot_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
