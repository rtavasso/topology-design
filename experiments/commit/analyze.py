"""Analysis of rung-3 results (experiments/commit/DESIGN.md §§4, 7).

    PYTHONPATH=src python experiments/commit/analyze.py results/kaggle_route-cm_stage1 [results/kaggle_route-cm_stage2] \
        [--gate results/kaggle_route-cm_stage1/gate.json] [--md out.md]

Stage 1: the commitment gate per phase-1 run (from the gate report when given) and the core-B learning table. Stage 2:
movement under each shift (shortest-correct on P-words at in4 / in8 under the new data, time to >= 0.95 from the end of
the phase-2 warm-up, program trajectory), per-seed coverage -> movement, the dose, filling in (pp8 by set placement),
and the stability controls. Verdicts are descriptive; DESIGN.md §7 fixes the thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "route"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "routefind"))
import analyze as A  # noqa: E402  (rung 1 helpers)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("rf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "routefind", "analyze.py"))
RF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RF)   # rung 2 helpers: learned, crit_step, prog_label, plus_one

MOVE_KEY = "min_acc_P"


def moved(r, thr=0.95):
    """Phase-2 outcome: shortest-correct committed accuracy on P-words >= thr at in4 and in8 at the end (clean eval when present)."""
    ev = r["evals"][-1]; e = ev.get("clean") or ev["eval"]
    return all((e.get(st, {}).get(MOVE_KEY) or 0) >= thr for st in ("in4", "in8"))


def move_step(r, thr=0.95):
    """First held-out step (from the end of the phase-2 warm-up) with every k <= 8 at shortest-correct >= thr, confirmed."""
    w = r["train"].get("warmup", 0)
    pts = [(c["step"], min((c.get("min_acc_by_k") or {}).values(), default=0.0)) for c in r["curve"] if c.get("min_acc_by_k")]
    for i in range(len(pts) - 1):
        if pts[i][1] >= thr and pts[i + 1][1] >= thr:
            return max(0, pts[i][0] - w)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--gate", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    recs = A.load(args.folders)
    groups = {}
    for r in recs:
        groups.setdefault(r.get("arm") or "?", []).append(r)
    gate = json.load(open(args.gate)) if args.gate else None
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"# Rung-3 results: {len(recs)} runs\n")

    p1 = {g: rs for g, rs in groups.items() if g.endswith("_p1")}
    if p1:
        out("## Phase 1: commitment (clean evaluation; the gate of DESIGN §3)\n")
        out("| arm | n | learned | E / L / mixed-or-other / R | criterion | deterministic | gate passes |")
        out("|---|---|---|---|---|---|---|")
        for g, rs in sorted(p1.items()):
            lab = [RF.prog_label(r) for r in rs if RF.learned(r)]
            nE, nL = lab.count("E"), lab.count("L"); nX = len(lab) - nE - nL
            nR = sum(1 for r in rs if not RF.learned(r) and RF.plus_one(r))
            det = sum(1 for r in rs if (A.final(r).get("in8", {}).get("samp_min_acc_P") or 0) >= 0.95)
            gp = sum(1 for r in rs if gate and gate["gate"].get(r["name"], {}).get("passes")) if gate else "–"
            out(f"| {g} | {len(rs)} | {sum(RF.learned(r) for r in rs)} | {nE} / {nL} / {nX} / {nR} | {A.fmt_t95([RF.crit_step(r) for r in rs], rs[0]['train']['steps'])} | {det} | {gp} |")
        if gate:
            out("\nWhat each committed router's phase-2 targets can see (DESIGN §5.3): shift A = the δ0 oracle route on P-words; shift B = a shortest correct completion on PP words\n")
            out("| router | start | beam ε0.1 A | beam ε0.3 A | 16 samples A | own posterior A | margin HOP−PREV (mean [min,max]) | p(PREV) at the class | beam ε0.1 B | 16 samples B |")
            out("|---|---|---|---|---|---|---|---|---|---|")
            for name, sees in sorted(gate["sees"].items()):
                a, b = sees["A"], sees["B"]
                mg = a.get("margin_hop_minus_prev")
                f = lambda v: "–" if v is None else f"{v:.2f}"
                mgs = "–" if not mg else f"{mg['mean']:.1f} [{mg['min']:.1f}, {mg['max']:.1f}]"
                out(f"| {name} | {gate['gate'][name]['start']} | {f(a.get('beam_eps0.1'))} | {f(a.get('beam_eps0.3'))} | {f(a.get('samples_K16'))} | "
                    f"{f(a.get('own_posterior_on_oracle'))} | {mgs} | {f(a.get('p_prev_at_class'))} | {f(b.get('beam_eps0.1'))} | {f(b.get('samples_K16'))} |")

    cb = {g: rs for g, rs in groups.items() if g.endswith(("_d1", "_d0", "_d1p4")) and not g.startswith(("EX_", "EI16_", "PF_", "PFIS16_d0"))}
    cb.update({g: rs for g, rs in groups.items() if g in ("PFIS16_d1p4",)})
    if cb:
        out("\n## Core B: exploration for the reward signals (from scratch, constant lr, 5k)\n")
        out("| arm | n | learned | program | criterion | distinct routes / beam hit (last 1k) | ESS / explained variance |")
        out("|---|---|---|---|---|---|---|")
        for g, rs in sorted(cb.items()):
            lab = [RF.prog_label(r) for r in rs if RF.learned(r)]
            last = lambda r, k: np.mean([c["diag"][k] for c in r["curve"] if c.get("diag") and k in c["diag"] and c["step"] > r["train"]["steps"] - 1000] or [np.nan])
            d1 = [last(r, "distinct_P") for r in rs]; bh = [last(r, "beam_hit") for r in rs]
            ess = [last(r, "ess") for r in rs]; ev = [last(r, "ev") for r in rs]
            fr = lambda v: "–" if np.all(np.isnan(v)) else f"{np.nanmin(v):.2f}…{np.nanmax(v):.2f}"
            out(f"| {g} | {len(rs)} | {sum(RF.learned(r) for r in rs)} | {'/'.join(lab) or '–'} | {A.fmt_t95([RF.crit_step(r) for r in rs], rs[0]['train']['steps'])} | {fr(d1)} / {fr(bh)} | {fr(ess)} / {fr(ev)} |")

    p2 = {g: rs for g, rs in groups.items() if any(tag in g for tag in ("_EA", "_LA", "_ES", "_LS", "_LB")) or g.endswith("_d0") and g.startswith(("EX_", "EI16_", "PF_", "PFIS16_"))}
    if p2:
        out("\n## Phase 2: movement after commitment (shortest-correct on P-words under the new data; time from the end of warm-up)\n")
        out("| arm | n | moved (≥ 0.95 in4 & in8) | time to move | final in4 / in8 (mean) | program at end (E/L/mix/other) | pp8 at end | init routers |")
        out("|---|---|---|---|---|---|---|---|")
        for g, rs in sorted(p2.items()):
            fin = lambda r, st: (A.final(r).get(st, {}) if not r["evals"][-1].get("clean") else r["evals"][-1]["clean"].get(st, {}))
            i4 = np.mean([fin(r, "in4").get(MOVE_KEY) or 0 for r in rs]); i8 = np.mean([fin(r, "in8").get(MOVE_KEY) or 0 for r in rs])
            lab = [RF.prog_label(r) for r in rs]
            pp8 = np.mean([(A.final(r).get("pp8", {}).get("min_acc") or 0) for r in rs])
            inits = sorted({r["train"].get("init_from") or "scratch" for r in rs})
            out(f"| {g} | {len(rs)} | {sum(moved(r) for r in rs)}/{len(rs)} | {A.fmt_t95([move_step(r) for r in rs], rs[0]['train']['steps'])} | {i4:.2f} / {i8:.2f} | "
                f"{lab.count('E')}/{lab.count('L')}/{lab.count('mix')}/{lab.count('other') + lab.count('–')} | {pp8:.2f} | {len(inits)} |")
        if gate:
            out("\n### Per-seed: what the target saw at the checkpoint against whether the run moved (shift A)\n")
            rows = []
            for g, rs in sorted(p2.items()):
                if "_EA" not in g and "_LA" not in g:
                    continue
                sig = g.split("_")[0]
                for r in rs:
                    ck = r["train"].get("init_from")
                    name = next((n for n, gg in gate["gate"].items() if gg["ckpt"] == ck), None)
                    if not name or name not in gate["sees"]:
                        continue
                    a = gate["sees"][name]["A"]
                    cov = {"EI16": a.get("beam_eps0.1"), "EI16e3": a.get("beam_eps0.3"), "PFIS16": a.get("samples_K16"), "IW16": a.get("samples_K16"),
                           "EX": a.get("own_posterior_on_oracle"), "PF": 1.0, "RT": 1.0}.get(sig)
                    rows.append((g, r["train"]["seed"], cov, moved(r), move_step(r)))
            for g, sd, cov, mv, ms in rows:
                out(f"- {g} s{sd}: coverage {('–' if cov is None else f'{cov:.2f}')} → {'moved' if mv else 'not'} ({ms})")
            pts = [(c, m) for _, _, c, m, _ in rows if c is not None]
            if len(pts) >= 6 and any(m for _, m in pts) and any(not m for _, m in pts):
                pos = [c for c, m in pts if m]; neg = [c for c, m in pts if not m]
                auc = np.mean([[1.0 if p > n else (0.5 if p == n else 0.0) for n in neg] for p in pos])
                out(f"\nAUROC of coverage for movement over {len(pts)} runs: {auc:.2f}")
        pp = {g: rs for g, rs in p2.items() if "_LB" in g}
        if pp:
            out("\n### Filling in (shift B): pp8 set placement on shortest-correct PP routes (before-first / at-first / at-second / elsewhere)\n")
            for g, rs in sorted(pp.items()):
                hist = {}
                for r in rs:
                    for k, v in (A.final(r).get("pp8", {}).get("pp_set_placement") or {}).items():
                        hist[k] = hist.get(k, 0) + v
                out(f"- {g}: {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
    if args.md:
        open(args.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
