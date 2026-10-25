"""Preregistered analyses for the "priors" round (PREREGISTRATION.md). Written before any main run.

    python experiments/priors/analyze.py q1 results/kaggle_q1u/results results/kaggle_q1s/results
    python experiments/priors/analyze.py q2 results/kaggle_q2/results

Each writes <first results dir>/../<exp>_summary.md and _summary.json and prints the summary. Verdicts follow
PREREGISTRATION.md exactly; anything else is labelled exploratory.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fleet"))
sys.path.insert(0, os.path.join(HERE, "..", "walk"))

import importlib.util  # noqa: E402

from run_fleet import load_result  # noqa: E402

_spec = importlib.util.spec_from_file_location("walk_analyze", os.path.join(HERE, "..", "walk", "analyze.py"))
_wa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wa)
sign_test, holm, med, fmt, sustained_grok, verdict = (_wa.sign_test, _wa.holm, _wa.med, _wa.fmt, _wa.sustained_grok,
                                                      _wa.verdict)

P2 = 59 * 59
K_STAR = 3  # program-prior reference learner (E1, F13)
FINAL_WINDOW = 5000  # ungrokking: final test accuracy = mean over the last 5,000 steps of the post-shrink phase
UNCONVERGED = 0.1  # |mean(last 2,500) - mean(previous 2,500)| above this marks a run as not converged


def load(dirs, prefix):
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".json", ".json.gz")):
                r = load_result(os.path.join(d, f))
                if r["config"]["tag"].startswith(prefix):
                    out.append(r)
    return out


def npairs(frac):
    return int(round(frac * P2))


def ungrok_row(r):
    c, s, tj = r["config"], r["summary"], r["trajectory"]
    sh = s.get("shrink") or {}
    row = {"graph": c["tag"].split("-")[1], "width": c["tag"].split("-")[2], "frac": c["shrink_frac"],
           "seed": c["model_seed"], "shrink_step": sh.get("step"), "final": None, "drop_final": None,
           "converged": None, "collapses_after": None}
    if sh.get("step") is None:
        return row
    end = sh["step"] + c["shrink_steps"]
    win = [x for x in tj if end - FINAL_WINDOW < x["step"] <= end]
    a = [x["test_acc"] for x in win if end - FINAL_WINDOW < x["step"] <= end - FINAL_WINDOW // 2]
    b = [x["test_acc"] for x in win if x["step"] > end - FINAL_WINDOW // 2]
    if not win or tj[-1]["step"] < end:
        return row  # did not finish the post-shrink phase (should not happen)
    row["final"] = sum(x["test_acc"] for x in win) / len(win)
    dr = [x["drop_acc"] for x in win if x.get("drop_acc") is not None]
    row["drop_final"] = sum(dr) / len(dr) if dr else None
    row["converged"] = bool(a and b and abs(sum(a) / len(a) - sum(b) / len(b)) <= UNCONVERGED)
    post = [x["train_acc"] for x in tj if x["step"] > sh["step"]]
    row["collapses_after"] = sum(1 for i in range(1, len(post)) if post[i] < 0.9 <= post[i - 1])
    row["weight_norm_end"] = math.sqrt(sum(v * v for v in tj[-1]["weight_norms"]))
    return row


def level_threshold(levels: dict, ok) -> tuple:
    """Smallest level f with ok(levels[f]) true at f and at every larger level. Returns (f or None, censoring):
    censoring 'below' if f is the smallest level (the true threshold may be lower), 'above' if no level qualifies."""
    fs = sorted(levels)
    best = None
    for f in reversed(fs):
        if levels[f] is not None and ok(levels[f]):
            best = f
        else:
            break
    if best is None:
        return None, "above"
    return best, ("below" if best == fs[0] else "")


def q1(dirs):
    U = [ungrok_row(r) for r in load(dirs, "pQ1u")]
    S = load(dirs, "pQ1s")
    lines = ["# Q1 — is the data gap the prior or the search?\n",
             f"{len(U)} ungrokking runs, {len(S)} scratch runs.\n"]
    cells = defaultdict(list)
    for x in U:
        cells[(x["graph"], x["width"], x["frac"])].append(x)
    # ---- ungrokking table and n_crit
    lines += ["## Ungrokking: final test accuracy after training on the subset (median over seeds)\n",
              "| graph | width | " + " | ".join(f"{f:g} ({npairs(f)})" for f in sorted({k[2] for k in cells})) + " |",
              "|---|---|" + "---|" * len({k[2] for k in cells})]
    gw = sorted({(k[0], k[1]) for k in cells})
    crit = {}
    excl = {}
    for g, w in gw:
        levels, row_s = {}, []
        for f in sorted({k[2] for k in cells}):
            xs = [x for x in cells[(g, w, f)] if x["final"] is not None]
            excl[(g, w, f)] = len(cells[(g, w, f)]) - len(xs)
            m = med([x["final"] for x in xs]) if xs else None
            levels[f] = m
            nc = sum(1 for x in xs if not x["converged"])
            row_s.append(f"{fmt(m)}" + (f" ({len(xs)}/{len(cells[(g, w, f)])})" if excl[(g, w, f)] else "")
                         + ("*" if nc * 4 > len(xs) else ""))
        crit[(g, w)] = level_threshold(levels, lambda m: m >= 0.5)
        lines.append(f"| {g} | {w} | " + " | ".join(row_s) + " |")
    lines.append("\n(k/n) = runs that reached sustained grokking and were shrunk; * = more than a quarter of the "
                 "cell's runs not converged (test accuracy moved by > 0.1 over the last 5,000 steps).\n")
    # ---- scratch: grok fraction per lr and n_floor
    sc = defaultdict(list)
    for r in S:
        c = r["config"]
        g, w = c["tag"].split("-")[1], c["tag"].split("-")[2]
        sc[(g, w, c["lr"], c["train_frac"])].append(r)
    floor = {}
    lines += ["## Scratch: seeds grokking (first crossing of 0.95) per learning rate\n",
              "| graph | width | lr | " + " | ".join(f"{f:g}" for f in sorted({k[3] for k in sc})) + " |",
              "|---|---|---|" + "---|" * len({k[3] for k in sc})]
    for g, w in sorted({(k[0], k[1]) for k in sc}):
        best = None
        for lr in sorted({k[2] for k in sc if k[:2] == (g, w)}):
            levels = {}
            cellstr = []
            for f in sorted({k[3] for k in sc}):
                rs = sc.get((g, w, lr, f))
                if rs is None:
                    cellstr.append("")
                    continue
                k = sum(1 for r in rs if r["summary"]["grokked"])
                levels[f] = k / len(rs)
                cellstr.append(f"{k}/{len(rs)}")
            fl_, cen = level_threshold(levels, lambda v: v >= 2 / 3 - 1e-9)
            if fl_ is not None and (best is None or fl_ < best[0]):
                best = (fl_, cen, lr)
            lines.append(f"| {g} | {w} | {lr:g} | " + " | ".join(cellstr) + " |")
        floor[(g, w)] = best if best is not None else (None, "above", None)
    # ---- decomposition
    lines += ["", "## Decomposition\n",
              "| graph | width | n_crit (ungrokking) | n_floor (scratch, best lr) | S = n_floor / n_crit | search share "
              "log S / log(n_floor / 3) |", "|---|---|---|---|---|---|"]
    dec = {}

    def show(f, cen):
        if f is None:
            return "> top level"
        return ("≤ " if cen == "below" else "") + f"{npairs(f)} ({f:g})"

    for g, w in sorted(set(crit) | set(floor)):
        fc, cc = crit.get((g, w), (None, "above"))
        ff, cf, lr = floor.get((g, w), (None, "above", None))
        Sv = share = None
        clean = fc is not None and ff is not None and cc == "" and cf == ""
        if fc is not None and ff is not None:
            Sv = npairs(ff) / npairs(fc)
            share = math.log(Sv) / math.log(npairs(ff) / K_STAR) if Sv > 0 else None
        dec[(g, w)] = {"n_crit": npairs(fc) if fc else None, "crit_cens": cc, "n_floor": npairs(ff) if ff else None,
                       "floor_cens": cf, "floor_lr": lr, "S": Sv, "share": share, "clean": clean}
        lines.append(f"| {g} | {w} | {show(fc, cc)} | {show(ff, cf)}" + (f" at lr {lr:g}" if lr else "") +
                     f" | {fmt(Sv)}{'' if clean else ' (censored)'} | {fmt(share)} |")
    # ---- verdicts
    bs = dec.get(("sym", "base"), {})
    lv = {f: med([x["final"] for x in cells[("sym", "base", f)] if x["final"] is not None])
          for f in sorted({k[2] for k in cells if k[:2] == ("sym", "base")})}
    p1_test = 0.05 in lv and 0.35 in lv and lv[0.05] is not None and lv[0.35] is not None
    p1 = p1_test and lv[0.05] < 0.5 and lv[0.35] >= 0.9
    p2_test = bool(bs) and bs["clean"]
    p2 = p2_test and 1.5 <= bs["S"] <= 3.0 and bs["share"] < 0.25
    a_, s_ = dec.get(("asym", "base"), {}), dec.get(("sym", "base"), {})
    p3_test = bool(a_ and s_ and a_.get("clean") and s_.get("clean") and a_["n_floor"] > s_["n_floor"])
    p3 = p3_test and (a_["n_crit"] / s_["n_crit"]) < (a_["n_floor"] / s_["n_floor"])
    crits = {w: dec.get(("sym", w), {}).get("n_crit") for w in ("narrow", "base", "wide")}
    cens_ok = all(dec.get(("sym", w), {}).get("crit_cens") == "" for w in crits)
    order = "untestable"
    if all(v is not None for v in crits.values()) and cens_ok:
        n_, b_, w_ = crits["narrow"], crits["base"], crits["wide"]
        if n_ == b_ == w_:
            order = "flat"
        elif w_ <= b_ <= n_:
            order = "C (wider generalizes from less data)"
        elif n_ <= b_ <= w_:
            order = "R (narrower generalizes from less data)"
        else:
            order = "non-monotone"
    p4 = order.startswith("C")
    floors = [dec.get(("sym", w), {}) for w in ("narrow", "base", "wide")]
    p4b_test = all(d_.get("n_floor") is not None or d_.get("floor_cens") == "above" for d_ in floors) and bool(floors)
    p4b = p4b_test and all((d_.get("n_floor") is None) or (d_["n_floor"] > npairs(0.1) and d_["floor_cens"] != "below")
                           for d_ in floors)
    lines += ["", "## Verdicts (preregistered)",
              f"- **P1 (ungrokking exists, sym base):** median final test accuracy < 0.5 at 174 pairs and ≥ 0.9 at 1,218: "
              f"{fmt(lv.get(0.05))} and {fmt(lv.get(0.35))} → {verdict(p1, p1_test, 'cells missing')}",
              f"- **P2 (decomposition, sym base):** 1.5 ≤ S ≤ 3 and search share < 0.25: S = {fmt(bs.get('S'))}, share = "
              f"{fmt(bs.get('share'))} → {verdict(p2, p2_test, 'a threshold is censored at the edge of the grid')}",
              f"- **P3 (wiring is search, base width):** asym's floor above sym's, and n_crit(asym)/n_crit(sym) < "
              f"n_floor(asym)/n_floor(sym) → {verdict(p3, p3_test, 'censored, or asym floor not above sym floor')}",
              f"- **P4 (width, sym; prediction C from Huang et al.):** ordering of n_crit across narrow/base/wide = "
              f"{order} → {verdict(p4, order not in ('untestable',), 'censored thresholds')}",
              f"- **P4b (generic width does not close the gap):** every width's n_floor > 348 pairs (0.1) → "
              f"{verdict(p4b, p4b_test, 'missing cells')}",
              "- *Exploratory (json):* accuracy on the dropped pairs, weight norms, semi-grokked runs (final 0.1–0.9), "
              "collapses after the shrink, scratch grokking by the sustained criterion."]
    # exploratory: sustained-criterion scratch floors
    sus = defaultdict(lambda: [0, 0])
    for r in S:
        c = r["config"]
        k = (c["tag"].split("-")[1], c["tag"].split("-")[2], c["lr"], c["train_frac"])
        sus[k][1] += 1
        sus[k][0] += sustained_grok(r["trajectory"]) is not None
    js = {"ungrok": U, "decomposition": {f"{g}|{w}": v for (g, w), v in dec.items()},
          "P1": p1, "P2": p2, "P3": p3, "P4_order": order, "P4": p4, "P4b": p4b,
          "excluded": {f"{g}|{w}|{f}": v for (g, w, f), v in excl.items() if v},
          "scratch_sustained": {f"{g}|{w}|{lr}|{f}": v for (g, w, lr, f), v in sus.items()},
          "semi": sum(1 for x in U if x["final"] is not None and 0.1 <= x["final"] < 0.9)}
    return lines, js


def q2(dirs):
    allr = load(dirs, "pQ2")
    base = {r["config"]["model_seed"]: r for r in allr if r["config"]["tag"] == "pQ2base"}
    jumps = [r for r in allr if r["config"]["tag"] == "pQ2jump"]
    lines = ["# Q2 — does the drift direction matter? (I13)\n", f"{len(base)} base runs, {len(jumps)} jump runs.\n"]

    def tsus(r):
        t = sustained_grok(r["trajectory"])
        return r["config"]["steps"] if t is None else t

    cells = defaultdict(dict)  # (at, H) -> seed -> {"drift": t, "random": t}
    for r in jumps:
        c = r["config"]
        cells[(c["drift_jump_at"], c["drift_jump_h"])].setdefault(c["model_seed"], {})[c["drift_jump_dir"]] = r
    keys = sorted(cells)
    rows, pvals = [], []
    for k in keys:
        D, sv_d, sv_r = [], [], []
        for seed, d in cells[k].items():
            b = base.get(seed)
            if b is None or "drift" not in d or "random" not in d:
                continue
            tb0 = sustained_grok(b["trajectory"])
            if tb0 is not None and tb0 <= k[0]:
                continue  # base already grokked (sustained) before the jump: no information (as E2)
            td, tr_ = tsus(d["drift"]), tsus(d["random"])
            if sustained_grok(d["drift"]["trajectory"]) is None and sustained_grok(d["random"]["trajectory"]) is None:
                continue  # both censored
            tb = tsus(b)
            D.append(tr_ - td)
            sv_d.append(tb - td)
            sv_r.append(tb - tr_)
        pos, neg, p2s = sign_test(D)
        n = pos + neg
        p1s = sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else float("nan")  # one-sided, drift > random
        rows.append({"cell": k, "n": len(D), "D_med": med(D), "pos": pos, "neg": neg, "p_one": p1s,
                     "saved_drift_med": med(sv_d), "saved_random_med": med(sv_r)})
        pvals.append(p1s if not math.isnan(p1s) else 1.0)
    adj = holm(pvals) if pvals else []
    lines += ["| jump at | H | eligible pairs | median steps saved: drift / random | median D = t(random) − t(drift) | "
              "+/− | one-sided p | Holm p |", "|---|---|---|---|---|---|---|---|"]
    allD = []
    for row, pa in zip(rows, adj):
        row["p_holm"] = pa
        lines.append(f"| {row['cell'][0]} | {row['cell'][1]} | {row['n']} | {row['saved_drift_med']} / "
                     f"{row['saved_random_med']} | {row['D_med']} | +{row['pos']}/−{row['neg']} | {fmt(row['p_one'], 4)} | "
                     f"{fmt(pa, 4)} |")
    pooled = []
    for k in keys:
        for seed, d in cells[k].items():
            b = base.get(seed)
            if b is None or "drift" not in d or "random" not in d:
                continue
            tb0 = sustained_grok(b["trajectory"])
            if tb0 is not None and tb0 <= k[0]:
                continue
            if sustained_grok(d["drift"]["trajectory"]) is None and sustained_grok(d["random"]["trajectory"]) is None:
                continue
            pooled.append(tsus(d["random"]) - tsus(d["drift"]))
    sig = sum(1 for row in rows if row["n"] >= 5 and row["p_holm"] < 0.05)
    testable = sum(1 for row in rows if row["n"] >= 5) >= 3
    pm = med(pooled)
    p5 = testable and sig >= 3 and pm is not None and pm > 0
    p5_fail = pm is not None and (pm <= 0 or sig <= 1)
    v = verdict(p5, testable, "fewer than 3 cells with 5 eligible pairs")
    if testable and not p5 and not p5_fail:
        v = "INCONCLUSIVE (2 cells significant)"
    lines += ["", "## Verdict (preregistered)",
              f"- **P5 (direction matters):** drift jumps reach sustained grokking earlier than matched-norm random jumps "
              f"in ≥ 3 of 4 cells (one-sided sign test, Holm p < 0.05), pooled median D > 0: {sig} cells significant, "
              f"pooled median D = {pm} → {v}",
              "- *Exploratory (json):* random jumps vs base (does any kick help?); replication of E2's drift savings."]
    return lines, {"cells": [{**r, "cell": list(r["cell"])} for r in rows], "P5": p5, "pooled_D_median": pm}


def main():
    exp, dirs = sys.argv[1], sys.argv[2:]
    if exp in ("q1u", "q1s"):  # called by the Kaggle queue with one job's results: analyse Q1 from both jobs
        root = os.path.dirname(os.path.dirname(os.path.abspath(dirs[0].rstrip("/"))))
        dirs = [os.path.join(root, f"kaggle_{j}", "results") for j in ("q1u", "q1s")]
        exp = "q1"
    lines, js = {"q1": q1, "q2": q2}[exp](dirs)
    outdir = os.path.dirname(os.path.abspath(dirs[0].rstrip("/")))
    with open(os.path.join(outdir, f"{exp}_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(outdir, f"{exp}_summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
