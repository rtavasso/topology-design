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
FINAL_WINDOW = 5000  # sensitivity criterion: mean test accuracy over the last 5,000 steps after the shrink
HOLD = 2000  # matched criterion: test accuracy >= 0.95 at every eval for 2,000 steps (both sides)
CONV_GAP = 15000  # convergence: compare the last 2,500 steps with the 2,500 ending 15,000 steps earlier
UNCONVERGED = 0.1


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
    """One ungrokking run. Primary success (matched to scratch): test accuracy >= 0.95 at every eval in the last
    2,000 steps of the post-shrink phase; a run that diverged after the shrink fails. Not shrunk -> excluded."""
    c, s, tj = r["config"], r["summary"], r["trajectory"]
    sh = s.get("shrink") or {}
    row = {"graph": c["tag"].split("-")[1], "width": c["tag"].split("-")[2], "frac": c["shrink_frac"],
           "seed": c["model_seed"], "shrink_step": sh.get("step"), "success": None, "final": None, "drop_final": None,
           "converged": None, "collapses_after": None, "rule_frac_at_shrink": None, "wn_at_shrink": None}
    if sh.get("step") is None:
        return row
    end = sh["step"] + c["shrink_steps"]
    at = [x for x in tj if x["step"] == sh["step"]]
    if at:
        row["rule_frac_at_shrink"] = at[0].get("rule_frac_test")
        row["wn_at_shrink"] = math.sqrt(sum(v * v for v in at[0]["weight_norms"]))
    if s.get("diverged_at") is not None and s["diverged_at"] > sh["step"]:
        row.update(success=False, final=0.0, converged=True)
        return row
    if tj[-1]["step"] < end:
        return row  # did not finish the post-shrink phase (should not happen); excluded and counted
    row["success"] = all(x["test_acc"] >= 0.95 for x in tj if end - HOLD <= x["step"] <= end)
    win = [x for x in tj if end - FINAL_WINDOW < x["step"] <= end]
    row["final"] = sum(x["test_acc"] for x in win) / len(win)
    dr = [x["drop_acc"] for x in win if x.get("drop_acc") is not None]
    row["drop_final"] = sum(dr) / len(dr) if dr else None
    last = [x["test_acc"] for x in tj if end - 2500 < x["step"] <= end]
    early = [x["test_acc"] for x in tj if end - CONV_GAP - 2500 < x["step"] <= end - CONV_GAP]
    row["converged"] = bool(last and early and abs(sum(last) / len(last) - sum(early) / len(early)) <= UNCONVERGED)
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


def interval(f, cens, top):
    """A threshold as an interval of pair counts, for comparisons that respect censoring."""
    if f is None:
        return (npairs(top) + 0.5, math.inf)
    if cens == "below":
        return (0.0, float(npairs(f)))
    return (float(npairs(f)), float(npairs(f)))


def scratch_success(r, budget):
    """Matched criterion: sustained onset t (>= 0.95 held 2,000 steps) with t + 2,000 <= budget."""
    t = sustained_grok(r["trajectory"], 0.95, HOLD)
    return t is not None and t + HOLD <= budget


def q1(dirs):
    U = [ungrok_row(r) for r in load(dirs, "pQ1u")]
    S = load(dirs, "pQ1s")
    lines = ["# Q1 — is the data gap the prior or the search?\n",
             f"{len(U)} ungrokking runs, {len(S)} scratch runs.\n"]
    cells = defaultdict(list)
    for x in U:
        cells[(x["graph"], x["width"], x["frac"])].append(x)
    ufracs = sorted({k[2] for k in cells})
    gw = sorted({(k[0], k[1]) for k in cells})
    crit, crit_sens, flags = {}, {}, {}
    lines += ["## Ungrokking: seeds keeping the rule (test ≥ 0.95 throughout the last 2,000 steps) / seeds shrunk\n",
              "| graph | width | " + " | ".join(f"{npairs(f)}" for f in ufracs) + " |",
              "|---|---|" + "---|" * len(ufracs)]
    for g, w in gw:
        lev, lev_sens, row_s = {}, {}, []
        n_unconv = n_inc = 0
        for f in ufracs:
            xs = [x for x in cells[(g, w, f)] if x["success"] is not None]
            k = sum(1 for x in xs if x["success"])
            lev[f] = (k / len(xs)) if xs else None
            lev_sens[f] = med([x["final"] for x in xs]) if xs else None
            n_unconv += sum(1 for x in xs if not x["converged"])
            n_inc += len(xs)
            row_s.append(f"{k}/{len(xs)}" + (f" of {len(cells[(g, w, f)])}" if len(xs) < len(cells[(g, w, f)]) else ""))
        crit[(g, w)] = level_threshold(lev, lambda v: v >= 0.5)
        crit_sens[(g, w)] = level_threshold(lev_sens, lambda v: v >= 0.5)
        flags[(g, w)] = {"unconverged": n_unconv, "runs": n_inc}
        lines.append(f"| {g} | {w} | " + " | ".join(row_s) + " |")
    lines.append("\nColumns are subset sizes in pairs. 'of n' = runs in the cell, when some never reached sustained "
                 "grokking before the deadline (excluded). A run that diverged after the shrink counts as not keeping "
                 "the rule.\n")
    # ---- scratch
    sc = defaultdict(list)
    for r in S:
        c = r["config"]
        sc[(c["tag"].split("-")[1], c["tag"].split("-")[2], c["lr"], c["train_frac"])].append(r)
    sfracs = sorted({k[3] for k in sc})
    floor, floor_half, floor_sens = {}, {}, {}
    lines += ["## Scratch: seeds with sustained grokking (≥ 0.95 held 2,000 steps) within the budget, per learning rate\n",
              "| graph | width | lr | " + " | ".join(f"{npairs(f)}" for f in sfracs) + " |",
              "|---|---|---|" + "---|" * len(sfracs)]
    for g, w in sorted({(k[0], k[1]) for k in sc}):
        lrs = sorted({k[2] for k in sc if k[:2] == (g, w)})
        best = {"full": None, "half": None, "sens": None}
        for lr in lrs:
            levs = {"full": {}, "half": {}, "sens": {}}
            cs = []
            for f in sfracs:
                rs = sc.get((g, w, lr, f))
                if rs is None:
                    cs.append("")
                    continue
                cap = rs[0]["config"]["steps"]
                kf = sum(scratch_success(r, cap) for r in rs)
                kh = sum(scratch_success(r, cap // 2) for r in rs)
                ks = sum(1 for r in rs if r["summary"]["grokked"])
                levs["full"][f], levs["half"][f], levs["sens"][f] = kf / len(rs), kh / len(rs), ks / len(rs)
                cs.append(f"{kf}/{len(rs)}")
            for key in best:
                fl_, cen = level_threshold(levs[key], lambda v: v >= 2 / 3 - 1e-9)
                if fl_ is not None and (best[key] is None or fl_ < best[key][0]):
                    best[key] = (fl_, cen, lr)
            lines.append(f"| {g} | {w} | {lr:g} | " + " | ".join(cs) + " |")
        top_lr = max(lrs)
        for key, dst in (("full", floor), ("half", floor_half), ("sens", floor_sens)):
            dst[(g, w)] = best[key] if best[key] is not None else (None, "above", None)
        flags.setdefault((g, w), {})
        flags[(g, w)]["best_lr_at_edge"] = floor[(g, w)][2] == top_lr
        flags[(g, w)]["budget_sensitive"] = floor[(g, w)][0] != floor_half[(g, w)][0]
    # ---- decomposition
    dec = {}

    def show(f, cen):
        if f is None:
            return "> top"
        return ("≤ " if cen == "below" else "") + f"{npairs(f)}"

    lines += ["", "## Decomposition (matched criteria)\n",
              "| graph | width | n_crit | n_floor (best lr) | S = n_floor / n_crit | search share | flags |",
              "|---|---|---|---|---|---|---|"]
    for g, w in sorted(set(crit) | set(floor)):
        fc, cc = crit.get((g, w), (None, "above"))
        ff, cf, lr = floor.get((g, w), (None, "above", None))
        clean = fc is not None and ff is not None and cc == "" and cf == ""
        Sv = npairs(ff) / npairs(fc) if (fc is not None and ff is not None) else None
        share = math.log(Sv) / math.log(npairs(ff) / K_STAR) if Sv else None
        fs_, cs_ = crit_sens.get((g, w), (None, "above"))
        ffs, cfs, _ = floor_sens.get((g, w), (None, "above", None))
        S_sens = npairs(ffs) / npairs(fs_) if (fs_ is not None and ffs is not None) else None
        fl = flags.get((g, w), {})
        flag_s = ", ".join(k for k in ("best_lr_at_edge", "budget_sensitive") if fl.get(k))
        if fl.get("runs") and fl.get("unconverged", 0) * 4 > fl["runs"]:
            flag_s += (", " if flag_s else "") + "unconverged>25%"
        dec[(g, w)] = {"n_crit": npairs(fc) if fc else None, "crit_cens": cc, "crit_level": fc,
                       "n_floor": npairs(ff) if ff else None, "floor_cens": cf, "floor_level": ff, "floor_lr": lr,
                       "S": Sv, "share": share, "clean": clean, "S_sensitivity_mismatched": S_sens,
                       "n_floor_half_budget": npairs(floor_half[(g, w)][0]) if floor_half.get((g, w), (None,))[0] else None,
                       "flags": flag_s}
        lines.append(f"| {g} | {w} | {show(fc, cc)} | {show(ff, cf)}" + (f" at lr {lr:g}" if lr else "") +
                     f" | {fmt(Sv)}{'' if clean else ' (censored)'} | {fmt(share)} | {flag_s} |")
    lines.append("\nSearch share = log S / log(n_floor / 3) is reported as a measurement, not a verdict.\n")
    # ---- verdicts
    bs = dec.get(("sym", "base"), {})
    provisional = bool(bs.get("flags"))
    lv = {f: [x["success"] for x in cells[("sym", "base", f)] if x["success"] is not None] for f in ufracs
          if ("sym", "base", f) in cells}
    frac_ok = lambda f: (sum(lv[f]) / len(lv[f])) if lv.get(f) else None
    p1_test = frac_ok(0.05) is not None and frac_ok(0.35) is not None
    p1 = p1_test and frac_ok(0.05) < 0.5 and frac_ok(0.35) >= 0.75
    p2_test = bool(bs) and bs["clean"]
    p2 = None
    if p2_test:
        p2 = "SUPPORTED" if bs["S"] <= 3.0 else ("FALSIFIED" if bs["S"] >= 4.0 else "INCONCLUSIVE (3 < S < 4)")
    a_, s_ = dec.get(("asym", "base"), {}), dec.get(("sym", "base"), {})
    p3_test = bool(a_ and s_ and a_.get("clean") and s_.get("clean") and a_["n_floor"] > s_["n_floor"])
    p3 = None
    if p3_test:
        rc, rf = a_["n_crit"] / s_["n_crit"], a_["n_floor"] / s_["n_floor"]
        p3 = "SUPPORTED" if rc < rf else ("INCONCLUSIVE (tie)" if rc == rf else "FALSIFIED")
    iv = {w: interval(crit.get(("sym", w), (None, "above"))[0], crit.get(("sym", w), (None, "above"))[1], max(ufracs))
          for w in ("narrow", "base", "wide") if ("sym", w) in crit}
    order = "untestable"
    if len(iv) == 3:
        le = lambda a, b: iv[a][1] <= iv[b][0]
        lt = lambda a, b: iv[a][1] < iv[b][0]
        exact = all(iv[w][0] == iv[w][1] for w in iv)
        if exact and iv["narrow"] == iv["base"] == iv["wide"]:
            order = "flat"
        elif le("wide", "base") and le("base", "narrow") and (lt("wide", "base") or lt("base", "narrow")):
            order = "C (wider generalizes from less data)"
        elif le("narrow", "base") and le("base", "wide") and (lt("narrow", "base") or lt("base", "wide")):
            order = "R (narrower generalizes from less data)"
        else:
            order = "non-monotone or indeterminate (ties/censoring)"
    floors = [dec.get(("sym", w), {}) for w in ("narrow", "base", "wide")]
    p4b_test = len([d_ for d_ in floors if d_]) == 3
    p4b = p4b_test and all((d_["n_floor"] is None) or (d_["n_floor"] > npairs(0.1) and d_["floor_cens"] != "below")
                           for d_ in floors)
    tag = " (provisional: " + bs.get("flags", "") + ")" if provisional else ""
    lines += ["## Verdicts (preregistered)",
              f"- **P1 (ungrokking exists, sym base):** fewer than half the seeds keep the rule at 174 pairs, at least "
              f"three quarters at 1,218: {fmt(frac_ok(0.05))} and {fmt(frac_ok(0.35))} → {verdict(p1, p1_test, 'cells missing')}",
              f"- **P2 (search accounts for at most ~3×, sym base):** S ≤ 3 supported, S ≥ 4 falsified: S = {fmt(bs.get('S'))}"
              f" → {p2 if p2_test else 'UNTESTABLE (a threshold is censored)'}{tag}",
              f"- *Replication check (Varma et al.: S ≈ 2):* S ≥ 1.5 → {('yes' if bs.get('S') and bs['S'] >= 1.5 else 'no') if p2_test else 'untestable'}",
              f"- **P3 (wiring acts on search, base):** n_crit(asym)/n_crit(sym) < n_floor(asym)/n_floor(sym) → "
              f"{p3 if p3_test else 'UNTESTABLE (censored, or asym floor not above sym floor)'}",
              f"- **P4 (width; prediction C):** ordering of n_crit = {order} → "
              f"{'SUPPORTED' if order.startswith('C') else ('UNTESTABLE' if order == 'untestable' else 'NOT SUPPORTED')}",
              f"- **P4b (generic width does not close the gap):** every width's n_floor > 348 pairs → "
              f"{verdict(p4b, p4b_test, 'missing widths')}",
              "- *Sensitivity:* S under the mismatched criteria (median final ≥ 0.5 / first crossing) is in the json.",
              "- *Exploratory (json):* dropped-pair accuracy, weight norms, rule fraction at the shrink, semi-grokked "
              "runs, collapses after the shrink, half-budget floors."]
    js = {"ungrok": U, "decomposition": {f"{g}|{w}": v for (g, w), v in dec.items()},
          "flags": {f"{g}|{w}": v for (g, w), v in flags.items()},
          "P1": p1, "P2": p2, "P3": p3, "P4_order": order, "P4b": p4b,
          "semi": sum(1 for x in U if x["final"] is not None and 0.1 <= x["final"] < 0.9),
          "excluded_not_shrunk": sum(1 for x in U if x["shrink_step"] is None)}
    return lines, js


def q2(dirs):
    allr = load(dirs, "pQ2")
    base = {r["config"]["model_seed"]: r for r in allr if r["config"]["tag"] == "pQ2base"}
    jumps = [r for r in allr if r["config"]["tag"] == "pQ2jump"]
    lines = ["# Q2 — does the drift direction matter? (I13)\n", f"{len(base)} base runs, {len(jumps)} jump runs.\n"]

    def tsus(r):
        t = sustained_grok(r["trajectory"])
        return r["config"]["steps"] if t is None else t

    cells = defaultdict(dict)  # (at, H) -> seed -> {dir: run}
    for r in jumps:
        c = r["config"]
        cells[(c["drift_jump_at"], c["drift_jump_h"])].setdefault(c["model_seed"], {})[c["drift_jump_dir"]] = r
    keys = sorted(cells)

    def pairs(k, null):
        out = []
        for seed, d in sorted(cells[k].items()):
            b = base.get(seed)
            if b is None or "drift" not in d or null not in d:
                continue
            tb0 = sustained_grok(b["trajectory"])
            if tb0 is not None and tb0 <= k[0]:
                continue  # base already grokked (sustained) before the jump: no information (as E2)
            if sustained_grok(d["drift"]["trajectory"]) is None and sustained_grok(d[null]["trajectory"]) is None:
                continue  # both censored
            out.append({"D": tsus(d[null]) - tsus(d["drift"]), "saved_drift": tsus(b) - tsus(d["drift"]),
                        "saved_null": tsus(b) - tsus(d[null])})
        return out

    res = {}
    for null in ("radial", "random", "rescale"):
        rows, pvals = [], []
        for k in keys:
            xs = pairs(k, null)
            D = [x["D"] for x in xs]
            pos, neg, _ = sign_test(D)
            n = pos + neg
            p1s = sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else float("nan")
            rows.append({"cell": list(k), "n": len(D), "D_med": med(D), "pos": pos, "neg": neg, "p_one": p1s,
                         "saved_drift_med": med([x["saved_drift"] for x in xs]),
                         "saved_null_med": med([x["saved_null"] for x in xs])})
            pvals.append(p1s if not math.isnan(p1s) else 1.0)
        for row, pa in zip(rows, holm(pvals) if pvals else []):
            row["p_holm"] = pa
        pooled = [x["D"] for k in keys for x in pairs(k, null)]
        res[null] = {"cells": rows, "pooled_D_median": med(pooled), "pooled_n": len(pooled)}
        lines += [f"### drift vs {null}" + (" (primary null)" if null == "radial" else " (secondary)"), "",
                  "| jump at | H | eligible pairs | median steps saved vs base: drift / null | median D = t(null) − t(drift) |"
                  " +/− | one-sided p | Holm p |", "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['cell'][0]} | {row['cell'][1]} | {row['n']} | {row['saved_drift_med']} / "
                         f"{row['saved_null_med']} | {row['D_med']} | +{row['pos']}/−{row['neg']} | {fmt(row['p_one'], 4)} | "
                         f"{fmt(row['p_holm'], 4)} |")
        lines.append(f"\nPooled median D = {med(pooled)} over {len(pooled)} pairs.\n")
    r = res["radial"]
    sig = sum(1 for row in r["cells"] if row["n"] >= 5 and row["p_holm"] < 0.05)
    testable = sum(1 for row in r["cells"] if row["n"] >= 5) >= 3
    pm = r["pooled_D_median"]
    if not testable:
        v = "UNTESTABLE (fewer than 3 cells with 5 eligible pairs)"
    elif sig >= 3 and pm is not None and pm > 0:
        v = "SUPPORTED"
    elif pm is not None and pm <= 0:
        v = "FALSIFIED"
    else:
        v = f"INCONCLUSIVE ({sig} cells significant, pooled median D > 0)"
    lines += ["## Verdict (preregistered)",
              f"- **P5 (direction matters, beyond norm changes):** drift jumps reach sustained grokking earlier than "
              f"radial-matched null jumps in ≥ 3 of 4 cells (one-sided sign test, Holm p < 0.05) with pooled median D > 0;"
              f" falsified if pooled median D ≤ 0: {sig} cells significant, pooled median D = {pm} → {v}",
              "- *Secondary:* the random and rescale-only nulls above; random vs base and E2 replication in the json."]
    return lines, {"nulls": res, "P5": v}


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
