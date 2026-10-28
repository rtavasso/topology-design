"""Analysis of rung-1 results (experiments/route/DESIGN.md §§9–10).

    PYTHONPATH=src python experiments/route/analyze.py results/kaggle_route-main [--md out.md]

Per arm (routed and neural; LF's stability exit LR and its training-depth evaluation as their own rows): route /
answer / marginal accuracy and steps by stratum (run mean ± s.e.); the preregistered paired comparisons (P1–P8) as
run-level seed bootstraps with per-seed counts; the residual study (p(HALT) and entropy by remaining distance, the
probe R²); pooling for selection (accuracy on P-at-depth-d words along training); steps to criterion by k;
extrapolation over training. Verdicts here are descriptive; §10 fixes the thresholds.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

STRATA_ORDER = ([f"in{k}" for k in range(1, 9)] + [f"len{k}" for k in (10, 12, 16, 20, 24, 32)]
                + ["pp4", "pp6", "pp8", "alt4", "alt6", "alt8", "deepP8", "pp16", "pp32", "alt16", "alt32"])
LR_PRIMARY = "0.01"
CRIT = 0.95


def load(folders):
    recs = []
    for fo in folders:
        for p in sorted(glob.glob(os.path.join(fo, "results", "*.json.gz")) + glob.glob(os.path.join(fo, "*.json.gz"))):
            with gzip.open(p, "rt") as f:
                recs.append(json.load(f))
    return recs


def group_key(r):
    cfg = r["cfg"]; tag = r.get("arm", "?"); extras = []
    if r["name"].startswith("pilot"):
        extras.append(f"lr={cfg.get('lr')}")
        if "lam" in cfg and r.get("router"):
            extras.append(f"lam={cfg['lam']:g}")
    return tag + ("[" + ",".join(extras) + "]" if extras else "")


def final(r):
    return r["evals"][-1]["eval"]


def mean_se(x):
    x = np.asarray([v for v in x if v is not None], float)
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


def derived_rows(r):
    rows = []
    fin = final(r)
    if "eval_T_train" in r:
        twin = dict(r); twin["evals"] = [{"eval": r["eval_T_train"]}]
        rows.append((group_key(r).replace("LF", f"LF@T{r['model']['T']}", 1), twin))
    if any("kl_exit" in v for v in fin.values() if isinstance(v, dict)):
        ins = {}
        for k, v in fin.items():
            if isinstance(v, dict) and LR_PRIMARY in v.get("kl_exit", {}):
                ins[k] = {**{kk: vv for kk, vv in v.items() if kk != "kl_exit"}, "acc": v["kl_exit"][LR_PRIMARY]["acc"],
                          "q_steps": v["kl_exit"][LR_PRIMARY]["q_steps"]}
        twin = dict(r); twin["evals"] = [{"eval": ins}]
        rows.append((group_key(r).replace("LF", "LR", 1), twin))
    return rows


def build_groups(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
        for g, twin in derived_rows(r):
            groups[g].append(twin)
    return groups


def table(groups, out, field, title, fmt="{:.2f}"):
    out(f"\n## {title} by stratum (run mean ± s.e.; n runs)\n")
    out("| arm | n | " + " | ".join(STRATA_ORDER) + " |")
    out("|---|---|" + "---|" * len(STRATA_ORDER))
    for g, rs in sorted(groups.items()):
        cells = []
        for st in STRATA_ORDER:
            m, se = mean_se([final(r).get(st, {}).get(field) for r in rs])
            cells.append((fmt.format(m) + (f"±{se:.2f}" if field != "q_steps" else "")) if not np.isnan(m) else "–")
        if all(c == "–" for c in cells):
            continue
        out(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")


def seed_of(r):
    return int(r["train"].get("seed", r["cfg"].get("seed", 0)))


def paired(groups, a, b, fa, fb=None):
    fb = fa if fb is None else fb
    da = {seed_of(r): fa(r) for r in groups.get(a, [])}
    db = {seed_of(r): fb(r) for r in groups.get(b, [])}
    common = sorted(s for s in set(da) & set(db) if da[s] is not None and db[s] is not None)
    return [da[s] for s in common], [db[s] for s in common]


def p_gt(x, y, n_boot=20000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0:
        return (float("nan"), float("nan")), float("nan")
    d = x - y
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(1)
    return (float((m > 0).mean()), float((m < 0).mean())), float(d.mean())


def at(st, field):
    return lambda r: final(r).get(st, {}).get(field)


def counts(groups, out, field, strata, lo=0.6, hi=0.9):
    out(f"\n### Per-seed counts of {field} (≥ {hi} / {lo}–{hi} / < {lo})\n")
    for g, rs in sorted(groups.items()):
        cells = []
        for st in strata:
            v = [final(r).get(st, {}).get(field) for r in rs]
            v = [x for x in v if x is not None]
            if v:
                cells.append(f"{st}: {sum(x >= hi for x in v)}/{sum(lo <= x < hi for x in v)}/{sum(x < lo for x in v)} of {len(v)}")
        if cells:
            out(f"- {g}: " + "; ".join(cells))


def preregistered(groups, out):
    out("\n## Preregistered comparisons (DESIGN.md §10; run-level bootstrap over paired seeds)\n")
    def line(label, a, b, fa, fb=None):
        x, y = paired(groups, a, b, fa, fb)
        if not x:
            out(f"- {label}: – (arms not both present)"); return
        (pg, pl), d = p_gt(x, y)
        out(f"- {label}: n = {len(x)}, mean(A − B) = {d:+.3f}, P(A > B) = {pg:.3f}, P(A < B) = {pl:.3f}")
    for st in ("in8", "pp8", "alt8", "len16", "len32", "pp16", "alt32"):
        line(f"P1/P3 route_acc@{st}: RM − RMT", "RM", "RMT", at(st, "route_acc"))
        line(f"P2/P3/P5 acc@{st}: RM − LR", "RM", "LR", at(st, "acc"))
        line(f"P5 acc@{st}: LR − S8", "LR", "S8", at(st, "acc"))
    for st in ("in4", "in8", "len16"):
        line(f"P6 steps@{st}: RMl0 − RM", "RMl0", "RM", at(st, "q_steps"))
        line(f"P6 acc@{st}: RMl0 − RM", "RMl0", "RM", at(st, "acc"))
    for st in ("in8", "pp8"):
        line(f"P7 acc@{st}: RMidx − RM", "RMidx", "RM", at(st, "acc"))
    counts(groups, out, "route_acc", ("in8", "pp8", "alt8", "deepP8", "len16", "len32", "pp16", "alt32"))
    counts(groups, out, "acc", ("in8", "pp8", "alt8", "len16", "len32", "pp16", "alt32"))


def residual(groups, out):
    out("\n## Residual study (DESIGN.md P8): p(HALT) and entropy along the committed route by remaining distance; probe R²\n")
    out("| arm | stratum | p_halt d=0 | d=1 | d≥2 | dead | entropy d=0 | d≥2 | probe R² |")
    out("|---|---|---|---|---|---|---|---|---|")
    for g, rs in sorted(groups.items()):
        if not rs[0].get("router"):
            continue
        for st in ("in8", "len16", "len32", "pp8", "alt32"):
            vals = {f: mean_se([final(r).get(st, {}).get(f) for r in rs])[0] for f in
                    ("p_halt_d0", "p_halt_d1", "p_halt_d2p", "p_halt_dead", "entropy_d0", "entropy_d2p")}
            pr = mean_se([final(r).get("probe_r2") for r in rs])[0]
            out(f"| {g} | {st} | " + " | ".join(f"{vals[f]:.2f}" if not np.isnan(vals[f]) else "–" for f in
                ("p_halt_d0", "p_halt_d1", "p_halt_d2p", "p_halt_dead", "entropy_d0", "entropy_d2p")) + f" | {pr:.2f} |")


def t95_of(rec, k, key="acc_by_k", min_n=250):
    cur = rec["curve"]
    g = lambda d: d.get(str(k), d.get(k))
    blocks, hits, n, end = [], 0.0, 0, None
    for c in cur:
        a = g(c.get(key, {})); cnt = g(c.get("n_by_k", {})) if "n_by_k" in c else (1 if a is not None else 0)
        if a is None or not cnt:
            continue
        hits += a * cnt; n += cnt; end = c["step"]
        if n >= min_n:
            blocks.append((end, hits / n)); hits, n = 0.0, 0
    if n:
        blocks.append((end, hits / n))
    for i, (s, a) in enumerate(blocks):
        if a < CRIT:
            continue
        nxt = blocks[i + 1][1] if i + 1 < len(blocks) else None
        if nxt is None or nxt >= CRIT:
            return s
    return None


def cmedian(vals):
    if not vals:
        return None
    m = float(np.median([np.inf if x is None else x for x in vals]))
    return None if not np.isfinite(m) else m


def fmt_t95(vals, budget):
    v = [x for x in vals if x is not None]; cens = len(vals) - len(v); med = cmedian(vals)
    if med is None:
        return f"> {budget}" + (f" ({cens}/{len(vals)} censored)" if v else "")
    return f"{int(med)}" + (f" [{min(v)}–{'>' + str(budget) if cens else max(v)}]" if len(vals) > 1 else "") + (f" ({cens}/{len(vals)} > {budget})" if cens else "")


def firsts(groups, out):
    out(f"\n## Steps to criterion by k (training-batch accuracy ≥ {CRIT} on ≥ 250 examples, confirmed by the next block; "
        "route accuracy for routed arms, answer accuracy for neural arms; censoring-aware median [min–max])\n")
    out("| arm | n | " + " | ".join(f"k={k}" for k in range(1, 9)) + " |")
    out("|---|---|" + "---|" * 8)
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if r.get("curve")]
        if not rs or g.startswith("LR") or "@" in g:
            continue
        budget = rs[0]["train"]["steps"]
        out(f"| {g} | {len(rs)} | " + " | ".join(fmt_t95([t95_of(r, k) for r in rs], budget) for k in range(1, 9)) + " |")


def pooling(groups, out):
    out("\n## Pooling for selection (neural arms): final-window training accuracy on words with a P at depth d, and on all-H words (d = 0)\n")
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if r.get("curve") and "acc_by_Pdepth" in r["curve"][-1]]
        if not rs:
            continue
        last = [r["curve"][-5:] for r in rs]
        cells = []
        for d in range(0, 9):
            v = [np.mean([c["acc_by_Pdepth"][str(d)] for c in L if str(d) in c["acc_by_Pdepth"]]) for L in last
                 if any(str(d) in c["acc_by_Pdepth"] for c in L)]
            n = int(np.mean([np.mean([c["n_by_Pdepth"].get(str(d), 0) for c in L]) for L in last]))
            cells.append(f"d={d}: {np.mean(v):.2f} (n/100 steps {n})" if v else f"d={d}: –")
        out(f"- {g}: " + "; ".join(cells))


def over_training(groups, out):
    out("\n## Extrapolation over training: answer accuracy at len16 / pp8 / alt32 by evaluation step (run means)\n")
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if len(r.get("evals", [])) > 1 and "@" not in g and not g.startswith("LR")]
        if not rs:
            continue
        steps = [e["step"] for e in rs[0]["evals"]]
        rs = [r for r in rs if len(r["evals"]) == len(steps)]
        line = []
        for st in ("len16", "pp8", "alt32"):
            a = np.array([[e["eval"].get(st, {}).get("acc", np.nan) for e in r["evals"]] for r in rs])
            line.append(f"{st} " + " ".join(f"{v:.2f}" for v in np.nanmean(a, 0)))
        out(f"- {g} (n={len(rs)}) steps {steps}: " + " | ".join(line))


def heads(groups, out):
    out("\n## Router heads (final): HOP/PREV/HALT column norms and biases; index tables\n")
    for g, rs in sorted(groups.items()):
        hs = [r.get("head") for r in rs if r.get("head")]
        if not hs:
            continue
        if "w_norm" in hs[0]:
            out(f"- {g}: " + "; ".join(f"|w|=({h['w_norm'][0]:.2f},{h['w_norm'][1]:.2f},{h['w_norm'][2]:.2f}) b=({h['b'][0]:+.2f},{h['b'][1]:+.2f},{h['b'][2]:+.2f})" for h in hs[:6]))
        elif "table" in hs[0]:
            out(f"- {g}: π by step (run 0, first 10): " + " ".join("(" + ",".join(f"{x:.2f}" for x in row) + ")" for row in hs[0]["table"][:10]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    recs = load(args.folders)
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"# Rung-1 results: {len(recs)} runs\n")
    groups = build_groups(recs)
    table(groups, out, "acc", "Answer accuracy (committed route for routed arms)")
    table(groups, out, "route_acc", "Route accuracy (committed route equals the oracle route)")
    table(groups, out, "marg_acc", "Marginal accuracy (argmax of the exact route mixture)")
    table(groups, out, "q_steps", "Steps per query (committed route / loops)", fmt="{:.1f}")
    preregistered(groups, out)
    residual(groups, out)
    firsts(groups, out)
    pooling(groups, out)
    over_training(groups, out)
    heads(groups, out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
