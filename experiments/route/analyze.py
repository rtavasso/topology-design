"""Analysis of rung-1 results (experiments/route/DESIGN.md §§9–10, revision 2).

    PYTHONPATH=src python experiments/route/analyze.py results/kaggle_route-main [--md out.md]

Per arm (routed and neural; LF's stability exit LR, its training-depth evaluation and its oracle-depth reading as
their own rows): route / answer / marginal accuracy and steps by stratum (run mean ± s.e.), split by words with and
without a P; the preregistered comparisons (P1–P9) as run-level seed bootstraps with per-seed counts; the residual
study (p(HALT) by remaining distance); the first deviation from the oracle route; pooling for selection (held-out
accuracy on P-at-depth-d words); steps to criterion by k and the SET / PREV flip times on the held-out criterion
set; extrapolation over training. Verdicts here are descriptive; §10 fixes the thresholds.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

STRATA_ORDER = ([f"in{k}" for k in range(1, 9)] + ["none3", "none4", "none8"] + [f"len{k}" for k in (10, 12, 16, 20, 24, 32)]
                + ["pp4", "pp6", "pp8", "alt4", "alt6", "alt8", "deepP8", "pp16", "pp32", "alt16", "alt32"])
LR_PRIMARY = "0.01"
CRIT = 0.95
ZIPF = (lambda k: np.arange(1, k + 1) ** -1.5 / (np.arange(1, k + 1) ** -1.5).sum())(8)


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
    """LF: its training-depth evaluation, the stability exit (LR) and the oracle-depth reading as separate rows."""
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
    if any("acc_at_k" in v for v in fin.values() if isinstance(v, dict)):
        for fld, tag in (("acc_at_k", "LF@k"), ("acc_at_k1", "LF@k+1"), ("acc_any", "LF@any")):
            ins = {k: {"acc": v[fld], "q_steps": float("nan")} for k, v in fin.items() if isinstance(v, dict) and fld in v}
            twin = dict(r); twin["evals"] = [{"eval": ins}]
            rows.append((group_key(r).replace("LF", tag, 1), twin))
    return rows


def build_groups(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
        for g, twin in derived_rows(r):
            groups[g].append(twin)
    return groups


def table(groups, out, field, title, fmt="{:.2f}", strata=STRATA_ORDER):
    out(f"\n## {title} by stratum (run mean ± s.e.; n runs)\n")
    out("| arm | n | " + " | ".join(strata) + " |")
    out("|---|---|" + "---|" * len(strata))
    for g, rs in sorted(groups.items()):
        cells = []
        for st in strata:
            m, se = mean_se([final(r).get(st, {}).get(field) for r in rs])
            cells.append((fmt.format(m) + (f"±{se:.2f}" if "steps" not in field else "")) if not np.isnan(m) else "–")
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


def zipf_in_range(field):
    def f(r):
        v = [final(r).get(f"in{k}", {}).get(field) for k in range(1, 9)]
        return None if any(x is None for x in v) else float(np.dot(ZIPF, v))
    return f


def counts(groups, out, field, strata, lo=0.5, hi=0.95):
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
    out("\n## Preregistered comparisons (DESIGN.md §10; run-level bootstrap over paired seeds; A − B)\n")
    def line(label, a, b, fa, fb=None):
        x, y = paired(groups, a, b, fa, fb)
        if not x:
            out(f"- {label}: – (arms not both present)"); return
        (pg, pl), d = p_gt(x, y)
        out(f"- {label}: n = {len(x)}, mean(A − B) = {d:+.3f}, P(A > B) = {pg:.3f}, P(A < B) = {pl:.3f}")
    def solo(label, a, fa):
        v = [fa(r) for r in groups.get(a, [])]; v = [x for x in v if x is not None]
        if v:
            out(f"- {label}: n = {len(v)}, mean {np.mean(v):.3f}, min {min(v):.3f}, max {max(v):.3f}")
    out("\n### P1 (the deferred rule is learned from the marginal; RM-T the speed reference)")
    for st in ("in4", "in8"):
        solo(f"P1 RM route_acc_P@{st}", "RM", at(st, "route_acc_P"))
        line(f"P1 route_acc_P@{st}: RM − RMT", "RM", "RMT", at(st, "route_acc_P"))
    solo("P1 RM Zipf in-range route_acc", "RM", zipf_in_range("route_acc"))
    out("\n### P2 (length: by construction once P1 holds)")
    for st in ("len16", "len32"):
        line(f"P2 route_acc@{st} − in8 (RM)", "RM", "RM", at(st, "route_acc"), at("in8", "route_acc"))
        line(f"P2 route_acc@{st} − in8 (RMd0)", "RMd0", "RMd0", at(st, "route_acc"), at("in8", "route_acc"))
    out("\n### P3 (new orders: PP runs reach pending count 2, unseen in training, under deferral 1)")
    for st in ("pp4", "pp6", "pp8", "pp16", "pp32"):
        solo(f"P3 RM route_acc@{st}", "RM", at(st, "route_acc"))
        line(f"P3 route_acc@{st} − route_acc_P@in8 (RM)", "RM", "RM", at(st, "route_acc"), at("in8", "route_acc_P"))
    line("P3 route_acc@pp8: RM − RMd0 (deferral-0 control, PP local there)", "RM", "RMd0", at("pp8", "route_acc"))
    for st in ("alt8", "deepP8"):
        line(f"frequency strata route_acc@{st} − route_acc_P@in8 (RM)", "RM", "RM", at(st, "route_acc"), at("in8", "route_acc_P"))
    out("\n### P5 (the looped block on the same tokens; LF-d0 the local-trigger control)")
    for arm in ("LF", "LFd0", "LR", "LF@k", "LF@k+1", "LF@any", "S8"):
        for st in ("in4", "in8"):
            solo(f"P5 {arm} acc_P@{st}", arm, at(st, "acc_P")); solo(f"P5 {arm} acc_noP@{st}", arm, at(st, "acc_noP"))
        solo(f"P5 {arm} acc@none8", arm, at("none8", "acc"))
    for st in ("in4", "in8"):
        line(f"P5 acc_P@{st}: LFd0 − LF", "LFd0", "LF", at(st, "acc_P"))
        line(f"P5 acc_P@{st}: LF@any − LF", "LF@any", "LF", at(st, "acc_P"))
        line(f"P5 acc_P@{st}: RM − LF", "RM", "LF", at(st, "acc_P"))
    for st in ("len16", "pp8"):
        line(f"P5 acc@{st}: RM − LR", "RM", "LR", at(st, "acc"))
        line(f"P5 acc@{st}: LR − S8", "LR", "S8", at(st, "acc"))
    out("\n### P7 (the distinguishing quantity: the state-blind reference 0.84 in both readings)")
    for arm in ("RMidx", "RM", "RMs", "RMd0", "RMT"):
        solo(f"P7 {arm} Zipf in-range acc (committed)", arm, zipf_in_range("acc"))
        solo(f"P7 {arm} Zipf in-range marg_acc", arm, zipf_in_range("marg_acc"))
    line("P7 Zipf in-range acc: RM − RMidx", "RM", "RMidx", zipf_in_range("acc"))
    line("P7 acc_P@in8: RM − RMidx", "RM", "RMidx", at("in8", "acc_P"))
    out("\n### P9 (F4 null: the symbol-only router; committed route accuracy on P-words is 0 by construction)")
    for st in ("in4", "in8", "pp8"):
        solo(f"P9 RMs route_acc_P@{st}", "RMs", at(st, "route_acc_P" if st.startswith("in") else "route_acc"))
        solo(f"P9 RMs acc_P@{st}", "RMs", at(st, "acc_P" if st.startswith("in") else "acc"))
        solo(f"P9 RMs marg_acc_P@{st}", "RMs", at(st, "marg_acc_P" if st.startswith("in") else "marg_acc"))
        solo(f"P9 RMs e_steps_P − oracle_steps_P@{st}", "RMs",
             lambda r, st=st: (lambda v: None if v.get("e_steps_P" if st.startswith("in") else "e_steps") is None else
                               v.get("e_steps_P" if st.startswith("in") else "e_steps") - v.get("oracle_steps_P" if st.startswith("in") else "oracle_steps"))(final(r).get(st, {})))
    line("P9 route_acc_noP@in8: RMs − RM", "RMs", "RM", at("in8", "route_acc_noP"))
    counts(groups, out, "route_acc", ("in4", "in8", "pp4", "pp8", "len16", "len32", "pp32", "alt32"))
    counts(groups, out, "route_acc_P", ("in4", "in8"))
    counts(groups, out, "acc_P", ("in4", "in8"))


def residual(groups, out):
    out("\n## Residual study (P8): p(HALT) and entropy along the committed route by remaining distance\n")
    out("| arm | stratum | p_halt d=0 | d=1 | d≥2 | dead | entropy d=0 | d≥2 |")
    out("|---|---|---|---|---|---|---|---|")
    for g, rs in sorted(groups.items()):
        if not rs[0].get("router"):
            continue
        for st in ("in8", "len32", "pp8", "alt32"):
            vals = {f: mean_se([final(r).get(st, {}).get(f) for r in rs])[0] for f in
                    ("p_halt_d0", "p_halt_d1", "p_halt_d2p", "p_halt_dead", "entropy_d0", "entropy_d2p")}
            out(f"| {g} | {st} | " + " | ".join(f"{vals[f]:.2f}" if not np.isnan(vals[f]) else "–" for f in
                ("p_halt_d0", "p_halt_d1", "p_halt_d2p", "p_halt_dead", "entropy_d0", "entropy_d2p")) + " |")


def deviations(groups, out):
    out("\n## First deviation from the oracle route (routed arms; oracle action > taken action, summed over runs)\n")
    for g, rs in sorted(groups.items()):
        if not rs[0].get("router"):
            continue
        cells = []
        for st in ("in8", "pp8", "len32"):
            conf = defaultdict(int); n = 0
            for r in rs:
                for k, v in final(r).get(st, {}).get("first_dev", {}).items():
                    conf[k] += v; n += v
            if n:
                cells.append(f"{st} ({n} failures): " + ", ".join(f"{k} {v}" for k, v in sorted(conf.items(), key=lambda kv: -kv[1])))
        if cells:
            out(f"- {g}: " + "; ".join(cells))


def t95_of(rec, k, key="acc_by_k"):
    """First held-out evaluation step with accuracy >= CRIT at k, confirmed by the next; None if never."""
    pts = [(c["step"], (c.get(key) or {}).get(str(k), (c.get(key) or {}).get(k))) for c in rec["curve"]]
    pts = [(s, a) for s, a in pts if a is not None]
    for i, (s, a) in enumerate(pts):
        if a >= CRIT and (i + 1 >= len(pts) or pts[i + 1][1] >= CRIT):
            return s
    return None


def flip_of(rec, action, thr=0.5):
    for c in rec["curve"]:
        if c.get("act_acc", {}).get(action, 0.0) >= thr:
            return c["step"]
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
    out(f"\n## Steps to criterion by k (held-out criterion set, accuracy ≥ {CRIT}, confirmed by the next 100-step point; "
        "route accuracy for routed arms, answer accuracy for neural arms; censoring-aware median [min–max]) and the SET / PREV flip "
        "times (first point with teacher-forced accuracy ≥ 0.5 on that action)\n")
    out("| arm | n | " + " | ".join(f"k={k}" for k in range(1, 9)) + " | SET flip | PREV flip |")
    out("|---|---|" + "---|" * 10)
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if r.get("curve")]
        if not rs or g.startswith("LR") or "@" in g:
            continue
        budget = rs[0]["train"]["steps"]
        flips = [fmt_t95([flip_of(r, a) for r in rs], budget) if rs[0].get("router") else "–" for a in ("SET", "PREV")]
        out(f"| {g} | {len(rs)} | " + " | ".join(fmt_t95([t95_of(r, k) for r in rs], budget) for k in range(1, 9)) + f" | {flips[0]} | {flips[1]} |")


def pooling(groups, out):
    out("\n## Pooling for selection (P4): held-out accuracy (last 5 points) on words whose deepest P is at depth d, and on all-H words (d = 0)\n")
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if r.get("curve") and "acc_by_Pdepth" in r["curve"][-1]]
        if not rs:
            continue
        last = [r["curve"][-5:] for r in rs]
        cells = []
        for d in range(0, 9):
            v = [np.mean([c["acc_by_Pdepth"][str(d)] for c in L if str(d) in c["acc_by_Pdepth"]]) for L in last
                 if any(str(d) in c["acc_by_Pdepth"] for c in L)]
            cells.append(f"d={d}: {np.mean(v):.2f}" if v else f"d={d}: –")
        out(f"- {g} (n={len(rs)}): " + "; ".join(cells))


def over_training(groups, out):
    out("\n## Extrapolation over training: route accuracy (routed) / answer accuracy (neural) at len16, pp8, len32 by evaluation step (run means)\n")
    for g, rs in sorted(groups.items()):
        rs = [r for r in rs if len(r.get("evals", [])) > 1 and "@" not in g and not g.startswith("LR")]
        if not rs:
            continue
        steps = [e["step"] for e in rs[0]["evals"]]
        rs = [r for r in rs if len(r["evals"]) == len(steps)]
        fld = "route_acc" if rs[0].get("router") else "acc"
        line = []
        for st in ("len16", "pp8", "len32"):
            a = np.array([[e["eval"].get(st, {}).get(fld, np.nan) for e in r["evals"]] for r in rs])
            line.append(f"{st} " + " ".join(f"{v:.2f}" for v in np.nanmean(a, 0)))
        out(f"- {g} (n={len(rs)}) steps {steps}: " + " | ".join(line))


def heads(groups, out):
    out("\n## Router heads (final): HOP/PREV/SET/HALT column norms and biases; index tables\n")
    for g, rs in sorted(groups.items()):
        hs = [r.get("head") for r in rs if r.get("head")]
        if not hs:
            continue
        if "w_norm" in hs[0]:
            out(f"- {g}: " + "; ".join("|w|=(" + ",".join(f"{x:.2f}" for x in h["w_norm"]) + ") b=(" + ",".join(f"{x:+.2f}" for x in h["b"]) + ")" for h in hs[:6]))
        elif "table" in hs[0]:
            out(f"- {g}: π by step (run 0, first 12): " + " ".join("(" + ",".join(f"{x:.2f}" for x in row) + ")" for row in hs[0]["table"][:12]))


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
    ins = [f"in{k}" for k in range(3, 9)] + ["len16", "len32", "deepP8"]
    table(groups, out, "route_acc_P", "Route accuracy on words WITH a P", strata=ins)
    table(groups, out, "acc_P", "Answer accuracy on words WITH a P", strata=ins)
    table(groups, out, "acc_noP", "Answer accuracy on all-H words", strata=ins)
    preregistered(groups, out)
    residual(groups, out)
    deviations(groups, out)
    firsts(groups, out)
    pooling(groups, out)
    over_training(groups, out)
    heads(groups, out)
    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
