"""The derive step of rung 4 (experiments/rewrite/DESIGN.md §5): everything computable from the dataset before any router
is trained. Writes derive/derived.json and derive/derived.md (tables), from which §6's floors, rungs and ceilings are read.

    PYTHONPATH=src python3 experiments/rewrite/derive/derive.py --data data/rewrite --workers 2

Contents:
  1. realized (world, k, binding, GP1, D*_any) tables of the training stream and the eval sets; distinct (t, s) per world;
     reachable-ball sizes; tie multiplicity (shortest state / action sequences, trace classes)
  2. the cue-blind ceiling per world on eval5: mean over instances of [non-binding] × (share of full-world shortest
     proofs, as action sequences, whose rules all lie in W) — what a router that ignores the presence tokens and picks a
     full-world shortest proof uniformly can score; read against P8
  3. the symbolic ladder per world × k × binding × GP1 on every eval set: greedy-σ (d = 1), lookahead-2, lookahead-3 with
     the lexicographic tie-break (stored with the records) and with a random tie-break (3 draws; in brackets), and the
     oracle (d = k; always 1 by construction, checked on a sample) — the floors for P2 and the rungs for P4
  4. the recombination table for Q1: the (rule, local pattern) decisions of the canonical proofs in the training stream
     per world; for every held-out world's binding instances in eval5, the share whose canonical proof uses a decision
     never taken in any training world (not recombination; reported apart)
  5. Q2 at sizes 6–7: the share of eval7 instances whose every shortest proof uses a prefix position ≥ 9 (unaddressable
     by the categorical head trained at size ≤ 5); coverage of the eval7 canonical decisions by training keys
     (pattern, rule) [structural] against (pattern, flat position, rule) [absolute]
  6. RT-post's targets: mean entropy of the per-state action target and the trace-class count per world (the mixture
     RT-post is trained toward; P6 is read against the targets themselves)
  7. RT-examples' ceiling per world: share of eval5 instances whose canonical proof's axioms (and, in brackets, some
     shortest proof's axioms) lie within the axioms used by two example proofs drawn from the same set (5 draws)
  8. wandering mass under a uniform policy: PF's prior success probability p_success per k (training records with PF)
  9. the strata and references the pass rules were amended onto after the pass-rule audit (DESIGN §17): for every held-out
     world's binding instances in eval5, whether a proof within T = 2k + 2 exists inside a training world contained in W
     (if none does, the instance *needs the unseen combination* — P2's stratum) and the outcome of the training-world
     reference policy under the mask (P2's non-composing comparator); the ladder's own matched ratio between sizes 6–7
     and size ≤ 5 (P3's normaliser); for the training worlds' eval7 instances, whether no route within T uses only
     prefix positions ≤ 8 (must fail for the categorical head — P3's second clause)

    ... --only-9 recomputes section 9 alone and merges it into the existing derived.json / derived.md.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from topology_design.rewrite.data import dec_action  # noqa: E402
from topology_design.rewrite.proxy import ladder_rollout  # noqa: E402
from topology_design.rewrite.rules import HALT, RULE_AX, step, valid_actions, world_from_name  # noqa: E402
from topology_design.rewrite.search import FULL, bfs, enumerate_shortest, reverse_distances, shortest_dag  # noqa: E402
from topology_design.rewrite.term import is_leaf, nodes, parse, subterm_at, to_str  # noqa: E402

TRAIN_WORLDS = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
ALL_WORLDS = ["A", "C", "I", "D", "AC", "AI", "AD", "CI", "CD", "ID", "ACI", "ACD", "AID", "CID", "ACID"]
HELD_OUT = [w for w in ALL_WORLDS if w not in TRAIN_WORLDS]


def read(path):
    """PF targets are cut down to their p_success at load: this step reads only their count and that scalar, and the full
    targets (≈ 4 GB over the training worlds) copied into the forked workers exhausted an 8 GB machine (a worker was
    OOM-killed and Pool.map then waited forever)."""
    out = []
    with gzip.open(path, "rt") as f:
        for l in f:
            r = json.loads(l)
            if "pf" in r:
                r["pf"] = {"p_success": r["pf"]["p_success"]}
            out.append(r)
    return out


def load(data, tag, worlds):
    out = {}
    for w in worlds:
        p = os.path.join(data, f"{tag}_{w}.jsonl.gz")
        if os.path.exists(p):
            out[w] = read(p)
    return out


# ---------------- patterns ----------------

def pattern(u, depth=2):
    """The local shape of a subterm: atoms other than 0 abstracted to x, subtrees below `depth` to *."""
    if is_leaf(u):
        return "0" if u == "0" else "x"
    if depth == 0:
        return "*"
    return f"({pattern(u[0], depth - 1)}+{pattern(u[1], depth - 1)})"


def decisions(rec):
    """(rule, pattern at the node, flat prefix index) for every action of the canonical proof."""
    W, cap = world_from_name(rec["world"]), rec["cap"]
    u = parse(rec["t"])
    out = []
    for a in rec["canon"][:-1]:
        rule, path = dec_action(a)
        idx = [p for p, _, _ in nodes(u)].index(path)
        out.append((rule, pattern(subterm_at(u, path)), idx))
        u = step(u, rule, path, W, cap)
    return out


# ---------------- per-instance heavy quantities (workers) ----------------

def _cue_blind(args):
    """Share of full-world shortest action sequences inside W (0 for binding instances)."""
    rec, limit = args
    if rec["binding"]:
        return 0.0
    W = world_from_name(rec["world"])
    t, s = parse(rec["t"]), parse(rec["s"])
    dist, succ, _ = bfs(t, FULL, rec["cap"], radius=rec["k"], stop_at=s)
    dgoal = reverse_distances(succ, s)
    if t not in dgoal:
        return None
    dag = shortest_dag(succ, dgoal, t)
    seqs = enumerate_shortest(dag, t, s, limit=limit, actions=True)
    inW = sum(all(RULE_AX[a[0]] in W for a in q) for q in seqs)
    return inW / len(seqs)


def _ladder_random(args):
    rec, draws, seed = args
    W, cap = world_from_name(rec["world"]), rec["cap"]
    t, s = parse(rec["t"]), parse(rec["s"])
    T = 2 * rec["k"] + 2
    out = {}
    for d in (1, 2, 3):
        ok = 0
        for j in range(draws):
            rng = random.Random(seed * 1000 + j)
            ok += ladder_rollout(t, s, W, cap, d, T, tie="random", rng=rng)["ok"]
        out[str(d)] = ok / draws
    return out


def _oracle_check(args):
    rec = args
    W, cap = world_from_name(rec["world"]), rec["cap"]
    return ladder_rollout(parse(rec["t"]), parse(rec["s"]), W, cap, rec["k"], 2 * rec["k"] + 2)["ok"]


def _unaddressable(args):
    """Does every shortest proof use a prefix position >= 9 at some step? (DP over the shortest DAG)"""
    rec = args
    W, cap = world_from_name(rec["world"]), rec["cap"]
    t, s = parse(rec["t"]), parse(rec["s"])
    dist, succ, _ = bfs(t, W, cap, radius=rec["k"], stop_at=s)
    dgoal = reverse_distances(succ, s)
    dag = shortest_dag(succ, dgoal, t)
    memo = {}

    def addressable(u):              # some shortest path from u to s using positions <= 8 only
        if u == s:
            return True
        if u in memo:
            return memo[u]
        paths = [p for p, _, _ in nodes(u)]
        ok = False
        for v, acts in dag[u].items():
            if any(paths.index(p) <= 8 for _, p in acts) and addressable(v):
                ok = True
                break
        memo[u] = ok
        return ok
    return not addressable(t)


def _some_proof_axioms(args):
    """The family of axiom sets over all shortest proofs (as action sequences, limited)."""
    rec, limit = args
    W, cap = world_from_name(rec["world"]), rec["cap"]
    t, s = parse(rec["t"]), parse(rec["s"])
    dist, succ, _ = bfs(t, W, cap, radius=rec["k"], stop_at=s)
    dgoal = reverse_distances(succ, s)
    dag = shortest_dag(succ, dgoal, t)
    fams = set()
    for q in enumerate_shortest(dag, t, s, limit=limit, actions=True):
        fams.add("".join(sorted({RULE_AX[a[0]] for a in q})))
    return sorted(fams)


# ---------------- tables ----------------

def cell_table(recs, keys):
    out = {}
    for r in recs:
        key = "|".join(str(k(r)) for k in keys)
        out[key] = out.get(key, 0) + 1
    return out


def ladder_table(recs, rand=None):
    """{world|binding|GP1|k: {n, d1, d2, d3, [d1r, d2r, d3r]}}"""
    out = {}
    for i, r in enumerate(recs):
        key = f"{r['world']}|{int(r['binding'])}|{int(r['first']['GP1'])}|{r['k']}"
        c = out.setdefault(key, {"n": 0, "d1": 0, "d2": 0, "d3": 0, "d1r": 0.0, "d2r": 0.0, "d3r": 0.0})
        c["n"] += 1
        for d in (1, 2, 3):
            c[f"d{d}"] += r["ladder"][str(d)]
            if rand is not None:
                c[f"d{d}r"] += rand[i][str(d)]
    for c in out.values():
        for d in (1, 2, 3):
            c[f"d{d}"] = c[f"d{d}"] / c["n"]
            c[f"d{d}r"] = c[f"d{d}r"] / c["n"]
    return out


def md_table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


TRAIN_NAMES = list(TRAIN_WORLDS)
HELD_BINDING = ["AI", "CI", "ACI", "ACD", "AID", "CID"]          # the held-out worlds that have a binding stratum (not the full world)


def _p2_world(args):
    """For one held-out world's eval5 binding instances: needs-the-unseen-combination flag and the training-world
    reference's outcome (masked). One process per world, with the full-world actions cached across instances."""
    from topology_design.rewrite.proxy import goal_distances, trainworld_rollout
    w, recs = args
    W, full = world_from_name(w), world_from_name("ACID")
    subs = [u for u in TRAIN_NAMES if set(u) < set(w)]
    cache, cap = {}, (recs[0]["cap"] if recs else 5)
    assert all(r["cap"] == cap for r in recs)

    def acts(u):
        a = cache.get(u)
        if a is None:
            a = valid_actions(u, full, cap)
            cache[u] = a
        return a

    out = {}
    for r in recs:
        t, s, T = parse(r["t"]), parse(r["s"]), 2 * r["k"] + 2
        dist = {u: goal_distances(s, world_from_name(u), r["cap"], actions=acts) for u in TRAIN_NAMES}
        within = [u for u in subs if dist[u].get(t) is not None and dist[u][t] <= T]
        out[f"{r['t']}|{r['s']}"] = {"comb": not within, "trainw": bool(trainworld_rollout(t, s, W, r["cap"], T, dist)),
                                     "shortest_in_sub": any(dist[u].get(t) == r["k"] for u in subs)}
    return w, out


def _must_fail(rec):
    """No route from t to s of at most T = 2k + 2 steps that uses prefix positions <= 8 only: the categorical head, whose
    rows 9–12 are never labelled at size <= 5, cannot solve the instance within the horizon. None = search cut."""
    W, cap = world_from_name(rec["world"]), rec["cap"]
    t, s, T = parse(rec["t"]), parse(rec["s"]), 2 * rec["k"] + 2
    seen, frontier = {t}, [t]
    for _ in range(T):
        nxt = []
        for u in frontier:
            idx = {p: i for i, (p, _, _) in enumerate(nodes(u))}
            for _, path, v in valid_actions(u, W, cap):
                if idx[path] > 8 or v in seen:
                    continue
                if v == s:
                    return False
                seen.add(v); nxt.append(v)
                if len(seen) > 400000:
                    return None
        frontier = nxt
        if not frontier:
            break
    return True


def matched_ratio(rows7, rows5, ok):
    """analyze.py's P3 statistic for any per-instance outcome: success at sizes 6–7 over size-<=5 success on the same
    (world, k, binding) cells reweighted to eval7's composition."""
    c7, c5 = {}, {}
    for r in rows7:
        c7.setdefault((r["world"], r["k"], bool(r["binding"])), []).append(r)
    for r in rows5:
        if r["k"] <= 4:
            c5.setdefault((r["world"], r["k"], bool(r["binding"])), []).append(r)
    num = den = used = 0.0
    for key, rows in c7.items():
        if key not in c5:
            continue
        num += sum(bool(ok(r)) for r in rows)
        den += len(rows) * sum(bool(ok(r)) for r in c5[key]) / len(c5[key])
        used += len(rows)
    return (num / den if den else None), int(used)


def section9(ev, D, md, pool):
    lad = lambda rs, d: (sum(bool(r["ladder"][str(d)]) for r in rs) / len(rs)) if rs else None
    f2 = lambda x: "–" if x is None else f"{x:.3f}"
    # 9a. P2's stratum and its references
    res = dict(pool.map(_p2_world, [(w, [r for r in ev["eval5"][w] if r["binding"]]) for w in HELD_BINDING if w in ev["eval5"]]))
    D["p2_comb"] = res
    rows, summ = [], {}
    for cls, ws in (("pairs", ["AI", "CI"]), ("triples", ["ACI", "ACD", "AID", "CID"])):
        pooled_b, pooled_c, pooled_tw_b, pooled_tw_c = [], [], 0, 0
        for w in ws:
            if w not in res:
                continue
            b = [r for r in ev["eval5"][w] if r["binding"]]
            fl = [res[w][f"{r['t']}|{r['s']}"] for r in b]
            c = [r for r, f in zip(b, fl) if f["comb"]]
            tw_b, tw_c = sum(f["trainw"] for f in fl), sum(f["trainw"] for f in fl if f["comb"])
            rows.append([w, len(b), f"{sum(not f['comb'] for f in fl) / len(b):.2f}", len(c), f2(lad(c, 1)), f2(lad(c, 2)), f2(lad(c, 3)), f2(tw_c / len(c) if c else None), f2(lad(b, 1)), f2(tw_b / len(b))])
            pooled_b += b; pooled_c += c; pooled_tw_b += tw_b; pooled_tw_c += tw_c
        summ[cls] = {"binding_n": len(pooled_b), "comb_n": len(pooled_c), "d1": lad(pooled_c, 1), "d2": lad(pooled_c, 2), "d3": lad(pooled_c, 3),
                     "trainw": pooled_tw_c / len(pooled_c) if pooled_c else None, "binding_d1": lad(pooled_b, 1), "binding_d3": lad(pooled_b, 3),
                     "binding_trainw": pooled_tw_b / len(pooled_b) if pooled_b else None}
        rows.append([f"**{cls}**", len(pooled_b), f"{1 - len(pooled_c) / len(pooled_b):.2f}", len(pooled_c), f2(summ[cls]["d1"]), f2(summ[cls]["d2"]), f2(summ[cls]["d3"]), f2(summ[cls]["trainw"]),
                     f2(summ[cls]["binding_d1"]), f2(summ[cls]["binding_trainw"])])
    D["p2_comb_summary"] = summ
    md.append("\n## 9. Strata and references of the amended pass rules (DESIGN §17)\n\n### 9a. P2: held-out binding instances (eval5) that need the unseen combination — no proof within T = 2k + 2 inside a training world contained in W; "
              "rungs and the training-world reference (masked) on them\n\n" +
              md_table(rows, ["world", "binding n", "proof within T in a training sub-world", "needs the combination: n", "greedy d=1", "d=2", "d=3", "training-world reference", "all binding: greedy", "all binding: training-world reference"]))
    # 9b. P3: the ladder's own matched ratio, and the categorical head's must-fail set
    e5 = {w: [dict(r, world=w) for r in rs] for w, rs in ev["eval5"].items()}
    e7 = {w: [dict(r, world=w) for r in rs] for w, rs in ev["eval7"].items()}
    groups = {"training worlds": TRAIN_NAMES, "single-axiom": ["A", "C", "I", "D"], "training pairs": ["AC", "AD", "CD", "ID"],
              "held-out": ["AI", "CI", "ACI", "ACD", "AID", "CID", "ACID"]}
    rho, rows = {}, []
    for name, ws in groups.items():
        r7 = [r for w in ws for r in e7.get(w, [])]; r5 = [r for w in ws for r in e5.get(w, [])]
        vals = [matched_ratio(r7, r5, lambda r, d=d: r["ladder"][str(d)]) for d in (1, 2, 3)]
        rho[name] = {f"d{d}": v[0] for d, v in zip((1, 2, 3), vals)}
        rho[name]["n"] = vals[0][1]
        rows.append([name, vals[0][1]] + [f2(v[0]) for v in vals])
    D["p3_ladder_matched_ratio"] = rho
    md.append("\n### 9b. P3: the symbolic rungs' own matched ratio, sizes 6–7 over size ≤ 5 on the same (world, k, binding) cells (1 = equally hard; the rungs are rules over structure and do not know the size)\n\n" +
              md_table(rows, ["worlds", "eval7 n matched", "d=1", "d=2", "d=3"]))
    un = D.get("eval7_unaddressable", {})
    mf, rows = {}, []
    for w in TRAIN_NAMES:
        rs = [dict(r, world=w) for r in ev["eval7"].get(w, [])]
        flagged = [r for r in rs if un.get(w, {}).get(f"{r['t']}|{r['s']}")]
        out = pool.map(_must_fail, flagged, chunksize=4) if flagged else []
        mf[w] = {f"{r['t']}|{r['s']}": o for r, o in zip(flagged, out)}
        rows.append([w, len(rs), len(flagged), sum(o is False for o in out), sum(o is True for o in out), sum(o is None for o in out)])
    D["eval7_mustfail"] = mf
    rows.append(["**all**"] + [sum(r[i] for r in rows) for i in range(1, 6)])
    md.append("\n### 9c. P3: eval7 instances of the training worlds that the categorical head cannot solve within T — no route of ≤ T = 2k + 2 steps on prefix positions ≤ 8 (the instances not listed in `eval7_mustfail` have an addressable shortest proof)\n\n" +
              md_table(rows, ["world", "n", "every shortest proof needs a position ≥ 9", "…but an addressable route within T exists", "must fail", "search cut"]))
    # 9d. the other strata's sizes
    tb = [r for w in TRAIN_NAMES for r in ev["eval5"].get(w, []) if r["binding"]]
    hb = [r for w in groups["held-out"] for r in ev["eval5"].get(w, []) if r["binding"]]
    D["strata_n"] = {"train_binding": len(tb), "train_binding_d1": lad(tb, 1), "train_binding_d3": lad(tb, 3), "heldout_binding": len(hb), "heldout_binding_d1": lad(hb, 1), "heldout_binding_d3": lad(hb, 3),
                     "heldout_comb": sum(v["comb_n"] for v in summ.values())}
    md.append(f"\n### 9d. Other strata of the amended rules (eval5): training worlds' binding stratum n = {len(tb)} (greedy {f2(lad(tb, 1))}, lookahead-3 {f2(lad(tb, 3))}); "
              f"held-out binding stratum n = {len(hb)} (greedy {f2(lad(hb, 1))}, lookahead-3 {f2(lad(hb, 3))}), of which {D['strata_n']['heldout_comb']} need the unseen combination.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "..", "data", "rewrite"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=2000, help="shortest-proof enumeration limit")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default=HERE, help="directory for derived.json and derived.md")
    ap.add_argument("--only-9", action="store_true", help="recompute section 9 alone and merge it into the existing outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.only_9:
        D = json.load(open(os.path.join(args.out, "derived.json")))
        text = open(os.path.join(args.out, "derived.md")).read()
        md = [text.split("\n## 9. ")[0].rstrip("\n")]
        ev = {tag: load(args.data, tag, ALL_WORLDS) for tag in ("eval5", "eval7")}
        with mp.Pool(args.workers) as pool:
            section9(ev, D, md, pool)
        D["seconds_section9"] = time.time() - t0
        json.dump(D, open(os.path.join(args.out, "derived.json"), "w"), indent=1)
        open(os.path.join(args.out, "derived.md"), "w").write("\n".join(md) + "\n")
        print("section 9 done", f"{time.time() - t0:.0f}s")
        return
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    train = load(args.data, "train", ALL_WORLDS)
    ev = {tag: load(args.data, tag, ALL_WORLDS) for tag in ("eval5", "eval7", "evalk78")}
    D = {"data_builds": meta.get("builds"), "k_weights": meta.get("k_weights"), "rates": meta.get("rates")}
    md = ["# Rung 4 derive step (generated by experiments/rewrite/derive/derive.py)\n"]

    # 1. realized tables
    def realized(by_world):
        out = {}
        for w, rs in by_world.items():
            out[w] = {"n": len(rs), "distinct_pairs": len({(r["t"], r["s"]) for r in rs}), "distinct_starts": len({r["t"] for r in rs}),
                      "binding": sum(r["binding"] for r in rs) / len(rs), "GP1": sum(r["first"]["GP1"] for r in rs) / len(rs),
                      "k": cell_table(rs, [lambda r: r["k"]]), "Dany": cell_table(rs, [lambda r: r["first"]["Dany"]]),
                      "ball_mean": sum(r["ball"] for r in rs) / len(rs), "ball_max": max(r["ball"] for r in rs),
                      "ties": {"state_seqs_mean": sum(r["n_shortest_states"] for r in rs) / len(rs),
                               "action_seqs_mean": sum(r["n_shortest_actions"] for r in rs) / len(rs),
                               "trace_classes_mean": sum(r["n_trace_classes"] for r in rs) / len(rs),
                               "share_unique_state_seq": sum(r["n_shortest_states"] == 1 for r in rs) / len(rs)},
                      "cells": cell_table(rs, [lambda r: r["k"], lambda r: int(r["binding"]), lambda r: int(r["first"]["GP1"]), lambda r: r["first"]["Dany"]])}
            if rs and "pf" in rs[0]:
                pfr = [r for r in rs if "pf" in r]
                out[w]["pf_n"] = len(pfr)
        return out
    D["realized"] = {"train": realized(train), **{tag: realized(by) for tag, by in ev.items()}}
    for tag in ["train", "eval5", "eval7", "evalk78"]:
        rows = []
        for w, r in D["realized"][tag].items():
            ks = r["k"]
            rows.append([w, r["n"], r["distinct_pairs"], f"{r['binding']:.2f}", f"{r['GP1']:.2f}",
                         " ".join(f"{k}:{ks[k] / r['n']:.2f}" for k in sorted(ks, key=int)),
                         " ".join(f"{d}:{r['Dany'][d] / r['n']:.2f}" for d in sorted(r["Dany"], key=int)),
                         f"{r['ball_mean']:.0f}", f"{r['ties']['trace_classes_mean']:.1f}", f"{r['ties']['share_unique_state_seq']:.2f}"])
        md.append(f"\n## 1. Realized strata: {tag}\n\n" + md_table(rows, ["world", "n", "distinct (t,s)", "binding", "GP1", "k shares", "D*_any shares", "ball", "trace classes", "unique proof"]))
    print("realized tables", f"{time.time() - t0:.0f}s", flush=True)

    with mp.Pool(args.workers) as pool:
        # 2. cue-blind ceiling on eval5
        cb = {}
        for w, rs in ev["eval5"].items():
            vals = pool.map(_cue_blind, [(r, args.limit) for r in rs], chunksize=8)
            ok = [v for v in vals if v is not None]
            nb = [r for r in rs if not r["binding"]]
            cb[w] = {"ceiling": sum(ok) / len(ok), "non_binding_share": len(nb) / len(rs),
                     "inW_share_among_non_binding": (sum(v for v, r in zip(vals, rs) if not r["binding"] and v is not None) / max(1, len(nb))),
                     "n": len(ok), "unreachable": len(vals) - len(ok)}
            # the binding stratum's ceiling is 0 by definition; the non-binding stratum's is the inW share
        D["cue_blind"] = cb
        md.append("\n## 2. Cue-blind ceiling per world (eval5): mean of [non-binding] × share of full-world shortest proofs inside W\n\n" +
                  md_table([[w, f"{c['ceiling']:.3f}", f"{c['non_binding_share']:.2f}", f"{c['inW_share_among_non_binding']:.3f}", c["n"]] for w, c in cb.items()],
                           ["world", "ceiling (all)", "non-binding share", "inW share | non-binding", "n"]))
        print("cue-blind", f"{time.time() - t0:.0f}s", flush=True)

        # 3. ladder per stratum, lexicographic (stored) and random tie-break
        lad = {}
        for tag, by in ev.items():
            lad[tag] = {}
            for w, rs in by.items():
                rand = pool.map(_ladder_random, [(r, args.draws, i) for i, r in enumerate(rs)], chunksize=8)
                lad[tag][w] = ladder_table(rs, rand)
            print("ladder", tag, f"{time.time() - t0:.0f}s", flush=True)
        D["ladder"] = lad
        for tag in lad:
            rows = []
            for w in lad[tag]:
                for key, c in sorted(lad[tag][w].items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[3]), kv[0])):
                    _, b, g, k = key.split("|")
                    rows.append([w, b, g, k, c["n"], f"{c['d1']:.2f} [{c['d1r']:.2f}]", f"{c['d2']:.2f} [{c['d2r']:.2f}]", f"{c['d3']:.2f} [{c['d3r']:.2f}]"])
            md.append(f"\n## 3. Symbolic ladder on {tag}: success of greedy-σ (d=1), lookahead-2, lookahead-3; lexicographic tie-break [random, {args.draws} draws]\n\n" +
                      md_table(rows, ["world", "binding", "GP1", "k", "n", "d=1", "d=2", "d=3"]))
        # P4's preregistered cells (review 2, S7): training worlds, GP1, k >= 4, by (world, k), where the rungs separate
        sep = {}
        for w in TRAIN_WORLDS:
            rs = ev["eval5"].get(w, [])
            for k in (4, 5, 6):
                cell = [r for r in rs if r["first"]["GP1"] and r["k"] == k]
                if len(cell) < 10:
                    continue
                d = {dd: sum(r["ladder"][str(dd)] for r in cell) / len(cell) for dd in (1, 2, 3)}
                sep[f"{w}|{k}"] = {"n": len(cell), "d1": d[1], "d2": d[2], "d3": d[3], "separated": bool(d[2] - d[1] >= 0.1 and d[3] - d[2] >= 0.1)}
        D["p4_cells"] = sep
        md.append("\n### P4 cells as preregistered in §6 (eval5, training worlds, GP1, k ≥ 4, by (world, k); lexicographic rungs; separated = d2 − d1 ≥ 0.1 and d3 − d2 ≥ 0.1)\n\n" +
                  md_table([[key, c["n"], f"{c['d1']:.2f}", f"{c['d2']:.2f}", f"{c['d3']:.2f}", "yes" if c["separated"] else "no"] for key, c in sep.items()],
                           ["world|k", "n", "d=1", "d=2", "d=3", "separated"]) +
                  f"\n\nSeparated cells: {sum(c['separated'] for c in sep.values())} of {len(sep)}.")
        # P4's amended cells (DESIGN §16, set from this table before any run): the same rule on every (world, binding, GP1, k)
        # cell of the training worlds at k >= 4 with n >= 10 — on GP1 cells the d = 1 and d = 2 rungs both sit near zero, so
        # the lower separation never holds there and the cells where it does are the binding non-GP1 ones
        sep2 = {}
        for w in TRAIN_WORLDS:
            rs = ev["eval5"].get(w, [])
            for k in (4, 5, 6):
                for b in (0, 1):
                    for g in (0, 1):
                        cell = [r for r in rs if int(bool(r["binding"])) == b and int(bool(r["first"]["GP1"])) == g and r["k"] == k]
                        if len(cell) < 10:
                            continue
                        d = {dd: sum(r["ladder"][str(dd)] for r in cell) / len(cell) for dd in (1, 2, 3)}
                        sep2[f"{w}|{b}|{g}|{k}"] = {"n": len(cell), "d1": d[1], "d2": d[2], "d3": d[3], "separated": bool(d[2] - d[1] >= 0.1 and d[3] - d[2] >= 0.1)}
        D["p4_cells_amended"] = sep2
        chosen = [key for key, c in sep2.items() if c["separated"]]
        pooled = [r for key in chosen for r in ev["eval5"][key.split("|")[0]]
                  if f"{key.split('|')[0]}|{int(bool(r['binding']))}|{int(bool(r['first']['GP1']))}|{r['k']}" == key]
        D["p4_amended_pooled"] = {"cells": chosen, "n": len(pooled),
                                  **{f"d{dd}": (sum(r["ladder"][str(dd)] for r in pooled) / len(pooled) if pooled else None) for dd in (1, 2, 3)}}
        # the same cells under the random tie-break (review of §16: the lower separation is tie-break-dependent)
        fr = lambda x: "—" if x is None else f"{x:.2f}"
        for key, c in sep2.items():
            lc = lad["eval5"][key.split("|")[0]].get(key, {})
            c.update({f"d{dd}r": lc.get(f"d{dd}r") for dd in (1, 2, 3)})
            c["separated_random"] = bool(c["d1r"] is not None and c["d2r"] - c["d1r"] >= 0.1 and c["d3r"] - c["d2r"] >= 0.1)
        nsum = sum(sep2[key]["n"] for key in chosen) or 1
        D["p4_amended_pooled"].update({f"d{dd}r": (sum((sep2[key][f"d{dd}r"] or 0.0) * sep2[key]["n"] for key in chosen) / nsum if chosen else None) for dd in (1, 2, 3)})
        D["p4_amended_pooled"]["cells_lower_separation_fails_random"] = [key for key in chosen if not sep2[key]["separated_random"]]
        md.append("\n### P4 cells as amended in §16 (eval5, training worlds, every (world, binding, GP1, k) cell at k ≥ 4 with n ≥ 10; same rungs and separation rule; [random tie-break] beside)\n\n" +
                  md_table([[key, c["n"], f"{c['d1']:.2f} [{fr(c['d1r'])}]", f"{c['d2']:.2f} [{fr(c['d2r'])}]", f"{c['d3']:.2f} [{fr(c['d3r'])}]", "yes" if c["separated"] else "no", "yes" if c["separated_random"] else "no"] for key, c in sep2.items()],
                           ["world|binding|GP1|k", "n", "d=1", "d=2", "d=3", "separated", "separated [random]"]) +
                  f"\n\nSeparated cells: {len(chosen)} of {len(sep2)} ({', '.join(chosen)}); pooled n = {len(pooled)}, rungs "
                  + (", ".join(f"d={dd} {D['p4_amended_pooled'][f'd{dd}']:.2f} [{fr(D['p4_amended_pooled'][f'd{dd}r'])}]" for dd in (1, 2, 3)) if pooled else "—")
                  + f"; under the random tie-break the lower separation fails on {len(D['p4_amended_pooled']['cells_lower_separation_fails_random'])} of these cells. P4 is read on them.")
        # unscored sensitivity sets (review of §16): upper separation only; garden path anywhere on the canonical proof
        def pooled_rungs(cells_):
            rs_ = [r for key in cells_ for r in ev["eval5"][key.split("|")[0]]
                   if f"{key.split('|')[0]}|{int(bool(r['binding']))}|{int(bool(r['first']['GP1']))}|{r['k']}" == key]
            return {"cells": cells_, "n": len(rs_), **{f"d{dd}": (sum(r["ladder"][str(dd)] for r in rs_) / len(rs_) if rs_ else None) for dd in (1, 2, 3)}}
        upper = [key for key, c in sep2.items() if c["d3"] - c["d2"] >= 0.1]
        D["p4_upper_only"] = pooled_rungs(upper)
        gpa = {}
        for w in TRAIN_WORLDS:
            rs = ev["eval5"].get(w, [])
            for k in (4, 5, 6):
                for b in (0, 1):
                    cell = [r for r in rs if int(bool(r["binding"])) == b and r.get("secondary", {}).get("canon_GP1_any") and r["k"] == k]
                    if len(cell) < 10:
                        continue
                    d = {dd: sum(r["ladder"][str(dd)] for r in cell) / len(cell) for dd in (1, 2, 3)}
                    gpa[f"{w}|{b}|gp-any|{k}"] = {"n": len(cell), "d1": d[1], "d2": d[2], "d3": d[3], "separated": bool(d[2] - d[1] >= 0.1 and d[3] - d[2] >= 0.1)}
        gpa_chosen = [key for key, c in gpa.items() if c["separated"]]
        gpa_rs = [r for key in gpa_chosen for r in ev["eval5"][key.split("|")[0]]
                  if int(bool(r["binding"])) == int(key.split("|")[1]) and r.get("secondary", {}).get("canon_GP1_any") and r["k"] == int(key.split("|")[3])]
        D["p4_gp_anywhere"] = {"cells": gpa_chosen, "n_cells_total": len(gpa), "n": len(gpa_rs),
                               **{f"d{dd}": (sum(r["ladder"][str(dd)] for r in gpa_rs) / len(gpa_rs) if gpa_rs else None) for dd in (1, 2, 3)}}
        md.append("\n### P4 sensitivity sets (unscored; §16): upper separation only (d3 − d2 ≥ 0.1): "
                  + f"{len(upper)} of {len(sep2)} cells, n = {D['p4_upper_only']['n']}, rungs " + (", ".join(f"{D['p4_upper_only'][f'd{dd}']:.2f}" for dd in (1, 2, 3)) if upper else "—")
                  + f". Garden path anywhere on the canonical proof, cells (world, binding, k) with n ≥ 10 under the same rule: {len(gpa_chosen)} of {len(gpa)} cells ({', '.join(gpa_chosen)}), n = {D['p4_gp_anywhere']['n']}, rungs "
                  + (", ".join(f"{D['p4_gp_anywhere'][f'd{dd}']:.2f}" for dd in (1, 2, 3)) if gpa_rs else "—") + ".")
        # oracle check on a sample
        sample = [r for rs in ev["eval5"].values() for r in [x for x in rs if x["k"] <= 5][:10]]      # the deep lookahead is costly
        oc = pool.map(_oracle_check, sample, chunksize=8)
        D["oracle_check"] = {"n": len(oc), "ok": sum(oc)}
        md.append(f"\nOracle rung (d = k) on a sample of {len(oc)} eval instances: {sum(oc)} succeed.")

        # 4. recombination table (Q1)
        train_dec = {}
        for w, rs in train.items():
            if w not in TRAIN_WORLDS:                 # the held-out worlds' files are RT-all's stream, not the training worlds'
                continue
            for r in rs:
                for rule, pat, idx in decisions(r):
                    train_dec.setdefault((rule, pat), set()).add(w)
        D["train_decisions"] = {f"{rule} @ {pat}": sorted(ws) for (rule, pat), ws in sorted(train_dec.items())}
        rec_tab = {}
        for w, rs in ev["eval5"].items():
            if w not in HELD_OUT:
                continue
            bind = [r for r in rs if r["binding"]]
            novel = 0
            novel_keys = {}
            for r in bind:
                keys = decisions(r)
                missing = [k for k in keys if (k[0], k[1]) not in train_dec]
                if missing:
                    novel += 1
                    for k in missing:
                        novel_keys[f"{k[0]} @ {k[1]}"] = novel_keys.get(f"{k[0]} @ {k[1]}", 0) + 1
            rec_tab[w] = {"binding_n": len(bind), "novel_decision_share": novel / max(1, len(bind)),
                          "novel_keys": dict(sorted(novel_keys.items(), key=lambda kv: -kv[1])[:10])}
        D["recombination"] = rec_tab
        md.append("\n## 4. Recombination (Q1): held-out worlds' binding instances (eval5) whose canonical proof uses a (rule, local pattern) never taken in any training world\n\n" +
                  md_table([[w, c["binding_n"], f"{c['novel_decision_share']:.3f}", "; ".join(f"{k} ({v})" for k, v in list(c["novel_keys"].items())[:4])] for w, c in rec_tab.items()],
                           ["world", "binding n", "share with a novel decision (not recombination)", "most frequent novel decisions"]))
        print("recombination", f"{time.time() - t0:.0f}s", flush=True)

        # 5. Q2 at sizes 6–7
        q2 = {}
        train_keys_struct = {(rule, pat) for (rule, pat) in train_dec}
        train_keys_abs = set()
        for w, rs in train.items():
            if w not in TRAIN_WORLDS:
                continue
            for r in rs:
                for rule, pat, idx in decisions(r):
                    train_keys_abs.add((rule, pat, idx))
        unaddr = {}
        for w, rs in ev["eval7"].items():
            un = pool.map(_unaddressable, rs, chunksize=8)
            unaddr[w] = {f"{r['t']}|{r['s']}": bool(u) for r, u in zip(rs, un)}
            first = [decisions(r)[0] for r in rs]
            q2[w] = {"n": len(rs), "unaddressable_share": sum(un) / len(rs),
                     "first_decision_covered_struct": sum((f[0], f[1]) in train_keys_struct for f in first) / len(rs),
                     "first_decision_covered_abs": sum((f[0], f[1], f[2]) in train_keys_abs for f in first) / len(rs),
                     "first_decision_position_ge9": sum(f[2] >= 9 for f in first) / len(rs)}
        D["q2"] = q2
        D["eval7_unaddressable"] = unaddr
        md.append("\n## 5. Q2 at sizes 6–7 (eval7): instances whose every shortest proof needs a prefix position ≥ 9; coverage of the first decision by training keys\n\n" +
                  md_table([[w, c["n"], f"{c['unaddressable_share']:.3f}", f"{c['first_decision_position_ge9']:.3f}", f"{c['first_decision_covered_struct']:.3f}", f"{c['first_decision_covered_abs']:.3f}"] for w, c in q2.items()],
                           ["world", "n", "unaddressable (all proofs)", "first decision at pos ≥ 9", "covered (pattern, rule)", "covered (pattern, position, rule)"]))
        print("q2", f"{time.time() - t0:.0f}s", flush=True)

        # 6. RT-post targets
        import math
        pt = {}
        for w, rs in train.items():
            if w not in TRAIN_WORLDS:
                continue
            ents, n = 0.0, 0
            for r in rs:
                for p in r["post"]:
                    ents += -sum(q * math.log(q) for q in p["target"].values() if q > 0); n += 1
            pt[w] = {"mean_target_entropy_nats": ents / n, "mean_trace_classes": sum(r["n_trace_classes"] for r in rs) / len(rs),
                     "share_multi_class": sum(r["n_trace_classes"] > 1 for r in rs) / len(rs)}
        D["post_targets"] = pt
        md.append("\n## 6. RT-post targets on the training stream: mean per-state target entropy, trace classes per instance\n\n" +
                  md_table([[w, f"{c['mean_target_entropy_nats']:.3f}", f"{c['mean_trace_classes']:.2f}", f"{c['share_multi_class']:.2f}"] for w, c in pt.items()],
                           ["world", "target entropy (nats)", "trace classes", "share > 1 class"]))

        # 7. RT-examples ceiling
        exc = {}
        rng = random.Random(0)
        for w, rs in ev["eval5"].items():
            fams = pool.map(_some_proof_axioms, [(r, args.limit) for r in rs], chunksize=8)
            canon_ax = ["".join(sorted({RULE_AX[dec_action(a)[0]] for a in r["canon"][:-1]})) for r in rs]
            ok_c, ok_s, tot = 0, 0, 0
            ok_cb, tot_b = 0, 0
            for i, r in enumerate(rs):
                for _ in range(5):
                    ex = rng.sample(range(len(rs)), 2)
                    used = set(canon_ax[ex[0]]) | set(canon_ax[ex[1]])
                    c_ok = set(canon_ax[i]) <= used
                    s_ok = any(set(f) <= used for f in fams[i])
                    ok_c += c_ok; ok_s += s_ok; tot += 1
                    if r["binding"]:
                        ok_cb += c_ok; tot_b += 1
            exc[w] = {"ceiling_canon": ok_c / tot, "ceiling_some_proof": ok_s / tot, "ceiling_canon_binding": ok_cb / max(1, tot_b), "binding_n": tot_b // 5}
        D["examples_ceiling"] = exc
        md.append("\n## 7. RT-examples ceiling per world (eval5): share of instances whose canonical proof's axioms lie within two random example proofs' axioms [some shortest proof's]; and on the binding stratum\n\n" +
                  md_table([[w, f"{c['ceiling_canon']:.3f}", f"{c['ceiling_some_proof']:.3f}", f"{c['ceiling_canon_binding']:.3f}", c["binding_n"]] for w, c in exc.items()],
                           ["world", "ceiling (canonical)", "[some proof]", "binding stratum", "binding n"]))
        print("examples ceiling", f"{time.time() - t0:.0f}s", flush=True)

    # 8. wandering mass
    wm = {}
    for w, rs in train.items():
        if w not in TRAIN_WORLDS:
            continue
        for r in rs:
            if "pf" in r:
                c = wm.setdefault(str(r["k"]), [0.0, 0])
                c[0] += r["pf"]["p_success"]; c[1] += 1
    D["uniform_policy_success_by_k"] = {k: v[0] / v[1] for k, v in sorted(wm.items(), key=lambda kv: int(kv[0]))}
    md.append("\n## 8. Uniform policy's success probability (PF's prior p_success) by k, training worlds\n\n" +
              md_table([[k, f"{v:.4f}"] for k, v in D["uniform_policy_success_by_k"].items()], ["k", "p_success"]))
    with mp.Pool(args.workers) as pool:
        section9(ev, D, md, pool)
    D["seconds"] = time.time() - t0
    json.dump(D, open(os.path.join(args.out, "derived.json"), "w"), indent=1)
    open(os.path.join(args.out, "derived.md"), "w").write("\n".join(md) + "\n")
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
