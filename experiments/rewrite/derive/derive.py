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
from topology_design.rewrite.rules import HALT, RULE_AX, step, world_from_name  # noqa: E402
from topology_design.rewrite.search import FULL, bfs, enumerate_shortest, reverse_distances, shortest_dag  # noqa: E402
from topology_design.rewrite.term import is_leaf, nodes, parse, subterm_at, to_str  # noqa: E402

TRAIN_WORLDS = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
ALL_WORLDS = ["A", "C", "I", "D", "AC", "AI", "AD", "CI", "CD", "ID", "ACI", "ACD", "AID", "CID", "ACID"]
HELD_OUT = [w for w in ALL_WORLDS if w not in TRAIN_WORLDS]


def read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "..", "data", "rewrite"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=2000, help="shortest-proof enumeration limit")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
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
        # oracle check on a sample
        sample = [r for rs in ev["eval5"].values() for r in [x for x in rs if x["k"] <= 5][:10]]      # the deep lookahead is costly
        oc = pool.map(_oracle_check, sample, chunksize=8)
        D["oracle_check"] = {"n": len(oc), "ok": sum(oc)}
        md.append(f"\nOracle rung (d = k) on a sample of {len(oc)} eval instances: {sum(oc)} succeed.")

        # 4. recombination table (Q1)
        train_dec = {}
        for w, rs in train.items():
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
        for r in rs:
            if "pf" in r:
                c = wm.setdefault(str(r["k"]), [0.0, 0])
                c[0] += r["pf"]["p_success"]; c[1] += 1
    D["uniform_policy_success_by_k"] = {k: v[0] / v[1] for k, v in sorted(wm.items(), key=lambda kv: int(kv[0]))}
    md.append("\n## 8. Uniform policy's success probability (PF's prior p_success) by k, training worlds\n\n" +
              md_table([[k, f"{v:.4f}"] for k, v in D["uniform_policy_success_by_k"].items()], ["k", "p_success"]))
    D["seconds"] = time.time() - t0
    json.dump(D, open(os.path.join(HERE, "derived.json"), "w"), indent=1)
    open(os.path.join(HERE, "derived.md"), "w").write("\n".join(md) + "\n")
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
