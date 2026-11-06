"""Instance generation by layer sampling, strata, rejection, and the dataset records (DESIGN §1 Instances, §3, §5; review 1
B1, B2, B5, S1, S2, M6).

An instance is (W, t, s) with k = d_W(t, s). Generation: sample t (size uniform over the range, shape uniform over
bracketings, leaves i.i.d. over the five atoms); BFS from t under W to the radius; sample k from K_WEIGHTS restricted to the
non-empty layers; draw s uniformly from layer k. The record carries everything the arms and the strata need, precomputed:
the canonical (lexicographically first) shortest proof, the RT-post posterior over shortest proofs, the binding test
(d_full < d_W), the first decision's GP1 / D*_sep / D*_any, the symbolic ladder's outcomes (lexicographic tie-break), the
shortest-proof counts and trace classes, and (on request) the PF flat-prior posterior.

Actions are encoded as "rule@path" ("" is the root) or "HALT"; terms as fully parenthesised strings."""

from __future__ import annotations

import math
import random

from .posterior import pf_posterior, shortest_posterior
from .proxy import decision, ladder_rollout, ted
from .rules import HALT, RULES, world_name
from .search import FULL, bfs, count_shortest, distance, enumerate_shortest, reverse_distances, shortest_dag, trace_classes
from .term import ATOMS, parse, size, to_str

K_WEIGHTS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}
RULE_INDEX = {r: i for i, r in enumerate(RULES)}
CATALAN = [1, 1, 2, 5, 14, 42, 132, 429, 1430]


# ---------------- encoding ----------------

def enc_action(a) -> str:
    return HALT if a == HALT else f"{a[0]}@{a[1]}"


def dec_action(s: str):
    if s == HALT:
        return HALT
    rule, path = s.split("@")
    return (rule, path)


# ---------------- sampling ----------------

def sample_shape(rng: random.Random, n: int):
    """A uniformly random bracketing of n leaves (placeholders None)."""
    if n == 1:
        return None
    weights = [CATALAN[i - 1] * CATALAN[n - i - 1] for i in range(1, n)]
    i = rng.choices(range(1, n), weights=weights)[0]
    return (sample_shape(rng, i), sample_shape(rng, n - i))


def fill_leaves(rng: random.Random, shape, atoms=ATOMS):
    if shape is None:
        return rng.choice(atoms)
    return (fill_leaves(rng, shape[0], atoms), fill_leaves(rng, shape[1], atoms))


def sample_term(rng: random.Random, sizes, atoms=ATOMS):
    n = rng.choice(list(sizes))
    return fill_leaves(rng, sample_shape(rng, n), atoms)


def sample_k(rng: random.Random, available, k_weights=K_WEIGHTS):
    ks = [k for k in k_weights if k in available]
    if not ks:
        return None
    return rng.choices(ks, weights=[k_weights[k] for k in ks])[0]


# ---------------- the canonical proof ----------------

def canonical_proof(dag: dict, start, goal) -> list:
    """The lexicographically first shortest proof: at every state the (rule order, path)-smallest action that stays on a
    shortest path."""
    out, u = [], start
    while u != goal:
        best = None
        for v, acts in dag[u].items():
            for a in acts:
                key = (RULE_INDEX[a[0]], a[1])
                if best is None or key < best[0]:
                    best = (key, a, v)
        out.append(best[1])
        u = best[2]
    out.append(HALT)
    return out


# ---------------- one instance: a cheap candidate phase, then completion ----------------

def make_candidate(rng: random.Random, W, cap: int, sizes, radius: int = 6, k_weights=K_WEIGHTS, k_set=None, t=None):
    """Phase 1 (cheap): sample t, BFS under W to the radius, layer-sample k and s, reverse distances, the binding test
    (full-world BFS to radius k − 1 only) and the depth-1 quantities (GP1, tie1, B1). Returns a candidate dict with the
    partial record under "rec" and the search context, or None when t has no layer in the target set."""
    if t is None:
        t = sample_term(rng, sizes)
    dist, succ, complete = bfs(t, W, cap, radius=radius)
    layers = {}
    for u, d in dist.items():
        if d >= 1:
            layers.setdefault(d, []).append(u)
    if k_set is not None:
        layers = {k: v for k, v in layers.items() if k in k_set}
        k_weights = {k: k_weights.get(k, 1) for k in k_set}
    k = sample_k(rng, layers, k_weights)
    if k is None:
        return None
    s = rng.choice(sorted(layers[k], key=to_str))
    dgoal = reverse_distances(succ, s)
    assert dgoal[t] == k
    d_full = (distance(t, s, FULL, cap, radius=k - 1) if k > 1 else None) if W != FULL else None
    bind = d_full is not None and d_full < k
    first = decision(t, s, W, cap, dgoal, sigma=ted, depth=False)
    rec = {
        "world": world_name(W), "t": to_str(t), "s": to_str(s), "k": k, "size_t": size(t), "size_s": size(s), "cap": cap,
        "d_full": d_full if bind else None, "binding": bool(bind),
        "first": {key: first[key] for key in ("n_cand", "n_correct", "B1", "GP1", "tie1")},
        "ball": len(dist),
    }
    return {"rec": rec, "t": t, "s": s, "k": k, "W": W, "cap": cap, "succ": succ, "dgoal": dgoal}


def complete_instance(cand: dict, secondary: bool = False, pf: bool = False, trace_limit: int = 5000) -> dict:
    """Phase 2 (expensive): the shortest DAG, the canonical proof, the RT-post posterior, shortest-proof counts and trace
    classes, the first decision's D*_sep / D*_any, the ladder's outcomes, the secondary strata and the PF posterior."""
    rec, t, s, k, W, cap, succ, dgoal = (cand[key] for key in ("rec", "t", "s", "k", "W", "cap", "succ", "dgoal"))
    dag = shortest_dag(succ, dgoal, t)
    canon = canonical_proof(dag, t, s)
    post = shortest_posterior(t, s, W, cap, succ, dgoal)
    n_state, n_action = count_shortest(dag, t, s)
    classes = trace_classes(enumerate_shortest(dag, t, s, limit=trace_limit, actions=True))
    first = decision(t, s, W, cap, dgoal, sigma=ted)
    T = 2 * k + 2
    ladder = {d: ladder_rollout(t, s, W, cap, d, T)["ok"] for d in (1, 2, 3)}
    rec.update({
        "canon": [enc_action(a) for a in canon],
        "post": [{"state": to_str(u), "weight": info["weight"], "target": {enc_action(a): p for a, p in info["target"].items()}}
                 for u, info in post.items()],
        "n_shortest_states": n_state, "n_shortest_actions": n_action, "n_trace_classes": len(classes),
        "trace_classes_truncated": n_action > trace_limit,
        "ladder": {str(d): ok for d, ok in ladder.items()},
    })
    rec["first"].update({key: first[key] for key in ("Dsep", "Dany")})
    if secondary:
        # along the canonical proof: the max of D*_any and whether any decision is GP1; over shortest proofs: the minimum
        # over proofs of the per-proof maximum of D*_any (dynamic programme over the DAG)
        u = t
        dany_max, gp_any = 0, False
        per_state = {}
        for a in canon[:-1]:
            info = decision(u, s, W, cap, dgoal, sigma=ted)
            per_state[u] = info
            dany_max = max(dany_max, info["Dany"]); gp_any = gp_any or info["GP1"]
            u = next(v for v, acts in dag[u].items() if a in acts)
        for v in dag:
            if v != s and v not in per_state:
                per_state[v] = decision(v, s, W, cap, dgoal, sigma=ted)
        memo = {}

        def best_max(v):
            if v == s:
                return 0
            if v in memo:
                return memo[v]
            m = min(max(per_state[v]["Dany"], best_max(w)) for w in dag[v])
            memo[v] = m
            return m
        rec["secondary"] = {"canon_Dany_max": dany_max, "canon_GP1_any": gp_any, "min_over_proofs_Dany_max": best_max(t),
                            "dag_states": len(dag)}
    if pf:
        _, succ_full, ok = bfs(t, W, cap)
        if ok:
            r = pf_posterior(t, s, W, cap, T, succ_full)
            states = sorted(r["states"].items(), key=lambda kv: -kv[1]["visits"])
            total = sum(info["visits"] for _, info in states)
            kept, acc = [], 0.0
            for u, info in states:
                kept.append({"state": to_str(u), "visits": info["visits"],
                             "target": {enc_action(a): p for a, p in info["target"].items()}})
                acc += info["visits"]
                if acc >= 0.999 * total and len(kept) >= 8 or len(kept) >= 256:
                    break
            rec["pf"] = {"p_success": r["p_success"], "n_states": len(r["states"]), "visits_total": total, "states": kept}
    return rec


def make_instance(rng: random.Random, W, cap: int, sizes, radius: int = 6, k_weights=K_WEIGHTS, k_set=None,
                  secondary: bool = False, pf: bool = False, trace_limit: int = 5000, t=None) -> dict | None:
    """One layer-sampled instance under W (both phases), or None when t has no layer in the target set."""
    cand = make_candidate(rng, W, cap, sizes, radius=radius, k_weights=k_weights, k_set=k_set, t=t)
    if cand is None:
        return None
    return complete_instance(cand, secondary=secondary, pf=pf, trace_limit=trace_limit)


# ---------------- rejection to the stratum weights ----------------

def acceptance_rates(counts: dict, target_binding: float, target_gp1: float, grid: int = 100, floor: float = 0.05) -> dict:
    """Given warm-up counts of the categories B (binding), G (GP1, not binding), P (neither) and BG (binding and GP1),
    find acceptance probabilities (q for G, p for P; B always accepted) that maximise throughput subject to
    binding share ≥ target_binding and GP1 share ≥ target_gp1, with q, p ≥ floor so that no category (and no proof
    length) disappears from the stream. If infeasible, the point that maximises the smaller share-to-target ratio."""
    rB, rG, rP, rBG = (counts.get(key, 0) for key in ("B", "G", "P", "BG"))
    best, best_feasible = None, None
    lo = int(round(floor * grid))
    for qi in range(lo, grid + 1):
        q = qi / grid
        for pi in range(lo, grid + 1):
            p = pi / grid
            tot = rB + q * rG + p * rP
            if tot <= 0:
                continue
            sb = rB / tot
            sg = (rBG + q * rG) / tot
            ratio = min(sb / target_binding if target_binding > 0 else math.inf, sg / target_gp1 if target_gp1 > 0 else math.inf)
            if ratio >= 1:
                if best_feasible is None or tot > best_feasible[0]:
                    best_feasible = (tot, q, p, sb, sg)
            if best is None or ratio > best[0] + 1e-12 or (abs(ratio - best[0]) <= 1e-12 and tot > best[1]):
                best = (ratio, tot, q, p, sb, sg)
    if best_feasible is not None:
        tot, q, p, sb, sg = best_feasible
        return {"q_gp1": q, "p_plain": p, "feasible": True, "expected_binding": sb, "expected_gp1": sg}
    ratio, tot, q, p, sb, sg = best
    return {"q_gp1": q, "p_plain": p, "feasible": False, "expected_binding": sb, "expected_gp1": sg}


def category(rec: dict) -> str:
    if rec["binding"]:
        return "B"
    return "G" if rec["first"]["GP1"] else "P"


def accept(rec: dict, rates: dict, rng: random.Random) -> bool:
    c = category(rec)
    if c == "B":
        return True
    return rng.random() < (rates["q_gp1"] if c == "G" else rates["p_plain"])


# ---------------- realized tables ----------------

def realized_table(records: list) -> dict:
    """Counts by world × k × binding × GP1 × D*_any, plus per-world distinct (t, s) and distinct t (M6)."""
    table, pairs, starts = {}, {}, {}
    for r in records:
        key = (r["world"], r["k"], int(r["binding"]), int(r["first"]["GP1"]), r["first"]["Dany"])
        table[key] = table.get(key, 0) + 1
        pairs.setdefault(r["world"], set()).add((r["t"], r["s"]))
        starts.setdefault(r["world"], set()).add(r["t"])
    return {"cells": {"|".join(map(str, k)): v for k, v in table.items()},
            "distinct_pairs": {w: len(v) for w, v in pairs.items()}, "distinct_starts": {w: len(v) for w, v in starts.items()},
            "n": {w: sum(v for k, v in table.items() if k[0] == w) for w in pairs}}


def summarize(records: list) -> dict:
    """Per world: n, binding share, GP1 share, k histogram, mean ball size, ladder success per rung."""
    out = {}
    by_world = {}
    for r in records:
        by_world.setdefault(r["world"], []).append(r)
    for w, rs in sorted(by_world.items()):
        n = len(rs)
        ks = {}
        for r in rs:
            ks[r["k"]] = ks.get(r["k"], 0) + 1
        out[w] = {"n": n, "binding": sum(r["binding"] for r in rs) / n, "GP1": sum(r["first"]["GP1"] for r in rs) / n,
                  "k": {str(k): ks[k] / n for k in sorted(ks)}, "ball_mean": sum(r["ball"] for r in rs) / n,
                  "ladder": {d: sum(r["ladder"][d] for r in rs) / n for d in ("1", "2", "3")},
                  "Dany": {str(d): sum(r["first"]["Dany"] == d for r in rs) / n for d in range(1, 10)}}
    return out
