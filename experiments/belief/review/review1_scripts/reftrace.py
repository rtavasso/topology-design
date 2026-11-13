"""Reference trajectories and per-decision reference distributions (reviewer's implementation)."""
import numpy as np
from core import *


def argmax_edge(ins, i, vec, order="prelim"):
    """First maximal edge in the author's sort order, among edges with positive mass. Returns local edge index."""
    e0 = ins.estart[i]
    idxs = [j for j in range(len(vec)) if vec[j] > 0]
    if order == "prelim":
        key = lambda j: (int(ins.er[e0 + j]), int(ins.ed[e0 + j]), ins.ep[e0 + j])
    else:                                                          # power.py: (2 r + inner, x, path)
        key = lambda j: (2 * int(ins.er[e0 + j]) + (1 if ins.ep[e0 + j] else 0), int(ins.ed[e0 + j]), ins.ep[e0 + j])
    idxs.sort(key=key)
    best = None
    for j in idxs:
        if best is None or vec[j] > vec[best] + 1e-12:
            best = j
    return best


def masked(v, zero_idx):
    if v is None:
        return None
    v = v.copy()
    if len(zero_idx):
        v[list(zero_idx)] = 0.0
    return renorm(v)


def refs(ins, i, b, P, A, rej_here, train=None, extra=True):
    """All reference distributions at a decision. Vectors over the edges of node i."""
    e0, e1 = ins.estart[i], ins.estart[i + 1]
    rules = ins.er[e0:e1]
    out = {}
    out["q"] = ins.mix(i, ins.wmask(i, b, P, A))
    frozen = ins.mix(i, ins.wmask(i, b))
    out["frozen"] = frozen
    out["m"] = masked(frozen, rej_here)
    if extra:
        rule_rej = [j for j in range(e1 - e0) if A >> int(rules[j]) & 1]
        out["rm"] = masked(frozen, rule_rej)                                   # rule-level mask on the frozen belief
        out["qA"] = ins.mix(i, ins.wmask(i, b, 0, A))                           # condition on rejections only
        out["qP"] = masked(ins.mix(i, ins.wmask(i, b, P, 0)), rej_here)         # condition on accepts only, + action mask
        out["qP_rm"] = masked(ins.mix(i, ins.wmask(i, b, P, 0)), rule_rej)      # accepts + rule-level mask
    return out


def run(ins, policy="q", order="prelim", extra=True, max_extra_calls=40, true=None, hook=None):
    """Trajectory under a reference policy. Returns (records, summary). policy: q | mask | oracle | callable(recs)->vec."""
    true = ins.true if true is None else true
    v, b, P, A, calls, rej = 0, ins.B, 0, 0, 0, 0
    rej_here = {}
    recs = []
    last = None
    while v != ins.s and b > 0 and calls < ins.B + max_extra_calls:
        rh = rej_here.get(v, set())
        R = refs(ins, v, b, P, A, rh, extra=extra)
        if hook is not None:
            hook(ins, v, b, P, A, rh, R)
        R.update(node=v, b=b, P=P, A=A, calls=calls, last=last, rej_here=set(rh),
                 true_live=bool(ins.D[true, v] <= b))
        if policy == "q":
            sc = R["q"]
        elif policy == "mask":
            sc = R["m"]
        elif policy == "oracle":
            wm = np.zeros(NW, dtype=bool); wm[true] = True
            sc = ins.mix(v, wm) if ins.D[true, v] <= b else None
        else:
            sc = policy(ins, R)
        if sc is None:
            break
        recs.append(R)
        j = argmax_edge(ins, v, sc, order)
        e = ins.estart[v] + j
        R["choice"] = j
        calls += 1
        r = int(ins.er[e])
        if HAS[true, r]:
            v, b, P = int(ins.ed[e]), b - 1, P | (1 << r)
            last = "acc"
        else:
            rej += 1
            rej_here.setdefault(v, set()).add(j)
            A |= 1 << r
            last = "rej"
    return recs, dict(ok=v == ins.s, calls=calls, rej=rej)


def beta(recs_list, key_p, key0="m", key1="q", sel=None):
    """Pooled projection coefficient. recs_list: iterable of decision dicts. key_p may be a function rec -> vector."""
    num = den = 0.0
    n = 0
    for r in recs_list:
        if sel is not None and not sel(r):
            continue
        r0, r1 = r[key0], r[key1]
        p = key_p(r) if callable(key_p) else r[key_p]
        if r0 is None or r1 is None or p is None:
            continue
        d = r1 - r0
        num += float((p - r0) @ d); den += float(d @ d); n += 1
    return (num / den if den > 0 else float("nan")), den, n
