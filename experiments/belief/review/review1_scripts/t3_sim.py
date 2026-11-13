"""Brute-force check of the Bayes-predictor claim (DESIGN §4, §6): simulate the generative process on fixed problems,
tabulate the empirical label frequencies given (state, budget, known-present, known-absent), compare with q.

Process per episode on a fixed problem (u, s, B): W ~ P(W | u, s); at every decision the label is one uniformly sampled
action optimal in W if W can still finish within the remaining budget, otherwise the decision is DROPPED; a proposer picks
an action; the kernel accepts iff its rule is in W.  Proposers:
  uniform     uniform over valid actions not yet rejected at this state                      (world-blind)
  qsample     a sample from q                                                                 (world-blind, history-dependent)
  verdictdep  q-greedy until the first rejection, then uniform over actions of rules not known absent   (world-blind)
  oracle50    with prob. 0.5 an optimal action in the true world, else uniform               (NOT world-blind)
  success     the qsample proposer, but only episodes that reach the goal are kept            (world-blind proposer, world-dependent selection)
  teacherlbl  the proposal IS the sampled label (teacher forcing on the target)               (NOT world-blind)
Reported per proposer: count-weighted mean TV(empirical, q) over keys with >= MINN labelled decisions, the same for a null
in which labels are drawn from q itself with the same counts, and the largest |z| over (key, action) cells.
Also: the same data keyed WITHOUT the remaining budget, against q at the modal budget."""
import random, sys, time, collections
import numpy as np
from core import *
from reftrace import *

NEP = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NI = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MINN = 400
MAXCALLS = 12
rng0 = random.Random(5)
insts = []
while len(insts) < NI:
    ins = Inst(rng0, draw_world=False)
    if 150 <= ins.n <= 1200:
        insts.append(ins)
nrng = np.random.default_rng(0)


def simulate(ins, proposer, nep, seed):
    rng = random.Random(seed)
    cons_idx = np.nonzero(ins.cons)[0]
    cum = np.cumsum(ins.post0[cons_idx])
    qcache, optcache = {}, {}
    tab = collections.defaultdict(lambda: collections.Counter())
    dropped = labelled = 0
    estart, er, ed = ins.estart, ins.er, ins.ed

    def q_of(v, b, P, A):
        key = (v, b, P, A)
        if key not in qcache:
            x = ins.mix(v, ins.wmask(v, b, P, A))
            qcache[key] = (None if x is None else (np.nonzero(x > 0)[0].tolist(), np.cumsum(x[x > 0]).tolist(), x))
        return qcache[key]

    def opt_of(v, w):
        key = (v, w)
        if key not in optcache:
            optcache[key] = np.nonzero(ins.optmat(v)[w])[0].tolist()
        return optcache[key]

    def draw(idx, cumw):
        x = rng.random() * cumw[-1]
        for j, c in zip(idx, cumw):
            if x <= c:
                return j
        return idx[-1]

    for ep in range(nep):
        w = int(cons_idx[min(np.searchsorted(cum, rng.random()), len(cons_idx) - 1)])
        v, b, P, A, calls = 0, ins.B, 0, 0, 0
        rej = {}
        buf = []
        while v != ins.s and b > 0 and calls < MAXCALLS:
            ne = int(estart[v + 1] - estart[v])
            live = ins.D[w, v] <= b
            lab = None
            if live:
                o = opt_of(v, w)
                lab = o[rng.randrange(len(o))]
                buf.append(((v, b, P, A), lab))
            else:
                dropped += 1
            rh = rej.get(v, ())
            if proposer in ("uniform",):
                cand = [j for j in range(ne) if j not in rh]
                a = cand[rng.randrange(len(cand))] if cand else None
            elif proposer in ("qsample", "success"):
                qq = q_of(v, b, P, A)
                a = None if qq is None else draw(qq[0], qq[1])
            elif proposer == "verdictdep":
                if A == 0:
                    qq = q_of(v, b, P, A)
                    a = None if qq is None else int(np.argmax(qq[2]))
                else:
                    cand = [j for j in range(ne) if not (A >> int(er[estart[v] + j])) & 1]
                    a = cand[rng.randrange(len(cand))] if cand else None
            elif proposer == "oracle50":
                if live and rng.random() < 0.5:
                    o = opt_of(v, w); a = o[rng.randrange(len(o))]
                else:
                    cand = [j for j in range(ne) if j not in rh]
                    a = cand[rng.randrange(len(cand))] if cand else None
            elif proposer == "teacherlbl":
                a = lab
            if a is None:
                break
            calls += 1
            e = estart[v] + a
            r = int(er[e])
            if HAS[w, r]:
                v, b, P = int(ed[e]), b - 1, P | (1 << r)
            else:
                rej.setdefault(v, set()).add(a)
                A |= 1 << r
        if proposer == "success" and v != ins.s:
            continue
        for key, lab in buf:
            tab[key][lab] += 1
            labelled += 1
    return tab, labelled, dropped


def compare(ins, tab, with_budget=True):
    tvs, nulls, ws, zmax, ncell, nkeys = [], [], [], 0.0, 0, 0
    if not with_budget:
        merged = collections.defaultdict(lambda: collections.Counter())
        modal = {}
        for (v, b, P, A), c in tab.items():
            merged[(v, P, A)].update(c)
            n = sum(c.values())
            if n > modal.get((v, P, A), (0, None))[0]:
                modal[(v, P, A)] = (n, b)
        items = [((v, modal[(v, P, A)][1], P, A), c) for (v, P, A), c in merged.items()]
    else:
        items = list(tab.items())
    for (v, b, P, A), c in items:
        n = sum(c.values())
        if n < MINN:
            continue
        q = ins.mix(v, ins.wmask(v, b, P, A))
        emp = np.zeros(len(q))
        for a, k in c.items():
            emp[a] = k
        emp /= n
        tvs.append(tv(emp, q)); ws.append(n)
        nulls.append(tv(nrng.multinomial(n, q / q.sum()) / n, q))
        sd = np.sqrt(np.maximum(q * (1 - q), 1e-12) / n)
        sel = (q > 0) & (q < 1)
        z = np.abs(emp - q)[sel] / sd[sel]
        if (emp[q == 0] > 0).any():
            zmax = float("inf")
        if len(z):
            zmax = max(zmax, float(z.max())); ncell += int(sel.sum())
        nkeys += 1
    ws = np.array(ws, dtype=float)
    return float((np.array(tvs) * ws).sum() / ws.sum()), float((np.array(nulls) * ws).sum() / ws.sum()), zmax, ncell, nkeys


print(f"{NI} problems, {NEP} episodes per proposer per problem; keys with >= {MINN} labelled decisions")
print("problem | proposer | labelled decisions | dropped (true world cannot finish) | keys | cells | mean TV(empirical, q) | null (labels drawn from q) | max |z|")
agg = collections.defaultdict(list)
for pi, ins in enumerate(insts):
    for prop in ("uniform", "qsample", "verdictdep", "oracle50", "success", "teacherlbl"):
        t0 = time.time()
        tab, lab, drop = simulate(ins, prop, NEP, 100 + pi)
        t, nl, zm, nc, nk = compare(ins, tab)
        agg[prop].append((t, nl, zm))
        print(f"{pi} (k={ins.k}, {ins.n} nodes, {int(ins.cons.sum())} worlds) | {prop} | {lab} | {drop} | {nk} | {nc} | {t:.4f} | {nl:.4f} | {zm:.1f} | {time.time()-t0:.0f}s", flush=True)
        if prop in ("uniform", "qsample"):
            t2, nl2, zm2, nc2, nk2 = compare(ins, tab, with_budget=False)
            agg[prop + ", keyed without budget"].append((t2, nl2, zm2))
            print(f"{pi} | {prop}, keyed without the remaining budget | | | {nk2} | {nc2} | {t2:.4f} | {nl2:.4f} | {zm2:.1f}", flush=True)
print("\nsummary (mean over problems): proposer | mean TV | null | worst max|z|")
for k, v in agg.items():
    a = np.array(v)
    print(f"{k} | {a[:,0].mean():.4f} | {a[:,1].mean():.4f} | {a[:,2].max():.1f}")
