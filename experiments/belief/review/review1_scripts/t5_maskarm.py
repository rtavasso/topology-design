"""What is the Bayes-optimal predictor of the MASK ARM (DESIGN §5: sees state, goal [and remaining budget], nothing about the
past; actions rejected at this state are masked at the output), and what beta does it get?  The design requires |beta| < 0.15
for this arm, 'or the instrument is broken'.

Per problem: Monte Carlo over the generative process with a world-blind proposer mix (half the episodes sample from q, half
are uniform over valid actions not yet rejected at the state).  Training rows of the mask arm: (key = (state, budget),
mask R = actions rejected at this state, label = uniform over optimal actions in the true world [soft]; rows where the true
world cannot finish are dropped).  Its cross-entropy minimiser under the output mask is fitted per key by an MM iteration
(theta_a <- Y_a / sum_{R not containing a} n_R / S_R(theta)).  It is then scored on the q-greedy reference trajectory like
any model.  A token-arm key (state, budget, P, A) is fitted from the same episodes as a check of the machinery (must give
beta ~ 1)."""
import random, sys, time, collections
import numpy as np
from core import *
from reftrace import *

NI = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MINC = 30
FRAC_UNI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5      # share of episodes with the uniform proposer
rng0 = random.Random(0)


def mm_fit(groups, ne, iters=200):
    """groups: {frozenset R: y vector}. Returns theta (unnormalised scores over the ne actions)."""
    Y = np.zeros(ne)
    Rs, ns = [], []
    for R, y in groups.items():
        Y += y
        m = np.ones(ne, dtype=bool)
        if R:
            m[list(R)] = False
        Rs.append(m); ns.append(y.sum())
    Rs = np.array(Rs); ns = np.array(ns)
    th = np.where(Y > 0, 1.0, 0.0)
    th /= th.sum()
    for _ in range(iters):
        S = (Rs * th[None, :]).sum(1)                       # mass allowed under each mask
        denom = (Rs * (ns / np.maximum(S, 1e-300))[:, None]).sum(0)
        new = np.where(Y > 0, Y / np.maximum(denom, 1e-300), 0.0)
        new /= new.sum()
        if np.abs(new - th).max() < 1e-10:
            th = new
            break
        th = new
    return th


acc = collections.defaultdict(float)
t0 = time.time()
for it in range(NI):
    ins = Inst(rng0)
    recs, _ = run(ins, "q", order="power", extra=True)
    scored = [d for d in recs if d["calls"] > 0 and d["m"] is not None]
    if not scored:
        continue
    want_vb = {(d["node"], d["b"]) for d in scored}
    want_v = {d["node"] for d in scored}
    want_tok = {(d["node"], d["b"], d["P"], d["A"]) for d in scored}
    g_vb = collections.defaultdict(dict); g_v = collections.defaultdict(dict); g_tok = {}
    rng = random.Random(1000 + it)
    cons_idx = np.nonzero(ins.cons)[0]
    cum = np.cumsum(ins.post0[cons_idx])
    qcache, softcache = {}, {}
    estart, er, ed = ins.estart, ins.er, ins.ed
    for ep in range(NEP):
        w = int(cons_idx[min(np.searchsorted(cum, rng.random()), len(cons_idx) - 1)])
        uniform = (ep * FRAC_UNI) % 1.0 + FRAC_UNI >= 1.0 if 0 < FRAC_UNI < 1 else FRAC_UNI >= 1
        v, b, P, A, calls = 0, ins.B, 0, 0, 0
        rej = {}
        while v != ins.s and b > 0 and calls < 12:
            ne = int(estart[v + 1] - estart[v])
            rh = rej.get(v, ())
            if ins.D[w, v] <= b and (v in want_v):
                if (v, w) not in softcache:
                    o = ins.optmat(v)[w].astype(float)
                    softcache[(v, w)] = o / o.sum()
                y = softcache[(v, w)]
                R = frozenset(rh)
                if (v, b) in want_vb:
                    d = g_vb[(v, b)]
                    d[R] = d.get(R, 0) + y
                d = g_v[v]
                d[R] = d.get(R, 0) + y
                kt = (v, b, P, A)
                if kt in want_tok:
                    g_tok[kt] = g_tok.get(kt, 0) + y
            if uniform:
                cand = [j for j in range(ne) if j not in rh]
                if not cand:
                    break
                a = cand[rng.randrange(len(cand))]
            else:
                key = (v, b, P, A)
                if key not in qcache:
                    x = ins.mix(v, ins.wmask(v, b, P, A))
                    qcache[key] = None if x is None else (np.nonzero(x > 0)[0], np.cumsum(x[x > 0]))
                qq = qcache[key]
                if qq is None:
                    break
                a = int(qq[0][min(np.searchsorted(qq[1], rng.random() * qq[1][-1]), len(qq[0]) - 1)])
            calls += 1
            e = estart[v] + a
            r = int(er[e])
            if HAS[w, r]:
                v, b, P = int(ed[e]), b - 1, P | (1 << r)
            else:
                rej.setdefault(v, set()).add(a)
                A |= 1 << r
    for d in scored:
        v, b = d["node"], d["b"]
        ne = len(d["q"])
        dq = d["q"] - d["m"]
        e = float(dq @ dq)
        acc["den_all"] += e; acc["n_all"] += 1
        for name, groups in (("vb", g_vb.get((v, b))), ("v", g_v.get(v))):
            if not groups or sum(y.sum() for y in groups.values()) < MINC:
                continue
            th = mm_fit(groups, ne)
            p = masked(th, d["rej_here"])
            if p is None:
                continue
            acc[f"den_{name}"] += e; acc[f"num_{name}"] += float((p - d["m"]) @ dq); acc[f"n_{name}"] += 1
            acc[f"tvm_{name}"] += tv(p, d["m"]); acc[f"tvq_{name}"] += tv(p, d["q"])
            for cut, ok in (("acc", d["last"] == "acc"), ("rej", d["last"] == "rej")):
                if ok:
                    acc[f"den_{name}_{cut}"] += e; acc[f"num_{name}_{cut}"] += float((p - d["m"]) @ dq)
            # reference candidates on the same decisions, for comparison
            for cand in ("qP", "rm"):
                if d[cand] is not None:
                    acc[f"num_{name}_{cand}"] += float((d[cand] - d["m"]) @ dq)
        kt = (v, b, d["P"], d["A"])
        if kt in g_tok and g_tok[kt].sum() >= MINC:
            p = g_tok[kt] / g_tok[kt].sum()
            acc["den_tok"] += e; acc["num_tok"] += float((p - d["m"]) @ dq); acc["n_tok"] += 1; acc["tvq_tok"] += tv(p, d["q"])
    if (it + 1) % 25 == 0:
        print(f"... {it+1} problems, {time.time()-t0:.0f}s", flush=True)

print(f"\n{NI} problems, {NEP} episodes each (share {FRAC_UNI} uniform proposals, rest q-sampling); scored post-verdict decisions on the q-greedy trajectory: {int(acc['n_all'])}")
for name, label in (("vb", "mask arm, input = (state, goal, remaining budget)"), ("v", "mask arm, input = (state, goal) only"), ):
    n = acc[f"n_{name}"]
    print(f"{label}: fitted on {int(n)} decisions ({acc['den_'+name]/acc['den_all']:.2f} of the signal energy); "
          f"beta = {acc['num_'+name]/acc['den_'+name]:.3f}; after an accept {acc['num_'+name+'_acc']/max(acc['den_'+name+'_acc'],1e-9):.3f}, after a rejection {acc['num_'+name+'_rej']/max(acc['den_'+name+'_rej'],1e-9):.3f}; "
          f"mean TV to m {acc['tvm_'+name]/n:.3f}, to q {acc['tvq_'+name]/n:.3f}; on the same decisions beta[qP] {acc['num_'+name+'_qP']/acc['den_'+name]:.3f}, beta[rule mask] {acc['num_'+name+'_rm']/acc['den_'+name]:.3f}")
n = acc["n_tok"]
print(f"check — token-arm key (state, budget, P, A), empirical label mean: {int(n)} decisions ({acc['den_tok']/acc['den_all']:.2f} of energy); beta = {acc['num_tok']/acc['den_tok']:.3f}; mean TV to q {acc['tvq_tok']/n:.4f}")
