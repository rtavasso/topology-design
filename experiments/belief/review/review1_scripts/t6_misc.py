"""Smaller checks: bootstrap intervals for beta of the rule-mask family; dependence of q on the remaining budget; q-greedy
decisions where the true world can no longer finish; which (rule, verdict) evidence the small training sets can show;
cost split of one instance."""
import random, sys, time, pickle, collections
import numpy as np
from core import *
from reftrace import *

D = pickle.load(open("dec_0_600.pkl", "rb"))
post = [d for d in D if d["calls"] > 0 and d["m"] is not None]
ninst = 600
g = np.random.default_rng(0)
print("== bootstrap over instances (600 instances, seed 0), 2000 resamples: beta of lookup-type 'models' ==")
for key in ("rm", "lookup1.0", "lookup2.0", "qP", "qP_rm", "qA"):
    num = np.zeros(ninst); den = np.zeros(ninst)
    for d in post:
        if d.get(key) is None:
            continue
        dq = d["q"] - d["m"]
        num[d["inst"]] += float((d[key] - d["m"]) @ dq); den[d["inst"]] += float(dq @ dq)
    idx = g.integers(0, ninst, size=(2000, ninst))
    bs = num[idx].sum(1) / den[idx].sum(1)
    print(f"{key:10s} beta {num.sum()/den.sum():.3f}  95% interval [{np.quantile(bs,0.025):.3f}, {np.quantile(bs,0.975):.3f}]  sd {bs.std():.3f} (sd at 2,000 instances ~ {bs.std()*np.sqrt(ninst/2000):.3f})")

# step statistic: sampling error
by = collections.defaultdict(list)
for d in D:
    by[d["inst"]].append(d)
num = np.zeros(ninst); den = np.zeros(ninst); numq = np.zeros(ninst)
gg = np.random.default_rng(1)
def noisy(p, sigma=0.9):
    w = (p + 1e-3) * np.exp(sigma * gg.standard_normal(p.shape)); return w / w.sum()
for i, recs in by.items():
    recs.sort(key=lambda d: d["calls"])
    for j in range(1, len(recs)):
        d, pv = recs[j], recs[j - 1]
        if d["last"] != "rej":
            continue
        a = pv["choice"]
        q0 = masked(pv["q"], [a])
        if q0 is None or d.get("rm") is None or pv.get("rm") is None:
            continue
        dq = d["q"] - q0
        p0 = masked(pv["rm"], [a])
        if p0 is None:
            continue
        num[i] += float((d["rm"] - p0) @ dq); den[i] += float(dq @ dq)
        pn, pp = noisy(d["q"]), noisy(pv["q"])
        pp0 = masked(pp, [a])
        numq[i] += float((pn - pp0) @ dq)
idx = g.integers(0, ninst, size=(2000, ninst))
bs = num[idx].sum(1) / den[idx].sum(1); bq = numq[idx].sum(1) / den[idx].sum(1)
print(f"\nbeta_step: rule mask {num.sum()/den.sum():.3f} (boot sd {bs.std():.3f}; at 2,000 instances ~{bs.std()*np.sqrt(ninst/2000):.3f}); "
      f"noisy exact updater (log-normal sigma 0.9 on both steps) {numq.sum()/den.sum():.3f} (boot sd {bq.std():.3f}; at 2,000 ~{bq.std()*np.sqrt(ninst/2000):.3f})")

# dependence of q on the remaining budget, and dead-world decisions
rng = random.Random(0)
tvs_up, tvs_dn, n_dead, n_all, t_ball, t_dist = [], [], 0, 0, 0.0, 0.0
for it in range(200):
    ins = Inst(rng)
    recs, _ = run(ins, "q", order="power", extra=False)
    for d in recs:
        n_all += 1; n_dead += not d["true_live"]
        v, b, P, A = d["node"], d["b"], d["P"], d["A"]
        if b + 1 <= ins.B:
            x = ins.mix(v, ins.wmask(v, b + 1, P, A))
            tvs_up.append(tv(x, d["q"]))
        if b - 1 >= 1:
            x = ins.mix(v, ins.wmask(v, b - 1, P, A))
            if x is not None:
                tvs_dn.append(tv(x, d["q"]))
tvs_up, tvs_dn = np.array(tvs_up), np.array(tvs_dn)
print(f"\nq at the same (state, evidence) with one more accepted step of budget: mean TV {tvs_up.mean():.3f}, >0.05 on {np.mean(tvs_up>0.05):.2f} of {len(tvs_up)} decisions; "
      f"with one less: mean TV {tvs_dn.mean():.3f}, >0.05 on {np.mean(tvs_dn>0.05):.2f} of {len(tvs_dn)}")
print(f"q-greedy decisions at which the true world can no longer finish (no label would exist in training): {n_dead} of {n_all} ({n_dead/n_all:.3f})")

# what evidence can the small training sets show?
print("\n== training sets (nested, shuffle seeds 7..11): rules present in ALL / in NONE of the N training worlds; world sizes ==")
for sh in (7, 8, 9, 10, 11):
    order = list(range(1, 256)); random.Random(sh).shuffle(order)
    pool = order[64:]
    for N in (8, 16):
        t = np.array(pool[:N]) - 1
        cnt = HAS[t].sum(0)
        print(f"seed {sh} N={N}: per-rule count of training worlds containing it {cnt.tolist()} (rules {RULES}); never rejectable in training: {[RULES[j] for j in range(NR) if cnt[j]==N]}; never acceptable: {[RULES[j] for j in range(NR) if cnt[j]==0]}; sizes {sorted(WSIZE[t].tolist())}")

# cost split
import cProfile, pstats
rng = random.Random(3)
t0 = time.time()
for _ in range(30):
    ins = Inst(rng)
t1 = time.time()
rng = random.Random(3)
tb = 0.0
for _ in range(30):
    a = time.time(); ins = Inst.__new__(Inst)
    from topology_design.rewrite.data import sample_term
    while True:
        u = sample_term(rng, (2, 3, 4, 5)); k = rng.choice((2, 3, 4)); ins._ball(u, k + 2)
        sp = [i for i, d in enumerate(ins.fdist) if d == k]
        if sp:
            break
    rng.choice(sp); rng.random()
    tb += time.time() - a
print(f"\ncost per instance: total {(t1-t0)/30:.3f}s, of which building the full-world ball {tb/30:.3f}s")
