"""Q2 (coverage): recompute the generalising-vs-memorising separation by N, how often the memorising reference's fallback
fires, how much of gamma's denominator sits on fallback decisions, where sensible non-memorising learners sit on gamma,
leakage between held-out and training worlds, and the variation over draws of the world sets.
Protocol of the primary configuration follows power_coverage.py (instances: random.Random(2); held-out and nested
training sets: random.Random(7).shuffle), re-implemented."""
import random, sys, time, pickle, collections
import numpy as np
from core import *
from reftrace import *

NINST = int(sys.argv[1]) if len(sys.argv) > 1 else 800
SHUF = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [7, 8, 9, 10, 11]
GRID = (8, 16, 32, 64, 128)
CFG = {}
for sh in SHUF:
    order = list(range(1, 256))
    random.Random(sh).shuffle(order)
    held = np.zeros(NW, dtype=bool); held[np.array(order[:64]) - 1] = True
    pool = order[64:]
    trains = {}
    for N in GRID:
        t = np.zeros(NW, dtype=bool); t[np.array(pool[:N]) - 1] = True
        trains[N] = t
    CFG[sh] = (held, trains)


def prod_prior(tmask, N):
    nr = HAS[tmask].sum(0)
    rho = (nr + 1.0) / (N + 2.0)
    return np.prod(np.where(HAS, rho[None, :], 1 - rho[None, :]), axis=1)


def qtrain(ins, i, b, P, A, rh, tmask, q):
    """Author's memorising reference (power.py Instance.trace), re-implemented. Returns (vector, status):
    status 'nocons' (no training world consistent with (u, s): reference := q), 'ok', 'fb_mask' (no training world agrees
    with the evidence: frozen training belief with actions rejected at this state masked), 'fb_q' (reference := q)."""
    base = ins.cons & tmask
    if not base.any():
        return q, "nocons"
    v = ins.mix(i, ins.wmask(i, b, P, A, base=base))
    if v is not None:
        return v, "ok"
    v = masked(ins.mix(i, ins.wmask(i, b, base=base)), rh)
    if v is not None:
        return v, "fb_mask"
    return q, "fb_q"


def alt_fallbacks(ins, i, b, P, A, rh, tmask, q, rules):
    """Other things a memoriser could do when no training world agrees with the evidence."""
    base = ins.cons & tmask
    out = {}
    fr = ins.mix(i, ins.wmask(i, b, base=base))
    rr = [j for j in range(len(rules)) if A >> int(rules[j]) & 1]
    out["fb_rulemask"] = masked(fr, rr)                                  # frozen training belief, rejected RULES removed
    # nearest training worlds: those agreeing with the largest number of verdicts
    live = ins.wmask(i, b, base=base)
    if live.any():
        agree = np.zeros(NW, dtype=int)
        for r in range(NR):
            if P >> r & 1:
                agree += HAS[:, r]
            if A >> r & 1:
                agree += ~HAS[:, r]
        best = agree[live].max()
        near = live & (agree == best)
        out["fb_nearest"] = masked(ins.mix(i, near), rr)
    else:
        out["fb_nearest"] = None
    return out


rng = random.Random(2)
stats = {sh: {N: collections.defaultdict(float) for N in GRID} for sh in SHUF}
recs_primary = []          # per-decision records for the primary configuration (for bootstrap / further cuts)
count = {sh: 0 for sh in SHUF}
train_side = collections.defaultdict(float)
made = 0
t0 = time.time()
primary = SHUF[0]
while count[primary] < NINST:
    ins = Inst(rng)
    made += 1
    use = [sh for sh in SHUF if CFG[sh][0][ins.true] and (sh == primary or count[sh] < NINST)]
    in_train64 = CFG[primary][1][64][ins.true]
    if not use and not in_train64:
        continue
    recs, summ = run(ins, "q", order="power", extra=True)
    rules_in_ball = np.zeros(NR, dtype=bool); rules_in_ball[np.unique(ins.er)] = True
    for sh in use:
        count[sh] += 1
        held, trains = CFG[sh]
        for N in GRID:
            tm = trains[N]
            S = stats[sh][N]
            S["inst"] += 1
            S["inst_nocons"] += not (ins.cons & tm).any()
            # leakage: a training world identical to the true world on every rule that labels an edge of the ball
            same = (HAS[tm][:, rules_in_ball] == HAS[ins.true, rules_in_ball][None, :]).all(1).any()
            S["inst_equiv_train"] += bool(same)
            ever_contradicted = False
            pw = prod_prior(tm, N)
            for d in recs:
                i, b, P, A, rh, q = d["node"], d["b"], d["P"], d["A"], d["rej_here"], d["q"]
                e0, e1 = ins.estart[i], ins.estart[i + 1]
                rules = ins.er[e0:e1]
                qt, st = qtrain(ins, i, b, P, A, rh, tm, q)
                e = float(((q - qt) ** 2).sum()); t = tv(q, qt)
                S["dec"] += 1; S["tv"] += t; S["tv05"] += t > 0.05; S["den"] += e
                S["dec_" + st] += 1; S["den_" + st] += e; S["tv_" + st] += t
                first = d["calls"] == 0
                S["den_first"] += e * first; S["dec_first"] += first
                if st.startswith("fb"):
                    ever_contradicted = True
                # candidate learners projected on (qt -> q)
                dq = q - qt
                cands = {}
                cands["prod"] = ins.mix(i, ins.wmask(i, b, P, A), pw)                    # product of empirical marginals, exact update
                cands["mask_gen"] = d["m"]                                               # full prior, mask only
                cands["rm_gen"] = d["rm"]                                                # full prior, rule mask
                if st == "fb_mask":
                    alt = alt_fallbacks(ins, i, b, P, A, rh, tm, q, rules)
                    cands["mem_fb_rulemask"] = alt["fb_rulemask"] if alt["fb_rulemask"] is not None else qt
                    cands["mem_fb_nearest"] = alt["fb_nearest"] if alt["fb_nearest"] is not None else qt
                    cands["mem_fb_q"] = q
                else:
                    cands["mem_fb_rulemask"] = cands["mem_fb_nearest"] = cands["mem_fb_q"] = qt
                for nm, p in cands.items():
                    if p is None:
                        S["miss_" + nm] += 1
                        continue
                    S["num_" + nm] += float((p - qt) @ dq)
                    if st == "ok":
                        S["numok_" + nm] += float((p - qt) @ dq)
                if sh == primary and N in (8, 16, 32, 64):
                    recs_primary.append((N, ins.true, count[sh], st, first, e, t,
                                         float((cands["prod"] - qt) @ dq) if cands["prod"] is not None else 0.0))
            S["inst_contradicted"] += ever_contradicted
    if in_train64:
        # P1 is read on TRAINING worlds at N = 64: which posterior is 'exact' there?
        tm = CFG[primary][1][64]
        T = train_side
        T["inst"] += 1
        for d in recs:
            if d["calls"] == 0 or d["m"] is None:
                continue
            i, b, P, A, rh, q, m = d["node"], d["b"], d["P"], d["A"], d["rej_here"], d["q"], d["m"]
            base = ins.cons & tm
            qt = ins.mix(i, ins.wmask(i, b, P, A, base=base))
            mt = masked(ins.mix(i, ins.wmask(i, b, base=base)), rh)
            if qt is None or mt is None:
                T["skipped"] += 1
                continue
            T["dec"] += 1
            T["tv_q_qt"] += tv(q, qt); T["tv_m_mt"] += tv(m, mt); T["tv_q_m"] += tv(q, m); T["tv_qt_mt"] += tv(qt, mt)
            dq = q - m
            T["den_full"] += float(dq @ dq)
            T["num_qt_full"] += float((qt - m) @ dq)          # Bayes learner of its 64-world data, scored with full-prior refs
            T["num_mt_full"] += float((mt - m) @ dq)          # mask-only learner with the 64-world prior, same refs
            dt = qt - mt
            T["den_tr"] += float(dt @ dt)
            T["num_q_tr"] += float((q - mt) @ dt)             # generalising posterior scored with training-prior refs
            T["num_m_tr"] += float((m - mt) @ dt)
            # model-relative: each learner's own update against its own mask
            T["num_own_qt_full"] += float((qt - mt) @ dq)
print(f"generated {made} instances, {time.time()-t0:.0f}s; held-out instances per shuffle: {count}")

for sh in SHUF:
    print(f"\n=== world-set draw (shuffle seed {sh}) ===")
    print("N | inst | mean TV(q,q_train) | TV>0.05 | inst with no consistent train world | decisions: ok / fb_mask / fb_q / nocons | "
          "share of gamma denominator: ok / fb_mask | first-decision share of denominator | inst with train world equal to true on all rules in ball | inst where evidence contradicts every train world")
    for N in GRID:
        S = stats[sh][N]
        dec, den = S["dec"], S["den"]
        print(f"{N} | {int(S['inst'])} | {S['tv']/dec:.3f} | {S['tv05']/dec:.2f} | {S['inst_nocons']/S['inst']:.2f} | "
              f"{S['dec_ok']/dec:.2f} / {S['dec_fb_mask']/dec:.2f} / {S['dec_fb_q']/dec:.2f} / {S['dec_nocons']/dec:.2f} | "
              f"{S['den_ok']/den:.2f} / {S['den_fb_mask']/den:.2f} | {S['den_first']/den:.2f} | {S['inst_equiv_train']/S['inst']:.2f} | {S['inst_contradicted']/S['inst']:.2f}")
    print("gamma of candidate learners (0 = author's memorising reference, 1 = generalising posterior):")
    print("N | product-of-empirical-marginals posterior | same, on in-support decisions only | full-prior mask-only | full-prior rule mask | memoriser, fallback = rule mask | memoriser, fallback = nearest train worlds | memoriser, fallback = q")
    for N in GRID:
        S = stats[sh][N]
        den = S["den"]
        g = lambda k: S["num_" + k] / den
        print(f"{N} | {g('prod'):.2f} | {S['numok_prod']/max(S['den_ok'],1e-9):.2f} | {g('mask_gen'):.2f} | {g('rm_gen'):.2f} | {g('mem_fb_rulemask'):.2f} | {g('mem_fb_nearest'):.2f} | {g('mem_fb_q'):.2f}")

T = train_side
print(f"\n=== P1's setting: training-world instances at N = 64 (primary draw), {int(T['inst'])} instances, {int(T['dec'])} post-verdict decisions, skipped {int(T['skipped'])} ===")
print(f"mean TV: q vs q_train64 {T['tv_q_qt']/T['dec']:.3f}; m vs m_train64 {T['tv_m_mt']/T['dec']:.3f}; q vs m {T['tv_q_m']/T['dec']:.3f}; q_train64 vs m_train64 {T['tv_qt_mt']/T['dec']:.3f}")
print(f"beta, references = full-prior (q, m):  Bayes learner of its own 64-world data (q_train64) {T['num_qt_full']/T['den_full']:.3f}; "
      f"mask-only learner with the 64-world prior (m_train64) {T['num_mt_full']/T['den_full']:.3f}; own-baseline <q_t - m_t, q - m>/|q-m|^2 {T['num_own_qt_full']/T['den_full']:.3f}")
print(f"beta, references = training-prior (q_train64, m_train64):  generalising posterior q {T['num_q_tr']/T['den_tr']:.3f}; full-prior mask m {T['num_m_tr']/T['den_tr']:.3f}")
pickle.dump(dict(stats={sh: {N: dict(stats[sh][N]) for N in GRID} for sh in SHUF}, recs=recs_primary, train_side=dict(T)), open("cov.pkl", "wb"))
