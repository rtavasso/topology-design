"""Model-free reference numbers for design revision 3 (after design review 1): the re-aimed statistics on several draws of
the world sets, with planted mechanisms and noisy stand-ins scored under them before any rule is written (L48, L50).

A draw = 64 held-out worlds and nested training sets T_N from the other 191. H8, rho = 0.5 (uniform prior over worlds).
Decisions at which the true world can no longer finish within the budget are not scored (the trajectory ends there).

A decision is in support when some training world agrees with the problem, the remaining budget and all verdicts so far;
there q_T is the Bayes predictor of the training distribution.

MECHANISMS.  Four named ways of answering, each computable exactly:
    M  memorising posterior q_T (prior on the training worlds)      G  generalising posterior q (prior on all worlds)
    O  optimistic planner o: no belief; plan in the single largest world not yet contradicted (every rule not rejected)
    D  deletion: answer as before, with the rejected action or rule removed
The statistics are least-squares coefficients of the model's output (or its change) on the mechanisms' outputs (or changes),
pooled over decisions (or probes); linear in the model, so independent output noise does not bias them.

LEVELS, in-support decisions:  (p - q_T) on (q - q_T, o - q_T).   M = (0, 0), G = (1, 0), O = (0, 1).

CHANGES.  A rejection probe at a decision: a rule r not yet tested, with an action at this state, that some surviving world
lacks.  Same state, same budget; evidence E against E + "r rejected".  RM_r deletes r's actions and renormalises.
    dp = RM_r[p(. | E + r-)] - RM_r[p(. | E)]        the model's change beyond deleting r, from its own two outputs
  dp = 0 for D: a model that ignores the evidence, deletes the tried action, or deletes the rule (any frozen belief).
  Reference changes:  dq, dq_T, do formed the same way from q, q_T, o;  and, when a training world fits E but none
                      fits E + r- (the model may switch its base at that moment):
                      g = RM_r[q(E)] - RM_r[q_T(E)]   (training prior -> full prior),   h = RM_r[o(E)] - RM_r[q_T(E)]   (-> planner).
  Probe classes: "in" (a training world fits before and after), "crossing" (before only), "beyond" (neither).
  Out of support (crossing + beyond):  dp on (dq, do, g, h).   G = (1, 0, 0, 0);  O = (0, 1, 0, 0);
      M in support and G outside = (1, 0, 1, 0);  M in support and O outside = (0, 1, 0, 1).
  In support:  dp on (dq_T, dq, do).

Usage: PYTHONPATH=src python3 experiments/belief/refs3.py heldout|gate|cheap [instances] > refs3_<part>.md
  heldout: held-out worlds, full-prior greedy trajectory (fixed tie rule: first maximiser in sorted action order).
  gate:    training worlds drawn as the generator will draw them (problem first, then a consistent training world), the
           memorising posterior's own greedy trajectory, in-support probes only.
"""
import collections, math, random, sys
import numpy as np
sys.path.insert(0, "experiments/belief")
import power as P

GRID = (4, 8, 16, 32, 64)
DRAWS = (7, 8, 9, 10, 11)
NOISE = 0.12
LAMS = (0.0, 0.5, 1.0)


def vec(d, acts):
    return np.array([d.get(a, 0.0) for a in acts])


def rm(p, units, r):
    if p is None:
        return None
    x = np.where(units == r, 0.0, p)
    z = x.sum()
    return x / z if z > 1e-12 else None


def rule_mask(p, units, A):
    x = np.where([bool(A >> int(j) & 1) for j in units], 0.0, p)
    z = x.sum()
    return x / z if z > 1e-12 else None


def wscores(ins, v, worlds, wt):
    """Bayes mixture over optimal actions with an arbitrary prior weight per world."""
    sc = collections.defaultdict(float)
    for w in worlds:
        dv = ins.dw(v, w)
        o = [(c, x, p) for c, x, p in ins.out[v] if w >> ins.unit_of[c] & 1 and ins.dw(x, w) == dv - 1]
        for a in o:
            sc[a] += wt[w] / len(o)
    z = sum(sc.values())
    return {a: x / z for a, x in sc.items()} if z > 0 else {}


def nearest(cands, Pm, A):
    """Training worlds that agree with the most verdicts (the fallback of a memoriser with no exact match)."""
    if not cands:
        return []
    agree = lambda w: bin(w & Pm).count("1") + bin(~w & A & 255).count("1")
    best = max(agree(w) for w in cands)
    return [w for w in cands if agree(w) == best]


class Reg:
    """Pooled least squares of model changes on reference directions, with sums kept per world for the bootstrap."""

    def __init__(self, k):
        self.k = k
        self.G = collections.defaultdict(lambda: np.zeros((k, k)))
        self.b = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(k)))
        self.n = 0

    def add(self, world, dirs, models):
        D = [d for d in dirs]
        G = self.G[world]
        for i in range(self.k):
            for j in range(self.k):
                if D[i] is not None and D[j] is not None:
                    G[i, j] += float(D[i] @ D[j])
        for name, dp in models.items():
            if dp is None:
                continue
            bb = self.b[world][name]
            for i in range(self.k):
                if D[i] is not None:
                    bb[i] += float(dp @ D[i])
        self.n += 1

    def _solve(self, worlds, name):
        G = sum((self.G[w] for w in worlds), np.zeros((self.k, self.k)))
        b = sum((self.b[w][name] for w in worlds), np.zeros(self.k))
        if G[0, 0] < 1e-6:
            return np.full(self.k, np.nan)
        try:
            return np.linalg.lstsq(G, b, rcond=1e-9)[0]
        except np.linalg.LinAlgError:
            return np.full(self.k, np.nan)

    def coef(self, name):
        return self._solve(list(self.G), name) if self.G else np.full(self.k, np.nan)

    def sd(self, name, g, nb=300, f=lambda c: c[0]):
        W = list(self.G)
        if len(W) < 2:
            return float("nan")
        return float(np.nanstd([f(self._solve([W[i] for i in g.integers(0, len(W), len(W))], name)) for _ in range(nb)]))

    def energy(self, i=0):
        return sum(G[i, i] for G in self.G.values())

    def corr(self):
        G = sum(self.G.values(), np.zeros((self.k, self.k)))
        return G[0, 1] / math.sqrt(G[0, 0] * G[1, 1]) if G[0, 0] > 0 and G[1, 1] > 0 else float("nan")


def dchange(p0, p1, units, r):
    a, c = rm(p1, units, r), rm(p0, units, r)
    return None if a is None or c is None else a - c


def walk(ins, policy_worlds, on_decision):
    """Greedy trajectory under the posterior with the prior on `policy_worlds` (None = all worlds)."""
    v, b, Pm, A, calls = 0, ins.B, 0, 0, 0
    while v != ins.s and b > 0 and calls < ins.B + 40:
        if ins.dw(v, ins.true) > b:
            break
        alive = [w for w in ins.cons if ins.dw(v, w) <= b]
        live = [w for w in alive if (w & Pm) == Pm and (w & A) == 0]
        pol = live if policy_worlds is None else [w for w in live if w in policy_worlds]
        qd = ins.scores(v, pol)
        if not qd:
            break
        on_decision(v, alive, live, Pm, A, calls)
        a = max(sorted(qd), key=lambda t: qd[t])
        calls += 1
        j = ins.unit_of[a[0]]
        if ins.true >> j & 1:
            v, b, Pm = a[1], b - 1, Pm | (1 << j)
        else:
            A |= 1 << j


def calibrated_sigma(g):
    pts, r0 = [], random.Random(0)
    for _ in range(12):
        i0 = P.Instance(r0, "H8")
        pts.append(vec(i0.scores(0, i0.cons), i0.out[0]))
    return P.calibrate(pts * 40, NOISE, g)


def draw_sets(draw):
    order = list(range(1, 256))
    random.Random(draw).shuffle(order)
    heldout, pool = set(order[:64]), order[64:]
    trains = {N: set(pool[:N]) for N in GRID}
    margs = {}
    for N, T in trains.items():
        pr = [(sum(w >> j & 1 for w in T) + 1) / (len(T) + 2) for j in range(8)]
        margs[N] = {w: math.prod(pr[j] if w >> j & 1 else 1 - pr[j] for j in range(8)) for w in range(1, 256)}
    return heldout, trains, margs


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def tri(R, m, k=None):
    c = R.coef(m) if R.n else np.full(R.k, np.nan)
    return ", ".join(fmt(float(x)) for x in (c if k is None else c[:k]))


def heldout_part(n_inst):
    g = np.random.default_rng(0)
    SIG = calibrated_sigma(g)
    lev, out, ins_rows = [], [], []
    for draw in DRAWS:
        heldout, trains, margs = draw_sets(draw)
        rng = random.Random(100 + draw)
        RL = {N: Reg(2) for N in GRID}          # levels, in support: (q - q_T, o - q_T)
        RO = {N: Reg(4) for N in GRID}          # changes, out of support: (dq, do, g, h)
        RI = {N: Reg(3) for N in GRID}          # changes, in support: (dq_T, dq, do)
        cnt = {N: collections.Counter() for N in GRID}
        n = 0
        while n < n_inst:
            ins = P.Instance(rng, "H8")
            if ins.true not in heldout:
                continue
            n += 1

            def dec(v, alive, live, Pm, A, calls):
                acts = ins.out[v]
                units = np.array([ins.unit_of[c] for c, _x, _p in acts])
                q = vec(ins.scores(v, live), acts)
                o = vec(ins.scores(v, [255 & ~A]), acts)                # optimistic planner: every rule not yet rejected
                fz_full = rule_mask(vec(ins.scores(v, alive), acts), units, A)
                probes = []
                for r in sorted(set(units.tolist())):
                    if (Pm | A) >> r & 1:
                        continue
                    wr = [w for w in live if not w >> r & 1]
                    base = rm(q, units, r)
                    if not wr or len(wr) == len(live) or base is None:
                        continue
                    q1 = vec(ins.scores(v, wr), acts)
                    o1 = vec(ins.scores(v, [255 & ~(A | (1 << r))]), acts)
                    probes.append((r, wr, q1, base, o1, dchange(o, o1, units, r)))
                for N, T in trains.items():
                    c = cnt[N]
                    aliveT = [w for w in alive if w in T]
                    liveT = [w for w in live if w in T]
                    c["dec"] += 1
                    qT = vec(ins.scores(v, liveT), acts) if liveT else None
                    m0 = vec(wscores(ins, v, live, margs[N]), acts)
                    if liveT:
                        c["insup"] += 1
                        fzT = rule_mask(vec(ins.scores(v, aliveT), acts), units, A)
                        models = {"post": q - qT, "plan": o - qT, "marg": m0 - qT,
                                  "fzT": None if fzT is None else fzT - qT, "fzF": None if fz_full is None else fz_full - qT,
                                  "mixGO": 0.5 * (q + o) - qT, "splan": P.noisy(o, SIG, g) - qT}
                        for lam in LAMS:
                            models[f"s{lam}"] = P.noisy((1 - lam) * qT + lam * q, SIG, g) - qT
                        RL[N].add(ins.true, (q - qT, o - qT), models)
                    near0 = qT if liveT else (vec(ins.scores(v, nearest(aliveT, Pm, A)), acts) if aliveT else None)
                    for r, wr, q1, base, o1, do in probes:
                        dq = q1 - base
                        wrT = [w for w in wr if w in T]
                        qT1 = vec(ins.scores(v, wrT), acts) if wrT else None
                        mem0 = qT if liveT else q                 # training optimum where it is defined, exact outside
                        mem1 = qT1 if wrT else q1
                        near1 = qT1 if wrT else (vec(ins.scores(v, nearest(aliveT, Pm, A | (1 << r))), acts) if aliveT else None)
                        mb = rm(mem0, units, r)
                        models = {
                            "post": dq, "plan": do,
                            "mem_q": dchange(mem0, mem1, units, r),
                            "mem_plan": dchange(qT if liveT else o, qT1 if wrT else o1, units, r),     # training optimum in support, planner outside
                            "mem_near": dchange(near0, near1, units, r) if near0 is not None and near1 is not None else None,
                            "marg": dchange(m0, vec(wscores(ins, v, wr, margs[N]), acts), units, r),
                            "half": None if mb is None else dchange(mem0, 0.5 * mb + 0.5 * mem1, units, r),
                            "mixGO": None if do is None else 0.5 * dq + 0.5 * do,
                            "splan": dchange(P.noisy(o, SIG, g), P.noisy(o1, SIG, g), units, r),
                        }
                        if mb is not None:
                            for lam in LAMS:
                                models[f"s{lam}"] = dchange(P.noisy(mem0, SIG, g), P.noisy((1 - lam) * mb + lam * mem1, SIG, g), units, r)
                        if wrT:
                            c["p_in"] += 1
                            bT = rm(qT, units, r)
                            RI[N].add(ins.true, (None if bT is None else qT1 - bT, dq, do), models)
                        else:
                            bT = rm(qT, units, r) if liveT else None
                            c["p_cross" if liveT else "p_beyond"] += 1
                            ob = rm(o, units, r)
                            RO[N].add(ins.true, (dq, do, None if bT is None else base - bT, None if bT is None or ob is None else ob - bT), models)

            walk(ins, None, dec)
            ins._d = ins.R = ins.out = None
        for N in GRID:
            c, L, R, I = cnt[N], RL[N], RO[N], RI[N]
            f0, f1, fs = (lambda x: x[0]), (lambda x: x[1]), (lambda x: x[2])
            lev.append(f"| {draw} | {N} | {c['insup'] / c['dec']:.0%} | {L.energy(0) / n_inst:.3f} | {fmt(L.corr())} | {tri(L, 'post')} | {tri(L, 'plan')} | {tri(L, 'mixGO')} | {tri(L, 'marg')} | {tri(L, 'fzT')} | {tri(L, 'fzF')} | "
                       + " / ".join(tri(L, f"s{l}") for l in LAMS) + f" | {tri(L, 'splan')} | {fmt(L.sd('s0.5', g, f=f0), 3)}, {fmt(L.sd('s0.5', g, f=f1), 3)} |")
            out.append(f"| {draw} | {N} | {c['p_cross'] / n_inst:.1f} / {c['p_beyond'] / n_inst:.1f} | {R.energy() / n_inst:.3f} | {tri(R, 'post')} | {tri(R, 'plan')} | {tri(R, 'mem_q')} | {tri(R, 'mem_plan')} | {tri(R, 'mixGO')} | {tri(R, 'mem_near')} | {tri(R, 'marg')} | {tri(R, 'half')} | "
                       + " / ".join(tri(R, f"s{l}") for l in LAMS) + f" | {tri(R, 'splan')} | {fmt(R.sd('s0.5', g, f=f0), 3)}, {fmt(R.sd('s0.5', g, f=f1), 3)} |")
            ins_rows.append(f"| {draw} | {N} | {c['p_in'] / n_inst:.1f} | {I.energy(0) / n_inst:.3f} / {I.energy(1) / n_inst:.3f} | {tri(I, 'mem_q')} | {tri(I, 'post')} | {tri(I, 'plan')} | {tri(I, 'marg')} | {tri(I, 'half')} | "
                            f"{fmt(I.sd('s0.5', g, f=f0), 3)}, {fmt(I.sd('s0.5', g, f=f1), 3)}, {fmt(I.sd('s0.5', g, f=fs), 3)} |")
        print(f"<!-- draw {draw} done -->", flush=True)
    print(f"# Reference numbers for design revision 3 — held-out worlds (model-free)\n")
    print(f"H8, ρ = 0.5, Δ = 2, k ∈ {{2, 3, 4}}. {n_inst} held-out-world instances per draw; five draws of the world sets. Definitions in `refs3.py`. "
          f"Stand-ins: independent noise at TV {NOISE} from the target on each output. Sampling sd: bootstrap over held-out worlds, for the stand-in at 0.5.\n")
    print("## Levels at in-support decisions — which mechanism the output sits at\n")
    print("Coefficients of (p − q_T) on (q − q_T, o − q_T): the memorising posterior is (0, 0), the generalising posterior (1, 0), the optimistic planner (0, 1).\n")
    print("| draw | training worlds | decisions in support | signal per instance, Σ\\|q − q_T\\|² | correlation of the two directions | posterior | planner | half posterior, half planner | product-of-marginals prior, updated | training prior frozen + rule mask | full prior frozen + rule mask | stand-in between the two posteriors at 0 / 0.5 / 1 | noisy planner | sd of the two coefficients |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lev))
    print("\n## Changes at out-of-support rejection probes — what the model does where no training world fits\n")
    print("Coefficients of the model's change on (dq, do, g, h): the posterior's change beyond deletion, the optimistic planner's change, and the two switches of base at a crossing probe (training prior → full prior, training prior → planner). Deleting models are all zero by construction.\n")
    print("| draw | training worlds | probes per instance, crossing / beyond | signal per instance, Σ\\|dq\\|² | exact posterior | planner | training optimum in support, posterior outside | training optimum in support, planner outside | half posterior, half planner | memoriser → nearest training worlds | product-of-marginals prior | half-way updater | stand-in at 0 / 0.5 / 1 | noisy planner | sd of the coefficients on dq and do |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(out))
    print("\n## Changes at in-support rejection probes in held-out worlds (reported, not scored)\n")
    print("Coefficients on (dq_T, dq, do).\n")
    print("| draw | training worlds | probes per instance | signal per instance, Σ\\|dq_T\\|² / Σ\\|dq\\|² | memorising posterior | exact posterior | planner | product-of-marginals prior | half-way updater (memorising) | sd of the three coefficients |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(ins_rows))


def gate_part(n_inst):
    g = np.random.default_rng(1)
    SIG = calibrated_sigma(g)
    rows = []
    for draw in DRAWS[:3]:
        heldout, trains, margs = draw_sets(draw)
        for N, T in trains.items():
            rng = random.Random(1000 * draw + N)
            R = Reg(3)
            cnt = collections.Counter()
            n = tries = 0
            while n < n_inst:
                ins = P.Instance(rng, "H8")
                tries += 1
                cand = sorted(w for w in ins.cons if w in T)
                if not cand:
                    continue
                ins.true = rng.choice(cand)
                n += 1

                def dec(v, alive, live, Pm, A, calls):
                    acts = ins.out[v]
                    units = np.array([ins.unit_of[c] for c, _x, _p in acts])
                    liveT = [w for w in live if w in T]
                    qT, q = vec(ins.scores(v, liveT), acts), vec(ins.scores(v, live), acts)
                    o = vec(ins.scores(v, [255 & ~A]), acts)
                    m0 = vec(wscores(ins, v, live, margs[N]), acts)
                    for r in sorted(set(units.tolist())):
                        if (Pm | A) >> r & 1:
                            continue
                        wrT = [w for w in liveT if not w >> r & 1]
                        if len(wrT) == len(liveT):
                            continue
                        if not wrT:
                            cnt["leaves"] += 1
                            continue
                        bT, bq = rm(qT, units, r), rm(q, units, r)
                        if bT is None or bq is None:
                            continue
                        wr = [w for w in live if not w >> r & 1]
                        qT1, q1 = vec(ins.scores(v, wrT), acts), vec(ins.scores(v, wr), acts)
                        do = dchange(o, vec(ins.scores(v, [255 & ~(A | (1 << r))]), acts), units, r)
                        models = {"mem": qT1 - bT, "post": q1 - bq, "plan": do,
                                  "marg": dchange(m0, vec(wscores(ins, v, wr, margs[N]), acts), units, r),
                                  "half": 0.5 * (qT1 - bT)}
                        for lam in LAMS:
                            models[f"s{lam}"] = dchange(P.noisy(qT, SIG, g), P.noisy((1 - lam) * bT + lam * qT1, SIG, g), units, r)
                        cnt["probe"] += 1
                        R.add(ins.true, (qT1 - bT, q1 - bq, do), models)

                walk(ins, T, dec)
                ins._d = ins.R = ins.out = None
            tot = lambda m: float(R.coef(m).sum())
            rows.append(f"| {draw} | {N} | {n / tries:.0%} | {cnt['probe'] / n_inst:.1f} | {cnt['leaves'] / n_inst:.1f} | {R.energy(0) / n_inst:.3f} | "
                        f"{fmt(tot('mem'))} | {fmt(tot('post'))} | {fmt(tot('plan'))} | {fmt(tot('marg'))} | {fmt(tot('half'))} | " + " / ".join(fmt(tot(f's{l}')) for l in LAMS)
                        + f" | {fmt(R.sd('s0.5', g, f=lambda x: x.sum()), 3)} |")
        print(f"<!-- draw {draw} done -->", flush=True)
    print("# Reference numbers for design revision 3 — the gate, training worlds (model-free)\n")
    print(f"H8. {n_inst} instances per cell, three draws. Rejection probes that stay in support, on the memorising posterior's own greedy trajectory. "
          "The statistic is the sum of the three coefficients of the model's change on (dq_T, dq, do): updating by any of the three non-deleting mechanisms. Deleting models read 0 by construction.\n")
    print("| draw | training worlds | problems with a consistent training world | probes per instance | probes that would leave support (not used) | signal per instance, Σ\\|dq_T\\|² | memorising posterior | exact full posterior | planner | product-of-marginals prior | half-way updater | stand-in at 0 / 0.5 / 1 | sd |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


def cheap_part(n_inst):
    """Two cheap mechanisms that are not a posterior, scored under the same statistics (L50).
    optimistic planner: no belief; plan in the single largest world not yet contradicted (all rules minus the rejected ones),
        uniform over that world's optimal actions.
    rule table: after 'r rejected', multiply the mass of every remaining action by a factor that depends only on (r, the
        action's rule) -- an 8 x 8 table, fitted here on the exact posterior's own probes (the most favourable fit)."""
    import torch
    g = np.random.default_rng(2)
    # ---- fit the table on independent instances
    rng = random.Random(4242)
    fit, one = [], collections.defaultdict(float)
    for _ in range(n_inst):
        ins = P.Instance(rng, "H8")

        def dec(v, alive, live, Pm, A, calls):
            acts = ins.out[v]
            units = np.array([ins.unit_of[c] for c, _x, _p in acts])
            q = vec(ins.scores(v, live), acts)
            for r in sorted(set(units.tolist())):
                if (Pm | A) >> r & 1:
                    continue
                wr = [w for w in live if not w >> r & 1]
                base = rm(q, units, r)
                if not wr or len(wr) == len(live) or base is None:
                    continue
                q1 = vec(ins.scores(v, wr), acts)
                fit.append((r, units.copy(), base, q1))
                one["den"] += float((q1 - base) @ (q1 - base))
                one["blind"] += float((q - base) @ (q1 - base))       # a model that ignores the verdict, scored without RM_r on its second output

        walk(ins, None, dec)
        ins._d = ins.R = ins.out = None
    theta = torch.zeros(8, 8, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    data = [(r, torch.tensor(u), torch.tensor(b), torch.tensor(t)) for r, u, b, t in fit]

    def table(r, u, b, th):
        w = b * torch.exp(th[r][u])
        return w / w.sum()

    for _ in range(150):
        opt.zero_grad()
        loss = sum(((table(r, u, b, theta) - t) ** 2).sum() for r, u, b, t in data) / len(data)
        loss.backward()
        opt.step()
    th = theta.detach()
    base_loss = sum(float(((b - t) ** 2).sum()) for r, u, b, t in data) / len(data)
    rows = []
    for draw in DRAWS[:3]:
        heldout, trains, margs = draw_sets(draw)
        rng = random.Random(300 + draw)
        GA = {N: collections.defaultdict(float) for N in GRID}
        RO = {N: Reg(2) for N in GRID}
        RI = {N: Reg(2) for N in GRID}
        n = 0
        while n < n_inst:
            ins = P.Instance(rng, "H8")
            if ins.true not in heldout:
                continue
            n += 1

            def dec(v, alive, live, Pm, A, calls):
                acts = ins.out[v]
                units = np.array([ins.unit_of[c] for c, _x, _p in acts])
                q = vec(ins.scores(v, live), acts)
                opt0 = vec(ins.scores(v, [255 & ~A]), acts)
                for N, T in trains.items():
                    liveT = [w for w in live if w in T]
                    qT = vec(ins.scores(v, liveT), acts) if liveT else None
                    if liveT:
                        diff = q - qT
                        GA[N]["den"] += float(diff @ diff)
                        GA[N]["opt"] += float((opt0 - qT) @ diff)
                    for r in sorted(set(units.tolist())):
                        if (Pm | A) >> r & 1:
                            continue
                        wr = [w for w in live if not w >> r & 1]
                        base = rm(q, units, r)
                        if not wr or len(wr) == len(live) or base is None:
                            continue
                        q1 = vec(ins.scores(v, wr), acts)
                        dq = q1 - base
                        wopt = 255 & ~(A | (1 << r))
                        models = {"table": table(r, torch.tensor(units), torch.tensor(base), th).numpy() - base}
                        if wopt and ins.dw(v, wopt) < 99:
                            models["opt"] = dchange(opt0, vec(ins.scores(v, [wopt]), acts), units, r)
                        wrT = [w for w in wr if w in T]
                        if wrT:
                            bT = rm(qT, units, r)
                            if bT is not None:
                                RI[N].add(ins.true, (vec(ins.scores(v, wrT), acts) - bT, dq), models)
                        else:
                            gdir = (base - rm(qT, units, r)) if liveT and rm(qT, units, r) is not None else None
                            RO[N].add(ins.true, (dq, gdir), models)

            walk(ins, None, dec)
            ins._d = ins.R = ins.out = None
        for N in GRID:
            den = GA[N]["den"] or float("nan")
            co = lambda R, m: R.coef(m) if R.n else np.full(2, np.nan)
            rows.append(f"| {draw} | {N} | {GA[N]['opt'] / den:.2f} | {fmt(float(co(RO[N], 'opt')[0]))} | {fmt(float(co(RI[N], 'opt')[0]))}, {fmt(float(co(RI[N], 'opt')[1]))} | "
                        f"{fmt(float(co(RO[N], 'table')[0]))} | {fmt(float(co(RI[N], 'table')[0]))}, {fmt(float(co(RI[N], 'table')[1]))} |")
        print(f"<!-- draw {draw} done -->", flush=True)
    print("# Reference numbers for design revision 3 — two cheap mechanisms that hold no belief (model-free)\n")
    print(f"H8, {n_inst} held-out-world instances per draw, three draws. Definitions in `refs3.py` (`cheap_part`).\n")
    print(f"Rule table: fitted on {len(fit)} probes of the exact full-prior posterior from {n_inst} other instances; squared error per probe {base_loss:.4f} for plain deletion, {float(loss):.4f} with the fitted table.\n")
    print(f"Why RM_r is applied to both of the model's outputs: with dp = p(E + r−) − RM_r[p(E)], a model that ignores the verdict (p = q(E) both times) reads {one['blind'] / one['den']:.2f} on the single direction dq; with RM_r on both it reads 0 exactly.\n")
    print("Single-direction readings (levels: (p − q_T) on (q − q_T); changes: on (dq, g) out of support and (dq_T, dq) in support) — the statistics as they stood before the planner was named.\n")
    print("| draw | training worlds | optimistic planner: gamma | optimistic planner: beta_cf out of support | optimistic planner: in-support probes (dq_T, dq) | rule table: beta_cf out of support | rule table: in-support probes (dq_T, dq) |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    part = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    {"heldout": heldout_part, "gate": gate_part, "cheap": cheap_part}[part](n)
