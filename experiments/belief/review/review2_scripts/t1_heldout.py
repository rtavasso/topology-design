"""Held-out worlds: levels in support, changes out of support (dG, dO, g, h) and in support (dM, dG, dO), for the design's
planted models and for a list of cheap mechanisms the design did not plant. Classes by DESIGN §8 from a bootstrap over
held-out worlds.

Usage: PYTHONPATH=src python3 t1_heldout.py DRAW N_INST [stream=author|design] [seed]
  stream=author : the instance stream of refs3.py (Random(100 + draw); world from the full prior, kept if held out), so
                  that numbers can be compared one to one with refs3_heldout.md.
  stream=wide   : as design, but the held-out set is every world outside T_32 (223 worlds; a change this review proposes).
  stream=design : DESIGN §3 as written (problem first; world uniform over the held-out worlds that can solve it; a
                  problem none can solve is redrawn).
Writes outputs/heldout_<stream>_d<DRAW>_n<N>.pkl and prints a table.
"""
import sys, pickle, time
from r2lib import *

RHOS = (0.2, 0.65, 0.8, 0.9, 0.97)
TAUS = (0.3, 1.0)
NOISE_TV = 0.12
NOISY = ("G", "O", "fzF", "memG", "memNear", "M")


class Ctx:
    """Per-instance caches and per-draw training-set summaries."""

    def __init__(self, ins, trains, margs, tsum, true=0):
        self.ins, self.trains, self.margs, self.tsum, self.true = ins, trains, margs, tsum, true
        self.pc = {}

    def pcount(self, wid):
        c = self.pc.get(wid)
        if c is None:
            c = self.pc[wid] = shortest_path_counts(self.ins, wid)
        return c


def popcount(x):
    return bin(x).count("1")


def mech_free(d, P, A, cx):
    """Mechanisms that do not depend on the training set. Returns {name: vector or None}."""
    ins = d.ins
    fit = d.fit(P, A)
    out = {}
    G = d.mix(fit)
    oid = 255 & ~A
    O = d.single(oid)
    out["G"], out["O"] = G, O
    out["mixGO"] = None if G is None or O is None else 0.5 * G + 0.5 * O
    for rho in RHOS:
        out[f"Grho{rho}"] = d.mix(fit, prior_vec(rho))
    # planners without a belief
    if O is not None:
        Do = ins.D[oid - 1]
        dv = int(Do[d.v])
        pc = cx.pcount(oid)
        w = np.where(O > 0, pc[d.dst], 0.0)
        out["Opaths"] = w / w.sum() if w.sum() > 0 else None
        allowed = (A >> d.rules) & 1 == 0
        feas = allowed & (Do[d.dst] <= d.b - 1)
        gap = (Do[d.dst].astype(float) - (dv - 1))
        for tau in TAUS:
            w = np.where(feas, np.exp(-gap / tau), 0.0)
            out[f"Osoft{tau}"] = w / w.sum()
        out["feas"] = feas / feas.sum()
        # leave-one-out hedged planner
        acc, k = O.copy(), 1
        for j in range(NR):
            if (P | A) >> j & 1:
                continue
            x = d.single(oid & ~(1 << j))
            if x is not None:
                acc += x; k += 1
        out["LOO"] = acc / k
    else:
        for n in ["Opaths", "feas", "LOO"] + [f"Osoft{t}" for t in TAUS]:
            out[n] = None
    pe = d.single(P) if P else None
    out["pess"] = pe if pe is not None else O
    if fit.any():
        mj = HAS[fit].mean(0)
        wid = (sum(1 << j for j in range(NR) if mj[j] > 0.5) | P) & ~A
        x = d.single(wid)
        out["MAPplan"] = x if x is not None else O
    else:
        out["MAPplan"] = O
    fz = d.mix(d.alive)
    out["fzF"] = d.rmA(fz, A)
    out["qPrm"] = d.rmA(d.mix(d.fit(P, 0)), A)
    out["qA"] = d.mix(d.fit(0, A))
    nb = ins.cons & (d.Dv <= ins.B)
    if P:
        nb = nb & ((WORLDS & P) == P)
    if A:
        nb = nb & ((WORLDS & A) == 0)
    out["Gnb"] = d.mix(nb)
    U = d.uniform(A)
    X = out["fzF"]
    c = min(1.0, 0.34 * popcount(A))
    out["countflat"] = None if X is None or U is None else (1 - c) * X + c * U
    if X is not None:
        w = X ** (1 + popcount(A))
        out["sharpen"] = w / w.sum()
    else:
        out["sharpen"] = None
    out["supG"] = None if G is None else (G > 0) / (G > 0).sum()
    if G is not None:
        for a in (3.0, 0.5):
            w = np.where(G > 0, G ** a, 0.0)
            out[f"Gpow{a}"] = w / w.sum()
        hard = np.zeros_like(G); hard[argmax_first(ins, d.v, G)] = 1.0
        out["Ghard"] = hard
        # hedged planner: optimistic-optimal actions weighted by how many live worlds... (not used)
    else:
        out["Gpow3.0"] = out["Gpow0.5"] = out["Ghard"] = None
    even = cx.true % 2 == 0
    out["hetGD"] = G if even else out["fzF"]
    out["hetGO"] = G if even else O
    out["U"] = U
    return out


def nearest_mask(d, T, P, A):
    cand = np.nonzero(d.alive & T)[0]
    if len(cand) == 0:
        return None
    ids = cand + 1
    agree = np.array([popcount(int(w) & P) + popcount(~int(w) & A & 255) for w in ids])
    m = np.zeros(NW, bool)
    m[cand[agree == agree.max()]] = True
    return m


def mech_T(d, P, A, cx, N, free):
    T, ts = cx.trains[N], cx.tsum[N]
    fit = d.fit(P, A)
    fT = fit & T
    out = {}
    M = d.mix(fT)
    G, O = free["G"], free["O"]
    out["M"] = M
    out["memG"] = M if M is not None else G
    out["memO"] = M if M is not None else O
    nm = nearest_mask(d, T, P, A)
    out["memNear"] = M if M is not None else (d.mix(nm) if nm is not None else None)
    out["marg"] = d.mix(fit, cx.margs[N])
    out["fzT"] = d.rmA(d.mix(d.alive & T), A)
    out["memU"] = M if M is not None else free["U"]
    out["memF"] = M if M is not None else free["fzF"]
    out["Mleak0.2"] = None if out["memU"] is None or free["U"] is None else 0.8 * out["memU"] + 0.2 * free["U"]
    for nm_, base in (("Ounion", ts["union"]), ("Ointer", ts["inter"]), ("Omaj", ts["maj"])):
        x = d.single((base | P) & ~A)
        out[nm_] = x if x is not None else O
    out["mixMO"] = (0.5 * M + 0.5 * O) if (M is not None and O is not None) else O
    aT = d.alive & T
    if aT.any():
        dis = np.array([popcount(P & ~int(w)) + popcount(A & int(w)) for w in WORLDS])
        out["softM"] = d.mix(aT, np.exp(-2.0 * dis))
    else:
        out["softM"] = None
    return out, bool(fT.any())


def run(draw, n_inst, stream="author", seed=None, tol=0.0, quiet=False):
    held, trains, pool = draw_sets(draw)
    if stream == "wide":                     # every world outside T_32 is held out (223 worlds), the same set for all N <= 32
        held = ~trains[32]
    margs, tsum = {}, {}
    for N, T in trains.items():
        ids = np.nonzero(T)[0] + 1
        pr = [(sum(int(w) >> j & 1 for w in ids) + 1) / (len(ids) + 2) for j in range(NR)]
        margs[N] = np.array([math.prod(pr[j] if w >> j & 1 else 1 - pr[j] for j in range(NR)) for w in range(1, 256)])
        union = 0; inter = 255
        for w in ids:
            union |= int(w); inter &= int(w)
        maj = sum(1 << j for j in range(NR) if sum(int(w) >> j & 1 for w in ids) * 2 > len(ids))
        tsum[N] = dict(union=union, inter=inter, maj=maj, minsize=min(popcount(int(w)) for w in ids),
                       never_rej=[j for j in range(NR) if inter >> j & 1], never_acc=[j for j in range(NR) if not union >> j & 1])
    rng = random.Random((100 + draw) if seed is None else seed)
    g = np.random.default_rng(1000 + draw)
    # calibrate the noise on first-decision posteriors of a few instances
    r0 = random.Random(0)
    pts = []
    for _ in range(12):
        i0 = Inst(r0)
        pts.append(Dec(i0, 0, i0.B, 0, 0).mix(i0.cons))
    lo, hi = 0.0, 4.0
    for _ in range(18):
        mid = (lo + hi) / 2
        t = np.mean([tv(noisy(p, mid, g), p) for p in pts * 40])
        lo, hi = (mid, hi) if t < NOISE_TV else (lo, mid)
    SIG = (lo + hi) / 2

    RL = {N: Reg(2) for N in GRID}
    RL3 = {N: Reg(3) for N in GRID}          # levels with a uniform direction added (a repair this review tries)
    RO = {N: Reg(4) for N in GRID}
    RI = {N: Reg(3) for N in GRID}
    cnt = {N: collections.Counter() for N in GRID}
    gc = collections.Counter()
    n = tries = 0
    t0 = time.time()
    while n < n_inst:
        tries += 1
        if stream == "author":
            ins = Inst(rng)
            if not held[ins.true]:
                continue
            true = ins.true
        else:
            ins = Inst(rng, draw_world=False)
            cand = np.nonzero(ins.cons & held)[0]
            if len(cand) == 0:
                continue
            true = int(rng.choice(list(cand)))
        n += 1
        cx = Ctx(ins, trains, margs, tsum, true)

        def dec(d, calls):
            P, A = d.P, d.A
            gc["dec"] += 1
            f0 = mech_free(d, P, A, cx)
            G0, O0 = f0["G"], f0["O"]
            srt = np.sort(G0)[::-1]
            if len(srt) > 1 and srt[0] - srt[1] < 1e-9:
                gc["dec_with_tied_argmax"] += 1
                if srt[0] != srt[1]:
                    gc["dec_tie_broken_by_float_noise"] += 1
            # probes (full pool), before the author's extra condition
            prs = []
            for r in d.probes():
                gc["probe_def"] += 1
                if d.rm(G0, r) is None:
                    gc["probe_G_all_on_r"] += 1
                    continue
                A1 = A | (1 << r)
                f1 = mech_free(d, P, A1, cx)
                prs.append((r, A1, f1))
                if d.rm(O0, r) is None:
                    gc["probe_O_all_on_r"] += 1
            t0s, t1s = {}, {}
            for N in GRID:
                c = cnt[N]
                c["dec"] += 1
                tN, insup = mech_T(d, P, A, cx, N, f0)
                t0s[N] = (tN, insup)
                M0 = tN["M"]
                if insup:
                    c["insup"] += 1
                    models = {k: (v - M0) for k, v in {**f0, **tN}.items() if v is not None}
                    for k in NOISY:
                        x = {**f0, **tN}.get(k)
                        if x is not None:
                            models["n:" + k] = noisy(x, SIG, g) - M0
                    RL[N].add(true, (G0 - M0, O0 - M0), models)
                    RL3[N].add(true, (G0 - M0, O0 - M0, f0["U"] - M0), models)
                for r, A1, f1 in prs:
                    tN1, insup1 = mech_T(d, P, A1, cx, N, f1)
                    a0, a1 = {**f0, **tN}, {**f1, **tN1}
                    dch = lambda x0, x1: (None if (d.rm(x0, r) is None or d.rm(x1, r) is None) else d.rm(x1, r) - d.rm(x0, r))
                    models = {k: dch(a0[k], a1[k]) for k in a0}
                    mb = d.rm(a0["memG"], r)
                    models["half"] = None if mb is None else dch(a0["memG"], 0.5 * mb + 0.5 * a1["memG"])
                    for k in NOISY:
                        if a0.get(k) is not None and a1.get(k) is not None:
                            models["n:" + k] = dch(noisy(a0[k], SIG, g), noisy(a1[k], SIG, g))
                    if mb is not None:
                        models["n:half"] = dch(noisy(a0["memG"], SIG, g), noisy(0.5 * mb + 0.5 * a1["memG"], SIG, g))
                    dG, dO = models["G"], models["O"]
                    if insup1:
                        c["p_in"] += 1
                        RI[N].add(true, (models["M"], dG, dO), models)
                    else:
                        kind = "p_cross" if insup else "p_beyond"
                        c[kind] += 1
                        bT = d.rm(M0, r) if insup else None
                        gdir = None if bT is None else d.rm(G0, r) - bT
                        ob = d.rm(O0, r)
                        hdir = None if (bT is None or ob is None) else ob - bT
                        if dO is None:
                            c[kind + "_noO"] += 1
                            c["e_noO"] += float(dG @ dG)
                        c["e_all"] += float(dG @ dG)
                        if insup and bT is None:
                            c["cross_M_all_on_r"] += 1
                        # L34 counts: is this probe input trivially outside training?
                        ts = tsum[N]
                        nrej = popcount(A1)
                        if nrej > 8 - ts["minsize"]:
                            c["oos_more_rej_than_any_training_episode"] += 1
                        if r in ts["never_rej"]:
                            c["oos_r_never_rejected_in_training"] += 1
                        if any(P >> j & 1 for j in ts["never_acc"]) or any(A1 >> j & 1 for j in ts["never_rej"]):
                            c["oos_some_token_never_in_training"] += 1
                        c["oos_nrej_sum"] += nrej
                        RO[N].add(true, (dG, dO, gdir, hdir), models)

        walk(ins, true, lambda d: d.mix(d.live), dec, tol=tol)
        ins.D = None
    res = dict(draw=draw, n=n_inst, stream=stream, tries=tries, RL=RL, RL3=RL3, RO=RO, RI=RI, cnt=cnt, gc=gc, tsum=tsum, sig=SIG,
               secs=time.time() - t0)
    return res


def strip(R):
    return dict(k=R.k, G={c: v for c, v in R.G.items()}, b={c: dict(v) for c, v in R.b.items()},
                e={c: dict(v) for c, v in R.e.items()}, n=R.n, nw=dict(R.nw))


if __name__ == "__main__":
    draw, n_inst = int(sys.argv[1]), int(sys.argv[2])
    stream = sys.argv[3] if len(sys.argv) > 3 else "author"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
    res = run(draw, n_inst, stream, seed)
    out = dict(res)
    for k in ("RL", "RL3", "RO", "RI"):
        out[k] = {N: strip(R) for N, R in res[k].items()}
    tag = f"{stream}_d{draw}_n{n_inst}" + (f"_s{seed}" if seed is not None else "")
    pickle.dump(out, open(f"{os.path.dirname(os.path.abspath(__file__))}/outputs/heldout_{tag}.pkl", "wb"))
    print(f"draw {draw}, {n_inst} instances ({stream} stream), {res['secs']:.0f} s, noise sigma {res['sig']:.3f}; counts {dict(res['gc'])}")
