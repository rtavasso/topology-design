"""Smaller checks on the change statistic (held-out worlds, out-of-support probes).

(i)   How much RM_r amplifies: the mass each mechanism puts on the probed rule before the probe (1 / (1 - mass) is the
      factor by which RM_r scales whatever the model has on the other actions).
(ii)  DESIGN §5: 'applied only to the earlier one, a model that ignores the verdict reads 0.09 instead of 0'.
(iii) Is the statistic unbiased under output noise?  dp is NOT linear in the model's outputs (RM_r renormalises).
      Stand-ins = mechanism + multiplicative noise at TV 0.12 / 0.20 / 0.30, coefficients on (dG, dO, g, h).
(iv)  A mixture of OUTPUTS (0.5 G + 0.5 O) against the mixture of CHANGES (0.5 dG + 0.5 dO) that refs3.py plants.
Usage: PYTHONPATH=src python3 t6_misc.py DRAW N_INST
"""
import sys
from r2lib import *

draw, n_inst = int(sys.argv[1]), int(sys.argv[2])
held, trains, pool = draw_sets(draw)
g = np.random.default_rng(99)
SIGS = {0.12: 0.43, 0.20: 0.75, 0.30: 1.25}
rng = random.Random(100 + draw)
NS = (4, 16, 32)
RO = {N: Reg(4) for N in NS}
mass = {N: collections.defaultdict(list) for N in NS}
one = collections.Counter()
tvs = collections.defaultdict(list)
n = 0
while n < n_inst:
    ins = Inst(rng)
    if not held[ins.true]:
        continue
    n += 1

    def dec(d, calls):
        P, A = d.P, d.A
        G0, O0 = d.mix(d.live), d.single(255 & ~A)
        fz0 = d.rmA(d.mix(d.alive), A)
        for r in d.probes():
            A1 = A | (1 << r)
            G1, O1 = d.mix(d.fit(P, A1)), d.single(255 & ~A1)
            bG = d.rm(G0, r)
            if bG is None:
                continue
            dG = G1 - bG
            dO = None if d.rm(O0, r) is None else O1 - d.rm(O0, r)
            one["den"] += float(dG @ dG)
            one["blind_one_sided"] += float((G0 - bG) @ dG)        # p(E + r-) = p(E) = G(E); RM_r only on the earlier output
            fz1 = d.rmA(d.mix(d.alive), A1)
            for N in NS:
                T = trains[N]
                M0, M1 = d.mix(d.live & T), d.mix(d.fit(P, A1) & T)
                if M1 is not None:
                    continue
                kind = "cross" if M0 is not None else "beyond"
                bT = d.rm(M0, r) if M0 is not None else None
                gdir = None if bT is None else bG - bT
                ob = d.rm(O0, r)
                hdir = None if bT is None or ob is None else ob - bT
                e = float(dG @ dG)
                mr = lambda p: float(p[d.rules == r].sum())
                mass[N]["G"].append((mr(G0), e)); mass[N]["O"].append((mr(O0), e))
                if M0 is not None:
                    mass[N]["M (crossing)"].append((mr(M0), e))
                dch = lambda x0, x1: (None if (x0 is None or x1 is None or d.rm(x0, r) is None or d.rm(x1, r) is None) else d.rm(x1, r) - d.rm(x0, r))
                mem0 = M0 if M0 is not None else G0
                models = {"G": dG, "mix of outputs 0.5G+0.5O": dch(0.5 * G0 + 0.5 * O0, 0.5 * G1 + 0.5 * O1),
                          "mix of changes 0.5dG+0.5dO": None if dO is None else 0.5 * dG + 0.5 * dO}
                for lv, sg in SIGS.items():
                    a, b_ = noisy(G0, sg, g), noisy(G1, sg, g)
                    if N == 4:
                        tvs[lv].append(tv(a, G0))
                    models[f"G + noise {lv}"] = dch(a, b_)
                    models[f"O + noise {lv}"] = dch(noisy(O0, sg, g), noisy(O1, sg, g))
                    models[f"M-then-G + noise {lv}"] = dch(noisy(mem0, sg, g), noisy(G1, sg, g))
                    models[f"non-reviser (frozen prior + rule mask) + noise {lv}"] = dch(noisy(fz0, sg, g), noisy(fz1, sg, g)) if fz0 is not None and fz1 is not None else None
                    models[f"ignores the verdict, same noise on both {lv}"] = dch(a, a)
                RO[N].add(ins.true, (dG, dO, gdir, hdir), models)

    walk(ins, ins.true, lambda d: d.mix(d.live), dec)
    ins.D = None

print(f"# draw {draw}, {n_inst} held-out-world instances")
print(f"(ii) verdict-blind model, RM_r on the earlier output only, single direction dG: {one['blind_one_sided'] / one['den']:.3f}  (DESIGN: 0.09)")
print("noise levels: mean TV to target " + ", ".join(f"{lv}: {np.mean(v):.3f}" for lv, v in tvs.items()))
for N in NS:
    print(f"\n## N = {N}: {RO[N].n} out-of-support probes")
    print("(i) mass on the probed rule before the probe; share of probes (share of dG signal) with mass >= 0.5 / >= 0.8 / >= 0.95:")
    for k, v in mass[N].items():
        a = np.array(v)
        print(f"    {k:14s} " + " / ".join(f"{(a[:, 0] >= t).mean():.2f} ({a[a[:, 0] >= t, 1].sum() / a[:, 1].sum():.2f})" for t in (0.5, 0.8, 0.95)))
    print("(iii), (iv) coefficients (b_G, b_O, s_G, s_O), bootstrap sd of b_G, b_O, residual share:")
    for nm in sorted({x for c in RO[N].b.values() for x in c}):
        c = RO[N].coef(nm); sd = np.nanstd(RO[N].boot(nm, g, nb=200), axis=0)
        print(f"    {nm:58s} ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}, {c[3]:+.2f})  sd {sd[0]:.3f}, {sd[1]:.3f}  resid {fmt(RO[N].resid_share(nm))}")
