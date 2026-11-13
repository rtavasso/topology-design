"""The gate (DESIGN §5, §8): in-support rejection probes on TRAINING worlds, along M's greedy trajectory;
U = a_M + a_G + a_O from the regression of dp on (dM, dG, dO). Scores the design's planted models and cheap or
miscalibrated ones. Two probe definitions are compared:
  A  (refs3.py, gate_part)  some fitting TRAINING world lacks r and some has it
  B  (DESIGN §5 as written, read on the full pool)  some fitting world of the 255 lacks r; in support iff a training world fits E + r-
Problems are drawn as §3 says: problem first, then a world uniform over the training worlds that can solve it.
Usage: PYTHONPATH=src python3 t4_gate.py DRAW N_INST
"""
import sys
from r2lib import *
from t1_heldout import mech_free, Ctx, popcount

draw, n_inst = int(sys.argv[1]), int(sys.argv[2])
held, trains, pool = draw_sets(draw)
g = np.random.default_rng(3000 + draw)
SIG = {0.12: 0.43, 0.20: 0.75}          # noise sigma for TV 0.12 (calibrated in t1_heldout.py) and about 0.20


def powr(p, a):
    if p is None:
        return None
    w = np.where(p > 0, p ** a, 0.0)
    return w / w.sum()


print(f"# gate, draw {draw}, {n_inst} training-world instances per size; U = a_M + a_G + a_O with 95 % bootstrap bounds over instances")
for N in GRID[:4]:
    T = trains[N]
    rng = random.Random(7000 * draw + N)
    R = {"A": Reg(3), "B": Reg(3)}
    cnt = collections.Counter()
    n = tries = 0
    while n < n_inst:
        ins = Inst(rng, draw_world=False)
        tries += 1
        cand = np.nonzero(ins.cons & T)[0]
        if len(cand) == 0:
            continue
        true = int(rng.choice(list(cand)))
        n += 1
        cx = Ctx(ins, {N: T}, None, None)

        def dec(d, calls):
            P, A = d.P, d.A
            fitT = d.live & T
            f0 = mech_free(d, P, A, cx)
            M0 = d.mix(fitT)
            prA = set(d.probes(pool=T))
            for r in d.probes():
                A1 = A | (1 << r)
                fit1T = d.fit(P, A1) & T
                if not fit1T.any():
                    cnt["leaves"] += 1
                    continue
                f1 = mech_free(d, P, A1, cx)
                M1 = d.mix(fit1T)
                bM = d.rm(M0, r)
                if bM is None or d.rm(f0["G"], r) is None:
                    cnt["undefined"] += 1
                    continue
                dch = lambda x0, x1: (None if (d.rm(x0, r) is None or d.rm(x1, r) is None) else d.rm(x1, r) - d.rm(x0, r))
                dM, dG, dO = M1 - bM, dch(f0["G"], f1["G"]), dch(f0["O"], f1["O"])
                U0, U1 = f0["U"], f1["U"]
                models = {"M": dM, "G": dG, "O": dO,
                          "half": 0.5 * dM,
                          "ignore (D)": dch(M0, M0),
                          "M then flatten half-way to uniform": dch(M0, 0.5 * bM + 0.5 * U1),
                          "M then uniform": dch(M0, U1),
                          "M then sharpen (RM[M]^2)": dch(M0, powr(bM, 2.0)),
                          "M then flatten (RM[M]^0.5)": dch(M0, powr(bM, 0.5)),
                          "M^3 (over-confident M)": dch(powr(M0, 3.0), powr(M1, 3.0)),
                          "M^0.5 (under-confident M)": dch(powr(M0, 0.5), powr(M1, 0.5)),
                          "0.7 M + 0.3 uniform": dch(0.7 * M0 + 0.3 * U0, 0.7 * M1 + 0.3 * U1),
                          "soft planner tau=1": dch(f0["Osoft1.0"], f1["Osoft1.0"]),
                          "feasible-uniform": dch(f0["feas"], f1["feas"]),
                          "LOO planner": dch(f0["LOO"], f1["LOO"]),
                          "version space (supG)": dch(f0["supG"], f1["supG"]),
                          "G rho=0.9": dch(f0["Grho0.9"], f1["Grho0.9"]),
                          "countflat": dch(f0["countflat"], f1["countflat"]),
                          }
                for lv, sg in SIG.items():
                    models[f"noisy M, TV {lv}"] = dch(noisy(M0, sg, g), noisy(M1, sg, g))
                    models[f"noisy ignore, TV {lv}"] = dch(noisy(M0, sg, g), noisy(M0, sg, g))
                key = (n, true)
                R["B"].add(n, (dM, dG, dO), models)
                cnt["B"] += 1
                if r in prA:
                    R["A"].add(n, (dM, dG, dO), models)
                    cnt["A"] += 1

        walk(ins, true, lambda d: d.mix(d.live & T), dec)
        ins.D = None
    print(f"\n## N = {N}: {n / tries:.0%} of problems have a consistent training world; probes per instance: definition A {cnt['A'] / n_inst:.1f}, B {cnt['B'] / n_inst:.1f}; "
          f"leave support {cnt['leaves'] / n_inst:.1f}; RM undefined {cnt['undefined']}; corr(dM,dG) {R['B'].corr(0, 1):.2f}, corr(dG,dO) {R['B'].corr(1, 2):.2f}, corr(dM,dO) {R['B'].corr(0, 2):.2f}")
    print("| model | U (def. A) | [95 %] | gate | U (def. B) | [95 %] | gate | (a_M, a_G, a_O) def. B |")
    print("|---|---|---|---|---|---|---|---|")
    names = sorted({nm for c in R["B"].b.values() for nm in c})
    for nm in names:
        cells = []
        for k in ("A", "B"):
            c = R[k].coef(nm)
            bs = R[k].boot(nm, g, nb=300).sum(1)
            lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
            cells.append(f"{c.sum():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'pass' if lo >= 0.5 else 'FAIL'}")
        cB = R["B"].coef(nm)
        print(f"| {nm} | {cells[0]} | {cells[1]} | ({cB[0]:+.2f}, {cB[1]:+.2f}, {cB[2]:+.2f}) |")
    sys.stdout.flush()
