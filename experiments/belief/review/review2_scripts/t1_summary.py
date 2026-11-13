"""Across draws: per model and N, the range of the scored coefficients and the list of DESIGN §8 classes (one per draw).
Usage: python3 t1_summary.py outputs/heldout_author_d*_n150.pkl"""
import sys, pickle, glob
from r2lib import *
from t1_report import load, cls_row

files = sorted(f for a in sys.argv[1:] for f in glob.glob(a))
g = np.random.default_rng(11)
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for f in files:
    res = pickle.load(open(f, "rb"))
    for N in GRID[:4]:
        L, O, I = load(res["RL"][N]), load(res["RO"][N]), load(res["RI"][N])
        eG = O.gram()[0, 0]
        for kind, R, zero in (("lev", L, "M"), ("oos", O, "D")):
            names = sorted({nm for c in R.b.values() for nm in c})
            for nm in names:
                c, lo, hi, k = cls_row(R, nm, g, zero)
                extra = (R.resid_share(nm), R.energy(nm) / eG if kind == "oos" else float("nan"))
                rows[(kind, nm)][N].append((c, lo, hi, k, extra))
        for nm in sorted({nm for c in I.b.values() for nm in c}):
            c = I.coef(nm)
            rows[("ins", nm)][N].append((c, None, None, "", (I.resid_share(nm), float("nan"))))
rng = lambda xs: f"{min(xs):+.2f}..{max(xs):+.2f}"
for kind, title in (("lev", "LEVELS in support (c_G, c_O)"), ("oos", "OUT-OF-SUPPORT changes (b_G, b_O)"), ("ins", "IN-SUPPORT changes: U = a_M + a_G + a_O (held-out worlds)")):
    print(f"\n## {title}; {len(files)} draws: " + ", ".join(f.split('/')[-1] for f in files))
    for (kd, nm), byN in sorted(rows.items()):
        if kd != kind:
            continue
        print(f"  {nm}")
        for N in GRID[:4]:
            v = byN.get(N, [])
            if not v:
                continue
            if kind == "ins":
                print(f"     N={N:2d}  a_M {rng([x[0][0] for x in v])}  a_G {rng([x[0][1] for x in v])}  a_O {rng([x[0][2] for x in v])}  U {rng([x[0].sum() for x in v])}")
            else:
                cl = " ".join(x[3] for x in v)
                res = [x[4][0] for x in v if not math.isnan(x[4][0])]
                en = [x[4][1] for x in v if not math.isnan(x[4][1])]
                print(f"     N={N:2d}  G {rng([x[0][0] for x in v])}  O {rng([x[0][1] for x in v])}  classes: {cl}"
                      + (f"  resid {rng(res)}" if res else "") + (f"  energy/dG {rng(en)}" if en else ""))
