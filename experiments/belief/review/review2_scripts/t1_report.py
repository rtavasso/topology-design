"""Tables from the pickles of t1_heldout.py: coefficients, 95 % bootstrap bounds over held-out worlds, DESIGN §8 classes,
residual shares, collinearity and cluster counts.

Usage: python3 t1_report.py outputs/heldout_*.pkl [--models a,b,c] [--part levels|oos|ins|cond|counts|all]
"""
import sys, pickle, glob
from r2lib import *


def load(R):
    r = Reg(R["k"])
    for c, v in R["G"].items():
        r.G[c] = v
    for c, v in R["b"].items():
        for n, x in v.items():
            r.b[c][n] = x
    for c, v in R["e"].items():
        for n, x in v.items():
            r.e[c][n] = x
    r.n, r.nw = R["n"], collections.Counter(R["nw"])
    return r


def cls_row(R, name, g, zero):
    c = R.coef(name)
    bs = R.boot(name, g)
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    return c, lo, hi, classify(lo[:2], hi[:2], zero)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
    part = opts.get("part", "all")
    want = opts.get("models")
    files = sorted(f for a in args for f in glob.glob(a))
    g = np.random.default_rng(5)
    for f in files:
        res = pickle.load(open(f, "rb"))
        draw, n = res["draw"], res["n"]
        print(f"\n# draw {draw}, {n} held-out-world instances, {res['stream']} stream  ({f.split('/')[-1]})")
        print(f"global counts: {dict(res['gc'])}")
        names = sorted({nm for N in GRID for c in res["RO"][N]["b"].values() for nm in c} | {nm for N in GRID for c in res["RL"][N]["b"].values() for nm in c})
        if want:
            names = [x for x in want.split(",") if x in names]
        if part in ("cond", "all"):
            print("\n## conditioning and clusters")
            print("| N | in-support decisions | corr(G-M, O-M) | OOS probes/inst cross / beyond | corr(dG, dO) | corr(g, h) | corr(dG, g) | cond. number of the 4x4 Gram | worlds with OOS probes | effective worlds (dG energy) | top-5 worlds' share of dG energy | Σ|dG|²/inst | in-support probes: corr(dM,dG), corr(dG,dO) |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for N in GRID:
                L, O, I, c = load(res["RL"][N]), load(res["RO"][N]), load(res["RI"][N]), res["cnt"][N]
                Gm = O.gram()
                e = sorted((G[0, 0] for G in O.G.values()), reverse=True)
                top5 = sum(e[:5]) / sum(e) if e and sum(e) > 0 else float("nan")
                ev = np.linalg.eigvalsh(Gm) if Gm[0, 0] > 0 else np.array([np.nan])
                print(f"| {N} | {c['insup'] / max(1, c['dec']):.0%} | {fmt(L.corr())} | {c['p_cross'] / n:.1f} / {c['p_beyond'] / n:.1f} | {fmt(O.corr(0, 1))} | {fmt(O.corr(2, 3))} | {fmt(O.corr(0, 2))} | "
                      f"{ev.max() / max(ev.min(), 1e-12):.0f} | {len(O.G)} | {O.neff():.1f} | {top5:.0%} | {Gm[0, 0] / n:.3f} | {fmt(I.corr(0, 1))}, {fmt(I.corr(1, 2))} |")
        if part in ("counts", "all"):
            print("\n## counts at out-of-support probes")
            print("| N | OOS probes | dO undefined (O wholly on r): cross / beyond | share of dG energy on those | crossing probes with M wholly on r | more rejections than any training episode can have | r never rejected in training | some verdict token never in training | mean #rejections in the list | rules in every training world / in none |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for N in GRID:
                c, ts = res["cnt"][N], res["tsum"][N]
                tot = c["p_cross"] + c["p_beyond"]
                if tot == 0:
                    continue
                print(f"| {N} | {tot} | {c['p_cross_noO']} / {c['p_beyond_noO']} | {c['e_noO'] / max(c['e_all'], 1e-12):.0%} | {c['cross_M_all_on_r']} | {c['oos_more_rej_than_any_training_episode'] / tot:.0%} | "
                      f"{c['oos_r_never_rejected_in_training'] / tot:.0%} | {c['oos_some_token_never_in_training'] / tot:.0%} | {c['oos_nrej_sum'] / tot:.2f} | {ts['never_rej']} / {ts['never_acc']} |")
        for N in GRID:
            L, O, I = load(res["RL"][N]), load(res["RO"][N]), load(res["RI"][N])
            if part in ("levels", "all"):
                print(f"\n## N = {N}: levels in support (c_G, c_O) [95 % bounds] class | residual share")
                for nm in names:
                    if not any(nm in c for c in L.b.values()):
                        continue
                    c, lo, hi, k = cls_row(L, nm, g, "M")
                    print(f"  {nm:12s} ({c[0]:+.2f}, {c[1]:+.2f})  [{lo[0]:+.2f},{hi[0]:+.2f}] [{lo[1]:+.2f},{hi[1]:+.2f}]  {k:6s} resid {fmt(L.resid_share(nm))}")
            if part in ("oos", "all") and N < 64:
                eG = O.gram()[0, 0]
                print(f"\n## N = {N}: out-of-support changes (b_G, b_O, s_G, s_O) [95 % bounds on b] class | residual share | Σ|dp|²/Σ|dG|²")
                for nm in names:
                    if not any(nm in c for c in O.b.values()):
                        continue
                    c, lo, hi, k = cls_row(O, nm, g, "D")
                    print(f"  {nm:12s} ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}, {c[3]:+.2f})  [{lo[0]:+.2f},{hi[0]:+.2f}] [{lo[1]:+.2f},{hi[1]:+.2f}]  {k:6s} resid {fmt(O.resid_share(nm))}  energy {O.energy(nm) / eG:.2f}")
            if part in ("ins", "all"):
                print(f"\n## N = {N}: in-support changes (a_M, a_G, a_O), sum")
                for nm in names:
                    if not any(nm in c for c in I.b.values()):
                        continue
                    c = I.coef(nm)
                    print(f"  {nm:12s} ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})  U = {c.sum():+.2f}  resid {fmt(I.resid_share(nm))}")
