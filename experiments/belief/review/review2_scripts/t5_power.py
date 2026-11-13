"""Conditioning, sampling error, bootstrap honesty and the cell-verdict rule.

A. From the pickles of t1_heldout.py: bootstrap sd (over held-out worlds) of the scored coefficients for noisy stand-ins
   (noise TV 0.12), at the run's size and scaled by sqrt(n / 4000); against DESIGN §5's claims and §9's 0.04 gate.
C. Bootstrap coverage with few clusters: a synthetic model whose coefficient on dG varies by held-out world
   (beta_w ~ N(mu, tau^2)), built on the real per-world Gram matrices; share of nominal-95 % percentile intervals
   that cover mu, and the classes returned.
D. The cell rule (8 of 10 runs, 2 seeds x 5 draws): probability of each cell verdict under a seed/draw spread.
Usage: python3 t5_power.py A outputs/heldout_*.pkl | C outputs/heldout_author_d8_n150.pkl ... | D
"""
import sys, pickle, glob
from r2lib import *
from t1_report import load

part = sys.argv[1]
files = sorted(f for a in sys.argv[2:] for f in glob.glob(a))
g = np.random.default_rng(17)

if part == "A":
    print("| file | N | OOS: sd(b_G), sd(b_O) for noisy G | noisy half-way | noisy O | noisy non-reviser (D) | scaled to 4,000: max over these | levels: sd(c_G), sd(c_O) noisy G | noisy M | scaled to 4,000 (max) | OOS probes | worlds with probes / effective |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        res = pickle.load(open(f, "rb"))
        n = res["n"]; k = math.sqrt(n / 4000)
        for N in GRID[:4]:
            O, L = load(res["RO"][N]), load(res["RL"][N])
            cells, mx = [], 0.0
            for nm in ("n:G", "n:half", "n:O", "n:fzF"):
                sd = np.nanstd(O.boot(nm, g, nb=300), axis=0)
                cells.append(f"{sd[0]:.3f}, {sd[1]:.3f}"); mx = max(mx, sd[0], sd[1])
            lc, lm = [], 0.0
            for nm in ("n:G", "n:M"):
                sd = np.nanstd(L.boot(nm, g, nb=300), axis=0)
                lc.append(f"{sd[0]:.3f}, {sd[1]:.3f}"); lm = max(lm, sd[0], sd[1])
            c = res["cnt"][N]
            print(f"| d{res['draw']} n{n} {res['stream']} | {N} | " + " | ".join(cells) + f" | **{mx * k:.3f}** | " + " | ".join(lc) + f" | {lm * k:.3f} | {c['p_cross'] + c['p_beyond']} | {len(O.G)} / {O.neff():.1f} |")

if part == "C":
    print("| file | N | worlds / effective | mu | tau (between-world sd) | coverage of the nominal 95 % interval | mean interval half-width | classes returned: G / mixed / D |")
    print("|---|---|---|---|---|---|---|---|")
    for f in files:
        res = pickle.load(open(f, "rb"))
        for N in (4, 16, 32):
            O = load(res["RO"][N])
            W = list(O.G)
            Gs = np.stack([O.G[c] for c in W])
            for mu, tau in ((0.7, 0.3), (0.7, 0.15), (0.1, 0.3)):
                cover = 0; hw = 0.0; cl = collections.Counter(); S = 400
                for _ in range(S):
                    beta = mu + tau * g.standard_normal(len(W))
                    bs = np.einsum("wij,w->wi", Gs[:, :, :1].repeat(1, axis=2), beta) if False else Gs[:, :, 0] * beta[:, None]
                    est = []
                    for t in range(200):
                        idx = g.integers(0, len(W), len(W))
                        est.append(np.linalg.lstsq(Gs[idx].sum(0), bs[idx].sum(0), rcond=1e-9)[0][:2])
                    est = np.array(est)
                    lo, hi = np.percentile(est, 2.5, axis=0), np.percentile(est, 97.5, axis=0)
                    cover += lo[0] <= mu <= hi[0]; hw += (hi[0] - lo[0]) / 2
                    cl[classify(lo, hi, "D")] += 1
                print(f"| d{res['draw']} | {N} | {len(W)} / {O.neff():.1f} | {mu} | {tau} | {cover / S:.2f} | {hw / S:.2f} | {cl['G'] / S:.2f} / {cl['mixed'] / S:.2f} / {cl['D'] / S:.2f} |")

if part == "D":
    print("Per-run class from one coefficient theta = mu + draw effect + seed effect and a 95 % half-width h: 'hi' (G or O) if theta - h >= 0.5; 'lo' (D or M) if theta + h <= 0.25; else mixed.")
    print("Cell verdict: a class held by >= 8 of 10 runs (5 draws x 2 seeds; 8 runs necessarily span >= 4 draws).\n")
    print("| true mean mu | seed sd | draw sd | h | P(run = lo / mixed / hi) | P(cell = lo) | P(cell = mixed) | P(cell = hi) | P(split) |")
    print("|---|---|---|---|---|---|---|---|---|")
    S = 20000
    for h in (0.04, 0.10):
        for ss, sdw in ((0.05, 0.0), (0.10, 0.05), (0.15, 0.10), (0.25, 0.10)):
            for mu in (0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 1.0):
                th = mu + sdw * g.standard_normal((S, 5, 1)) + ss * g.standard_normal((S, 5, 2))
                lo = (th + h <= 0.25); hi = (th - h >= 0.5); mx = ~lo & ~hi
                nlo, nhi, nmx = lo.sum((1, 2)), hi.sum((1, 2)), mx.sum((1, 2))
                pl, ph, pm = (nlo >= 8).mean(), (nhi >= 8).mean(), (nmx >= 8).mean()
                print(f"| {mu} | {ss} | {sdw} | {h} | {lo.mean():.2f} / {mx.mean():.2f} / {hi.mean():.2f} | {pl:.2f} | {pm:.2f} | {ph:.2f} | {1 - pl - ph - pm:.2f} |")
