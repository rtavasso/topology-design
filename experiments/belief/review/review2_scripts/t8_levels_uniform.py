"""Levels in support with a third direction (uniform - M): does an entropy control separate the generalising posterior
from diffuse or soft answers?  Reads the RL3 block pickled by t1_heldout.py.
Usage: python3 t8_levels_uniform.py outputs/heldout_author_d7_n150.pkl ..."""
import sys, pickle, glob
from r2lib import *
from t1_report import load
NAMES = ("G", "O", "U", "Osoft1.0", "Osoft0.3", "Gpow3.0", "Gpow0.5", "Ghard", "Grho0.8", "Grho0.9", "LOO", "fzF", "fzT", "Mleak0.2", "marg", "n:G", "n:M", "n:O")
for f in sorted(x for a in sys.argv[1:] for x in glob.glob(a)):
    res = pickle.load(open(f, "rb"))
    print(f"\n# draw {res['draw']}, {res['n']} instances: (c_G, c_O) with two directions  ->  (c_G, c_O, c_U) with the uniform direction added")
    for N in (4, 16, 32):
        L2, L3 = load(res["RL"][N]), load(res["RL3"][N])
        print(f"  N = {N}; corr(G-M, U-M) = {L3.corr(0, 2):.2f}, corr(O-M, U-M) = {L3.corr(1, 2):.2f}")
        for nm in NAMES:
            if not any(nm in c for c in L3.b.values()):
                continue
            a, b = L2.coef(nm), L3.coef(nm)
            print(f"    {nm:10s} ({a[0]:+.2f}, {a[1]:+.2f})  ->  ({b[0]:+.2f}, {b[1]:+.2f}, {b[2]:+.2f})  resid {fmt(L3.resid_share(nm))}")
