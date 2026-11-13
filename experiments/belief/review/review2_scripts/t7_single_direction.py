"""The optimistic planner under the ONE-direction statistics that preceded revision 3 (refs3_cheap.md): gamma = levels
on (G - M) alone; beta_cf out of support on (dG, g) alone. Computed from the Gram matrices pickled by t1_heldout.py.
Also the readable-range numbers of DESIGN §5 (signal per instance, probes per instance) at every N including 64.
Usage: python3 t7_single_direction.py outputs/heldout_author_d*_n150.pkl"""
import sys, pickle, glob
from r2lib import *
from t1_report import load
print("| draw | N | planner: gamma (one direction) | planner: beta_cf out of support (dG, g only) | in-support decisions | levels signal per instance | OOS probes per instance (cross + beyond) | OOS signal per instance | in-support probes per instance |")
print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(x for a in sys.argv[1:] for x in glob.glob(a)):
    res = pickle.load(open(f, "rb")); n = res["n"]
    for N in GRID:
        L, O = load(res["RL"][N]).gram(), load(res["RO"][N]).gram()
        c = res["cnt"][N]
        gam = L[0, 1] / L[0, 0]
        A = O[np.ix_([0, 2], [0, 2])]; b = O[[0, 2], 1]
        bcf = np.linalg.lstsq(A, b, rcond=1e-9)[0][0] if O[0, 0] > 1e-9 else float("nan")
        print(f"| {res['draw']} | {N} | {gam:.2f} | {bcf:.2f} | {c['insup'] / c['dec']:.0%} | {L[0, 0] / n:.3f} | {(c['p_cross'] + c['p_beyond']) / n:.1f} | {O[0, 0] / n:.3f} | {c['p_in'] / n:.1f} |")
