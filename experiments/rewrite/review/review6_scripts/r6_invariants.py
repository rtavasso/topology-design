"""Review 6 — internal-consistency invariants of the per-instance records (an evaluation-bug screen; the saved routers
are not in the repository, so no episode could be replayed through a model).

    python3 experiments/rewrite/review/review6_scripts/r6_invariants.py > /tmp/r6_invariants.txt
"""
import sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from r6_common import *

D = load()
viol = collections.Counter(); tot = collections.Counter()
for name, r in D["runs"].items():
    for tag, M in D["meta"].items():
        o = r["out"][tag]; k = np.array([m["k"] for m in M]); T = 2 * k + 2
        okm, oku, cm, cu, st = o["ok_m"], o["ok_u"], o["cat_m"], o["cat_u"], o["steps_m"]
        chk = {
            "ok_m <=> cat_m == ok": (okm == (cm == 0)),
            "ok_u <=> cat_u == ok": (oku == (cu == 0)),
            "ok_u => ok_m (an unmasked success uses valid actions only, so the masked argmax is the same)": (~oku | okm),
            "ok_m => k <= steps <= T": (~okm | ((st >= k) & (st <= T))),
            "masked category is never 'invalid'": (cm < 4),
            "ok_m => visited goal": (~okm | o["vis_m"]),
            "'no halt at goal' => visited and not ok (masked)": ((cm != 3) | (o["vis_m"] & ~okm)),
            "horizon => steps == T (masked)": ((cm != 1) | (st == T)),
            "first_canon => first_ok": (~o["first_canon"] | o["first_ok"]),
            "k = 1 and first_ok and not ok_m is possible only by a later failure (steps >= 1)": (~((k == 1) & o["first_ok"] & ~okm) | (st >= 1)),
            "k = 1 wrong halt at step 0 => first decision not correct": (~((cm == 2) & (st == 0)) | ~o["first_ok"]),
        }
        for nm, v in chk.items():
            viol[(tag, nm)] += int((~v).sum()); tot[(tag, nm)] += len(v)
        # the cells stored with the final point agree with the per-instance list
        cells = {}
        for i, m in enumerate(M):
            ck = f"{m['world']}|{int(m['binding'])}|{int(m['GP1'])}|{m['k']}"
            c = cells.setdefault(ck, [0, 0, 0]); c[0] += 1; c[1] += int(okm[i]); c[2] += int(oku[i])
        fc = r["final_cells"][tag]
        if {ck: tuple(v) for ck, v in cells.items()} != {ck: (v["n"], v["ok_m"], v["ok_u"]) for ck, v in fc.items()}:
            viol[(tag, "final cells == per-instance aggregation")] += 1
        tot[(tag, "final cells == per-instance aggregation")] += 1
for (tag, nm), n in sorted(tot.items()):
    print(f"{tag:8s} {nm}: {viol[(tag, nm)]} violations of {n}")
# ladder monotonicity is not an invariant (a deeper lookahead can do worse); report how often it fails, for the record
for tag, M in D["meta"].items():
    a = sum(m["ladder"]["1"] and not m["ladder"]["3"] for m in M); b = sum(m["ladder"]["2"] and not m["ladder"]["3"] for m in M)
    print(f"{tag}: greedy succeeds where lookahead-3 fails on {a} of {len(M)}; lookahead-2 succeeds where lookahead-3 fails on {b}")
