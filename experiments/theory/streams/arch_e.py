"""Round E: the candidate at ten seeds, against the reference that runs every stream. EXPLORATORY.

Candidate (from rounds A-D): register started as the exact one-position shift, sinusoidal keys, query normalised;
softmax over all streams, not renormalised; every stream runs until the gate's mean top probability passes 0.8 (or half
of training at the latest), after that only the top stream; on the branch task the gate also reads the raw data state.
Reference: the same network with every stream always run.

Predictions, written before the run (15 Nov 2026):
  G1/G2  program, 4 streams           candidate solved 10/10; median error within 2x of the reference.
  G3/G4  program, 8 streams, 5 prims  candidate solved 10/10; median error within 2x of the reference.
  G5/G7  branch, 4 streams            candidate solved at least 8/10; median within 2x of the reference.
  G6     branch, gate reads the NORMALISED data   solved count within 2 of G5's (round C's 5/5 against 3/5 was noise).
  G8     program, trained on lengths 3-5          candidate solved 10/10 at lengths 6 and 8.
  G9     branch, trained on lengths 3-5           candidate solved at least 8/10 at length 6 and at least 6/10 at length 8.
  Compute: streams run per pass, averaged over training, below 2.0 for G1 and G8, below 3.0 for G3, at most 2.5 for G5
  (which may fall back to the half-way switch: near the branch boundary the gate is never confident).
"""
import sys
from loopmoe import run

C = dict(progress="regsin", keys="sinus", score="softmax")
SP = dict(topk=1, auto=0.8)
BIG = dict(K=8, P=5, ntrain=80, B=8)
BR = dict(task="branch", ntrain=64, gate_h=True, ln_h=False)
LEN = dict(lens=(3, 4, 5), test_lens=(4, 6, 8))
ROWS = [("G1 program, 4 streams: candidate", dict(C, **SP)),
        ("G2 program, 4 streams: every stream run", dict(C)),
        ("G3 program, 8 streams: candidate", dict(C, **SP, **BIG)),
        ("G4 program, 8 streams: every stream run", dict(C, **BIG)),
        ("G5 branch: candidate (gate reads raw data)", dict(C, **SP, **BR)),
        ("G6 branch: candidate, gate reads normalised data", dict(C, **SP, **dict(BR, ln_h=True))),
        ("G7 branch: every stream run", dict(C, **BR)),
        ("G8 program, lengths 3-5 -> 6, 8: candidate", dict(C, **SP, **LEN)),
        ("G9 branch, lengths 3-5 -> 6, 8: candidate", dict(C, **SP, **dict(BR, ntrain=64), **LEN))]
if __name__ == "__main__":
    med = lambda v: sorted(v)[len(v) // 2]
    print("| variant | held-out error | solved (<1e-2) | length 6 | length 8 | solved at 6, at 8 | streams per pass: training, afterwards | dense steps of 3,000 |")
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for i in sys.argv[1:]:
        label, over = ROWS[int(i)]
        rs = [run(over, s) for s in range(10)]
        te = [r["test"] for r in rs]
        sw = [r["switch"] for r in rs if r["switch"] is not None]
        if "by_len" in rs[0]:
            c6 = [r["by_len"][6] for r in rs]; c8 = [r["by_len"][8] for r in rs]; c4 = [r["by_len"][4] for r in rs]
            lens = f"{med(c6):.1e} [{min(c6):.1e}, {max(c6):.1e}] | {med(c8):.1e} [{min(c8):.1e}, {max(c8):.1e}] | {sum(v < 1e-2 for v in c6)}/10, {sum(v < 1e-2 for v in c8)}/10"
            te = c4
        else:
            lens = "- | - | -"
        print(f"| {label} | {med(te):.1e} [{min(te):.1e}, {max(te):.1e}] | {sum(t < 1e-2 for t in te)}/10 | {lens} | "
              f"{med([r['train_ran'] for r in rs]):.2f}, {med([r['ran'] for r in rs]):.2f} | {(str(med(sw)) + ' [' + str(min(sw)) + ', ' + str(max(sw)) + ']') if sw else '-'} |", flush=True)
