"""Round C: a gate that must read the data state, and what that costs in stability. EXPLORATORY.

Branch task: the three primitives plus a token IF = "apply primitive 0 if <w, h> > 0, else primitive 1"; 64 of the 256
programs for training. Four streams, softmax scores, register reg0 unless stated. The teacher's branch is a hard
threshold; a student with a soft gate can only approach it.

Predictions, written before the run (15 Nov 2026):
  C0  gate does not read the data                     cannot branch: held-out error above 0.05, solved 0/5.
  C1  gate reads the normalised data state            at least 5x below C0; median below 1e-2.
  C2  gate reads the raw data state                   within 2x of C1 (states are O(1) here; the published argument for
                                                      normalising the selector's input is about depth growth).
  C3  C1 + penalty on large selection logits (1e-3)   within 2x of C1 (no round-off problem in double precision).
  C4  one query per pass (clock) + data               within 2x of C1.
  C5  query from the data, no register                can branch but not sequence: above 0.03, solved 0/5.
  C6  C1, dense for the first half then top-1         within 3x of C1.
  C7  C1, top-2 throughout                            within 3x of C1 in at least 4/5.
  C8  PROGRAM task, gate reads the normalised data    within 3x of round A's 2.7e-5 (reading the data costs nothing when
                                                      the progress signal is clean; A7).
  C9  C1 at learning rate 1e-2                        within 3x of C1.
  C10 C2 at learning rate 1e-2                        within 3x of C9 (same reason as C2).
  Gain of the trained network (largest singular value of d h_4 / d h_0 over 96 held-out cases): the median of C1 is
  within 20 % of C0's; the maximum of C1 is more than 1.5x C0's maximum (the gate's own term, test A2, lives near the
  branch boundary).
"""
import sys
from loopmoe import run

BR = dict(task="branch", ntrain=64, progress="reg0", gain=True)
ROWS = [("C0 gate blind to the data", dict(BR)),
        ("C1 gate reads normalised data", dict(BR, gate_h=True)),
        ("C2 gate reads raw data", dict(BR, gate_h=True, ln=False)),
        ("C3 C1 + logit penalty", dict(BR, gate_h=True, zloss=1e-3)),
        ("C4 clock + data", dict(BR, gate_h=True, progress="clock")),
        ("C5 data query, no register", dict(BR, gate_h=True, progress="data")),
        ("C6 C1 dense then top-1", dict(BR, gate_h=True, topk=1, anneal=0.5)),
        ("C7 C1 top-2", dict(BR, gate_h=True, topk=2)),
        ("C8 program task, gate reads data", dict(progress="reg0", gate_h=True, gain=True)),
        ("C9 C1 at lr 1e-2", dict(BR, gate_h=True, lr=1e-2)),
        ("C10 C2 at lr 1e-2", dict(BR, gate_h=True, ln=False, lr=1e-2))]
if __name__ == "__main__":
    med = lambda v: sorted(v)[len(v) // 2]
    print("| variant | training error | held-out error | solved (<1e-2) | gain: median | gain: max | teacher gain: median, max | streams run per pass |")
    print("|---|---|---|---|---|---|---|---|", flush=True)
    for i in sys.argv[1:]:
        label, over = ROWS[int(i)]
        rs = [run(over, s) for s in range(5)]
        te = [r["test"] for r in rs]
        print(f"| {label} | {med([r['train'] for r in rs]):.1e} | {med(te):.1e} [{min(te):.1e}, {max(te):.1e}] | {sum(t < 1e-2 for t in te)}/5 | "
              f"{med([r['gain_med'] for r in rs]):.2f} | {med([r['gain_max'] for r in rs]):.2f} [{max(r['gain_max'] for r in rs):.2f}] | "
              f"{med([r['tgain_med'] for r in rs]):.2f}, {med([r['tgain_max'] for r in rs]):.2f} | {med([r['ran'] for r in rs]):.2f} |", flush=True)
