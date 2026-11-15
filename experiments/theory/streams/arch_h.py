"""Round H: checks called for by the independent review (review/review_moe_in_loops.md). EXPLORATORY.

The review's own small runs (3-5 seeds) said: with the gate that also reads the data, top-1 from the FIRST step already
works (branch 5/5, program 3/3, eight streams 1/2); with a token-only gate it does not (1/5); sigmoid scores do as well
as softmax under dense-then-top-1 (3/3); and what a dense start supplies is an address and a one-to-one token -> stream
assignment, not a confident gate. This round reruns those at ten seeds and tests the last one directly.

Candidate network throughout (register started as the exact shift, sinusoidal keys, softmax, raw data read) unless stated.

Predictions, written before the run and after reading the review (15 Nov 2026):
  H1  program, 4 streams, gate reads the data, top-1 from the first step      solved at least 8/10.
  H2  program, 4 streams, token-only gate, top-1 from the first step          solved at most 3/10.
  H3  program, 8 streams, gate reads the data, top-1 from the first step      solved between 3 and 7 of 10.
  H5  branch, top-1 from the first step                                       solved at least 9/10; median within 2x of 1.1e-3.
  H6  program, 8 streams, gate reads the data, dense until confident, top-1   solved 10/10.
  H7  program, 8 streams, token-only, dense 10 % then top-1: sigmoid against softmax   both at least 9/10, medians within 2x.
  H8  register that starts still, token-only, 4 streams, dense for 90 / 150 / 200 / 300 steps then top-1, with a probe at
      the switch: the run ends solved exactly when, at the switch, the address was learned (attention on position t in
      over 90 % of passes) and each token had its own top stream - agreement in at least 34 of 40 runs; the share of such
      runs rises with the dense length; most runs that end solved had mean top probability below 0.8 at the switch.
  H9  the candidate (dense until confident, top-1) under five other teachers, program and branch   solved 10/10.
"""
import sys
from loopmoe import run

C = dict(progress="regsin", keys="sinus", score="softmax")
DATA = dict(gate_h=True, ln_h=False)
BIG = dict(K=8, P=5, ntrain=80, B=8)
BR = dict(task="branch", ntrain=64, **DATA)
med = lambda v: sorted(v)[len(v) // 2]


def row(label, over, seeds=range(10)):
    rs = [run(over, s) for s in seeds]
    te = [r["test"] for r in rs]
    print(f"| {label} | {med(te):.1e} [{min(te):.1e}, {max(te):.1e}] | {sum(t < 1e-2 for t in te)}/{len(te)} | "
          f"{med([r['used'] for r in rs])} | {med([r['train_ran'] for r in rs]):.2f} |", flush=True)


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "small":
        print("| variant | held-out error | solved (<1e-2) | streams used | streams per pass in training |\n|---|---|---|---|---|", flush=True)
        row("H1 program, 4 streams, gate reads data, top-1 from the first step", dict(C, **DATA, topk=1))
        row("H2 program, 4 streams, token-only gate, top-1 from the first step", dict(C, topk=1))
        row("H5 branch, top-1 from the first step", dict(C, **BR, topk=1))
        print("\n| H8: dense steps | address learned at switch | one-to-one at switch | mean top probability at switch | ends solved | solved agrees with (address and one-to-one) |\n|---|---|---|---|---|---|", flush=True)
        agree_all = 0
        for steps in (90, 150, 200, 300):
            rs = [run(dict(progress="reg0", topk=1, anneal=steps / 3000, probe=True), s) for s in range(10)]
            ok = [r["at_switch"]["addr"] > 0.9 and r["at_switch"]["one2one"] for r in rs]
            sol = [r["test"] < 1e-2 for r in rs]
            agree = sum(a == b for a, b in zip(ok, sol)); agree_all += agree
            print(f"| {steps} | {sum(r['at_switch']['addr'] > 0.9 for r in rs)}/10 | {sum(r['at_switch']['one2one'] for r in rs)}/10 | "
                  f"{med([r['at_switch']['conf'] for r in rs]):.2f} [{min(r['at_switch']['conf'] for r in rs):.2f}, {max(r['at_switch']['conf'] for r in rs):.2f}] | "
                  f"{sum(sol)}/10 | {agree}/10 |", flush=True)
        print(f"\nH8 agreement over all runs: {agree_all}/40", flush=True)
        print("\n| H9: other teachers | held-out error | solved (<1e-2) | streams used | streams per pass in training |\n|---|---|---|---|---|", flush=True)
        for name, over in (("program", dict(C, topk=1, auto=0.8)), ("branch", dict(C, **BR, topk=1, auto=0.8))):
            rs = [run(dict(over, tseed=t), 0) for t in (11, 12, 13, 14, 15)]
            te = [r["test"] for r in rs]
            print(f"| H9 {name}, teachers 11-15 | {med(te):.1e} [{min(te):.1e}, {max(te):.1e}] | {sum(t < 1e-2 for t in te)}/5 | {med([r['used'] for r in rs])} | {med([r['train_ran'] for r in rs]):.2f} |", flush=True)
    else:
        print("| variant | held-out error | solved (<1e-2) | streams used | streams per pass in training |\n|---|---|---|---|---|", flush=True)
        row("H3 program, 8 streams, gate reads data, top-1 from the first step", dict(C, **DATA, **BIG, topk=1))
        row("H6 program, 8 streams, gate reads data, dense until confident then top-1", dict(C, **DATA, **BIG, topk=1, auto=0.8))
        row("H7a program, 8 streams, token-only, dense 10 % then top-1, sigmoid", dict(C, **BIG, score="sigmoid", topk=1, anneal=0.1))
        row("H7b program, 8 streams, token-only, dense 10 % then top-1, softmax", dict(C, **BIG, topk=1, anneal=0.1))
