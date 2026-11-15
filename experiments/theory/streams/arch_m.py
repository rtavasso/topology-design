"""Round M: the streamed block against ONE monolithic looped block, at matched parameters and at matched compute.
EXPLORATORY. This is the comparison docs/moe_in_loops.md section 7 and ledger idea I37 say is missing.

What is compared. Everything outside the block is identical: the same register (started as the exact shift), the same
sinusoidal keys, the same attention read r_t of the original sequence, the same residual step, optimiser, learning rate,
3,000 steps, ten paired seeds. Only the block differs:
  streams (the candidate)      K streams V_k tanh(B_k h), gate = softmax(G r_t [+ MLP(r_t, h)]), dense until the gate's
                               mean top probability passes 0.8, then only the top stream is computed.
  one block, additive          V tanh(B h + C r_t + b): the instruction read is one more input.
  one block, per-unit gate     V (sigmoid(C r_t + b) * tanh(B h)): every hidden unit has its own soft gate, all are computed.
  one block, two hidden layers V tanh(B2 tanh(B1 [h; r_t] + b1) + b2)   (branch task: the candidate is also two stages deep,
                               a gate MLP of 24 units and then a stream of 12).
  one block, no attention      V tanh(B h + C [whole padded sequence; register] + b): no built-in way to address the
                               sequence - the literal "one big block that is re-fed its input".
Matching. Parameters: the monolith's width is chosen so that its trainable parameters equal the candidate's to within
3 % (program, 4 streams: 705 against 706 at width 29; 8 streams: 1,329 against 1,331 at width 58; branch: 1,169 against
1,140 at two layers of 23, or 1,155 at one layer of 50). Compute: hidden units evaluated per pass at run time - the
candidate evaluates 12 (program) or 36 (branch: 24 in the gate and 12 in the stream); the compute-matched monolith has
that many units and so fewer parameters. Neither side is tuned.

Training programs of lengths 3, 4, 5 (40 each; 80 each with five primitives; 64 each on the branch task). Outcomes, all on
the same trained network: error on held-out programs of length 4 (training programs removed), at lengths 6 and 8, and
at length 4 with inputs twice as large as in training. "Solved" = below 1e-2.

Predictions, written and committed before the run (15 Nov 2026). Rows are numbered as in ROWS.
  Q1  Recombination is not the streams' doing. Every parameter-matched monolith that gets the instruction read
      (M3, M4, M9, M10) solves held-out programs of length 4 in at least 8/10. (Its block only ever has to learn one
      function per token; held-out programs are new orders of the same tokens. This is I37's null.)
  Q2  Accuracy at matched parameters. Additive monolith: median error at length 4 at least 10x the candidate's
      (M3 against M1; M9 against M8) - it has to switch units on and off by saturating them. Per-unit gate: within 10x
      of the candidate (M4 against M1; M10 against M8).
  Q3  Matched compute. The 12-unit additive monolith solves at most 2/10 with three primitives (M5) and 0/10 with five
      (M11): three primitives need 18 directions.
  Q4  More passes than trained. Candidate: error at length 8 within 2x of its error at length 4 (M1, M8). Additive
      monolith: at least 5x its own length-4 error (M3). Per-unit gate: within 3x (M4).
  Q5  Inputs twice as large. Candidate solved in at least 8/10 (M1). Additive monolith solved in at most 5/10, with a
      median at least 5x the candidate's (M3).
  Q6  No attention. M7 solves held-out programs of length 4 in 0/10, and is above 0.03 at lengths 6 and 8.
  Q7  Branch task. Two-layer monolith at matched parameters (M13): median at length 4 at least 3x the candidate's (M12)
      and solved in at most 8/10; at length 8 at least 3x its own length-4 error, while the candidate stays within 2x.
      One hidden layer of 50 (M15): solved in at most 5/10. Compute-matched (M14): no better than M13.
  Q8  Running every stream (M2) is within 2x of the candidate (M1) on every outcome (round E).
  Q9  Five other teachers, one seed each (M1, M3, M4): the ordering of Q2 holds for at least 4 of 5 teachers.
What each outcome would mean. If Q1 holds, the case for streams cannot be "they recombine"; it rests on Q2-Q5: accuracy
per parameter, units computed per pass, and behaviour outside the trained range. If the monoliths also match there, the
streams buy only run-time compute.
"""
import sys
from loopmoe import run

C = dict(progress="regsin", keys="sinus", lens=(3, 4, 5), test_lens=(4, 6, 8), clean_test=True, scales=(2.0,))
SP = dict(topk=1, auto=0.8)
BIG = dict(K=8, P=5, ntrain=80, B=8)
BR = dict(task="branch", ntrain=64)
ROWS = [
    ("M1 streams, candidate", 12, dict(C, **SP)),
    ("M2 streams, every stream run", 48, dict(C)),
    ("M3 one block, additive, same parameters", 29, dict(C, block="mono_add", W=29)),
    ("M4 one block, per-unit gate, same parameters", 29, dict(C, block="mono_mul", W=29)),
    ("M5 one block, additive, same compute", 12, dict(C, block="mono_add", W=12)),
    ("M6 (unused)", 0, None),
    ("M7 one block, no attention, same parameters", 14, dict(C, block="mono_add", ctx="tape", W=14)),
    ("M8 8 streams: candidate", 12, dict(C, **SP, **BIG)),
    ("M9 8 streams: one block, additive, same parameters", 58, dict(C, **BIG, block="mono_add", W=58)),
    ("M10 8 streams: one block, per-unit gate, same parameters", 58, dict(C, **BIG, block="mono_mul", W=58)),
    ("M11 8 streams: one block, additive, same compute", 12, dict(C, **BIG, block="mono_add", W=12)),
    ("M12 branch: streams, candidate", 36, dict(C, **SP, **BR, gate_h=True, ln_h=False)),
    ("M13 branch: one block, two layers, same parameters", 46, dict(C, **BR, block="mono2", W=23)),
    ("M14 branch: one block, two layers, same compute", 36, dict(C, **BR, block="mono2", W=18)),
    ("M15 branch: one block, one layer, same parameters", 50, dict(C, **BR, block="mono_add", W=50)),
]
med = lambda v: sorted(v)[len(v) // 2]
cell = lambda v: f"{med(v):.1e} [{min(v):.1e}, {max(v):.1e}] {sum(x < 1e-2 for x in v)}/{len(v)}"

if __name__ == "__main__":
    if sys.argv[1] == "teachers":
        print("| teacher | M1 streams, candidate | M3 one block, additive | M4 one block, per-unit gate |\n|---|---|---|---|", flush=True)
        for t in (11, 12, 13, 14, 15):
            e = [run(dict(ROWS[i][2], tseed=t), 0)["by_len"][4] for i in (0, 2, 3)]
            print(f"| {t} | {e[0]:.1e} | {e[1]:.1e} | {e[2]:.1e} |", flush=True)
    else:
        print("| block | parameters | hidden units per pass | length 4, held out | length 6 | length 8 | length 4, inputs x2 |")
        print("|---|---|---|---|---|---|---|", flush=True)
        for i in sys.argv[1:]:
            label, units, over = ROWS[int(i)]
            rs = [run(over, s) for s in range(10)]
            b = lambda L: cell([r["by_len"][L] for r in rs])
            print(f"| {label} | {rs[0]['nparams']} | {units} | {b(4)} | {b(6)} | {b(8)} | {cell([r['scaled'][2.0][4] for r in rs])} |", flush=True)
