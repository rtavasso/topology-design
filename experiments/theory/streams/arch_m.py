"""Round M: the streamed block against ONE monolithic looped block, at matched parameters, with compute reported.
EXPLORATORY. This is the comparison docs/moe_in_loops.md section 7 and ledger idea I37 say is missing.

Revision 2 (15 Nov 2026), after an independent review of the design (review/review_round_m_design.md, verdict "run
after changes"). The first design and its predictions Q1-Q9 are in git at a737c25, committed before any run; they are
scored too. What changed: every arm gets its own learning rate (the review's one-seed probes showed the rate flips the
verdict for the per-unit-gate block); training error is reported so that a recombination gap can be read; errors are
recorded during training and three seeds run twice as long; out-of-range outcomes are scored as per-seed ratios, not
against the 1e-2 threshold; compute is reported as multiply-adds per pass; the "same compute" rows, which could not
work by arithmetic, are replaced by a width series; a two-layer block is added on the program task; the no-attention
block is run wide and read at trained length only.

What is compared. Everything outside the block is identical: the same register (started as the exact shift), the same
sinusoidal keys, the same attention read r_t of the original sequence, the same residual step, Adam, 3,000 steps, ten
paired seeds (same training programs, embeddings, inputs). Only the block differs:
  streams (the candidate)      K streams V_k tanh(B_k h), gate = softmax(G r_t [+ MLP(r_t, h)]), dense until the gate's
                               mean top probability passes 0.8, then only the top stream is computed.
  one block, additive          V tanh(B h + C r_t + b): the instruction read is one more input.
  one block, two hidden layers V tanh(B2 tanh(B1 [h; r_t] + b1) + b2).
  per-unit gate                V (sigmoid(C r_t + b) * tanh(B h)). NOT a monolith in I37's sense: it is the streams family
                               with W streams of width one, sigmoid scores, every stream run. It is here to separate
                               "multiplicative gating" from "grouping into streams and running one".
  one block, no attention      V tanh(B h + C [whole padded sequence; register] + b), wide (48 units, three times the
                               parameters): no built-in way to address the sequence. Read at trained length only.
Matching. Parameters within 3 % of the candidate's (checked by a test). Compute is reported, not matched: multiply-adds
per pass in the block at run time (candidate 176 with four streams; a parameter-matched monolith 580).
Learning rates. Three rates (1e-3, 3e-3, 1e-2) x three seeds that are not among the ten (100-102) for one arm per
family and task; each family runs at the rate with the lowest median TRAINING error at length 4. Width variants and the
eight-stream rows inherit their family's rate (a limit).

Training programs of lengths 3, 4, 5 (40 each; 80 each with five primitives; 64 each on the branch task). Outcomes, all
on the same trained network: error on training programs of length 4; on held-out programs of length 4 (training
programs removed); at lengths 6 and 8; at length 4 with inputs twice as large; the held-out error at steps 1,000 and
2,000; whether the register still addresses position t.

FIRST PREDICTIONS (Q1-Q9, a737c25) - kept for scoring; see git for the full text. In brief: Q1 parameter-matched
monoliths with the instruction read solve held-out programs in at least 8/10; Q2 additive at least 10x the candidate's
error, per-unit gate within 10x; Q3 12-unit monolith fails (row dropped: arithmetic); Q4 at length 8 the candidate
within 2x of its length-4 error, additive at least 5x its own, per-unit gate within 3x; Q5 inputs x2: candidate solved
at least 8/10, additive at most 5/10 and at least 5x the candidate; Q6 no-attention block 0/10 at length 4 and above
0.03 at lengths 6, 8 (length clause dropped: confounded); Q7 branch: two-layer at least 3x the candidate and at most
8/10, at length 8 at least 3x its own length-4 error, one layer at most 5/10; Q8 running every stream within 2x of the
candidate (row dropped: known from round E); Q9 ordering of Q2 holds for at least 4 of 5 other teachers.

REVISED PREDICTIONS (R1-R9), written after the design review and after seeing its one-seed probes (which already show
R1, R2 for four streams, R5 and R9 in one seed each - those four are confirmations at ten seeds, not blind):
  R1  No block has a recombination gap. Held-out over training error at length 4 is between 0.5 and 2 (median over
      seeds) for every block, the no-attention block included - that one fails to FIT (training error above 1e-2).
  R2  Accuracy at matched parameters, each arm at its own best rate. Additive (M3) and two-layer (M16) monoliths: at
      least 30x the candidate's error at length 4 (median of per-seed ratios), candidate better in at least 9/10 seeds.
      Per-unit gate (M4): within 3x of the candidate.
  R3  A bigger monolith does not close it: additive at twice the parameters (M3b, 58 units) still at least 10x.
  R4  Longer training does not close it: at 6,000 steps (three seeds) additive is still at least 30x the candidate.
  R5  Out of range, nobody is special. Length 8 over length 4, per seed: median below 2 for every block. Inputs x2 over
      x1: median between 3 and 30 for every block, and no block's median more than 3x another's. (This withdraws the
      fragility claims of Q4 and Q5.)
  R6  Branch task. Two-layer monolith at matched parameters (M13): at least 3x the candidate (M12), candidate better in
      at least 8/10; one layer (M15): at least 10x; two layers at twice the parameters (M13b): candidate still better in
      at least 8/10.
  R7  Eight streams, five primitives. Additive (M9) and two-layer (M17): at least 10x the candidate (M8); per-unit gate
      (M10): within 3x.
  R8  Five other teachers, one seed each: additive at least 10x the candidate for 5/5; per-unit gate within 3x for at
      least 4/5.
  R9  Per-unit gate at 18 units (M4a; 360 multiply-adds, twice the candidate's): at least 10x the candidate.
What each outcome would mean. R1 holding means this testbed cannot show streams recombining better than a monolith:
with a shared controller that hands the block one instruction per pass, recombination is free for any block - I37's
null. Then the question becomes what the block's form buys. If R2-R4 and R6-R7 hold with the per-unit gate matching the
candidate, the answer is: multiplicative gating buys accuracy per parameter (by one to two orders of magnitude), and
grouping into streams with one run buys compute (176 against 580 multiply-adds) at no loss. If the per-unit gate is
clearly better than the candidate, streams cost accuracy. If the additive monolith matches, streams buy only compute.
"""
import json
import os
import statistics
import sys
from loopmoe import DEF, run

C = dict(progress="regsin", keys="sinus", lens=(3, 4, 5), test_lens=(4, 6, 8), clean_test=True, scales=(2.0,), ckpts=(1000, 2000))
SP = dict(topk=1, auto=0.8)
BIG = dict(K=8, P=5, ntrain=80, B=8)
BR = dict(task="branch", ntrain=64)
ROWS = {
    "M1": ("streams, candidate", "cand", dict(C, **SP)),
    "M3": ("one block, additive, same parameters (29 units)", "add", dict(C, block="mono_add", W=29)),
    "M3a": ("one block, additive, 18 units", "add", dict(C, block="mono_add", W=18)),
    "M3b": ("one block, additive, twice the parameters (58 units)", "add", dict(C, block="mono_add", W=58)),
    "M16": ("one block, two layers, same parameters (16 + 16)", "two", dict(C, block="mono2", W=16)),
    "M4": ("per-unit gate, same parameters (29 units)", "mul", dict(C, block="mono_mul", W=29)),
    "M4a": ("per-unit gate, 18 units", "mul", dict(C, block="mono_mul", W=18)),
    "M7": ("one block, no attention, 48 units", "add", dict(C, block="mono_add", ctx="tape", W=48)),
    "M8": ("8 streams: candidate", "cand", dict(C, **SP, **BIG)),
    "M9": ("8 streams: one block, additive, same parameters (58 units)", "add", dict(C, **BIG, block="mono_add", W=58)),
    "M17": ("8 streams: one block, two layers, same parameters (26 + 26)", "two", dict(C, **BIG, block="mono2", W=26)),
    "M10": ("8 streams: per-unit gate, same parameters (58 units)", "mul", dict(C, **BIG, block="mono_mul", W=58)),
    "M12": ("branch: streams, candidate", "cand_br", dict(C, **SP, **BR, gate_h=True, ln_h=False)),
    "M13": ("branch: one block, two layers, same parameters (23 + 23)", "two_br", dict(C, **BR, block="mono2", W=23)),
    "M13b": ("branch: one block, two layers, twice the parameters (36 + 36)", "two_br", dict(C, **BR, block="mono2", W=36)),
    "M15": ("branch: one block, one layer, same parameters (50 units)", "add_br", dict(C, **BR, block="mono_add", W=50)),
}
SWEEP = {"cand": "M1", "add": "M3", "two": "M16", "mul": "M4", "cand_br": "M12", "two_br": "M13", "add_br": "M15"}
CAND = {"M3": "M1", "M3a": "M1", "M3b": "M1", "M16": "M1", "M4": "M1", "M4a": "M1", "M7": "M1",
        "M9": "M8", "M17": "M8", "M10": "M8", "M13": "M12", "M13b": "M12", "M15": "M12"}
HERE = os.path.dirname(os.path.abspath(__file__))
med = statistics.median


def madds(over):
    """Multiply-adds per pass inside the block at run time (the shared register and attention read are excluded)."""
    c = dict(DEF); c.update(over)
    d, mv = c["d"], c["mv"]
    if c["block"] == "streams":
        return c["K"] * mv + (c["H"] * (mv + d) + c["K"] * c["H"] if c["gate_h"] else 0) + 2 * c["w"] * d
    cin = mv if c["ctx"] == "attn" else 8 * (c["P"] + (c["task"] == "branch")) + c["mc"]
    W = c["W"]
    return W * (d + cin) + W * W + d * W if c["block"] == "mono2" else W * (d + cin) + d * W


def rates():
    out = {}
    for f in os.listdir(HERE):
        if f.startswith("arch_m_lr_") and f.endswith(".json"):
            out.update(json.load(open(os.path.join(HERE, f))))
    return out


def fmt(v):
    return f"{med(v):.1e} [{min(v):.1e}, {max(v):.1e}]"


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sweep":                      # sweep <tag> <family> ...
        best = {}
        print("| family (arm) | rate | training error, length 4 (median of 3 seeds) | held-out, length 4 |\n|---|---|---|---|", flush=True)
        for fam in sys.argv[3:]:
            res = {}
            for lr in (1e-3, 3e-3, 1e-2):
                rs = [run(dict(ROWS[SWEEP[fam]][2], lr=lr), s) for s in (100, 101, 102)]
                res[lr] = med([r["train_by_len"][4] for r in rs])
                print(f"| {fam} ({SWEEP[fam]}) | {lr:g} | {res[lr]:.1e} | {med([r['by_len'][4] for r in rs]):.1e} |", flush=True)
            best[fam] = min(res, key=res.get)
        json.dump(best, open(os.path.join(HERE, f"arch_m_lr_{sys.argv[2]}.json"), "w"))
        print("\nchosen:", best, flush=True)
    elif mode == "main":                     # main <tag> <row> ...
        lr = rates()
        store = {}
        print("| block | rate | parameters | multiply-adds per pass | training, length 4 | held out, length 4 | held out / training | length 8 / length 4 | inputs x2 / x1 | at step 1,000; 2,000 | address kept |")
        print("|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
        for key in sys.argv[3:]:
            label, fam, over = ROWS[key]
            rs = [run(dict(over, lr=lr[fam]), s) for s in range(10)]
            store[key] = [dict(train4=r["train_by_len"][4], te4=r["by_len"][4], te6=r["by_len"][6], te8=r["by_len"][8], x2=r["scaled"][2.0][4],
                               c1=r["curve"][1000], c2=r["curve"][2000], addr=r["addr"], switch=r["switch"], nparams=r["nparams"]) for r in rs]
            s = store[key]
            print(f"| {key} {label} | {lr[fam]:g} | {s[0]['nparams']} | {madds(over)} | {fmt([r['train4'] for r in s])} | {fmt([r['te4'] for r in s])} | "
                  f"{med([r['te4'] / r['train4'] for r in s]):.2f} | {med([r['te8'] / r['te4'] for r in s]):.2f} | {med([r['x2'] / r['te4'] for r in s]):.1f} | "
                  f"{med([r['c1'] for r in s]):.1e}; {med([r['c2'] for r in s]):.1e} | {med([r['addr'] for r in s]):.2f} |", flush=True)
            json.dump(store, open(os.path.join(HERE, f"arch_m_results_{sys.argv[2]}.json"), "w"))
    elif mode == "long":                     # 6,000 steps, three seeds
        lr = rates()
        print("| block | held-out error, length 4, at step 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 |\n|---|---|---|---|---|---|---|", flush=True)
        for key in sys.argv[2:]:
            label, fam, over = ROWS[key]
            rs = [run(dict(over, lr=lr[fam], steps=6000, ckpts=(1000, 2000, 3000, 4000, 5000)), s) for s in range(3)]
            cells = " | ".join(f"{med([r['curve'][k] for r in rs]):.1e}" for k in (1000, 2000, 3000, 4000, 5000))
            print(f"| {key} {label} | {cells} | {med([r['by_len'][4] for r in rs]):.1e} |", flush=True)
    elif mode == "teachers":
        lr = rates()
        print("| teacher | M1 streams, candidate | M3 one block, additive | M4 per-unit gate |\n|---|---|---|---|", flush=True)
        for t in (11, 12, 13, 14, 15):
            e = [run(dict(ROWS[k][2], lr=lr[ROWS[k][1]], tseed=t), 0)["by_len"][4] for k in ("M1", "M3", "M4")]
            print(f"| {t} | {e[0]:.1e} | {e[1]:.1e} | {e[2]:.1e} |", flush=True)
    elif mode == "paired":                   # per-seed ratios of each block to its candidate
        st = {}
        for f in sorted(os.listdir(HERE)):
            if f.startswith("arch_m_results_"):
                st.update(json.load(open(os.path.join(HERE, f))))
        print("| block | error / candidate's, length 4 (median of per-seed ratios) [min, max] | seeds where the candidate is better | same at length 8 | same with inputs x2 |\n|---|---|---|---|---|")
        for key, cand in CAND.items():
            if key in st and cand in st:
                rr = lambda f: [a[f] / b[f] for a, b in zip(st[key], st[cand])]
                r4 = rr("te4")
                print(f"| {key} {ROWS[key][0]} | {med(r4):.1f} [{min(r4):.1f}, {max(r4):.1f}] | {sum(x > 1 for x in r4)}/10 | {med(rr('te8')):.1f} | {med(rr('x2')):.1f} |")
