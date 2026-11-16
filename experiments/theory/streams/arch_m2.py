"""Round M2: follow-up to round M after the independent review of its results (review/review_round_m_results.md).
EXPLORATORY.

The review's main finding: round M's conclusion "a block with a soft gate on every unit matches the candidate, so
grouping into streams buys only compute" is not best against best. The three-seed selection put the candidate at a
constant rate of 0.01, where its error wanders (a noise floor); at 0.003 the eight-stream candidate was 3.5-7 times
better than the per-unit gate in the reviewer's three seeds, a three-stream candidate of 18 units beat the 18-unit
per-unit gate by 11-32 times, and on the branch task a per-unit gate that reads the data came within 1.1-3 times of
the candidate. All of that is three seeds. This round reruns it at ten.

Changes from round M. (1) The learning rate decays to zero on a cosine (every arm), which removes the constant-rate
noise floor and the register drift it caused; (2) rates 0.003 and 0.01 are swept for EVERY row separately (three seeds,
100-102, training error at length 4), so nothing inherits a rate; (3) new rows: a candidate with three streams of six
units (18 units, no spare stream) against the 18-unit per-unit gate; on the branch task a per-unit gate whose logits
also read the data state through a small MLP, parameter-matched to the candidate.

Predictions, written and committed before the run, after reading the review and its three-seed leads (15 Nov 2026):
  S1  Four streams. Per-unit gate (N4) within 3x of the candidate (N1): median of per-seed ratios between 1/3 and 3.
  S2  Eight streams. The candidate (N8) is better than the per-unit gate (N10): median ratio at least 2, candidate
      better in at least 8/10. (The lead says 3.5-7x without decay; decay may let the slower block catch up, so I
      predict less.)
  S3  Eighteen units. The three-stream candidate (N1s) is better than the 18-unit per-unit gate (N4a): median ratio at
      least 5, better in at least 8/10.
  S4  Branch. The data-reading per-unit gate (N14) is within 3x of the candidate (N12), with the candidate better in at
      least 7/10; and it is at least 3x better than the two-layer block (N13).
  S5  The additive block (N3) is still at least 30x the candidate (N1) under decay.
  S6  With decay the length-8 jumps go: length 8 over length 4 is below 3 in at least 9/10 seeds for each of N1, N4,
      N8, N10.
  S7  Inputs twice as large: the candidate's and the per-unit gate's ratios (x2 over x1, medians) are within a factor
      3 of each other, with four streams and with eight.
  S8  Early in training the candidate is ahead: at step 1,000 its held-out error is lower than the per-unit gate's in
      at least 8/10 seeds, with four streams and with eight.
What each outcome would mean. S1 with S2 and S3: grouping into streams buys compute everywhere and accuracy where
streams are many or units few. S1 with S2 and S3 failing: grouping buys compute only. S4 says how much of the branch
lead belongs to gating and how much to streams.
"""
import json
import os
import sys
from arch_m import C, SP, BIG, BR, HERE, fmt, med
from loopmoe import DEF, run

D = dict(C, decay=True)
ROWS = {
    "N1": ("streams, candidate (4 streams of 12)", dict(D, **SP)),
    "N4": ("per-unit gate, same parameters (29 units)", dict(D, block="mono_mul", W=29)),
    "N3": ("one block, additive, same parameters (29 units)", dict(D, block="mono_add", W=29)),
    "N1s": ("streams, candidate, 3 streams of 6 (18 units)", dict(D, **SP, K=3, w=6)),
    "N4a": ("per-unit gate, 18 units", dict(D, block="mono_mul", W=18)),
    "N8": ("8 streams: candidate", dict(D, **SP, **BIG)),
    "N10": ("8 streams: per-unit gate, same parameters (58 units)", dict(D, **BIG, block="mono_mul", W=58)),
    "N12": ("branch: streams, candidate", dict(D, **SP, **BR, gate_h=True, ln_h=False)),
    "N14": ("branch: per-unit gate that reads the data, same parameters (27 units)", dict(D, **BR, block="mono_mul", W=27, gate_h=True, ln_h=False)),
    "N13": ("branch: one block, two layers, same parameters (23 + 23)", dict(D, **BR, block="mono2", W=23)),
}
CAND = {"N4": "N1", "N3": "N1", "N4a": "N1s", "N10": "N8", "N14": "N12", "N13": "N12"}


def madds(over):
    """Multiply-adds per pass inside the block at run time; the shared attention read is not included."""
    c = dict(DEF); c.update(over)
    d, mv, W = c["d"], c["mv"], c["W"]
    if c["block"] == "streams":
        return c["K"] * mv + (c["H"] * (mv + d) + c["K"] * c["H"] if c["gate_h"] else 0) + 2 * c["w"] * d
    if c["block"] == "mono2":
        return W * (d + mv) + W * W + d * W
    return W * (d + mv) + d * W + (c["Hg"] * (mv + d) + W * c["Hg"] if c["gate_h"] else 0)


def rates():
    out = {}
    for f in os.listdir(HERE):
        if f.startswith("arch_m2_lr_") and f.endswith(".json"):
            out.update(json.load(open(os.path.join(HERE, f))))
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sweep":                      # sweep <tag> <row> ...
        best = {}
        print("| row | rate | training error, length 4 (median of 3 seeds) | held-out, length 4 |\n|---|---|---|---|", flush=True)
        for key in sys.argv[3:]:
            res = {}
            for lr in (3e-3, 1e-2):
                rs = [run(dict(ROWS[key][1], lr=lr), s) for s in (100, 101, 102)]
                res[lr] = med([r["train_by_len"][4] for r in rs])
                print(f"| {key} | {lr:g} | {res[lr]:.1e} | {med([r['by_len'][4] for r in rs]):.1e} |", flush=True)
            best[key] = min(res, key=res.get)
        json.dump(best, open(os.path.join(HERE, f"arch_m2_lr_{sys.argv[2]}.json"), "w"))
    elif mode == "main":                     # main <tag> <row> ...
        lr = rates(); store = {}
        print("| block | rate | parameters | multiply-adds per pass | training, length 4 | held out, length 4 | held out / training | length 8 / length 4 | inputs x2 / x1 | at step 1,000; 2,000 | address kept |")
        print("|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
        for key in sys.argv[3:]:
            label, over = ROWS[key]
            rs = [run(dict(over, lr=lr[key]), s) for s in range(10)]
            s = store[key] = [dict(train4=r["train_by_len"][4], te4=r["by_len"][4], te6=r["by_len"][6], te8=r["by_len"][8], x2=r["scaled"][2.0][4],
                                   c1=r["curve"][1000], c2=r["curve"][2000], addr=r["addr"], switch=r["switch"], nparams=r["nparams"], train_ran=r["train_ran"]) for r in rs]
            print(f"| {key} {label} | {lr[key]:g} | {s[0]['nparams']} | {madds(over)} | {fmt([r['train4'] for r in s])} | {fmt([r['te4'] for r in s])} | "
                  f"{med([r['te4'] / r['train4'] for r in s]):.2f} | {med([r['te8'] / r['te4'] for r in s]):.2f} | {med([r['x2'] / r['te4'] for r in s]):.1f} | "
                  f"{med([r['c1'] for r in s]):.1e}; {med([r['c2'] for r in s]):.1e} | {med([r['addr'] for r in s]):.2f} |", flush=True)
            json.dump(store, open(os.path.join(HERE, f"arch_m2_results_{sys.argv[2]}.json"), "w"))
    elif mode == "paired":
        st = {}
        for f in sorted(os.listdir(HERE)):
            if f.startswith("arch_m2_results_"):
                st.update(json.load(open(os.path.join(HERE, f))))
        print("| block | error / its candidate's, length 4 (median of per-seed ratios) [min, max] | seeds where the candidate is better | same at step 1,000 | same at length 8 | same with inputs x2 | seeds with length 8 / length 4 above 3 |\n|---|---|---|---|---|---|---|")
        for key in ROWS:
            j = lambda k: sum(r["te8"] / r["te4"] > 3 for r in st[k])
            if key in CAND and key in st and CAND[key] in st:
                rr = lambda f: [a[f] / b[f] for a, b in zip(st[key], st[CAND[key]])]
                r4, r1 = rr("te4"), rr("c1")
                print(f"| {key} {ROWS[key][0]} | {med(r4):.2f} [{min(r4):.2f}, {max(r4):.1f}] | {sum(x > 1 for x in r4)}/10 | {med(r1):.1f}, candidate ahead {sum(x > 1 for x in r1)}/10 | {med(rr('te8')):.2f} | {med(rr('x2')):.2f} | {j(key)}/10 |")
            elif key in st:
                print(f"| {key} {ROWS[key][0]} | (candidate) | | | | | {j(key)}/10 |")
