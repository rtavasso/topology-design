"""Round D: more passes than trained. EXPLORATORY.

Program task, three primitives, four streams, dense softmax gates. Training programs of lengths 3, 4 and 5 (40 each,
passes = length); tested on 100 random programs each of length 4 (in range), 6 and 8.

Predictions, written before the run (15 Nov 2026):
  D1  one query per pass (clock), learned keys     lengths 6 and 8: above 0.03, solved 0/5 (queries and keys for passes
                                                   beyond the fifth were never trained).
  D2  register reg0, learned keys                  as D1 (the keys of positions 6-8 were never trained).
  D3  register reg0, fixed sinusoidal keys         in range solved 5/5; length 6 solved in at most 2/5 (training on three
                                                   lengths does not force the learned advance to be a pure shift).
  D4  register regrot, fixed sinusoidal keys       as D3.
  D5  register started as the exact shift (one position per pass, test A5), then trained   solved 5/5 at lengths 6 and 8;
                                                   error at length 8 less than 10x the error at length 4.
"""
import sys
from loopmoe import run

L = dict(lens=(3, 4, 5), test_lens=(4, 6, 8))
ROWS = [("D1 clock, learned keys", dict(L, progress="clock")),
        ("D2 reg0, learned keys", dict(L, progress="reg0")),
        ("D3 reg0, sinusoidal keys", dict(L, progress="reg0", keys="sinus")),
        ("D4 regrot, sinusoidal keys", dict(L, progress="regrot", keys="sinus")),
        ("D5 exact-shift register, sinusoidal keys", dict(L, progress="regsin", keys="sinus"))]
if __name__ == "__main__":
    med = lambda v: sorted(v)[len(v) // 2]
    print("| variant | training error | length 4 (in range) | length 6 | length 8 | solved at 6, at 8 |")
    print("|---|---|---|---|---|---|", flush=True)
    for i in sys.argv[1:]:
        label, over = ROWS[int(i)]
        rs = [run(over, s) for s in range(5)]
        cell = lambda Ln: (lambda v: f"{med(v):.1e} [{min(v):.1e}, {max(v):.1e}]")([r["by_len"][Ln] for r in rs])
        print(f"| {label} | {med([r['train'] for r in rs]):.1e} | {cell(4)} | {cell(6)} | {cell(8)} | "
              f"{sum(r['by_len'][6] < 1e-2 for r in rs)}/5, {sum(r['by_len'][8] < 1e-2 for r in rs)}/5 |", flush=True)
