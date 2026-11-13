"""Cross-check the numpy distance DP against a direct per-world BFS; timing; headline entropy numbers."""
import random, time, sys
import numpy as np
from core import *

rng = random.Random(0)
t0 = time.time()
insts = [Inst(rng) for _ in range(12)]
print("gen time per instance", (time.time() - t0) / 12)
bad = 0; tot = 0
for ins in insts[:6]:
    for w in random.Random(1).sample(range(1, 256), 40):
        d = brute_dw(ins, w, ins.B)
        d2 = int(ins.D[w - 1, 0])
        tot += 1
        if (d is None and d2 <= ins.B) or (d is not None and d != d2):
            bad += 1
            print("MISMATCH", w, d, d2)
print("brute-force distance check: mismatches", bad, "of", tot)
