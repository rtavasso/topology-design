"""Review experiment 1: is the dense start needed once the address is given?
Top-1 FROM SCRATCH (no dense phase) with (a) the candidate's exact-shift register, program task; (b) the same on the
branch task; (c) one learned query per pass (clock). Uses loopmoe.run unchanged."""
import sys, time
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/streams")
from loopmoe import run

C = dict(progress="regsin", keys="sinus", score="softmax")
BR = dict(task="branch", ntrain=64, gate_h=True, ln_h=False)
ARMS = {"a": ("candidate register, top-1 from scratch, program", dict(C, topk=1)),
        "b": ("candidate register, top-1 from scratch, branch", dict(C, topk=1, **BR)),
        "c": ("clock, learned keys, top-1 from scratch, program", dict(progress="clock", topk=1)),
        "e": ("reg0, SIGMOID scores, dense 10 % then top-1, program", dict(progress="reg0", score="sigmoid", topk=1, anneal=0.1)),
        "f": ("candidate register, top-1 from scratch, PROGRAM task, gate also reads raw data", dict(C, topk=1, gate_h=True, ln_h=False)),
        "g": ("candidate register, top-1 from scratch, 8 streams 5 prims, gate also reads raw data", dict(C, topk=1, gate_h=True, ln_h=False, K=8, P=5, ntrain=80, B=8)),
        "d": ("candidate register, top-1 from scratch, 8 streams 5 prims", dict(C, topk=1, K=8, P=5, ntrain=80, B=8))}
arm, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
label, over = ARMS[arm]
for s in seeds:
    t0 = time.time()
    r = run(over, s)
    print(f"{label} | seed {s} | train {r['train']:.1e} | held-out {r['test']:.1e} | purity {r['purity']:.2f} | used {r['used']} | ran(train) {r['train_ran']:.2f} | {time.time()-t0:.0f}s", flush=True)
