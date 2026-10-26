# Design tool for the generator-of-worlds round (docs/generator_design.md §12). Usage: python count_isa.py P LMAX [ops]
"""Enumerate straight-line programs over Z_p with 2 registers (r0=x, r1=y initially), output r0.
Instructions: dest <- src1 OP src2 for OP in {add, sub, mul}; dest <- f(src) for f in {neg, inc}.
Counts distinct register states and distinct output functions by minimal program length; estimates how many
random examples identify a function among those reachable."""
import sys, time, numpy as np
p = int(sys.argv[1]); LMAX = int(sys.argv[2]); OPS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["add","sub","mul","neg","inc"]
X, Y = np.meshgrid(np.arange(p), np.arange(p), indexing="ij"); X, Y = X.ravel(), Y.ravel()
start = np.stack([X, Y]).astype(np.int16)            # state: (2 registers, p*p table)
binops = {"add": lambda a,b: (a+b)%p, "sub": lambda a,b: (a-b)%p, "mul": lambda a,b: (a*b)%p}
unops = {"neg": lambda a: (-a)%p, "inc": lambda a: (a+1)%p}
instrs = []
for d in (0,1):
    for o in OPS:
        if o in binops:
            for s1 in (0,1):
                for s2 in (0,1): instrs.append(("b", o, d, s1, s2))
        elif o in unops:
            for s in (0,1): instrs.append(("u", o, d, s, None))
def apply(st, ins):
    k, o, d, s1, s2 = ins
    out = st.copy()
    out[d] = binops[o](st[s1], st[s2]) if k == "b" else unops[o](st[s1])
    return out
key = lambda st: st.tobytes()
seen_states = {key(start)}; frontier = [start]
seen_out = {start[0].tobytes(): 0}
print(f"p={p} ops={OPS} |instructions|={len(instrs)}  total functions Z_p^2->Z_p = p^(p^2) = {p}^{p*p}")
t0 = time.time()
for L in range(1, LMAX + 1):
    nxt = []
    for st in frontier:
        for ins in instrs:
            s2 = apply(st, ins); k = key(s2)
            if k not in seen_states:
                seen_states.add(k); nxt.append(s2)
                ko = s2[0].tobytes()
                if ko not in seen_out: seen_out[ko] = L
    frontier = nxt
    newf = sum(1 for v in seen_out.values() if v == L)
    print(f"L={L}: programs {len(instrs)**L:.3g}, new states {len(nxt)}, new output functions {newf}, cumulative functions {len(seen_out)}  ({time.time()-t0:.0f}s)", flush=True)
    if not nxt: break
# identifiability: examples needed to single out a function among all reachable ones
F = np.frombuffer(b"".join(seen_out.keys()), dtype=np.int16).reshape(len(seen_out), p*p)
rng = np.random.default_rng(0)
for k in (1, 2, 3, 4, 6, 8, 12):
    amb = []
    for _ in range(300):
        i = rng.integers(len(F)); idx = rng.choice(p*p, size=min(k, p*p), replace=False)
        amb.append(int((F[:, idx] == F[i, idx]).all(1).sum()))
    print(f"  {k} examples: median #functions consistent = {int(np.median(amb))}, P(unique) = {np.mean(np.array(amb)==1):.2f}")
