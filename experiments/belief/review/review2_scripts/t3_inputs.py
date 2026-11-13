"""Input-side check (ledger L34): are the scored inputs unusual relative to the training rows in a trivial way?

(a) Training rows: simulate the generator of DESIGN §3/§7 (proposer mix 1/2 uniform, 1/2 optimistic-optimal) on training
    worlds T_N and tabulate, per row, the number of rejected rules in the evidence, of accepted rules, and the slack
    b - d_O(v) (budget minus the optimistic distance).
(b) Scored inputs: the same marginals at the out-of-support probes of held-out worlds (evidence E + r-), on G's greedy
    trajectory, weighted by count and by the signal |dG|^2; whether the probed rule has any mass under the optimistic
    planner (if not, only the uniform proposer can have produced such a rejection in training).
(c) G's greedy trajectory under the proposer mix: per-step proposal probability of the action G-greedy takes.
Usage: PYTHONPATH=src python3 t3_inputs.py DRAW N N_PROBLEMS
"""
import sys
from r2lib import *

draw, N, NP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
held, trains, pool = draw_sets(draw)
T = trains[N]
rng = random.Random(4000 + draw)


def pc(x):
    return bin(x).count("1")


# ---------- (a) training rows
rows = collections.Counter(); slack = collections.Counter(); rejO = collections.Counter(); ends = collections.Counter()
nprob = 0
while nprob < NP:
    ins = Inst(rng, draw_world=False)
    cand = [int(w) for w in np.nonzero(ins.cons & T)[0]]
    if not cand:
        continue
    nprob += 1
    cache = {}
    for _ in range(100):
        true = rng.choice(cand)
        v, b, P, A = 0, ins.B, 0, 0
        while True:
            if v == ins.s:
                ends["goal"] += 1; break
            if b == 0:
                ends["budget"] += 1; break
            if ins.D[true, v] > b:
                ends["true world dead"] += 1; break
            rows[(pc(A), pc(P))] += 1
            oid = 255 & ~A
            slack[int(b - ins.D[oid - 1, v])] += 1
            c = cache.get(v)
            if c is None:
                d = Dec(ins, v, b, 0, 0)
                c = cache[v] = (d.rules, d.dst, d.ok)
            rules, dst, ok = c
            allowed = np.nonzero((A >> rules) & 1 == 0)[0]
            oopt = np.nonzero(ok[oid - 1])[0]
            if rng.random() < 0.5:
                j = int(rng.choice(list(allowed))); src = "uniform"
            else:
                j = int(rng.choice(list(oopt))); src = "optimistic"
            r = int(rules[j])
            if HAS[true, r]:
                v, b, P = int(dst[j]), b - 1, P | (1 << r)
            else:
                A |= 1 << r
                rejO[("rejected rule had optimistic mass", bool(ok[oid - 1][rules == r].any()))] += 1
    ins.D = None
tot = sum(rows.values())
print(f"# draw {draw}, N = {N}: {NP} training problems x 100 episodes, {tot} training rows")
print("episode ends:", {k: f"{v / sum(ends.values()):.2f}" for k, v in ends.items()})
nr = collections.Counter(); na = collections.Counter()
for (a, p), c in rows.items():
    nr[a] += c; na[p] += c
print("training rows by number of rejected rules in the evidence:", {k: f"{nr[k] / tot:.3f}" for k in sorted(nr)})
print("training rows by number of accepted rules:", {k: f"{na[k] / tot:.3f}" for k in sorted(na)})
print("training rows by slack b - d_O(v):", {k: f"{slack[k] / tot:.3f}" for k in sorted(slack)})
tr = sum(rejO.values())
print("training rejections whose rule had optimistic mass:", {str(k[1]): f"{v / tr:.2f}" for k, v in rejO.items()})

# ---------- (b), (c) scored inputs
rng2 = random.Random(100 + draw)
pn = collections.Counter(); pe = collections.Counter(); po = collections.Counter(); ps = collections.Counter()
steps = []; n = 0
while n < NP:
    ins = Inst(rng2)
    if not held[ins.true]:
        continue
    n += 1

    def dec(d, calls):
        P, A = d.P, d.A
        G0 = d.mix(d.live); oid = 255 & ~A
        O0 = d.single(oid)
        allowed = (A >> d.rules) & 1 == 0
        j = argmax_first(ins, d.v, G0)
        steps.append((0.5 / allowed.sum() + 0.5 * O0[j], O0[j] > 0, 1.0 / allowed.sum()))
        for r in d.probes():
            A1 = A | (1 << r)
            if (d.fit(P, A1) & T).any():
                continue
            b0, G1 = d.rm(G0, r), d.mix(d.fit(P, A1))
            if b0 is None:
                continue
            e = float(((G1 - b0) ** 2).sum())
            pn[pc(A1)] += 1; pe[pc(A1)] += e
            k = bool(O0[d.rules == r].sum() > 0)
            po[k] += 1; po[("e", k)] += e
            ps[int(d.b - ins.D[oid - 1, d.v])] += 1

    walk(ins, ins.true, lambda d: d.mix(d.live), dec)
    ins.D = None
tp, te = sum(pn.values()), sum(pe.values())
print(f"\nout-of-support probes on {NP} held-out-world instances: {tp}")
print("by number of rejected rules in the probe's evidence (count share / signal share):", {k: f"{pn[k] / tp:.3f} / {pe[k] / te:.3f}" for k in sorted(pn)})
print("probed rule has optimistic mass (count share / signal share):", {str(k): f"{po[k] / tp:.2f} / {po[('e', k)] / te:.2f}" for k in (True, False)})
print("by slack:", {k: f"{ps[k] / tp:.3f}" for k in sorted(ps)})
st = np.array(steps)
print(f"\nG-greedy steps: {len(st)}; the action taken is optimistic-optimal on {st[:, 1].mean():.1%}; proposal probability under the mix: median {np.median(st[:, 0]):.3f}, "
      f"10th percentile {np.percentile(st[:, 0], 10):.3f}, minimum {st[:, 0].min():.4f}; uniform proposer alone: median {np.median(st[:, 2]):.3f}")
