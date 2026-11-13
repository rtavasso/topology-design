"""Brute-force simulation of the generative process of DESIGN §3/§7 as written, against the memorising posterior M.

Per problem (u, s, k): world uniform over the pool's worlds that can reach s from u within B = k + 2; each proposal comes
with probability 1/2 from (i) uniform over valid actions of rules not yet rejected, (ii) uniform over the actions optimal
if every rule not yet rejected were present; label = one action uniform over the true world's optimal actions; the episode
ends at the goal, when the budget is used, or when the true world can no longer finish (no row is generated there).
Empirical label frequencies per model input (state, budget, known-present, known-absent) are compared with M (prior on the
pool) and, for scale, with G (prior on all worlds). Null: the same counts with labels drawn from M itself.

Variants (to see what breaks the claim):
  spec        as written
  nodrop      the episode continues, and rows are generated, after the true world can no longer finish within the budget
              (label: uniform over the actions that reduce the true world's distance, while that distance is <= B);
              an input at which M is undefined counts as TV = 1                    -- violates (a)
  success     only episodes that reach the goal are kept                           -- violates (c)
Usage: PYTHONPATH=src python3 t2_gen.py DRAW N EPISODES [nprob]
"""
import sys, time
from r2lib import *

draw, N, EP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
NPROB = int(sys.argv[4]) if len(sys.argv) > 4 else 6
held, trains, pool = draw_sets(draw)
T = trains[N]
prng = random.Random(555)
g = np.random.default_rng(7)


def simulate(ins, variant, rng):
    cand = [int(w) for w in np.nonzero(ins.cons & T)[0]]
    cache = {}

    def node(v):
        c = cache.get(v)
        if c is None:
            d = Dec(ins, v, ins.B, 0, 0)
            c = cache[v] = dict(rules=[int(x) for x in d.rules], dst=[int(x) for x in d.dst], ok=d.ok, opt={}, allow={})
        return c

    def opt_edges(v, w):               # w = world index 0..254
        c = node(v)
        o = c["opt"].get(w)
        if o is None:
            o = c["opt"][w] = [int(j) for j in np.nonzero(c["ok"][w])[0]]
        return o

    def allowed(v, A):
        c = node(v)
        o = c["allow"].get(A)
        if o is None:
            o = c["allow"][A] = [j for j, r in enumerate(c["rules"]) if not A >> r & 1]
        return o

    counts = collections.defaultdict(lambda: collections.Counter())
    nrows = 0
    stats = collections.Counter()
    for _ in range(EP):
        true = rng.choice(cand)
        v, b, P, A = 0, ins.B, 0, 0
        rows = []
        while v != ins.s and b > 0:
            dead = ins.D[true, v] > b
            if dead and variant != "nodrop":
                break
            if dead and ins.D[true, v] >= 99:          # (a) violated: still a row while the true world's distance is known
                break
            lab = rng.choice(opt_edges(v, true))
            rows.append(((v, b, P, A), lab))
            if rng.random() < 0.5:
                j = rng.choice(allowed(v, A))
            else:
                j = rng.choice(opt_edges(v, (255 & ~A) - 1))
            c = cache[v]
            r = c["rules"][j]
            if HAS[true, r]:
                v, b, P = c["dst"][j], b - 1, P | (1 << r)
            else:
                A |= 1 << r
        ok = v == ins.s
        stats["solved"] += ok
        stats["rows"] += len(rows)
        stats["nrej"] += bin(A).count("1")
        if variant == "success" and not ok:
            continue
        for key, lab in rows:
            counts[key][lab] += 1
    return counts, stats


def compare(ins, counts, rng_np):
    tvM = tvG = tvN = wsum = 0.0
    zmax = 0.0
    ncell = nkeys = 0
    for (v, b, P, A), cn in counts.items():
        n = sum(cn.values())
        if n < 200:
            continue
        d = Dec(ins, v, b, P, A)
        M = d.mix(d.live & T)
        G = d.mix(d.live)
        if M is None:
            tvM += n * 1.0; wsum += n; continue          # a row at an input where M is undefined would be a contradiction
        emp = np.zeros(d.nE)
        for j, c in cn.items():
            emp[j] = c / n
        null = rng_np.multinomial(n, M / M.sum()) / n
        tvM += n * tv(emp, M); tvG += n * tv(emp, G); tvN += n * tv(null, M); wsum += n
        nkeys += 1
        for j in range(d.nE):
            if n * M[j] >= 5 and M[j] < 1:
                z = (emp[j] - M[j]) * n / math.sqrt(n * M[j] * (1 - M[j]))
                zmax = max(zmax, abs(z)); ncell += 1
            elif M[j] == 0 and emp[j] > 0:
                zmax = max(zmax, 99.0)
    return tvM / wsum, tvN / wsum, tvG / wsum, zmax, ncell, nkeys


if __name__ == "__main__":
    print(f"draw {draw}, pool of {N} training worlds, {EP} episodes per problem and variant")
    print("| problem | variant | consistent pool worlds | rows / episode | solved | inputs with >= 200 rows | mean TV(empirical, M) | null | mean TV(empirical, G) | largest abs(z) (cells) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    done = 0
    while done < NPROB:
        ins = Inst(prng, draw_world=False)
        nc = int((ins.cons & T).sum())
        if nc < 2:
            continue
        done += 1
        for variant in ("spec", "nodrop", "success"):
            t0 = time.time()
            counts, st = simulate(ins, variant, random.Random(1000 * done + len(variant)))
            a, b_, c, z, ncell, nk = compare(ins, counts, g)
            print(f"| {done} (k={ins.k}, ball {ins.n}) | {variant} | {nc} | {st['rows'] / EP:.2f} | {st['solved'] / EP:.2f} | {nk} | {a:.4f} | {b_:.4f} | {c:.4f} | {z:.1f} ({ncell}) |", flush=True)
        ins.D = None
