"""Preliminary, exact, model-free numbers for the hidden-world carrier (DESIGN §3–§4), computed before the design is fixed
(ledger L36: compute the cells before fixing the rules that read them).

Carrier: terms over a, b, c, d, 0 with +, size cap 5; eight directed rewrite rules. A world is a non-empty set of *units*
(H8: each rule is a unit; H4: the four axioms are units). The kernel accepts an action iff its rule's unit is in the world.

Generator (world-blind in u, s): u ~ the rung-4 term sampler; k ~ uniform{2,3,4}; s ~ uniform on the full-world sphere of
radius k around u; W ~ independent Bernoulli(rho) per unit, conditioned on d_W(u, s) <= k + DELTA.
Exact posterior over worlds: P(W | u, s, evidence) ∝ rho^|W| (1-rho)^(n-|W|) · 1[d_W(u,s) <= k+DELTA] · 1[W agrees with every verdict].
Step target: uniform over the actions that are optimal in the true world. Its Bayes predictor:
    q(a | v, evidence) = sum_W P(W | ...) 1[a optimal in W at v] / #optimal_W(v)      (worlds that can still finish in budget)

Reference policies (all exact):
    oracle       knows W
    q-greedy     argmax q, belief updated on every verdict (what a model that learned the posterior would do)
    mask-greedy  belief over worlds frozen at P(W | u, s); an action rejected at a state is not proposed there again
Usage: PYTHONPATH=src python3 experiments/belief/prelim_info.py [n] > experiments/belief/prelim_info.md
"""
import math, random, sys, time, collections
from collections import deque
from topology_design.rewrite.rules import RULES, valid_actions
from topology_design.rewrite.search import FULL
from topology_design.rewrite.data import sample_term

CAP, DELTA = 5, 2
KS = (2, 3, 4)
RIDX = {r: i for i, r in enumerate(RULES)}
UNITS = {"H8": [(r,) for r in RULES], "H4": [("Af", "Ab"), ("C",), ("If", "IbR", "IbL"), ("Df", "Db")]}


def build_ball(u, radius):
    idx, nodes, dist, out = {u: 0}, [u], [0], []
    q = deque([u])
    while q:
        v = q.popleft()
        i = idx[v]
        while len(out) <= i:
            out.append([])
        if dist[i] >= radius:
            continue
        for rule, path, x in valid_actions(v, FULL, CAP):
            if x not in idx:
                idx[x] = len(nodes); nodes.append(x); dist.append(dist[i] + 1); q.append(x)
            out[i].append((RIDX[rule], idx[x], path))
    while len(out) < len(nodes):
        out.append([])
    return nodes, dist, out


class Instance:
    def __init__(self, rng, units, rho):
        self.nu = len(units)
        self.unit_of = [None] * len(RULES)
        for j, grp in enumerate(units):
            for r in grp:
                self.unit_of[RIDX[r]] = j
        nw = (1 << self.nu) - 1
        self.ALL = (1 << nw) - 1
        self.M = []                                   # per rule: bitset of the worlds containing its unit
        for r in range(len(RULES)):
            j = self.unit_of[r]
            self.M.append(sum(1 << (w - 1) for w in range(1, nw + 1) if w >> j & 1))
        self.prior = [0.0] + [rho ** bin(w).count("1") * (1 - rho) ** (self.nu - bin(w).count("1")) for w in range(1, nw + 1)]
        while True:
            u = sample_term(rng, (2, 3, 4, 5))
            self.k = rng.choice(KS)
            self.B = self.k + DELTA
            nodes, dist, out = build_ball(u, self.B)
            sphere = [i for i, d in enumerate(dist) if d == self.k]
            if sphere:
                break
        self.n, self.out, self.s = len(nodes), out, rng.choice(sphere)
        prev = [0] * self.n
        prev[self.s] = self.ALL
        self.R = [prev]
        for _ in range(self.B):
            new = list(prev)
            for i in range(self.n):
                acc = prev[i]
                for r, x, _p in out[i]:
                    t = prev[x]
                    if t:
                        acc |= t & self.M[r]
                new[i] = acc
            self.R.append(new)
            prev = new
        self._d = {}
        self.cons = [w for w in range(1, nw + 1) if self.dw(0, w) <= self.B]
        z = sum(self.prior[w] for w in self.cons)
        self.post0 = {w: self.prior[w] / z for w in self.cons}
        x = rng.random(); acc = 0.0
        for w, p in self.post0.items():
            acc += p
            if x <= acc:
                break
        self.true = w

    def dw(self, i, w):
        key = (i, w)
        if key not in self._d:
            d = 99
            for j, layer in enumerate(self.R):
                if layer[i] >> (w - 1) & 1:
                    d = j
                    break
            self._d[key] = d
        return self._d[key]

    def opt(self, i, w):
        dv = self.dw(i, w)
        return [(r, x, p) for r, x, p in self.out[i] if w >> self.unit_of[r] & 1 and self.dw(x, w) == dv - 1]

    def scores(self, i, worlds, weights):
        sc = collections.defaultdict(float)
        for w in worlds:
            o = self.opt(i, w)
            for r, x, p in o:
                sc[(r, x, p)] += weights[w] / len(o)
        z = sum(sc.values())
        return {a: v / z for a, v in sc.items()} if z > 0 else {}

    def train_scores(self, v, b, P, A, rejected, train):
        """The memorising reference: the posterior with the prior restricted to the training worlds. When no training world
        agrees with the evidence it falls back to its unrevised belief with the rejected actions masked, then to all worlds."""
        base = [w for w in self.cons if w in train] or self.cons
        live = [w for w in base if self.dw(v, w) <= b]
        upd = [w for w in live if (w & P) == P and (w & A) == 0]
        if upd:
            return self.scores(v, upd, self.post0)
        m = {a: x for a, x in self.scores(v, live, self.post0).items() if a not in rejected[v]}
        if m:
            z = sum(m.values())
            return {a: x / z for a, x in m.items()}
        return self.scores(v, [w for w in self.cons if self.dw(v, w) <= b and (w & P) == P and (w & A) == 0], self.post0)

    def run(self, policy, train=None):
        v, b, P, A, calls, rej = 0, self.B, 0, 0, 0, 0
        rejected = collections.defaultdict(set)
        gaps = []
        while v != self.s and b > 0 and calls < self.B + 40:
            live = [w for w in self.cons if self.dw(v, w) <= b]
            upd = [w for w in live if (w & P) == P and (w & A) == 0]
            q = self.scores(v, upd, self.post0)
            m = {a: x for a, x in self.scores(v, live, self.post0).items() if a not in rejected[v]}
            zm = sum(m.values())
            m = {a: x / zm for a, x in m.items()} if zm > 0 else {}
            if policy == "q" and calls > 0 and q and m:
                gaps.append(0.5 * sum(abs(q.get(a, 0) - m.get(a, 0)) for a in set(q) | set(m)))
            sc = q if policy == "q" else m
            if train is not None:
                qt = self.train_scores(v, b, P, A, rejected, train)
                if policy == "q" and q and qt:
                    gaps.append(0.5 * sum(abs(q.get(a, 0) - qt.get(a, 0)) for a in set(q) | set(qt)))
                if policy == "qtrain":
                    sc = qt
            if policy == "oracle":
                sc = self.scores(v, [self.true], {self.true: 1.0}) if self.dw(v, self.true) <= b else {}
            if not sc:
                break
            a = max(sorted(sc), key=lambda t: sc[t])
            calls += 1
            j = self.unit_of[a[0]]
            if self.true >> j & 1:
                v, b, P = a[1], b - 1, P | (1 << j)
            else:
                rej += 1; rejected[v].add(a); A |= 1 << j
        return dict(ok=v == self.s, calls=calls, rej=rej, gaps=gaps)


def H(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def table(name, rho, n, seed=0):
    rng = random.Random(seed)
    rows, t0 = [], time.time()
    for _ in range(n):
        ins = Instance(rng, UNITS[name], rho)
        r = dict(k=ins.k, n_nodes=ins.n, ncons=len(ins.cons), Hw=H(ins.post0.values()), size=bin(ins.true).count("1"),
                 dtrue=ins.dw(0, ins.true))
        q0 = ins.scores(0, ins.cons, ins.post0)
        r["H_hidden"] = H(q0.values())
        r["H_known"] = math.log2(len(ins.opt(0, ins.true)))
        r["rej1"] = sum(p for a, p in q0.items() if not ins.true >> ins.unit_of[a[0]] & 1)
        for pol in ("oracle", "q", "mask"):
            res = ins.run(pol)
            r[pol] = res
        rows.append(r)
    m = lambda f: sum(f(r) for r in rows) / len(rows)
    gaps = [g for r in rows for g in r["q"]["gaps"]]
    cells = [f"{name}, ρ = {rho}", f"{m(lambda r: r['Hw']):.2f}", f"{m(lambda r: r['ncons']):.0f} of {(1 << len(UNITS[name])) - 1}",
             f"{m(lambda r: r['size']):.1f} of {len(UNITS[name])}", f"{m(lambda r: r['H_known']):.2f} / {m(lambda r: r['H_hidden']):.2f}", f"{m(lambda r: r['rej1']):.2f}"]
    for pol in ("oracle", "q", "mask"):
        cells.append(f"{m(lambda r: r[pol]['ok']):.2f} / {m(lambda r: r[pol]['calls']):.1f} / {m(lambda r: r[pol]['rej']):.1f}")
    cells.append(f"{sum(g > 0.05 for g in gaps) / max(1, len(gaps)):.0%} of {len(gaps)}; mean TV {sum(gaps) / max(1, len(gaps)):.2f}")
    cells.append(f"{m(lambda r: r['n_nodes']):.0f}; {(time.time() - t0) / n:.2f}s")
    return "| " + " | ".join(cells) + " |"


def coverage_table(n_train, n, rho=0.5, seed=0):
    """Held-out worlds: the generalising posterior (prior over all 255 worlds) against the memorising one (prior over the
    training worlds only), on instances whose true world is not a training world."""
    rng = random.Random(seed)
    nw = (1 << 8) - 1
    train = set(random.Random(1000 + n_train).sample(range(1, nw + 1), n_train))
    rows = []
    while len(rows) < n:
        ins = Instance(rng, UNITS["H8"], rho)
        if ins.true in train:
            continue
        a, b = ins.run("q", train=train), ins.run("qtrain", train=train)
        rows.append((a, b, len([w for w in ins.cons if w in train])))
    m = lambda f: sum(f(r) for r in rows) / len(rows)
    gaps = [g for r in rows for g in r[0]["gaps"]]
    return (f"| {n_train} | {m(lambda r: r[2]):.0f} | {m(lambda r: r[0]['ok']):.2f} / {m(lambda r: r[0]['calls']):.1f} / {m(lambda r: r[0]['rej']):.1f} | "
            f"{m(lambda r: r[1]['ok']):.2f} / {m(lambda r: r[1]['calls']):.1f} / {m(lambda r: r[1]['rej']):.1f} | "
            f"{sum(g > 0.05 for g in gaps) / len(gaps):.0%} of {len(gaps)}; mean TV {sum(gaps) / len(gaps):.2f} |")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    print("# Hidden-world carrier: preliminary exact numbers\n")
    print(f"{n} instances per row, seed 0; k ∈ {{2, 3, 4}}, accepted-step budget k + {DELTA} (a run fails when an accepted non-optimal step uses the budget up). No model. Definitions in `prelim_info.py`.\n")
    print("| carrier | world entropy given (u, s), bits | consistent worlds | true world size | first-step entropy known / hidden, bits | P(first proposal rejected) | oracle: solved / kernel calls / rejections | q-greedy | mask-greedy | decisions after a verdict where exact and mask differ by TV > 0.05 | ball nodes; time per instance |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rho in (("H4", 0.5), ("H8", 0.35), ("H8", 0.5), ("H8", 0.75)):
        print(table(name, rho, n), flush=True)
    KS = (4, 5)
    print(f"\nLonger proofs, k ∈ {{4, 5}} ({max(40, n // 4)} instances per row):\n")
    print("| carrier | world entropy given (u, s), bits | consistent worlds | true world size | first-step entropy known / hidden, bits | P(first proposal rejected) | oracle: solved / kernel calls / rejections | q-greedy | mask-greedy | decisions after a verdict where exact and mask differ by TV > 0.05 | ball nodes; time per instance |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rho in (("H8", 0.35), ("H8", 0.5)):
        print(table(name, rho, max(40, n // 4)), flush=True)
    KS = (2, 3, 4)
    print(f"\nHeld-out worlds (H8, ρ = 0.5, k ∈ {{2, 3, 4}}, {max(100, n // 2)} instances per row whose true world is not a training world). "
          "The generalising posterior puts its prior on all 255 worlds; the memorising posterior on the training worlds only:\n")
    print("| training worlds | training worlds consistent with (u, s) | generalising: solved / kernel calls / rejections | memorising | decisions where the two differ by TV > 0.05 |")
    print("|---|---|---|---|---|")
    for nt in (8, 16, 32, 64, 128):
        print(coverage_table(nt, max(100, n // 2)), flush=True)

