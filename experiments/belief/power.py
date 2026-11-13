"""Power of the belief experiment's two scored questions (DESIGN §1), without a model.

A stand-in "model" is a point on the line between the two exact references plus estimation noise:
    Q1   p = (1 - lam) * mask-only update  +  lam * exact update            (lam = share of the exact update it makes)
    Q2   p = (1 - gam) * memorising posterior  +  gam * generalising one    (held-out worlds)
    noise: p' ∝ (p + 1e-3) * exp(sigma * z), z ~ N(0, 1) per valid action; sigma set so that mean TV(p', p) is a chosen level.
Rung 4's routers sat at TV 0.10–0.17 from their targets (RESULTS §5, p6_check), which is the realistic range.

Two statistics are evaluated:
    vote   share of informative decisions (references differ by TV > 0.05) on which p' is nearer the first reference's rival
           (DESIGN rev 0's rule)
    beta   the pooled projection of the model's departure onto the exact one: sum <p' - ref0, ref1 - ref0> / sum |ref1 - ref0|^2
           over ALL decisions; it estimates lam (or gam) and needs no threshold.
Sampling error is by bootstrap over instances (Q1) and over held-out worlds (Q2).

Carriers differ only in which (rule, at-root / below-root) classes share a hidden unit; H10 and H12 are probes of how the
signal grows with hidden bits on the existing executor, not proposals.

Usage: PYTHONPATH=src python3 experiments/belief/power.py > experiments/belief/power.md
"""
import math, random, sys, time, collections
from collections import deque
import numpy as np
from topology_design.rewrite.rules import RULES, valid_actions
from topology_design.rewrite.search import FULL
from topology_design.rewrite.data import sample_term

CAP, DELTA, KS = 5, 2, (2, 3, 4)
RIDX = {r: i for i, r in enumerate(RULES)}
R, I = "root", "inner"
CARRIERS = {
    "H4": [["Af", "Ab"], ["C"], ["If", "IbR", "IbL"], ["Df", "Db"]],
    "H6": [["Af"], ["Ab"], ["C"], ["If", "IbR", "IbL"], ["Df"], ["Db"]],
    "H8": [[r] for r in RULES],
    "H10": [["Af"], ["Ab"], [("C", R)], [("C", I)], ["If"], [("IbR", R)], [("IbR", I)], ["IbL"], ["Df"], ["Db"]],
    "H12": [[("Af", R)], [("Af", I)], ["Ab"], [("C", R)], [("C", I)], ["If"], [("IbR", R)], [("IbR", I)], [("IbL", R)], [("IbL", I)], ["Df"], ["Db"]],
}


def unit_map(spec):
    u = [None] * 16
    for j, grp in enumerate(spec):
        for item in grp:
            if isinstance(item, str):
                u[2 * RIDX[item]] = u[2 * RIDX[item] + 1] = j
            else:
                u[2 * RIDX[item[0]] + (item[1] == I)] = j
    assert None not in u
    return u


def build_ball(u, radius):
    idx, nodes, dist, out = {u: 0}, [u], [0], [[]]
    q = deque([u])
    while q:
        v = q.popleft()
        i = idx[v]
        if dist[i] >= radius:
            continue
        for rule, path, x in valid_actions(v, FULL, CAP):
            if x not in idx:
                idx[x] = len(nodes); nodes.append(x); dist.append(dist[i] + 1); out.append([]); q.append(x)
            out[i].append((2 * RIDX[rule] + (1 if path else 0), idx[x], path))
    return nodes, dist, out


class Instance:
    def __init__(self, rng, carrier, rho=0.5):
        self.unit_of = unit_map(CARRIERS[carrier])
        self.nu = len(CARRIERS[carrier])
        nw = (1 << self.nu) - 1
        ALL = (1 << nw) - 1
        Mu = [sum(1 << (w - 1) for w in range(1, nw + 1) if w >> j & 1) for j in range(self.nu)]
        self.M = [Mu[self.unit_of[c]] for c in range(16)]
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
        prev[self.s] = ALL
        self.R = [prev]
        for _ in range(self.B):
            new = list(prev)
            for i in range(self.n):
                acc = prev[i]
                for c, x, _p in out[i]:
                    t = prev[x]
                    if t:
                        acc |= t & self.M[c]
                new[i] = acc
            self.R.append(new)
            prev = new
        self._d = {}
        bits = self.R[self.B][0]
        self.cons = [w for w in range(1, nw + 1) if bits >> (w - 1) & 1]
        pr = {w: rho ** bin(w).count("1") * (1 - rho) ** (self.nu - bin(w).count("1")) for w in self.cons}
        z = sum(pr.values())
        self.post0 = {w: p / z for w, p in pr.items()}
        x = rng.random(); acc = 0.0
        for w, p in self.post0.items():
            acc += p
            if x <= acc:
                break
        self.true = w

    def dw(self, i, w):
        key = (i, w)
        d = self._d.get(key)
        if d is None:
            d = 99
            for j, layer in enumerate(self.R):
                if layer[i] >> (w - 1) & 1:
                    d = j
                    break
            self._d[key] = d
        return d

    def scores(self, i, worlds):
        sc = collections.defaultdict(float)
        for w in worlds:
            dv = self.dw(i, w)
            o = [(c, x, p) for c, x, p in self.out[i] if w >> self.unit_of[c] & 1 and self.dw(x, w) == dv - 1]
            for a in o:
                sc[a] += self.post0[w] / len(o)
        z = sum(sc.values())
        return {a: v / z for a, v in sc.items()} if z > 0 else {}

    def trace(self, train=None):
        """Decisions along the exact-posterior-greedy trajectory: the exact update q, the mask-only update m, and (with a
        training set) the memorising posterior qt, each as a vector over the valid actions at that state."""
        v, b, P, A, calls = 0, self.B, 0, 0, 0
        rejected = collections.defaultdict(set)
        recs = []
        base = ([w for w in self.cons if w in train] or self.cons) if train is not None else None
        while v != self.s and b > 0 and calls < self.B + 40:
            live = [w for w in self.cons if self.dw(v, w) <= b]
            upd = [w for w in live if (w & P) == P and (w & A) == 0]
            q = self.scores(v, upd)
            if not q:
                break
            m = {a: x for a, x in self.scores(v, live).items() if a not in rejected[v]}
            zm = sum(m.values())
            acts = self.out[v]
            vec = lambda d, z=1.0: np.array([d.get(a, 0.0) / z for a in acts])
            rec = dict(after=calls > 0, q=vec(q), m=vec(m, zm) if zm > 0 else None)
            if train is not None:
                tl = [w for w in base if self.dw(v, w) <= b]
                tu = [w for w in tl if (w & P) == P and (w & A) == 0]
                qt = self.scores(v, tu)
                if not qt:
                    mm = {a: x for a, x in self.scores(v, tl).items() if a not in rejected[v]}
                    z2 = sum(mm.values())
                    qt = {a: x / z2 for a, x in mm.items()} if z2 > 0 else q
                rec["qt"] = vec(qt)
            recs.append(rec)
            a = max(sorted(q), key=lambda t: q[t])
            calls += 1
            j = self.unit_of[a[0]]
            if self.true >> j & 1:
                v, b, P = a[1], b - 1, P | (1 << j)
            else:
                rejected[v].add(a); A |= 1 << j
        return recs


def noisy(p, sigma, g):
    w = (p + 1e-3) * np.exp(sigma * g.standard_normal(p.shape))
    return w / w.sum()


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def calibrate(points, level, g):
    lo, hi = 0.0, 4.0
    sample = points[:600]
    for _ in range(18):
        mid = (lo + hi) / 2
        t = np.mean([tv(noisy(p, mid, g), p) for p in sample])
        lo, hi = (mid, hi) if t < level else (lo, mid)
    return (lo + hi) / 2


def stats(groups, ref0, ref1, lam, sigma, g, n_boot=300):
    """groups: list of lists of decisions (one list per instance or per world). Returns beta, vote, and their bootstrap sd."""
    num, den, votes, ninf = [], [], [], []
    for grp in groups:
        a = b = c = d = 0.0
        for r in grp:
            r0, r1 = r[ref0], r[ref1]
            p = noisy((1 - lam) * r0 + lam * r1, sigma, g) if sigma > 0 else (1 - lam) * r0 + lam * r1
            diff = r1 - r0
            a += float((p - r0) @ diff); b += float(diff @ diff)
            if tv(r0, r1) > 0.05:
                d += 1; c += tv(p, r1) < tv(p, r0)
        num.append(a); den.append(b); votes.append(c); ninf.append(d)
    num, den, votes, ninf = map(np.array, (num, den, votes, ninf))
    beta, vote = num.sum() / den.sum(), votes.sum() / max(1.0, ninf.sum())
    idx = g.integers(0, len(groups), size=(n_boot, len(groups)))
    bs = num[idx].sum(1) / den[idx].sum(1)
    vs = votes[idx].sum(1) / np.maximum(1.0, ninf[idx].sum(1))
    return beta, vote, bs.std(), vs.std()


def gather(carrier, n, seed, heldout=None, trains=None):
    rng = random.Random(seed)
    out, t0, made = [], time.time(), 0
    while len(out) < n:
        ins = Instance(rng, carrier)
        made += 1
        if heldout is not None and ins.true not in heldout:
            continue
        if trains is None:
            out.append((ins, ins.trace()))
        else:
            out.append((ins, {N: ins.trace(train=T) for N, T in trains.items()}))
        ins._d = ins.R = ins.out = None          # the caches are large; only the traces and the posterior are kept
    return out, (time.time() - t0) / made


LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE = (0.0, 0.05, 0.10, 0.15, 0.20)

def probe_row(name, n, g):
    d, sec = gather(name, n, 0)
    grp = [[r for r in recs if r["after"] and r["m"] is not None] for _, recs in d]
    pp = [r for x in grp for r in x]
    sep = np.array([tv(r["q"], r["m"]) for r in pp]) if pp else np.array([0.0])
    ent = np.mean([-sum(p * math.log2(p) for p in ins.post0.values()) for ins, _ in d])
    energy = sum(float(((r["q"] - r["m"]) ** 2).sum()) for r in pp) / n
    sgn = calibrate([r["q"] for r in pp], 0.12, g)
    b, v, sb, sv = stats(grp, "m", "q", 1.0, sgn, g)
    print(f"| {name} | {len(CARRIERS[name])} | {ent:.2f} | {len(pp) / n:.1f} | {np.mean(sep > 0.05):.0%} | {sep.mean():.3f} | {energy:.3f} | {sb * math.sqrt(n / 2000):.3f} | {sec:.2f} |", flush=True)


if __name__ == "__main__":
    g = np.random.default_rng(0)
    if len(sys.argv) > 1 and sys.argv[1] == "probe":          # only the probe rows named, e.g. probe H10:150 H12:40
        for item in sys.argv[2:]:
            name, n = item.split(":")
            probe_row(name, int(n), g)
        sys.exit(0)
    print("# Power of the belief experiment's scored questions (model-free)\n")
    print("Definitions in `power.py`. A stand-in model is a mixture of the two exact references plus noise; rung 4's routers sat 0.10–0.17 (TV) from their targets.\n")

    # ---- Q1 on the existing carrier
    data, sec = gather("H8", 600, 0)
    groups = [[r for r in recs if r["after"] and r["m"] is not None] for _, recs in data]
    pts = [r for grp in groups for r in grp]
    sep = np.array([tv(r["q"], r["m"]) for r in pts])
    print("## Q1 (updating) on H8, the existing executor\n")
    print(f"{len(data)} instances, {len(pts)} decisions after a verdict ({len(pts) / len(data):.1f} per instance). "
          f"Exact against mask-only update: TV > 0.05 on {np.mean(sep > 0.05):.0%}, > 0.10 on {np.mean(sep > 0.10):.0%}, > 0.20 on {np.mean(sep > 0.20):.0%}; "
          f"mean {sep.mean():.3f}; mean on the informative ones {sep[sep > 0.05].mean():.2f}. {sec:.2f} s per instance.\n")
    print("Each cell: beta / vote share, for a model that makes the share lam of the exact update, at each noise level:\n")
    print("| model noise (TV to its own target) | " + " | ".join(f"lam = {l}" for l in LAMS) + " | sd of beta, sd of vote at 2,000 instances |")
    print("|---|" + "---|" * (len(LAMS) + 1))
    qpts = [r["q"] for r in pts]
    for lv in NOISE:
        sg = calibrate(qpts, lv, g) if lv > 0 else 0.0
        cells, sds = [], None
        for lam in LAMS:
            b, v, sb, sv = stats(groups, "m", "q", lam, sg, g)
            cells.append(f"{b:.2f} / {v:.2f}")
            sds = (sb, sv)
        k = math.sqrt(len(groups) / 2000)
        print(f"| {lv:.2f} | " + " | ".join(cells) + f" | {sds[0] * k:.3f}, {sds[1] * k:.3f} |")
    print()

    # ---- Q2 on the existing carrier
    order = list(range(1, 256))
    random.Random(7).shuffle(order)
    heldout, pool = set(order[:64]), order[64:]
    trains = {N: set(pool[:N]) for N in (16, 64, 160)}
    data2, sec2 = gather("H8", 500, 1, heldout=heldout, trains=trains)
    print("## Q2 (coverage) on H8\n")
    print(f"{len(data2)} instances whose true world is one of 64 held-out worlds; training sets nested, drawn from the other 191. "
          f"Bootstrap over held-out worlds. {sec2:.2f} s per generated instance.\n")
    print("| training worlds | decisions | generalising against memorising: TV > 0.05 / mean TV | beta at gam = 0 / 0.5 / 1, noise 0.12 | vote at gam = 0 / 0.5 / 1 | sd of beta, sd of vote (64 worlds, these instances) |")
    print("|---|---|---|---|---|---|")
    sg = None
    for N in (16, 64, 160):
        byw = collections.defaultdict(list)
        allp = []
        for ins, tr in data2:
            byw[ins.true] += tr[N]
            allp += tr[N]
        if sg is None:
            sg = calibrate([r["q"] for r in allp], 0.12, g)
        sep = np.array([tv(r["q"], r["qt"]) for r in allp])
        res = [stats(list(byw.values()), "qt", "q", gam, sg, g) for gam in (0.0, 0.5, 1.0)]
        print(f"| {N} | {len(allp)} | {np.mean(sep > 0.05):.0%} / {sep.mean():.3f} | " + " / ".join(f"{r[0]:.2f}" for r in res) + " | " +
              " / ".join(f"{r[1]:.2f}" for r in res) + f" | {res[1][2]:.3f}, {res[1][3]:.3f} |")
    print()

    # ---- would more hidden bits help?
    print("## Signal against hidden bits (probe)\n")
    print("Same problems (seed 0); the carriers differ only in how the sixteen (rule, root / below-root) classes are grouped into hidden units.\n")
    print("| carrier | hidden units | world entropy given the problem, bits | decisions after a verdict per instance | TV(exact, mask) > 0.05 | mean TV | signal per instance, sum of squared differences | sd of beta at noise 0.12, 2,000 instances | s per instance |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, n in (("H4", 300), ("H6", 300), ("H8", 300), ("H10", 150), ("H12", 40)):
        probe_row(name, n, g)
