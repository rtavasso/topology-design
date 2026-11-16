"""Review probe 1 (no training): structure of the training-world graphs, references, and what the training worlds
determine in `select` for a learner that sees only which primitive wins (signs), not trigger magnitudes."""
import sys, itertools, time
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/worlds")
import torch
from worlds import DEF, Teacher, sample_worlds, references, err, all_worlds
t0 = time.time()
P = 6

def cut_vertices(P, edges):
    def conn(nodes, edges):
        nodes = set(nodes); adj = {i: set() for i in nodes}
        for a, b in edges:
            if a in nodes and b in nodes: adj[a].add(b); adj[b].add(a)
        s = next(iter(nodes)); seen = {s}; st = [s]
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen: seen.add(v); st.append(v)
        return len(seen) == len(nodes)
    return [v for v in range(P) if not conn(set(range(P)) - {v}, edges)]

print("== training graphs: world split depends on (seed, cover) only; cut vertices => select not determined even for linear bids")
for cover in (7, 9, 11, 13):
    row = []
    for seed in range(12):
        tr, te = sample_worlds(P, cover, seed)
        row.append(len(cut_vertices(P, tr.tolist())))
    print(f"cover {cover}: #cut vertices per seed 0..11: {row}  -> splits with a cut vertex: {sum(r>0 for r in row)}/12")
for seed in (0, 1):
    for cover in (7, 9):
        tr, te = sample_worlds(P, cover, seed)
        print(f"seed {seed} cover {cover} train {tr.tolist()}")
a = {tuple(x) for x in sample_worlds(P, 7, 0)[0].tolist()}; b = {tuple(x) for x in sample_worlds(P, 9, 0)[0].tolist()}
print("nested 7 in 9 at seed 0?", a <= b, " shared:", len(a & b))

def forced_fraction(T, tr, te, x):
    """fraction of (held-out world, state) where the teacher's choice follows from the trained choices by transitivity"""
    s = x @ T.w.T                                    # [N, P]
    tot = forced = 0
    edges = tr.tolist()
    for a, b in te.tolist():
        for i in range(x.shape[0]):
            hi, lo = (a, b) if s[i, a] > s[i, b] else (b, a)
            reach = {hi}; st = [hi]
            while st:
                u = st.pop()
                for p, q in edges:
                    for (u1, v1) in ((p, q), (q, p)):
                        if u1 == u and s[i, v1] < s[i, u] and v1 not in reach:
                            reach.add(v1); st.append(v1)
            forced += lo in reach; tot += 1
    return forced / tot

class TT(Teacher):
    def step(self, worlds, h, policy="true"):
        if policy == "blind":       # knows nothing about the world: mean of all P primitives
            f = torch.tanh(torch.einsum("pij,mbj->mbpi", self.A, h)).mean(2) * (1 if self.task == "select" else 2)
            return h + self.eps * f
        return super().step(worlds, h, policy)

print("\n== references on held-out worlds (error 1.0 = predicting no change). cover 9, seed 0")
print("task teacher blend one blind(mean of all 6) wrongworld | select: fraction of held-out decisions forced by transitivity (cover 7 / 9 / 11)")
for task in ("select", "sum"):
    for ts in (10, 11, 12):
        c = dict(DEF, task=task, tseed=ts); T = TT(c)
        tr, te = sample_worlds(P, 9, 0)
        x = torch.randn(len(te), 2048, 6, generator=torch.Generator().manual_seed(0))
        tgt = T.run(te, x, 4)
        r = references(c, te)
        blind = err(T.run(te, x, 4, "blind"), tgt, x).item()
        g = torch.Generator().manual_seed(1)
        wrong = tr[torch.randint(len(tr), (len(te),), generator=g)]
        ww = err(T.run(wrong, x, 4), tgt, x).item()
        ff = ""
        if task == "select":
            ff = " / ".join(f"{forced_fraction(T, *sample_worlds(P, cv, 0), torch.randn(200, 6, generator=torch.Generator().manual_seed(3))):.2f}" for cv in (7, 9, 11))
        print(f"{task} {ts} {r['blend']:.3f} {r['one']:.3f} {blind:.3f} {ww:.3f} | {ff}")

print("\n== counterexample (select): rescale the triggers of one side of a cut vertex; all trained choices unchanged, held-out change")
done = 0
for seed in range(12):
    tr, te = sample_worlds(P, 7, seed)
    cv = cut_vertices(P, tr.tolist())
    if not cv or done >= 2: continue
    v = cv[0]; rest = set(range(P)) - {v}
    adj = {i: set() for i in rest}
    for p, q in tr.tolist():
        if p in rest and q in rest: adj[p].add(q); adj[q].add(p)
    s0 = next(iter(rest)); comp = {s0}; st = [s0]
    while st:
        u = st.pop()
        for w_ in adj[u]:
            if w_ not in comp: comp.add(w_); st.append(w_)
    T = Teacher(dict(DEF, task="select")); T2 = Teacher(dict(DEF, task="select"))
    for p in comp: T2.w[p] = T.w[v] + 4.0 * (T.w[p] - T.w[v])
    x = torch.randn(1, 4000, 6, generator=torch.Generator().manual_seed(0))
    agree_tr = min((T.choice(tr[i:i+1], x) == T2.choice(tr[i:i+1], x)).double().mean().item() for i in range(len(tr)))
    cross = [w for w in te.tolist() if (w[0] in comp) != (w[1] in comp) and v not in w]
    ag = [(T.choice(torch.tensor([w]), x) == T2.choice(torch.tensor([w]), x)).double().mean().item() for w in cross]
    xb = torch.randn(len(cross), 2048, 6, generator=torch.Generator().manual_seed(0)); cw = torch.tensor(cross)
    e = err(T2.run(cw, xb, 4), T.run(cw, xb, 4), xb).item()
    print(f"seed {seed} cover 7 train {tr.tolist()} cut vertex {v} side {sorted(comp)}: min agreement on trained worlds {agree_tr:.3f}; "
          f"agreement on {len(cross)} crossing held-out worlds {[round(a,2) for a in ag]}; 4-pass error of the alternative on them {e:.3f}")
    done += 1
print("time", round(time.time() - t0, 1))
