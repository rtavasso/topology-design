"""Branching hypotheses along a proof, measured in bits, on the rewrite carrier (rw-only proofs over +, 0 with axioms A, C, I, D).

Formal objects (everything exact; nothing is trained here).
  A point in a proof is a state u (a term) with goal s in world W. Its hypotheses are the valid actions at u.
  The posterior over next steps p_W(. | u) is the first-step marginal of the uniform distribution over shortest proofs u -> s
  (the RT-post target, DESIGN §4). Its entropy H(p_W(. | u)) is the branching entropy at that point.

Part A — the world is known (what rung 4's data looks like).
  A1  chain rule (a check on the code and the data): sum over DAG states of weight(u) * H(next state | u) = log2(#shortest
      state sequences).
  A2  where the entropy sits along the sequence: branching entropy by step index.
  A3  how much of the path entropy is reordering: H(path) = H(trace class) + H(path | class), classes = proofs modulo
      swapping independent steps (Foata normal form).
  A4  proposition (no computation needed): with W known and the executor deterministic, the result of trying an action is a
      function of (u, s, W), so it carries 0 bits; and conditioning the posterior on "action a is not taken" is p with a
      removed and renormalised. A kernel verdict can then only be used as a mask.

Part B — the world is hidden (the reasoner knows u, s and the eight rules, not which axioms the kernel has).
  Prior: uniform over the worlds in which s is reachable from u in at most R steps (the consistent set; R = 8, the data's
  proofs have at most 6). Belief over next steps: the mixture m = sum_W prior(W) p_W. A trial: propose a ~ m; the kernel
  accepts iff a's axiom is in the true world. Then
      information gained about the world      I(o; W) = H_b(P(accept))            (o is a function of W)
      exact updated belief                    q = sum_W prior(W | o) p_W
      mask-only update                        m~ = m with a removed (reject) or m unchanged (accept), renormalised
      what the verdict says beyond the mask   TV(q, m~), KL(q || m~)
      mass on the truly optimal first steps   under m, under m~, under q

Usage: PYTHONPATH=src python3 experiments/rewrite/analysis/branching_info.py data/rewrite [n_per_world] > .../branching_info.md
"""
import gzip, json, math, random, sys, collections
from topology_design.rewrite.rules import WORLDS, RULE_AX, world_name, world_from_name, valid_actions, step, HALT
from topology_design.rewrite.search import bfs, reverse_distances, shortest_dag, enumerate_shortest, trace_classes, FULL
from topology_design.rewrite.posterior import shortest_posterior
from topology_design.rewrite.term import parse
from topology_design.rewrite.data import dec_action, enc_action

R = 8


def H(p):
    return -sum(x * math.log2(x) for x in p.values() if x > 0)


def Hb(x):
    return 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def kl(p, q):
    return sum(x * math.log2(x / q[k]) for k, x in p.items() if x > 0)


def load(data, tag="eval5"):
    out = []
    for W in WORLDS:
        with gzip.open(f"{data}/{tag}_{world_name(W)}.jsonl.gz", "rt") as f:
            out += [json.loads(l) for l in f]
    return out


def first_posterior(t, s, W, cap):
    """p_W(. | t) as {action string: prob}, or None when s is not reachable from t within R steps under W."""
    if t == s:
        return {HALT: 1.0}
    dist, succ, _ = bfs(t, W, cap, radius=R, stop_at=s)
    if s not in dist:
        return None
    dgoal = reverse_distances(succ, s)
    post = shortest_posterior(t, s, W, cap, succ=succ, dgoal=dgoal)
    return {enc_action(a): p for a, p in post[t]["target"].items()}


def part_a(recs, rng, n_exact=400):
    print("## Part A — the world is known\n")
    multi = [r for r in recs if r["k"] >= 1]
    # A1 + A2 on a sample: rebuild next-state distributions from the stored posterior
    sample = rng.sample(multi, 600)
    worst = 0.0
    by_step = collections.defaultdict(lambda: collections.defaultdict(float))     # k -> step index -> expected entropy
    cnt = collections.Counter()
    for r in sample:
        W, cap, t, s = world_from_name(r["world"]), r["cap"], parse(r["t"]), parse(r["s"])
        post = {parse(e["state"]): e for e in r["post"]}
        depth = {t: 0}
        frontier = [t]
        tot = 0.0
        while frontier:
            nxt = []
            for u in frontier:
                e = post[u]
                if HALT in e["target"]:
                    continue
                ns = collections.defaultdict(float)
                for a, p in e["target"].items():
                    rule, path = dec_action(a)
                    v = step(u, rule, path, W, cap)
                    ns[v] += p
                    if v not in depth:
                        depth[v] = depth[u] + 1
                        nxt.append(v)
                tot += e["weight"] * H(ns)
                by_step[r["k"]][depth[u]] += e["weight"] * H(e["target"])
            frontier = nxt
        cnt[r["k"]] += 1
        worst = max(worst, abs(tot - math.log2(r["n_shortest_states"])))
    print(f"**A1 chain rule.** On {len(sample)} sampled instances, |Σ_u weight(u)·H(next state | u) − log2(#shortest state sequences)| ≤ {worst:.2e}. The stored posteriors are consistent with the stored proof counts.\n")
    print("**A2 branching entropy by step** (bits; expected over the uniform-over-shortest-proofs walk; the action entropy, which includes several actions producing one state):\n")
    ks = sorted(cnt)
    print("| proof length k | n | " + " | ".join(f"step {j + 1}" for j in range(max(ks))) + " | whole path |")
    print("|---|---|" + "---|" * (max(ks) + 1))
    for k in ks:
        cells = [f"{by_step[k][j] / cnt[k]:.2f}" if j < k else "" for j in range(max(ks))]
        print(f"| {k} | {cnt[k]} | " + " | ".join(cells) + f" | {sum(by_step[k].values()) / cnt[k]:.2f} |")
    print()
    # A3 on all records from stored counts, exact class entropy on a sample
    n1 = [r for r in recs if r["n_shortest_actions"] > 1]
    one_class = sum(r["n_trace_classes"] == 1 for r in n1)
    print(f"**A3 reordering against different proofs.** Of {len(recs)} instances, {len(n1)} ({len(n1) / len(recs):.0%}) have more than one shortest proof. "
          f"In {one_class} of those ({one_class / len(n1):.0%}) every shortest proof is the same proof up to the order of independent steps (one trace class).\n")
    ex = rng.sample([r for r in n1 if r["n_shortest_actions"] <= 3000], n_exact)
    hp = hc = 0.0
    byk = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in ex:
        W, cap, t, s = world_from_name(r["world"]), r["cap"], parse(r["t"]), parse(r["s"])
        dist, succ, _ = bfs(t, W, cap, radius=R, stop_at=s)
        dag = shortest_dag(succ, reverse_distances(succ, s), t)
        seqs = enumerate_shortest(dag, t, s, limit=5000, actions=True)
        cl = trace_classes(seqs)
        n = len(seqs)
        a, b = math.log2(n), H({c: v / n for c, v in cl.items()})
        hp += a; hc += b
        byk[r["k"]][0] += a; byk[r["k"]][1] += b; byk[r["k"]][2] += 1
    print(f"Exact split on {len(ex)} sampled instances with several shortest proofs (uniform over shortest action sequences): path entropy {hp / len(ex):.2f} bits = "
          f"{hc / len(ex):.2f} bits of *which proof* (trace class) + {(hp - hc) / len(ex):.2f} bits of *which order*. Share that is order: {(hp - hc) / hp:.0%}.\n")
    alias = sum(math.log2(r["n_shortest_actions"]) - math.log2(r["n_shortest_states"]) for r in ex) / len(ex)
    print(f"Of the same instances' path entropy, {alias:.2f} bits come from several actions producing the same next state (log2 #action sequences − log2 #state sequences); "
          f"the rest are proofs that pass through different terms.\n")
    print("| k | n | path entropy | which proof | which order |")
    print("|---|---|---|---|---|")
    for k in sorted(byk):
        a, b, n = byk[k]
        print(f"| {k} | {n} | {a / n:.2f} | {b / n:.2f} | {(a - b) / n:.2f} |")
    print()
    print("**A4 proposition.** With the world known and a deterministic executor, the outcome of trying any action is a function of (state, goal, world): it carries 0 bits. "
          "Conditioning the posterior over next steps on \"a is not taken\" gives the posterior with a removed and renormalised. "
          "So on data like rung 4's, a kernel verdict fed back can be used only as a mask; anything more a model does with it is computation the model could have done without it.\n")


def part_b(recs, rng, n_per_world):
    print("## Part B — the world is hidden\n")
    rows = []
    for W in WORLDS:
        pool = [r for r in recs if r["world"] == world_name(W) and r["k"] >= 2]
        for r in rng.sample(pool, n_per_world):
            cap, t, s = r["cap"], parse(r["t"]), parse(r["s"])
            posts = {}
            for V in WORLDS:
                p = first_posterior(t, s, V, cap)
                if p is not None:
                    posts[V] = p
            assert W in posts
            true = posts[W]
            prior = {V: 1.0 / len(posts) for V in posts}
            mix = lambda pr: {a: sum(pr[V] * posts[V].get(a, 0.0) for V in pr) for a in set().union(*[posts[V] for V in pr])}
            m = mix(prior)
            opt = set(true)
            row = dict(world=world_name(W), k=r["k"], n_cons=len(posts), Hw=math.log2(len(posts)), H_known=H(true), H_hidden=H(m),
                       mass0=sum(m.get(a, 0) for a in opt))
            ig = tvv = klv = mass_mask = mass_q = prej = hw_after = 0.0
            for a, pa in m.items():
                if pa <= 0:
                    continue
                ax = RULE_AX[dec_action(a)[0]]
                p_acc = sum(pr for V, pr in prior.items() if ax in V)
                acc = ax in W
                post_w = {V: pr for V, pr in prior.items() if (ax in V) == acc}
                z = sum(post_w.values())
                post_w = {V: pr / z for V, pr in post_w.items()}
                q = mix(post_w)
                if acc:
                    mt = m
                else:
                    rest = {b: x for b, x in m.items() if b != a}
                    zz = sum(rest.values())
                    mt = {b: x / zz for b, x in rest.items()} if zz > 0 else {}
                ig += pa * Hb(p_acc)
                prej += pa * (not acc)
                hw_after += pa * math.log2(len(post_w))
                if mt:
                    tvv += pa * tv(q, mt)
                    klv += pa * kl(q, mt)
                    mass_mask += pa * sum(mt.get(b, 0) for b in opt)
                mass_q += pa * sum(q.get(b, 0) for b in opt)
            row.update(ig=ig, tv=tvv, kl=klv, mass_mask=mass_mask, mass_q=mass_q, prej=prej, hw_after=hw_after)
            rows.append(row)
    mean = lambda key, rs=rows: sum(r[key] for r in rs) / len(rs)
    print(f"{len(rows)} instances ({n_per_world} per true world, proof length ≥ 2). Prior: uniform over the worlds in which the goal is reachable within {R} steps.\n")
    print(f"- **Uncertainty about the world:** {mean('n_cons'):.1f} consistent worlds on average, {mean('Hw'):.2f} bits.")
    print(f"- **Branching entropy at the first step:** {mean('H_known'):.2f} bits when the world is known, {mean('H_hidden'):.2f} bits when it is hidden.")
    print(f"- **One trial** (an action drawn from the current belief, verdict from the true kernel): rejected with probability {mean('prej'):.2f}; "
          f"{mean('ig'):.2f} bits about the world expected; world entropy after it {mean('hw_after'):.2f} bits.")
    print(f"- **What the verdict says beyond a mask:** total variation between the exact update and the mask-only update {mean('tv'):.3f}; KL {mean('kl'):.3f} bits.")
    print(f"- **Belief on the truly optimal first steps:** {mean('mass0'):.3f} before the trial, {mean('mass_mask'):.3f} after a mask-only update, {mean('mass_q'):.3f} after the exact update.\n")
    print("| true world | n | consistent worlds | bits about the world per trial | P(reject) | TV(exact, mask) | mass on optimal: before | mask | exact |")
    print("|---|---|---|---|---|---|---|---|---|")
    for W in WORLDS:
        rs = [r for r in rows if r["world"] == world_name(W)]
        print(f"| {world_name(W)} | {len(rs)} | {mean('n_cons', rs):.1f} | {mean('ig', rs):.2f} | {mean('prej', rs):.2f} | {mean('tv', rs):.3f} | {mean('mass0', rs):.2f} | {mean('mass_mask', rs):.2f} | {mean('mass_q', rs):.2f} |")
    print()
    big = sum(r["tv"] > 0.05 for r in rows) / len(rows)
    print(f"Instances on which the exact update differs from the mask by TV > 0.05 in expectation: {big:.0%}.\n")


if __name__ == "__main__":
    data = sys.argv[1]
    npw = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = random.Random(0)
    recs = load(data)
    print("# Branching hypotheses along a proof, in bits (rewrite carrier, eval5)\n")
    print("Exact computations on the dataset and the executor; no model involved. Definitions in `branching_info.py`.\n")
    part_a(recs, rng)
    part_b(recs, rng, npw)
