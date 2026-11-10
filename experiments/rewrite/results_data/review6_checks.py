"""The author's own recount of review 6's main new numbers before acting on them (not the reviewer's code).
Usage: PYTHONPATH=src python3 experiments/rewrite/results_data/review6_checks.py <results dir> <data dir> <derived.json>"""
import gzip, json, sys
sys.path.insert(0, "experiments/rewrite"); sys.path.insert(0, "src")
import analyze as A

res, data, der = sys.argv[1], sys.argv[2], sys.argv[3]
runs = A.load_results(res)
rc, rsh = A.by_seed(runs, "RT-canon"), A.by_seed(runs, "RT-shuffled")

# (1) eval5 training-world instances that are verbatim training instances
seen = set()
for W in A.TRAIN_WORLDS:
    want = {(e["t"], e["s"]) for e in map(json.loads, gzip.open(f"{data}/eval5_{W}.jsonl.gz", "rt"))}
    with gzip.open(f"{data}/train_{W}.jsonl.gz", "rt") as f:
        for l in f:
            r = json.loads(l)
            if (r["t"], r["s"]) in want:
                seen.add((W, r["t"], r["s"]))
is_seen = lambda p: (p["world"], p["t"], p["s"]) in seen
rows = A.instances(rc[0], "eval5", world=A.TRAIN_WORLDS)
print(f"(1) eval5 training-world instances that are training instances: {sum(map(is_seen, rows))} of {len(rows)}")
for k in (None, 4, 5, 6):
    sel = lambda r, s_: [p for p in A.instances(r, "eval5", world=A.TRAIN_WORLDS) if (k is None or p["k"] == k) and is_seen(p) == s_]
    print(f"    k={k or 'all'}: seen n={len(sel(rc[0], True))} RT-canon {A.summarize_seeds([A.rate(sel(r, True)) for r in rc])} | unseen n={len(sel(rc[0], False))} {A.summarize_seeds([A.rate(sel(r, False)) for r in rc])}")

# (2) P4 on the §16 cells without the repeated pairs
cells = {tuple(c) if isinstance(c, list) else c for c in json.load(open(der)).get("p4_cells16", [])} if False else None
def cellkey(p): return (p["world"], p["binding"], p["GP1"], p["k"])
allr = A.instances(rc[0], "eval5", world=A.TRAIN_WORLDS)
by = {}
for p in allr:
    if p["k"] >= 4: by.setdefault(cellkey(p), []).append(p)
sep = [c for c, v in by.items() if len(v) >= 10 and A.ladder_rate(v, 2) - A.ladder_rate(v, 1) >= 0.1 - 1e-9 and A.ladder_rate(v, 3) - A.ladder_rate(v, 2) >= 0.1 - 1e-9]
in16 = lambda p: p["k"] >= 4 and cellkey(p) in sep
s16 = lambda r, s_: [p for p in A.instances(r, "eval5", world=A.TRAIN_WORLDS) if in16(p) and (s_ is None or is_seen(p) == s_)]
u = s16(rc[0], False)
vals = [A.rate(s16(r, False)) for r in rc]
print(f"(2) §16 cells: {len(sep)} cells, n={len(s16(rc[0], None))}; unseen n={len(u)}: RT-canon {A.summarize_seeds(vals)}; rungs {A.ladder_rate(u, 1):.2f}/{A.ladder_rate(u, 2):.2f}/{A.ladder_rate(u, 3):.2f}; >= d3 in {sum(v >= A.ladder_rate(u, 3) for v in vals)}/10")

# (3) RT-shuffled against RT-canon on held-out worlds
HELD = A.PAIRS_HELD + A.TRIPLES + A.FULL
for nm, kw in (("held-out, all", dict(world=HELD)), ("full world", dict(world=A.FULL)), ("triples non-binding", dict(world=A.TRIPLES, binding=False))):
    for key in ("ok_m", "ok_u"):
        c = [A.rate(A.instances(r, "eval5", **kw), key) for r in rc]; s = [A.rate(A.instances(r, "eval5", **kw), key) for r in rsh]
        print(f"(3) {nm} [{key}] n={len(A.instances(rc[0], 'eval5', **kw))}: RT-canon {A.summarize_seeds(c)}; RT-shuffled {s[0]:.2f} {s[1]:.2f}; RT-canon seeds below both: {sum(x < min(s) for x in c)}/10")

# (4) one-step instances at sizes 6–7 with no node past depth 4 and no growth
def depth(t):
    d = m = 0
    for ch in t:
        if ch == "(": d += 1; m = max(m, d)
        elif ch == ")": d -= 1
    return m
leaves = lambda t: sum(ch in "abcd0" for ch in t)
def sub(r, grow, deep):
    out = []
    for p in A.instances(r, "eval7", world=A.TRAIN_WORLDS, k=1):
        g = leaves(p["s"]) > leaves(p["t"]); d = max(depth(p["t"]), depth(p["s"])) > 4
        if g == grow and d == deep: out.append(p)
    return out
for grow in (False, True):
    for deep in (False, True):
        rows = sub(rc[0], grow, deep)
        wh = sum(sum(p["cat_m"] == "wrong halt" for p in sub(r, grow, deep)) for r in rc) / max(1, len(rows) * len(rc))
        print(f"(4) eval7 k=1, growth={grow}, some node deeper than 4={deep}: n={len(rows)} RT-canon {A.summarize_seeds([A.rate(sub(r, grow, deep)) for r in rc])} wrong halt {wh:.2f}")
