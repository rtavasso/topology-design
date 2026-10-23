"""Independent recomputation of E1 from raw run files (does not use experiments/walk/analyze.py)."""
import glob, gzip, json, math, sys, dataclasses
from collections import defaultdict
import numpy as np

REPO = "/home/claude/topology-design"
sys.path.insert(0, REPO + "/src")
from topology_design import fleet, program_prior as pp

runs = [json.load(gzip.open(f)) for f in sorted(glob.glob(REPO + "/results/kaggle_e1/results/*.json.gz"))]
print("n runs", len(runs))
streams = [r for r in runs if r["config"]["tag"].startswith("wE1-")]
floors = [r for r in runs if r["config"]["tag"].startswith("wE1floor-")]
print("streams", len(streams), "floors", len(floors))

P = 59
N_TRAIN = 1392
UNIFORM = N_TRAIN * math.log(P)


def mkcfg(c):
    fields = {f.name for f in dataclasses.fields(fleet.FleetConfig)}
    kw = {k: v for k, v in c.items() if k in fields}
    for k, v in kw.items():
        if isinstance(v, list):
            kw[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return fleet.FleetConfig(**kw)


# ---------------- streams: check structure ----------------
cells = defaultdict(list)
for r in streams:
    c = r["config"]
    graph = c["tag"].split("-")[1]
    b = c["arrive_every"]
    assert c["tag"] == f"wE1-{graph}-b{b}", c["tag"]
    assert c["arm"] == {"sym": "pk000000", "asym": "pk012000"}[graph]
    pq = r["summary"]["prequential"]
    n = pq["n"]
    assert len(n) == 41 and n[0] == 140 and all(x == 32 for x in n[1:40]) and n[40] == 4, n
    assert sum(n) == N_TRAIN
    assert c["steps"] == 40 * b, (c["steps"], b)
    cells[(graph, c["lr"], b)].append(r)
print("cells", {k: len(v) for k, v in sorted(cells.items())})

# ---------------- program prior on the same arrival orders ----------------
T, logm, exprs = pp.enumerate_functions(P, max_size=7)
print("program class size", len(T))
a_grid = np.repeat(np.arange(P), P)
b_grid = np.tile(np.arange(P), P)
true_fn = (a_grid + b_grid) % P

pp_cache = {}


def prior_for(cfg):
    key = cfg.data_seed
    if key in pp_cache:
        return pp_cache[key]
    task = fleet.build_task(cfg)
    tr = task["tr"].numpy()
    y = task["y"].numpy()
    assert len(tr) == N_TRAIN
    assert (y == true_fn).all()
    arr, inc, K = fleet._arrivals([cfg], len(tr), "cpu")
    inc = inc[0].numpy()
    assert K == 40
    xb = [tr[inc == k] for k in range(K + 1)]
    yb = [y[x] for x in xb]
    res = pp.prequential(T, logm, xb, yb, P, eps=1e-3, mu=0.5, true_fn=true_fn)
    # k*: one at a time in arrival order (rank order)
    import torch
    g = torch.Generator().manual_seed(40_000 + cfg.data_seed)
    order = torch.randperm(len(tr), generator=g).numpy()
    seq = tr[order]
    # check order consistent with inc
    rank = np.empty(len(tr), int); rank[order] = np.arange(len(tr))
    assert (inc == np.where(rank < 140, 0, (rank - 140) // 32 + 1)).all()
    one = pp.prequential(T, logm, [seq[i:i + 1] for i in range(40)], [y[seq[i:i + 1]] for i in range(40)], P,
                         eps=1e-3, mu=0.5, true_fn=true_fn)
    kstar = next(i + 1 for i, v in enumerate(one["post_true"]) if v >= 0.99)
    out = {"nats": res["nats"], "total": sum(res["nats"]), "kstar": kstar, "post1": one["post_true"][:6]}
    pp_cache[key] = out
    return out


print("\n=== program prior per data seed ===")
for r in streams:
    prior_for(mkcfg(r["config"]))
for s, v in sorted(pp_cache.items()):
    print(f"seed {s}: total {v['total']:.2f} nats, k*={v['kstar']}, post_true first {np.round(v['post1'], 4)}")
print(f"uniform {UNIFORM:.1f}")

# ---------------- GD code lengths ----------------
print("\n=== GD code length per cell ===")
ratios = []
for k in sorted(cells):
    tots = []
    for r in sorted(cells[k], key=lambda r: r["config"]["model_seed"]):
        c = r["config"]
        tot = sum(r["summary"]["prequential"]["nats"])
        ref = prior_for(mkcfg(c))["total"]
        ratios.append(tot / ref)
        tots.append(tot)
    print(k, "seeds", [round(t) for t in tots], "median", round(float(np.median(tots)), 1))
print(f"ratio GD/prior range {min(ratios):.2f} - {max(ratios):.2f}")

# ---------------- production point ----------------
print("\n=== production point ===")
for k in sorted(cells):
    pts = []
    for r in sorted(cells[k], key=lambda r: r["config"]["model_seed"]):
        acc = r["summary"]["prequential"]["acc"]
        n = r["summary"]["prequential"]["n"]
        pt = None
        for i in range(1, len(acc)):
            if acc[i] >= 0.9:
                pt = int(sum(n[:i]))
                break
        pts.append(pt if pt is not None else N_TRAIN)
        cens = [p is None for p in [pt]]
    ncens = sum(1 for p in pts if p == N_TRAIN)
    print(k, pts, "median", float(np.median(pts)), f"censored {ncens}/5")
# sensitivity: any increment with acc exactly 0.9 or last (n=4) increment
print("acc==0.9 exactly anywhere?", any(abs(a - 0.9) < 1e-9 for r in streams for a in r["summary"]["prequential"]["acc"]))
print("last-increment (n=4) first-hit cases:")
for r in streams:
    acc = r["summary"]["prequential"]["acc"]
    first = next((i for i in range(1, 41) if acc[i] >= 0.9), None)
    if first == 40:
        print("  ", r["config"]["tag"], r["config"]["lr"], r["config"]["model_seed"])
# check the first-hit is sustained or not (ever drops after)
print("\nruns where acc drops below 0.9 after first hit:")
for r in streams:
    acc = r["summary"]["prequential"]["acc"]
    first = next((i for i in range(1, 41) if acc[i] >= 0.9), None)
    if first is not None and any(a < 0.9 for a in acc[first + 1:]):
        print("  ", r["config"]["tag"], r["config"]["lr"], r["config"]["model_seed"], "first", first,
              [round(a, 2) for a in acc[first:]])

# ---------------- data floor ----------------
print("\n=== data floor ===")
fl = defaultdict(list)
for r in floors:
    c = r["config"]
    graph = c["tag"].split("-")[1]
    fl[(graph, c["lr"], c["train_frac"])].append(r)
for k in sorted(fl):
    rows = []
    ok = 0
    for r in sorted(fl[k], key=lambda r: r["config"]["model_seed"]):
        s = r["summary"]
        t = r["trajectory"]
        # independent t_grok: first eval with test_acc >= 0.95
        tg = next((e["step"] for e in t if e["test_acc"] >= 0.95), None)
        fin = t[-1]["test_acc"]
        g = (tg is not None) and fin >= 0.9
        g_sum = s["t_grok"] is not None and s["final_test_acc"] >= 0.9
        ok += g
        rows.append((r["config"]["model_seed"], tg, s["t_grok"], round(fin, 3), round(s["final_test_acc"], 3), g, g_sum,
                     s["steps_run"]))
    print(k, "groks" if ok >= 2 else "no", f"{ok}/3", rows)
