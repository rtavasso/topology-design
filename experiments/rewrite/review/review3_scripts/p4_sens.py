import gzip, json, os, collections
D = "data/rewrite"
TRAIN = ["A","C","I","D","AC","AD","CD","ID"]
recs = []
for w in TRAIN:
    with gzip.open(os.path.join(D, f"eval5_{w}.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["k"] >= 4: recs.append(r)
def rungs(rs): n=len(rs); return tuple(sum(r["ladder"][str(d)] for r in rs)/n for d in (1,2,3))
def run(name, keyf, rule):
    cells = collections.defaultdict(list)
    for r in recs: cells[keyf(r)].append(r)
    big = {k:v for k,v in cells.items() if len(v) >= 10}
    chosen = [k for k,v in big.items() if rule(rungs(v))]
    pooled = [r for k in chosen for r in big[k]]
    d = rungs(pooled) if pooled else (None,)*3
    print(f"{name}: {len(chosen)}/{len(big)} cells, n={len(pooled)}, rungs={[round(x,3) if x is not None else None for x in d]}, window={round(d[2]-d[1],3) if pooled else None}")
    return chosen
std = lambda d: d[1]-d[0] >= 0.1 and d[2]-d[1] >= 0.1
k4 = lambda r: (r["world"], int(bool(r["binding"])), int(bool(r["first"]["GP1"])), r["k"])
run("§16 (0.1/0.1)", k4, std)
run("upper only (d3-d2>=0.1)", k4, lambda d: d[2]-d[1] >= 0.1)
run("0.05/0.1", k4, lambda d: d[1]-d[0] >= 0.05 and d[2]-d[1] >= 0.1)
run("0.05/0.05", k4, lambda d: d[1]-d[0] >= 0.05 and d[2]-d[1] >= 0.05)
run("GP-anywhere partition (world,binding,canon_GP1_any,k)", lambda r: (r["world"], int(bool(r["binding"])), int(bool(r["secondary"]["canon_GP1_any"])), r["k"]), std)
ch = run("GP-anywhere cells only, std rule", lambda r: (r["world"], int(bool(r["binding"])), int(bool(r["secondary"]["canon_GP1_any"])), r["k"]) if r["secondary"]["canon_GP1_any"] else ("x",), std)
print(ch)
run("no-world partition (binding,GP1,k)", lambda r: (int(bool(r["binding"])), int(bool(r["first"]["GP1"])), r["k"]), std)
run("(world,k) all strata", lambda r: (r["world"], r["k"]), std)
run("(world,binding,k)", lambda r: (r["world"], int(bool(r["binding"])), r["k"]), std)
