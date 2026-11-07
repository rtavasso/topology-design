import gzip, json, os, collections
D = "data/rewrite"
TRAIN = ["A","C","I","D","AC","AD","CD","ID"]
chosen = {('AC',1,0,4),('AD',1,0,4),('CD',1,0,5),('ID',0,0,4),('ID',1,1,4),('ID',1,0,5)}
pooled = []
for w in TRAIN:
    with gzip.open(os.path.join(D, f"eval5_{w}.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            key = (w, int(bool(r["binding"])), int(bool(r["first"]["GP1"])), r["k"])
            if key in chosen: pooled.append(r)
print("n", len(pooled))
print("record keys:", sorted(pooled[0].keys()))
print("first keys:", sorted(pooled[0]["first"].keys()))
sec = [k for k in pooled[0].keys() if "secondary" in k or "canon" in k.lower() or "gp" in k.lower() or "max" in k.lower()]
print("secondary-like keys:", sec)
# monotonicity of rungs at instance level
pat = collections.Counter((r["ladder"]["1"], r["ladder"]["2"], r["ladder"]["3"]) for r in pooled)
print("ladder patterns (d1,d2,d3):", dict(pat))
fail1 = [r for r in pooled if not r["ladder"]["1"]]
print("greedy fails:", len(fail1), "of which GP1:", sum(bool(r["first"]["GP1"]) for r in fail1), "non-GP1 share of greedy failures:", 1 - sum(bool(r['first']['GP1']) for r in fail1)/len(fail1))
# secondary strata, if present
for key in ("secondary",):
    if key in pooled[0]:
        print(key, pooled[0][key])
        gp_any = sum(bool(r[key].get("gp_any", r[key].get("GP_any", False))) for r in pooled)
        print("GP anywhere along canonical proof:", gp_any, "/", len(pooled))
        print("Dany_max distribution:", collections.Counter(r[key].get("dany_max", r[key].get("Dany_max")) for r in pooled))
        print("Dany_min_over_proofs distribution:", collections.Counter(r[key].get("dany_min_max", r[key].get("Dany_minmax")) for r in pooled))
# first-decision Dany / Dsep distributions
print("first Dany:", collections.Counter(r["first"]["Dany"] for r in pooled))
print("first Dsep:", collections.Counter(r["first"]["Dsep"] for r in pooled))
print("B1 (first move looks worse by proxy, non-strict?):", collections.Counter(r["first"]["B1"] for r in pooled))
print("tie1:", collections.Counter(r["first"]["tie1"] for r in pooled))
# size of t among pooled
print("size_t:", collections.Counter(r["size_t"] for r in pooled))
# per cell rungs and n
cells = collections.defaultdict(list)
for r in pooled: cells[(r["world"], int(bool(r["binding"])), int(bool(r["first"]["GP1"])), r["k"])].append(r)
for key, rs in sorted(cells.items()):
    n=len(rs); print(key, n, [round(sum(r["ladder"][str(d)] for r in rs)/n,3) for d in (1,2,3)], "d2ok&d3fail:", sum(r["ladder"]["2"] and not r["ladder"]["3"] for r in rs))
