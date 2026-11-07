import gzip, json, os, collections
D = "data/rewrite"
TRAIN = ["A","C","I","D","AC","AD","CD","ID"]
cells = collections.defaultdict(list)
prereg = collections.defaultdict(list)
allk4 = []
for w in TRAIN:
    with gzip.open(os.path.join(D, f"eval5_{w}.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["world"] != w: print("world mismatch", w, r["world"])
            k = r["k"]
            if k < 4: continue
            b = int(bool(r["binding"])); g = int(bool(r["first"]["GP1"]))
            cells[(w,b,g,k)].append(r)
            if g: prereg[(w,k)].append(r)
            allk4.append(r)
def rungs(rs):
    n = len(rs)
    return tuple(sum(r["ladder"][str(d)] for r in rs)/n for d in (1,2,3))
def sep(d): return d[1]-d[0] >= 0.1 and d[2]-d[1] >= 0.1
print("k values seen:", sorted({r["k"] for r in allk4}))
print("--- §6 cells (GP1, by world,k), n>=10 ---")
n6 = 0
for key in sorted(prereg):
    rs = prereg[key]
    if len(rs) < 10: continue
    n6 += 1
    d = rungs(rs); print(key, len(rs), [f"{x:.2f}" for x in d], "SEP" if sep(d) else "")
print("§6 cells with n>=10:", n6)
gp1 = [r for r in allk4 if r["first"]["GP1"]]
print("pooled GP1 k>=4: n", len(gp1), [f"{x:.3f}" for x in rungs(gp1)])
print("--- §16 cells (world,binding,GP1,k), n>=10 ---")
n16 = 0; chosen = []
for key in sorted(cells, key=lambda t:(TRAIN.index(t[0]),t[3],t[1],t[2])):
    rs = cells[key]
    if len(rs) < 10: continue
    n16 += 1
    d = rungs(rs)
    s = sep(d)
    if s: chosen.append(key)
    print(key, len(rs), [f"{x:.3f}" for x in d], "SEP" if s else "")
print("§16 cells with n>=10:", n16, "separated:", len(chosen), chosen)
pooled = [r for key in chosen for r in cells[key]]
print("pooled n", len(pooled), [f"{x:.3f}" for x in rungs(pooled)])
print("binding share", sum(bool(r["binding"]) for r in pooled)/len(pooled), "GP1 share", sum(bool(r["first"]["GP1"]) for r in pooled)/len(pooled))
# how many cells near the threshold (fragility)
print("--- cells near the threshold (|gap-0.1| < 0.05 on either side) ---")
for key in sorted(cells, key=lambda t:(TRAIN.index(t[0]),t[3],t[1],t[2])):
    rs = cells[key]
    if len(rs) < 10: continue
    d = rungs(rs)
    g1, g2 = d[1]-d[0], d[2]-d[1]
    if abs(g1-0.1) < 0.05 or abs(g2-0.1) < 0.05:
        print(key, len(rs), f"gap12={g1:.3f} gap23={g2:.3f}", "SEP" if sep(d) else "")
# per-seed strict-between window: 0.72-0.45
print("window", rungs(pooled)[2]-rungs(pooled)[1])
# per-cell sizes and k distribution among chosen
print({key: len(cells[key]) for key in chosen})
# what about the k>=4 cells with n 5..9 that would separate?
print("--- excluded cells n<10 that would separate ---")
for key in sorted(cells, key=lambda t:(TRAIN.index(t[0]),t[3],t[1],t[2])):
    rs = cells[key]
    if len(rs) >= 10 or len(rs) < 1: continue
    d = rungs(rs)
    if sep(d): print(key, len(rs), [f"{x:.2f}" for x in d])
