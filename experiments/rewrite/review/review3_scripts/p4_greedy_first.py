import gzip, json, os, collections, sys
sys.path.insert(0, "src")
from topology_design.rewrite.search import bfs, reverse_distances, world
from topology_design.rewrite.proxy import ladder_policy, ladder_rollout
from topology_design.rewrite.term import parse
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
stats = collections.Counter(); mism = 0
first_wrong_by_d = collections.Counter()
for r in pooled:
    t, s = parse(r["t"]), parse(r["s"]); W = world(r["world"]); cap = r["cap"]; k = r["k"]
    dist, succ, complete = bfs(t, W, cap, radius=k)
    dgoal = reverse_distances(succ, s)
    assert dgoal[t] == k
    for d in (1,2,3):
        a = ladder_policy(t, s, W, cap, d)
        first_ok = a is not None and dgoal.get(a[2]) == k - 1
        ok = ladder_rollout(t, s, W, cap, d, 2*k+2)["ok"]
        if ok != r["ladder"][str(d)]: mism += 1
        if not ok:
            stats[(d, "first_wrong" if not first_ok else "later")] += 1
print("stored-vs-recomputed ladder mismatches:", mism)
for d in (1,2,3):
    fw, lt = stats[(d,"first_wrong")], stats[(d,"later")]
    print(f"d={d}: failures {fw+lt}; first decision wrong {fw} ({fw/max(1,fw+lt):.2f}); failure later {lt} ({lt/max(1,fw+lt):.2f}); share of all 170 with later failure {lt/170:.2f}")
# secondary strata on the pooled set
sec = collections.Counter((r["secondary"]["canon_GP1_any"]) for r in pooled)
print("canon_GP1_any:", dict(sec))
print("canon_Dany_max:", dict(collections.Counter(r["secondary"]["canon_Dany_max"] for r in pooled)))
print("min_over_proofs_Dany_max:", dict(collections.Counter(r["secondary"]["min_over_proofs_Dany_max"] for r in pooled)))
