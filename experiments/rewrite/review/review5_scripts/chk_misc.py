import json, gzip, os
from eng import TRAIN, HELD_B, DATA
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
def rd(tag, w):
    with gzip.open(os.path.join(DATA, f"{tag}_{w}.jsonl.gz"), "rt") as f:
        return [json.loads(l) for l in f]
lad = lambda rs, d: sum(bool(r["ladder"][str(d)]) for r in rs) / len(rs)
E5 = {w: rd("eval5", w) for w in TRAIN + HELD_B + ["ACID"]}
tb = [r for w in TRAIN for r in E5[w] if r["binding"]]
hb = [r for w in HELD_B + ["ACID"] for r in E5[w] if r["binding"]]
print("training binding n", len(tb), "d1 %.3f d2 %.3f d3 %.3f" % (lad(tb, 1), lad(tb, 2), lad(tb, 3)))
print("held-out binding n", len(hb), "d1 %.3f d3 %.3f" % (lad(hb, 1), lad(hb, 3)), "ACID binding", sum(r["binding"] for r in E5["ACID"]))
pb = [r for w in ["AI", "CI"] for r in E5[w] if r["binding"]]; trb = [r for w in ["ACI", "ACD", "AID", "CID"] for r in E5[w] if r["binding"]]
print("pairs all binding n", len(pb), "d1 %.3f d3 %.3f" % (lad(pb, 1), lad(pb, 3)), "| triples n", len(trb), "d1 %.3f d3 %.3f" % (lad(trb, 1), lad(trb, 3)))
print("ACI eval7 binding:", sum(bool(r["binding"]) for r in rd("eval7", "ACI")), "/ 600")
exc = D["examples_ceiling"]
print("P7 ceiling (training binding, weighted): %.4f on n %d" % (sum(exc[w]["ceiling_canon_binding"] * exc[w]["binding_n"] for w in TRAIN) / sum(exc[w]["binding_n"] for w in TRAIN), sum(exc[w]["binding_n"] for w in TRAIN)))
# P6 references on the multi-class training instances with more than one shortest-path first action
n = 0; tvu = tvc = 0.0; uni = mode = 0; perw = {}
for w in TRAIN:
    for r in E5[w]:
        tg = next(p["target"] for p in r["post"] if p["state"] == r["t"])
        if r["n_trace_classes"] > 1 and len(tg) > 1:
            n += 1; perw[w] = perw.get(w, 0) + 1
            u = 1 / len(tg)
            tvu += 0.5 * sum(abs(u - p) for p in tg.values())
            c = r["canon"][0]
            tvc += 0.5 * (abs(1 - tg.get(c, 0)) + sum(p for a, p in tg.items() if a != c))
            uni += all(abs(p - u) < 1e-12 for p in tg.values())
            mode += tg.get(c, 0) >= max(tg.values()) - 1e-12
print("P6 instances", n, perw, "uniform-over-shortest-actions TV %.3f; canonical one-hot TV %.3f; target exactly uniform %.2f; canonical a mode %.2f" % (tvu / n, tvc / n, uni / n, mode / n))
# P7 'above the band' reachability: x > 0.954 m + 0.1 needs m < (1 - 0.1) / 0.954
print("P7 clause 1 'above' unreachable once RT-canon's unmasked median >=", round(0.9 / 0.954, 4))
# duplicates of (t, s) within a world (keys of the derived flags)
for tag in ("eval5", "eval7"):
    dup = {w: len(rs) - len({(r["t"], r["s"]) for r in rs}) for w, rs in ((w, rd(tag, w)) for w in TRAIN + HELD_B + ["ACID"])}
    print(tag, "duplicate (t,s) per world:", {w: d for w, d in dup.items() if d})
