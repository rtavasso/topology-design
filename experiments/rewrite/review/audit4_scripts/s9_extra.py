import json
from common import *
from s2_cueblind import dist_from
from s6_trainw import pol
from topology_design.rewrite.rules import world_from_name
from topology_design.rewrite.term import parse
s5 = json.load(open("s5_out.json")); s2 = json.load(open("s2_out.json"))
e5 = load("eval5"); e7 = load("eval7")
# (a) composition-needing sub-stratum: token-blind policies
for nm, ws in (("pairs", PAIRS), ("triples", TRIPLES)):
    acc = {"n": 0, "trainw_mask": 0, "minw_mask": 0, "fw_mask": 0, "minw_unmasked": 0, "d1": 0, "d2": 0, "d3": 0}
    per = {}
    for w in ws:
        W = world_from_name(w)
        b = [r for r in e5[w] if r["binding"]]
        b2 = [x for x, r in zip(s2[w], e5[w]) if r["binding"]]
        subs = [U for U in s5[w][0]["dU"] if U != w and U in TRAIN_WORLDS]
        m = 0
        for r, x, y in zip(b, s5[w], b2):
            if any(x["dU"][U] is not None for U in subs): continue
            t, s, cap, T = parse(r["t"]), parse(r["s"]), r["cap"], 2 * r["k"] + 2
            D = {cw: dist_from(s, cap, world_from_name(cw)) for cw in TRAIN_WORLDS}
            acc["n"] += 1; m += 1
            acc["trainw_mask"] += pol(t, s, W, cap, T, D); acc["minw_mask"] += x["minw_mask"]; acc["fw_mask"] += x["fw_m"]; acc["minw_unmasked"] += y["minw"]
            for d in (1, 2, 3): acc[f"d{d}"] += bool(r["ladder"][str(d)])
        per[w] = m
    n = acc.pop("n")
    print(nm, "needs-combination binding n =", n, per, {k: round(v / n, 3) for k, v in acc.items()})
# (b) ladder matched ratio with GP1 added to the matching key
def matched(worlds, d, gp):
    c7, c5 = {}, {}
    key = (lambda w, r: (w, r["k"], bool(r["binding"]), bool(r["first"]["GP1"]))) if gp else (lambda w, r: (w, r["k"], bool(r["binding"])))
    for w in worlds:
        for r in e7[w]: c7.setdefault(key(w, r), []).append(r)
        for r in e5[w]:
            if r["k"] <= 4: c5.setdefault(key(w, r), []).append(r)
    num = den = used = 0
    for k_, rows in c7.items():
        if k_ not in c5: continue
        num += sum(bool(r["ladder"][str(d)]) for r in rows); den += len(rows) * sum(bool(r["ladder"][str(d)]) for r in c5[k_]) / len(c5[k_]); used += len(rows)
    return round(num / den, 3), used
for nm, ws in (("train", TRAIN_WORLDS), ("held-out", PAIRS + TRIPLES + ["ACID"])):
    print(nm, "ladder matched ratio (world,k,binding):", [matched(ws, d, False) for d in (1, 2, 3)], " with GP1 in the key:", [matched(ws, d, True) for d in (1, 2, 3)])
# eval7 ACI binding count vs DESIGN §16 "0 / 600"
print("eval7 ACI binding instances:", sum(bool(r["binding"]) for r in e7["ACI"]), [(r["t"], r["s"], r["k"], r["d_full"]) for r in e7["ACI"] if r["binding"]])
print("eval5 ACI binding with size_t<=2:", sum(1 for r in e5["ACI"] if r["binding"] and r["size_t"] <= 2), "of", sum(1 for r in e5["ACI"] if r["binding"]))
