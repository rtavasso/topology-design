import json
from common import *
res = json.load(open("s2_out.json"))
def rt(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")
print("world | n | ceiling | FW unmasked | FW masked(strict) | FW masked | MINW (token-blind, unmasked) | bind n | FW_m bind | MINW bind | d1 bind | d3 bind")
for w in ALL:
    rows = res[w]; b = [r for r in rows if r["binding"]]
    c = DER["cue_blind"][w]["ceiling"]
    print(f"{w:5s} {len(rows)} ceil={c:.3f} fw_u={rt(rows,'fw_u'):.3f} fw_ms={rt(rows,'fw_ms'):.3f} fw_m={rt(rows,'fw_m'):.3f} minw={rt(rows,'minw'):.3f} | bind n={len(b)} fw_u={rt(b,'fw_u'):.3f} fw_m={rt(b,'fw_m'):.3f} minw={rt(b,'minw'):.3f} d1={rt(b,'d1'):.3f} d3={rt(b,'d3'):.3f}"
          f"  | exceed(ceil+.05): fw_m {'Y' if rt(rows,'fw_m')>c+0.05 else 'n'} minw {'Y' if rt(rows,'minw')>c+0.05 else 'n'}")
def pool(ws, f=lambda r: True):
    return [r for w in ws for r in res[w] if f(r)]
for nm, ws in (("pairs", PAIRS), ("triples", TRIPLES), ("train", TRAIN_WORLDS), ("held-out", PAIRS + TRIPLES)):
    b = pool(ws, lambda r: r["binding"]); nb = pool(ws, lambda r: not r["binding"])
    print(f"{nm} binding n={len(b)}: FW_u={rt(b,'fw_u'):.3f} FW_m={rt(b,'fw_m'):.3f} MINW={rt(b,'minw'):.3f} d1={rt(b,'d1'):.3f} d3={rt(b,'d3'):.3f} || non-binding n={len(nb)}: FW_u={rt(nb,'fw_u'):.3f} FW_ms={rt(nb,'fw_ms'):.3f} FW_m={rt(nb,'fw_m'):.3f} MINW={rt(nb,'minw'):.3f}")
# MINW: which worlds chosen when failing
from collections import Counter
for w in ["AC", "AD", "CD", "ID", "AI", "CI", "A", "D"]:
    print(w, "MINW chosen worlds:", Counter(r["minw_world"] for r in res[w]).most_common(6))
