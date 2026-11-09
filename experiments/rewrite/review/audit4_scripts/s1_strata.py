from common import *
from collections import Counter
e5 = load("eval5"); e7 = load("eval7"); k78 = load("evalk78", ["AC"])
def show(name, rs):
    ks = Counter(r["k"] for r in rs)
    print(f"{name}: n={len(rs)} d1={lad(rs,1):.3f} d2={lad(rs,2):.3f} d3={lad(rs,3):.3f} GP1={sum(r['first']['GP1'] for r in rs)/max(1,len(rs)):.2f} k={dict(sorted(ks.items()))}")
print("== P2 strata (eval5 binding)")
for w in PAIRS + TRIPLES + ["ACID"]:
    show(w + " binding", [r for r in e5[w] if r["binding"]])
pb = [r for w in PAIRS for r in e5[w] if r["binding"]]
tb = [r for w in TRIPLES + ["ACID"] for r in e5[w] if r["binding"]]
show("pairs binding pooled", pb); show("triples+full binding pooled", tb); show("held-out binding pooled (P5)", pb + tb)
for nm, rs in (("pairs", pb), ("triples", tb)):
    for lo, hi in ((1, 3), (4, 6)):
        sub = [r for r in rs if lo <= r["k"] <= hi]
        show(f"  {nm} binding k{lo}-{hi}", sub)
    for g in (0, 1):
        show(f"  {nm} binding GP1={g}", [r for r in rs if int(r['first']['GP1']) == g])
print("== training-world binding (P7 clause 1)")
trb = [r for w in TRAIN_WORLDS for r in e5[w] if r["binding"]]
for w in TRAIN_WORLDS:
    show(w + " binding", [r for r in e5[w] if r["binding"]])
show("training binding pooled", trb)
exc = DER["examples_ceiling"]
ws = [w for w in TRAIN_WORLDS if exc[w]["binding_n"] > 0]
print("weighted examples ceiling, training binding:", sum(exc[w]["ceiling_canon_binding"] * exc[w]["binding_n"] for w in ws) / sum(exc[w]["binding_n"] for w in ws))
for cls, wl in (("pair", PAIRS), ("triple+full", TRIPLES + ["ACID"])):
    ws = [w for w in wl if exc[w]["binding_n"] > 0]
    print("weighted examples ceiling", cls, sum(exc[w]["ceiling_canon_binding"] * exc[w]["binding_n"] for w in ws) / sum(exc[w]["binding_n"] for w in ws))
print("== per world eval5 all: ladder, cue-blind ceiling")
for w in ALL:
    rs = e5[w]
    nb = [r for r in rs if not r["binding"]]
    print(f"{w}: n={len(rs)} ceil={DER['cue_blind'][w]['ceiling']:.3f} d1={lad(rs,1):.3f} d2={lad(rs,2):.3f} d3={lad(rs,3):.3f} | non-binding n={len(nb)} d1={lad(nb,1):.3f} d3={lad(nb,3):.3f}")
print("== evalk78")
show("AC k78", k78["AC"])
print("== eval7 binding per world")
for w in ALL:
    show(w + " e7 binding", [r for r in e7[w] if r["binding"]])
