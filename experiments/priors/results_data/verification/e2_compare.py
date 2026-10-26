"""E2 drift-jump savings (500-step eval grid after step 1000) vs Q2 drift-arm savings (25-step grid). usage: python e2_compare.py ROOT"""
import sys, gzip, json, statistics as stt
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import q2_verify as q
e2 = json.load(gzip.open(f"{q.ROOT}/experiments/walk/results_data/e2_runs.json.gz"))
b = {r["config"]["model_seed"]: q.sustained(r["trajectory"]) for r in e2 if r["config"]["tag"] == "wE2-sym"}
st = [x["step"] for x in [r for r in e2 if r["config"]["tag"] == "wE2-sym"][0]["trajectory"]]
print("E2 eval steps between 1000 and 6000:", [s for s in st if 1000 <= s <= 6000])
J = defaultdict(dict)
for r in e2:
    c = r["config"]
    if c["tag"] == "wE2jump-sym":
        J[(c["drift_jump_at"], c["drift_jump_h"])][c["model_seed"]] = q.sustained(r["trajectory"])
for k in sorted(J):
    sv = [b[s] - J[k][s] for s in sorted(J[k]) if b[s] > k[0]]
    sv2 = []
    for s in sorted(q.arms[k]):
        tb = q.sustained(q.base[s]["trajectory"])
        if tb > k[0]:
            sv2.append(tb - q.sustained(q.arms[k][s]["drift"]["trajectory"]))
    print(k, "E2 savings", sv, "median", stt.median(sv), "| Q2 drift savings median", stt.median(sv2))
