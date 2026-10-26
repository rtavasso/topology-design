"""Per-arm savings vs base (t_base - t_arm) per cell, and E2 eval grid check. usage: python q2_arms.py ROOT"""
import sys, gzip, json, statistics as stt
sys.argv = sys.argv[:2]
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import q2_verify as q
for k in sorted(q.arms):
    out = []
    for d in ("drift", "radial", "random", "rescale"):
        sv = []
        for s, a in q.arms[k].items():
            tb = q.sustained(q.base[s]["trajectory"])
            if tb <= k[0]:
                continue
            ta = q.sustained(a[d]["trajectory"]); ta = 15000 if ta is None else ta
            sv.append(tb - ta)
        out.append(f"{d}: med {stt.median(sv)} [{min(sv)},{max(sv)}]")
    jm = {d: q.arms[k][610][d]["summary"]["jump"] for d in ("drift", "radial", "random", "rescale")}
    print(k, " | ".join(out))
    print("   jump info s610:", {d: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in j.items() if kk not in ("step", "h", "dir")} for d, j in jm.items()})
e2 = json.load(gzip.open(f"{q.ROOT}/experiments/walk/results_data/e2_runs.json.gz"))
r = [r for r in e2 if r["config"]["tag"] == "wE2-sym"][0]
st = [x["step"] for x in r["trajectory"]]
print("E2 eval steps sample:", st[:15], "...", st[-5:], "n", len(st), "eval_every", r["config"]["eval_every"], r["config"].get("dense_until"))
# seed 613 drift arm dip that E2 missed
a = q.arms[(1000, 250)][613]["drift"]["trajectory"]
print("s613 drift(1000,250) evals 1900-3000 with acc<0.95:", [(x["step"], round(x["test_acc"], 3)) for x in a if 1900 <= x["step"] <= 3000 and x["test_acc"] < 0.95][:20])
