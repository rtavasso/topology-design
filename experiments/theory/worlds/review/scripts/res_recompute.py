"""Recompute every number in RESULTS.md from the per-run JSON. Read-only."""
import json, os, random, statistics, sys
W = "/home/claude/topology-design/experiments/theory/worlds"
sys.path.insert(0, W)
from stage0 import regime
med = statistics.median
def L(task, cover, arm):
    f = f"{W}/stage2_{task}_{cover}_{arm}.json"
    return json.load(open(f)) if os.path.exists(f) else None
def boot(v, n=10000, seed=0):
    rnd = random.Random(seed)
    ms = sorted(med([v[rnd.randrange(len(v))] for _ in v]) for _ in range(n))
    return ms[int(0.05 * n)], ms[int(0.95 * n)]
def g(x): return f"{x:.3g}"
print("== main table, select 9")
for arm in ["mono","mono_wide","mono_wd0.001","mono_wd0.01","mono_wd0.1","unit","streams_self","streams_joint","tok_mono","tok_streams","oracle",
            "tok_mono_lr0.03","tok_streams_lr0.03","tok_streams_lr0.1","null_tok_mono","null_mono"]:
    d = L("select", 9, arm); rs = d["runs"]
    held = [r["held"] for r in rs]; seen=[r["seen"] for r in rs]
    ref = {k: [r["ref"][k] for r in rs] for k in rs[0]["ref"]}
    reg = {}
    for r in rs: reg[regime(r,"select")] = reg.get(regime(r,"select"),0)+1
    print(f"{arm:20s} cfg={d['cfg']} np={rs[0]['nparams']} seen {g(med(seen))} held {g(med(held))} [{g(min(held))},{g(max(held))}]",
          f"| >1: {sum(h>1 for h in held)} >allmean: {sum(h>a for h,a in zip(held,ref['allmean']))} >blend: {sum(h>a for h,a in zip(held,ref['blend']))}",
          f"<.5blend: {sum(h<=.5*a for h,a in zip(held,ref['blend']))} <trans: {sum(h<a for h,a in zip(held,ref['trans']))} <.5trans {sum(h<=.5*a for h,a in zip(held,ref['trans']))}",
          f"| med held/blend {g(med([h/a for h,a in zip(held,ref['blend'])]))} medh/medblend {g(med(held)/med(ref['blend']))} med held/trans {g(med([h/a for h,a in zip(held,ref['trans'])]))} medh/medtrans {g(med(held)/med(ref['trans']))}",
          f"| held/seen med {g(med([h/s for h,s in zip(held,seen)]))} medh/meds {g(med(held)/med(seen))} | {reg}")
rs = L("select",9,"mono")["runs"]
print("refs select 9:", {k: (g(med([r['ref'][k] for r in rs])), g(min(r['ref'][k] for r in rs)), g(max(r['ref'][k] for r in rs))) for k in rs[0]['ref']})
rs = L("select",7,"mono")["runs"]
print("refs select 7:", {k: (g(med([r['ref'][k] for r in rs])), g(min(r['ref'][k] for r in rs)), g(max(r['ref'][k] for r in rs))) for k in rs[0]['ref']})
# best weight decay per split
wds = [L("select",9,a)["runs"] for a in ["mono","mono_wd0.001","mono_wd0.01","mono_wd0.1"]]
bestwd = [min(x[i]["held"] for x in wds) for i in range(12)]
print("best-wd per split median", g(med(bestwd)), "ratio best-wd/mono med", g(med([b/m["held"] for b,m in zip(bestwd, wds[0])])))
def contrast(task, cover, A, B, key="held"):
    a, b = L(task,cover,A), L(task,cover,B)
    if not (a and b): return None
    rh = [x[key]/y[key] for x,y in zip(a["runs"], b["runs"])]
    lo, hi = boot(rh)
    return med(rh), lo, hi, sum(x<1 for x in rh), sum(x<=0.5 for x in rh), sum(x<=0.2 for x in rh), min(rh), max(rh), rh
pairs = [("C1","tok_streams","tok_mono"),("C2","streams_self","mono"),("C2'","streams_joint","mono"),("C3","tok_mono","mono"),("C4","tok_streams","streams_self"),
         ("unit/ss","unit","streams_self"),("unit/mono","unit","mono"),("null tok","null_tok_mono","tok_mono"),("null mono","null_mono","mono"),
         ("C1@.03","tok_streams_lr0.03","tok_mono_lr0.03"),("C1 ts.03/tm.01","tok_streams_lr0.03","tok_mono"),("C1 ts.01/tm.03","tok_streams","tok_mono_lr0.03"),
         ("C1 ts.1/tm.01","tok_streams_lr0.1","tok_mono"),("C1 ts.1/tm.03","tok_streams_lr0.1","tok_mono_lr0.03"),
         ("wd.1/mono","mono_wd0.1","mono"),("wide/mono","mono_wide","mono"),("C3wide","tok_mono","mono_wide"),("C2wide","streams_self","mono_wide"),("C2wd","streams_self","mono_wd0.1")]
for task, cover in [("select",9),("select",7),("sum",9)]:
    print(f"== contrasts {task} {cover}")
    for lab, A, B in pairs:
        c = contrast(task,cover,A,B)
        if c is None: continue
        s = contrast(task,cover,A,B,"seen")
        print(f"{lab:16s} held {c[0]:.3g} [{c[1]:.3g},{c[2]:.3g}] won {c[3]}/12 <=.5: {c[4]} <=.2: {c[5]} range [{c[6]:.3g},{c[7]:.3g}] | seen {s[0]:.3g} [{s[1]:.3g},{s[2]:.3g}] | held/seen-ratio per split med {med([h/x for h,x in zip(c[8],s[8])]):.3g}")
print("== per-split C1 select 9:", [round(x,2) for x in contrast("select",9,"tok_streams","tok_mono")[8]])
print("== per-split C1 select 7:", [round(x,2) for x in contrast("select",7,"tok_streams","tok_mono")[8]])
print("== per-split null tok 9:", [round(x,2) for x in contrast("select",9,"null_tok_mono","tok_mono")[8]])
print("== per-split null tok 7:", [round(x,2) for x in contrast("select",7,"null_tok_mono","tok_mono")[8]])
print("== sum 9")
for arm in ["mono","tok_mono","streams_self","tok_streams","unit","oracle"]:
    rs = L("sum",9,arm)["runs"]
    held = [r["held"] for r in rs]; seen=[r["seen"] for r in rs]
    bar = [min(r["ref"]["allmean"], r["ref"]["one"]) for r in rs]
    reg = {}
    for r in rs: reg[regime(r,"sum")] = reg.get(regime(r,"sum"),0)+1
    print(f"{arm:14s} seen {g(med(seen))} held {g(med(held))} within3x {sum(h<=3*s for h,s in zip(held,seen))} >1: {sum(h>1 for h in held)} <=.5bar: {sum(h<=.5*b for h,b in zip(held,bar))} med bar {g(med(bar))} {reg}")
print("== select 7 arms")
for arm in ["mono","tok_mono","streams_self","streams_joint","tok_streams","unit","oracle","tok_mono_lr0.03","tok_streams_lr0.03"]:
    rs = L("select",7,arm)["runs"]
    held = [r["held"] for r in rs]; seen=[r["seen"] for r in rs]; ref = {k: [r["ref"][k] for r in rs] for k in rs[0]["ref"]}
    reg = {}
    for r in rs: reg[regime(r,"select")] = reg.get(regime(r,"select"),0)+1
    print(f"{arm:20s} seen {g(med(seen))} held {g(med(held))} medh/meds {g(med(held)/med(seen))} held/blend {g(med(held)/med(ref['blend']))} <.5blend {sum(h<=.5*a for h,a in zip(held,ref['blend']))} <.5trans {sum(h<=.5*a for h,a in zip(held,ref['trans']))} {reg}")
