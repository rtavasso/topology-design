"""Extra analyses from the JSON: stage 0 recompute, additive excess (held - seen), C1 against the second tok_mono
initialisation, per-teacher breakdown, rate effect, one-pass diagnostics. Read-only."""
import json, os, random, statistics, sys, glob
W = "/home/claude/topology-design/experiments/theory/worlds"
sys.path.insert(0, W)
from stage0 import regime
med = statistics.median
def L(task, cover, arm): return json.load(open(f"{W}/stage2_{task}_{cover}_{arm}.json"))["runs"]
def boot(v, n=10000, seed=0):
    rnd = random.Random(seed)
    ms = sorted(med([v[rnd.randrange(len(v))] for _ in v]) for _ in range(n))
    return ms[int(0.05 * n)], ms[int(0.95 * n)]
print("== stage 0")
for f in sorted(glob.glob(f"{W}/stage0_*.json")):
    d = json.load(open(f)); name = os.path.basename(f)
    if "_lr_" in name: print(name, d); continue
    task = "sum" if "_sum_" in name else "select"
    for lab, rs in d.items():
        held=[r["held"] for r in rs]; seen=[r["seen"] for r in rs]
        if rs[0]["nheld"]==0: print(name, lab, "seen %.3g"%med(seen)); continue
        if "control" in name:
            print(name, lab, "seen %.3g held %.3g ratio per run"%(med(seen),med(held)), ["%.2f"%(h/s) for h,s in zip(held,seen)], "n", len(rs)); continue
        reg={}
        for r in rs: reg[regime(r,task)] = reg.get(regime(r,task),0)+1
        extra = ""
        if task=="select":
            extra = " <trans %d <=.5trans %d <=.5blend %d within3x %d"%(sum(r["held"]<r["ref"]["trans"] for r in rs), sum(r["held"]<=.5*r["ref"]["trans"] for r in rs), sum(r["held"]<=.5*r["ref"]["blend"] for r in rs), sum(r["held"]<=3*r["seen"] for r in rs))
        print(name, lab, "seen %.3g held %.3g"%(med(seen),med(held)), reg, extra)
print("== additive excess held - seen (per split), select")
for cover in (9,7):
    for A in ["tok_mono","null_tok_mono","tok_mono_lr0.03","tok_streams","tok_streams_lr0.03"]:
        rs = L("select",cover,A); ex=[r["held"]-r["seen"] for r in rs]
        print(cover, f"{A:20s} seen {med([r['seen'] for r in rs]):.4f} held {med([r['held'] for r in rs]):.4f} excess med {med(ex):.4f} [{min(ex):.4f},{max(ex):.4f}]")
    for A,B in [("tok_streams","tok_mono"),("tok_streams","null_tok_mono"),("tok_streams_lr0.03","tok_mono_lr0.03"),("null_tok_mono","tok_mono")]:
        a,b = L("select",cover,A),L("select",cover,B)
        dx=[(x["held"]-x["seen"])-(y["held"]-y["seen"]) for x,y in zip(a,b)]
        rx=[(x["held"]-x["seen"])/(y["held"]-y["seen"]) for x,y in zip(a,b)]
        dh=[x["held"]-y["held"] for x,y in zip(a,b)]; ds=[x["seen"]-y["seen"] for x,y in zip(a,b)]
        lo,hi=boot(rx)
        print(f"  {A}/{B}: excess ratio med {med(rx):.2f} [{lo:.2f},{hi:.2f}] A lower {sum(d<0 for d in dx)}/12 | diff held med {med(dh):+.4f} diff seen med {med(ds):+.4f} diff excess med {med(dx):+.4f}")
print("== C1 against the second tok_mono initialisation")
for cover in (9,7):
    a,b,c = L("select",cover,"tok_streams"),L("select",cover,"null_tok_mono"),L("select",cover,"tok_mono")
    r=[x["held"]/y["held"] for x,y in zip(a,b)]; lo,hi=boot(r)
    print(cover, "tok_streams/null_tok_mono %.2f [%.2f,%.2f] won %d  <=.5: %d"%(med(r),lo,hi,sum(x<1 for x in r),sum(x<=.5 for x in r)))
    r2=[x["held"]/(0.5*(y["held"]+z["held"])) for x,y,z in zip(a,b,c)]; lo,hi=boot(r2)
    print(cover, "tok_streams/mean(two tok_mono inits) %.2f [%.2f,%.2f] won %d"%(med(r2),lo,hi,sum(x<1 for x in r2)))
    r=[x["held"]/y["held"] for x,y in zip(a,c)]
    for t in range(3): print("   teacher", t, ["%.2f"%r[i] for i in range(t,12,3)])
print("== rate effect within arm (lr0.03 / lr0.01), held and seen")
for cover in (9,7):
    for A in ("tok_mono","tok_streams"):
        a,b=L("select",cover,A+"_lr0.03"),L("select",cover,A)
        r=[x["held"]/y["held"] for x,y in zip(a,b)]; s=[x["seen"]/y["seen"] for x,y in zip(a,b)]
        print(cover, A, "held %.2f (higher at 0.03 in %d/12) seen %.2f"%(med(r),sum(x>1 for x in r),med(s)))
print("== one-pass diagnostics select 9 (one_true, one_blend, r2, mass_on_pair)")
for A in ["mono","mono_wide","unit","streams_self","streams_joint","tok_mono","tok_streams"]:
    rs=L("select",9,A)
    print(f"{A:14s}", " ".join(f"{k} {med([r[k] for r in rs]):.2f}" for k in ("one_true","one_blend","r2","mass_on_pair")), "| closer to blend than truth in %d/12"%sum(r["one_blend"]<r["one_true"] for r in rs))
print("== sum 9 streams_self per split held vs half bar")
rs=L("sum",9,"streams_self"); print(["%.3f/%.3f"%(r["held"],0.5*min(r["ref"]["allmean"],r["ref"]["one"])) for r in rs])
print("== tuning tables")
for f in sorted(glob.glob(f"{W}/stage2_cfg_*.json")):
    d=json.load(open(f)); b=min(t["seen"] for t in d["table"])
    print(os.path.basename(f), "best", d["best"])
    for t in d["table"]: print("    ", t["cfg"], "seen %.3g (%.2fx best) held %.3g"%(t["seen"],t["seen"]/b,t["held"]), "<- within 2x" if t["seen"]<=2*b else "")
