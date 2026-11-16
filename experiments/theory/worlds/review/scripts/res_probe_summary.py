"""Summaries of the probe runs against the stored stage-2 runs, and the stage-0 8,000-step pairing. No training."""
import json, statistics
W = "/home/claude/topology-design/experiments/theory/worlds"
med = statistics.median
P = json.load(open("res_probe_runs.json"))
tm = json.load(open(f"{W}/stage2_select_9_tok_mono.json"))["runs"]; ts = json.load(open(f"{W}/stage2_select_9_tok_streams.json"))["runs"]
for lab, d in P.items():
    ss = sorted(map(int, d)); print(lab, "splits", ss, "training seconds %.0f" % sum(v["secs"] for v in d.values()))
    for other, name in ((tm, "tok_mono"), (ts, "tok_streams")):
        rh = [d[str(s)]["held"] / other[s]["held"] for s in ss]; rs = [d[str(s)]["seen"] / other[s]["seen"] for s in ss]
        ex = [(d[str(s)]["held"] - d[str(s)]["seen"], other[s]["held"] - other[s]["seen"]) for s in ss]
        print(f"   / stored {name}: held ratios {[round(x,2) for x in rh]} median {med(rh):.2f}; seen ratio median {med(rs):.2f}; excess probe vs stored {[(round(a,3),round(b,3)) for a,b in ex]}")
    print("   stored C1 on these splits:", [round(ts[s]["held"]/tm[s]["held"],2) for s in ss])
print("total probe training seconds (plus 20 s for the margin probe):", sum(v["secs"] for d in P.values() for v in d.values()))
a = json.load(open(f"{W}/stage0_fit_select_tok_mono.json"))["coverage 9, 8,000 steps"]; b = json.load(open(f"{W}/stage0_dial_select_tok_mono.json"))["coverage 9"]
print("tok_mono 8,000 / 4,000 steps (stage 0, six splits): seen ratio med %.2f held ratio med %.2f; held ratios %s" % (med([x["seen"]/y["seen"] for x,y in zip(a,b)]), med([x["held"]/y["held"] for x,y in zip(a,b)]), [round(x["held"]/y["held"],2) for x,y in zip(a,b)]))
mw = json.load(open(f"{W}/stage2_select_9_mono_wide.json"))["runs"]; m = json.load(open(f"{W}/stage2_select_9_mono.json"))["runs"]
print("mono wide/mono: seen %.2f held %.2f" % (med([x["seen"]/y["seen"] for x,y in zip(mw,m)]), med([x["held"]/y["held"] for x,y in zip(mw,m)])))
s9 = json.load(open(f"{W}/stage2_sum_9_mono.json"))["runs"]; print("sum mono held > 1 in", sum(r["held"]>1 for r in s9), "of 12; min", min(r["held"] for r in s9))
