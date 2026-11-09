"""With the positions the categorical head is labelled on (slots valid at cap 5): instances whose every SHORTEST proof
needs a never-labelled slot, and those with no route at all within T; then clause 2 re-run with the corrected set."""
import json, random, importlib.util
from eng import *
import eng
exec(open("chk_mustfail.py").read().split("t0 = time.time()")[0].split("D = json.load")[0])   # imports only
VALID = {tuple(x) for x in json.load(open("valid_slots_cap5.json"))}
own = json.load(open("own_mustfail.json"))
def ball(start, W, cap, radius, allow):
    dist = {start: 0}; fr = [start]
    for d in range(1, radius + 1):
        nx = []
        for u in fr:
            for rule, i, v in moves(u, cap):
                if AX[rule] not in W or not allow(rule, i) or v in dist: continue
                dist[v] = d; nx.append(v)
        fr = nx
        if not fr: break
    return dist
def route_within(t, s, W, cap, T, allow):
    F = ball(t, W, cap, (T + 1) // 2, allow)
    if s in F: return True
    B = ball(s, W, cap, T // 2, allow)
    small, big = (F, B) if len(F) < len(B) else (B, F)
    return any(v in big and small[v] + big[v] <= T for v in small)
allowS = lambda rule, i: (i, rule) in VALID
tot = [0, 0, 0]
for w in TRAIN:
    c = [0, 0, 0]
    for r in read("eval7", w):
        key = f"{r['t']}|{r['s']}"
        sh = not route_within(parse(r["t"]), parse(r["s"]), set(w), r["cap"], r["k"], allowS)
        own[w][key]["shortS"] = sh
        c[0] += sh; c[1] += own[w][key]["mfS"]; c[2] += sh and not own[w][key]["mfS"]
    eng._mv.clear()
    print(w, "every shortest proof needs a never-labelled slot:", c[0], "| no route within T:", c[1], "| detour within T exists:", c[2])
    tot = [a + b for a, b in zip(tot, c)]
print("ALL", tot)
json.dump(own, open("own_mustfail.json", "w"))
# clause 2 with the corrected must-fail set planted into derived
spec = importlib.util.spec_from_file_location("analyze", "REPO_ROOT/experiments/rewrite/analyze.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
E5 = {w: read("eval5", w) for w in TRAIN}; E7 = {w: read("eval7", w) for w in TRAIN}
def row(r, w, ok): return {"world": w, "t": r["t"], "s": r["s"], "k": r["k"], "binding": r["binding"], "GP1": r["first"]["GP1"], "ok_m": bool(ok), "ok_u": bool(ok), "ladder": r["ladder"]}
def run(arm, s, e5, e7=()): return {"cfg": {"arm": arm, "seed": s, "name": f"{arm}{s}"}, "final": {"eval5": {"per_instance": e5}, "eval7": {"per_instance": e7}}}
Dfix = {"eval7_mustfail": {w: {k: True for k, v in own[w].items() if v["mfS"]} for w in TRAIN}}
for name, okf in (("lookahead-3", lambda r: r["ladder"]["3"]), ("oracle", lambda r: True)):
    for fail in ("mfS", "shortS"):
        e5 = [row(r, w, okf(r)) for w in TRAIN for r in E5[w]]
        e7 = [row(r, w, okf(r) and not own[w][f"{r['t']}|{r['s']}"][fail]) for w in TRAIN for r in E7[w]]
        c = A.verdicts([run("RT-canon", 0, e5)] + [run("RT-absolute", s, e5, e7) for s in range(5)], Dfix)["P3"]["clause2"]; x = c["per_seed"][0]
        print(f"corrected must-fail set (n {x['must_fail_n']}), RT-absolute={name} failing exactly its {fail} set: rest n {x['rest_n']} R {x['rest_R']:.3f} rho {x['rest_rho']:.3f} bar {0.9*x['rest_rho']:.3f} -> {c['verdict']} ({c['ok']}/5)")
