"""analyze.py on synthetic run records built from the dataset (no training results): planted outcomes per scenario."""
import importlib.util, json, random, sys, copy
from common import *
spec = importlib.util.spec_from_file_location("rw_analyze", os.path.join(REPO, "experiments/rewrite/analyze.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
sets = {"eval5": load("eval5"), "eval7": load("eval7"), "evalk78": load("evalk78", ["AC"])}
s5 = json.load(open("s5_out.json"))
# no-composition flag for held-out binding instances: some proof within T inside a training sub-world
nocomp = {}
for w in PAIRS + TRIPLES:
    b = [r for r in sets["eval5"][w] if r["binding"]]
    subs = [U for U in s5[w][0]["dU"] if U != w and U in TRAIN_WORLDS]
    for r, x in zip(b, s5[w]):
        nocomp[(w, r["t"], r["s"])] = any(x["dU"][U] is not None for U in subs)
def make_run(arm, seed, okf):
    rng = random.Random(hash((arm, seed)) & 0xffff)
    final = {}
    for tag, by in sets.items():
        pis = []
        for w, rs in by.items():
            for r in rs:
                tf = next(p["target"] for p in r["post"] if p["state"] == r["t"])
                ok = bool(okf(tag, r, rng))
                pis.append({"world": w, "t": r["t"], "s": r["s"], "k": r["k"], "binding": r["binding"], "GP1": r["first"]["GP1"], "Dany": r["first"]["Dany"],
                            "size": r["size_t"], "ok_m": ok, "ok_u": ok, "cat_m": "ok" if ok else "horizon", "cat_u": "ok" if ok else "horizon", "steps_m": r["k"],
                            "first_ok": ok, "ladder": r["ladder"], "visited_goal_m": ok, "visited_goal_u": ok, "p_first": dict(tf), "first_canon": True,
                            "target_first": tf, "n_classes": r["n_trace_classes"]})
        final[tag] = {"per_instance": pis}
    cells = {}
    for pi in final["eval5"]["per_instance"]:
        c = cells.setdefault(f"{pi['world']}|{int(pi['binding'])}|{int(pi['GP1'])}|{pi['k']}", {"n": 0, "ok_m": 0}); c["n"] += 1; c["ok_m"] += pi["ok_m"]
    return {"cfg": {"arm": arm, "seed": seed, "name": f"{arm}_s{seed}", "steps": 100}, "curve": [{"step": 100, "loss": 0.1, "eval5": cells}], "final": final, "seconds": 60.0, "ei_found": [0, 0, 0]}
def section(md, a, b):
    return md[md.index(a):md.index(b)]
def scen(name, canon_f, all_f=lambda tag, r, rng: rng.random() < 0.95, extra=()):
    runs = [make_run("RT-canon", s, canon_f) for s in range(10)] + [make_run("RT-all", s, all_f) for s in range(3)]
    for arm, n, f in extra:
        runs += [make_run(arm, s, f) for s in range(n)]
    md = A.report(runs, DER)
    print(f"\n######## {name}")
    return md
bern = lambda p: (lambda tag, r, rng: rng.random() < p)
lad_f = lambda d: (lambda tag, r, rng: r["ladder"][str(d)])
def noncomposer(tag, r, rng):
    if tag == "eval5" and r["binding"] and r["world"] in PAIRS + TRIPLES:
        return nocomp[(r["world"], r["t"], r["s"])]
    return True
md = scen("S1 planted pass: RT-canon Bernoulli 0.78 everywhere, RT-all 0.95", bern(0.78)); print(section(md, "## P2", "## P3"))
md = scen("S2 planted fail: RT-canon = greedy rung", lad_f(1)); print(section(md, "## P2", "## P3")); print(section(md, "## P3", "## P4")[:1500])
md = scen("S3: RT-canon = lookahead-3 rung", lad_f(3)); print(section(md, "## P2", "## P3"))
md = scen("S4: non-composing router (solves a held-out binding instance iff a proof within T lies in a training sub-world; all else solved)", noncomposer); print(section(md, "## P2", "## P3"))
# P3 bias: per-instance skill = greedy rung at size<=5, greedy rung thinned by 0.85 at sizes 6-7
def degraded(tag, r, rng):
    return r["ladder"]["1"] and (tag != "eval7" or rng.random() < 0.85)
md = scen("S5 P3: router = greedy rung, thinned to 85 % at sizes 6-7", degraded, extra=[("RT-absolute", 5, degraded)]); print(section(md, "## P3", "## P4"))
# P5-P8 sections as printed
md = scen("S6 all arms present (Bernoulli)", bern(0.6), extra=[("EI16", 5, bern(0.3)), ("RT-examples", 3, bern(0.5)), ("RT-shuffled", 2, bern(0.4)), ("RT-post", 2, bern(0.6))])
print(section(md, "## P5", "## P6")); print(section(md, "## P6", "## P7")[:900]); print(section(md, "## P7", "## P8")); print(section(md, "## P8", "## Failure")[:2600]); print(md[md.index("## P1"):])
