"""Q1: what analyze.verdicts returns for symbolic policies planted as RT-canon / RT-absolute on the real eval instances."""
import json, sys, random, importlib.util
from eng import read, TRAIN, HELD_B
spec = importlib.util.spec_from_file_location("analyze", "REPO_ROOT/experiments/rewrite/analyze.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
D = json.load(open("REPO_ROOT/experiments/rewrite/derive/derived.json"))
flags = D["p2_comb"]
ALLW = ["A", "C", "I", "D", "AC", "AI", "AD", "CI", "CD", "ID", "ACI", "ACD", "AID", "CID", "ACID"]
E5 = {w: read("eval5", w) for w in ALLW}
E7 = {w: read("eval7", w) for w in TRAIN}
P2 = json.load(open("p2_policies.json"))
own = json.load(open("own_mustfail.json"))

def row(r, ok_m, ok_u=None):
    return {"world": r["world"], "t": r["t"], "s": r["s"], "k": r["k"], "binding": r["binding"], "GP1": r["first"]["GP1"], "ok_m": bool(ok_m), "ok_u": bool(ok_m if ok_u is None else ok_u), "ladder": r["ladder"]}
def run(arm, seed, e5, e7=()):
    return {"cfg": {"arm": arm, "seed": seed, "name": f"{arm}_s{seed}"}, "final": {"eval5": {"per_instance": list(e5)}, "eval7": {"per_instance": list(e7)}}}

# ---------- P2 ----------
def p2_rows(policy, rng=None, thin=1.0):
    """policy: 'd1'|'d2'|'d3'|'trainw'|'oracle'| a p2_policies key (planted by per-world count on the comb instances)."""
    out = []
    for w in HELD_B:
        cnt = 0
        for r in E5[w]:
            f = flags[w].get(f"{r['t']}|{r['s']}", {})
            if not (r["binding"] and f.get("comb")):
                out.append(row(r, True)); continue
            if policy in ("d1", "d2", "d3"): ok = r["ladder"][policy[1]]
            elif policy == "trainw": ok = f["trainw"]
            elif policy == "oracle": ok = True
            else:
                ok = cnt < round(P2[w][policy]); cnt += 1      # same count per world as the policy scored (order reversed in that script; only the count matters)
            if rng is not None and ok: ok = rng.random() < thin
            out.append(row(r, ok))
    return out
def p2_verdict(canon, rtall, thin=1.0, seeds=10):
    runs = [run("RT-canon", s, p2_rows(canon, random.Random(s) if thin < 1 else None, thin)) for s in range(seeds)]
    if rtall: runs += [run("RT-all", s, p2_rows(rtall)) for s in range(3)]
    v = A.verdicts(runs, D)["P2"]
    m = v["ok_m"]
    return f"pairs {A._median(m['pair_rates']):.3f} triples {A._median(m['triple_rates']):.3f} | pair_ok {m['pair_ok']}/10, >floor {m['pair_above_floor']}/10, triples>floor {m['triple_above_floor']}/10, RT-all reaches {m['rt_all_reaches']} -> {v['verdict']} {v.get('why') or v.get('cue') or ''}"
v0 = A.verdicts([run("RT-canon", 0, p2_rows("oracle"))], D)["P2"]
print("P2 refs from analyze:", {k: (v0[k]["n"], round(v0[k]["greedy"], 3), round(v0[k]["d3"], 3), round(v0[k]["trainw"], 3), round(v0[k]["floor"], 3)) for k in ("pairs", "triples")})
for canon in ("d1", "trainw", "d2", "d3", "oracle", "greedy-fallback", "greedy-last", "subs-only", "random-tier0"):
    print(f"P2 RT-canon={canon:16s} RT-all=oracle:", p2_verdict(canon, "oracle"))
print("P2 RT-canon=d1, RT-all=d3:", p2_verdict("d1", "d3"))
print("P2 RT-canon=d1, RT-all=d2:", p2_verdict("d1", "d2"))
print("P2 RT-canon=d1, no RT-all:", p2_verdict("d1", None))
print("P2 RT-canon=d3, no RT-all (pass without any ceiling):", p2_verdict("d3", None))
for thin in (0.99, 0.97, 0.95, 0.9, 0.8):
    print(f"P2 RT-canon=oracle thinned to {thin} per seed, RT-all=oracle:", p2_verdict("oracle", "oracle", thin))
for thin in (1.0, 0.98, 0.96):
    print(f"P2 RT-canon=d3 thinned to {thin}, RT-all=oracle:", p2_verdict("d3", "oracle", thin) if thin < 1 else p2_verdict("d3", "oracle"))

# ---------- P3 clause 1 ----------
def e5_rows(okf): return [row(dict(r, world=w), okf(r)) for w in TRAIN for r in E5[w]]
def e7_rows(okf, rng, keep, fail=None, key="mf8"):
    out = []
    for w in TRAIN:
        for r in E7[w]:
            ok = okf(r) and (rng.random() < keep)
            if fail and own[w][f"{r['t']}|{r['s']}"][fail]: ok = False
            out.append(row(dict(r, world=w), ok))
    return out
pol = {"greedy": lambda r: r["ladder"]["1"], "lookahead-2": lambda r: r["ladder"]["2"], "lookahead-3": lambda r: r["ladder"]["3"], "oracle": lambda r: True}
print()
for name, okf in pol.items():
    for keep in (1.0, 0.97, 0.95, 0.93, 0.90, 0.85):
        runs = [run("RT-canon", s, e5_rows(okf), e7_rows(okf, random.Random(100 + s), keep)) for s in range(10)]
        c = A.verdicts(runs, D)["P3"]["clause1"]
        x = c["per_seed"]
        print(f"P3c1 {name:11s} kept {keep:.2f} at sizes 6-7: R {min(q['R'] for q in x):.3f}-{max(q['R'] for q in x):.3f}, rung {sorted(set(q['rung'] for q in x))}, rho {x[0]['rho']:.3f}, bar {0.9*x[0]['rho']:.3f}; §17 ok {c['ok']}/10 -> {c['verdict']}; §6 (R>=0.9) {c['raw_ok']}/10")
# a router whose size<=5 base straddles the rung midpoint: oracle thinned at BOTH sizes to ~0.87
print()
for base in (0.86, 0.88):
    runs = []
    for s in range(10):
        rng = random.Random(500 + s)
        e5 = [row(dict(r, world=w), rng.random() < base) for w in TRAIN for r in E5[w]]
        e7 = [row(dict(r, world=w), rng.random() < base * 0.97) for w in TRAIN for r in E7[w]]
        runs.append(run("RT-canon", s, e5, e7))
    c = A.verdicts(runs, D)["P3"]["clause1"]
    print(f"P3c1 flat policy base {base}, keeps 0.97 at sizes 6-7: rungs {[q['rung'] for q in c['per_seed']]}, bars {[round(0.9*q['rho'],3) for q in c['per_seed']]}, R {[round(q['R'],3) for q in c['per_seed']]} -> ok {c['ok']}/10 {c['verdict']}")

# ---------- P3 clause 2 ----------
print()
for name in ("lookahead-3", "oracle"):
    okf = pol[name]
    for fail in ("mf8", "flag", "mf6", "mfS"):
        runs = [run("RT-canon", 0, e5_rows(okf))] + [run("RT-absolute", s, e5_rows(okf), e7_rows(okf, random.Random(s), 1.0, fail)) for s in range(5)]
        c = A.verdicts(runs, D)["P3"]["clause2"]
        x = c["per_seed"][0]
        print(f"P3c2 RT-absolute={name:11s} failing exactly its {fail} set (n {sum(own[w][k][fail] for w in TRAIN for k in own[w])}): must-fail n {x['must_fail_n']} rate {x['must_fail_rate']:.2f}; rest n {x['rest_n']} R {x['rest_R']:.3f} rho {x['rest_rho']:.3f} (rung {x['rest_rung']}) bar {0.9*x['rest_rho']:.3f} -> ok {c['ok']}/5 {c['verdict']}")
