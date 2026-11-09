"""Q3: would the planted tests catch an off-by-one count, a moved threshold or a swapped reading? Mutate a copy of
analyze.py in memory, run the three planted tests against it."""
import sys, types, importlib.util, traceback, os
sys.path.insert(0, "REPO_ROOT/src"); sys.path.insert(0, "REPO_ROOT/tests")
SRC = open("REPO_ROOT/experiments/rewrite/analyze.py").read()
import test_rewrite as TR
TESTS = [TR.test_p2_verdict_on_planted_outcomes, TR.test_p3_verdict_on_planted_outcomes, TR.test_p5_to_p8_verdicts_on_planted_outcomes]
def load(src):
    m = types.ModuleType("analyze_mut"); exec(compile(src, "analyze_mut.py", "exec"), m.__dict__); return m
def run(src):
    m = load(src); TR._analyze = lambda: m
    fails = []
    for t in TESTS:
        try: t()
        except Exception as e: fails.append(t.__name__.replace("test_", "").replace("_on_planted_outcomes", ""))
    return fails
assert run(SRC) == [], run(SRC)
M = [
 ("P2 pairs count 6/10 -> 5/10", '_count(n_pair, len(rc), 0.6)', '_count(n_pair, len(rc), 0.5)'),
 ("P2 pairs count 6/10 -> 7/10", '_count(n_pair, len(rc), 0.6)', '_count(n_pair, len(rc), 0.7)'),
 ("P2 triples count 6/10 -> 5/10", '_count(n_tri, len(rc), 0.6)', '_count(n_tri, len(rc), 0.5)'),
 ("P2 triples count 6/10 -> 9/10", '_count(n_tri, len(rc), 0.6)', '_count(n_tri, len(rc), 0.9)'),
 ("P2 falsifier <=3/10 -> <=4/10", 'math.floor(0.3 * m["n_seeds"] + 1e-9)', 'math.floor(0.4 * m["n_seeds"] + 1e-9)'),
 ("P2 falsifier <=3/10 -> <=0/10", 'math.floor(0.3 * m["n_seeds"] + 1e-9)', 'math.floor(0.0 * m["n_seeds"] + 1e-9)'),
 ("P2 bar >=0.7 -> >=0.65", 'rate_ >= 0.7', 'rate_ >= 0.65'),
 ("P2 bar >=0.7 -> >=0.74", 'rate_ >= 0.7', 'rate_ >= 0.74'),
 ("P2 margin 0.2 -> 0.3", 'rate_ - g >= 0.2', 'rate_ - g >= 0.3'),
 ("P2 margin 0.2 -> 0.16", 'rate_ - g >= 0.2', 'rate_ - g >= 0.16'),
 ("P2 triples > floor -> >=", 'x > rt_["floor"]', 'x >= rt_["floor"]'),
 ("P2 pairs > floor -> >=", 'x > rp["floor"]', 'x >= rp["floor"]'),
 ("P2 floor max -> greedy only", '"floor": max(ladder_rate(rows, 1), tw) if rows else None', '"floor": ladder_rate(rows, 1) if rows else None'),
 ("P2 floor max -> trainw only", '"floor": max(ladder_rate(rows, 1), tw) if rows else None', '"floor": tw if rows else None'),
 ("P2 RT-all read unmasked always", 'allp = [rate(comb(r, PAIRS_HELD), rd) for r in ra]', 'allp = [rate(comb(r, PAIRS_HELD), "ok_u") for r in ra]'),
 ("P2 RT-all median -> max", '"rt_all_reaches": pair_ok(_median(allp), rp["greedy"])', '"rt_all_reaches": pair_ok(max(allp) if allp else None, rp["greedy"])'),
 ("P2 RT-all gate ignores +0.2 (>=0.7 only)", '"rt_all_reaches": pair_ok(_median(allp), rp["greedy"])', '"rt_all_reaches": (_median(allp) or 0) >= 0.7'),
 ("P2 verdict from the unmasked reading", 'm = p2["ok_m"]', 'm = p2["ok_u"]'),
 ("P2 cue from the masked reading", '"composes under the stated cue" if p2["ok_u"]["passes"]', '"composes under the stated cue" if p2["ok_m"]["passes"]'),
 ("P2 comb filter dropped (all binding)", 'if (flags or {}).get(pi["world"], {}).get(_key(pi), {}).get("comb")]', 'if (flags or {}).get(pi["world"], {}).get(_key(pi), {}) is not None]'),
 ("P2 pass requires no triples clause", 'and (rt_["n"] == 0 or _count(n_tri, len(rc), 0.6))', ''),
 ("P3c1 count 7/10 -> 6/10", '_count(ok1, len(c1), 0.7)', '_count(ok1, len(c1), 0.6)'),
 ("P3c1 count 7/10 -> 5/10", '_count(ok1, len(c1), 0.7)', '_count(ok1, len(c1), 0.5)'),
 ("P3c1 falsifier <=3 -> <=4", 'ok1 <= math.floor(0.3 * len(c1) + 1e-9)', 'ok1 <= math.floor(0.4 * len(c1) + 1e-9)'),
 ("P3c1 falsifier <=3 -> <=5", 'ok1 <= math.floor(0.3 * len(c1) + 1e-9)', 'ok1 <= math.floor(0.5 * len(c1) + 1e-9)'),
 ("P3c2 count 4/5 -> 3/5", '_count(ok2, len(per), 0.8)', '_count(ok2, len(per), 0.6)'),
 ("P3c2 falsifier <=1/5 -> <=2/5", 'ok2 <= math.floor(0.2 * len(per) + 1e-9)', 'ok2 <= math.floor(0.4 * len(per) + 1e-9)'),
 ("P3c2 falsifier <=1/5 -> 0/5", 'ok2 <= math.floor(0.2 * len(per) + 1e-9)', 'ok2 <= 0'),
 ("P3c2 rest read masked", 'instances(r, "eval5", world=TRAIN_WORLDS), "ok_u")', 'instances(r, "eval5", world=TRAIN_WORLDS), "ok_m")'),
 ("P3c2 must-fail read masked", 'rate(must, "ok_u")', 'rate(must, "ok_m")'),
 ("P3c2 must-fail not removed from rest", 'rest = [pi for pi in e7 if mf.get(pi["world"], {}).get(_key(pi)) is not True]', 'rest = list(e7)'),
 ("P3c2 artefact tolerance 0.05 -> 0.5", 'x["must_fail_rate"] <= 0.05', 'x["must_fail_rate"] <= 0.5'),
 ("P3 factor 0.9 -> 0.8", '"ok": R >= 0.9 * usable[d][0]', '"ok": R >= 0.8 * usable[d][0]'),
 ("P3 factor 0.9 -> 0.96", '"ok": R >= 0.9 * usable[d][0]', '"ok": R >= 0.96 * usable[d][0]'),
 ("P3 nearest rung -> always d=3", 'd = min(usable, key=lambda d: (abs(usable[d][1] - base), d))', 'd = 3'),
 ("P3 nearest rung -> farthest", 'd = min(usable, key=lambda d: (abs(usable[d][1] - base), d))', 'd = max(usable, key=lambda d: (abs(usable[d][1] - base), d))'),
 ("P5 tolerance 0.1 -> 0.14", 'abs(m_ei - m_rt) <= 0.1', 'abs(m_ei - m_rt) <= 0.14'),
 ("P5 tolerance 0.1 -> 0.06", 'abs(m_ei - m_rt) <= 0.1', 'abs(m_ei - m_rt) <= 0.06'),
 ("P5 falsifier ignores RT above greedy", '(m_ei < g < m_rt)', '(m_ei < g)'),
 ("P5 read unmasked", 'm_rt, m_ei = _median([rate(sel(r)) for r in rc]), _median([rate(sel(r)) for r in rei])', 'm_rt, m_ei = _median([rate(sel(r), "ok_u") for r in rc]), _median([rate(sel(r), "ok_u") for r in rei])'),
 ("P5 EI median -> max", '_median([rate(sel(r)) for r in rei])\n', 'max([rate(sel(r)) for r in rei])\n'),
 ("P6 post count 7/10 -> 5/10", '_count(okp, len(sp), 0.7)', '_count(okp, len(sp), 0.5)'),
 ("P6 canon share 0.9 -> 0.8", 'x["canon"] >= 0.9', 'x["canon"] >= 0.8'),
 ("P6 canon share 0.9 -> 0.94", 'x["canon"] >= 0.9', 'x["canon"] >= 0.94'),
 ("P6 TV below uniform -> below 2x uniform", 'x["tv"] < x["tv_uniform"]', 'x["tv"] < 2 * x["tv_uniform"]'),
 ("P7 band 0.1 -> 0.15", 'within = sum(abs(x - target) <= 0.1 for x in ex_t)', 'within = sum(abs(x - target) <= 0.15 for x in ex_t)'),
 ("P7 band 0.1 -> 0.03", 'within = sum(abs(x - target) <= 0.1 for x in ex_t)', 'within = sum(abs(x - target) <= 0.03 for x in ex_t)'),
 ("P7 c1 3/3 -> 2/3", 'c1 = "holds" if within == len(ex_t)', 'c1 = "holds" if within * 3 >= 2 * len(ex_t)'),
 ("P7 c2 margin 0.05 -> 0.15", 'x < m_h - 0.05', 'x < m_h - 0.15'),
 ("P7 c2 margin 0.05 -> 0", 'x < m_h - 0.05', 'x < m_h'),
 ("P7 c2 3/3 -> 2/3", 'c2 = "holds" if below_h == len(ex_h)', 'c2 = "holds" if below_h * 3 >= 2 * len(ex_h)'),
 ("P7 falsifier 2/3 -> 3/3", '"falsified" if ge_h * 3 >= 2 * len(ex_h)', '"falsified" if ge_h == len(ex_h)'),
 ("P7 no ceiling factor", 'target = ceil * m_t', 'target = m_t'),
 ("P7 held-out read masked", '_median([rate(hb(r), "ok_u") for r in rc])', '_median([rate(hb(r), "ok_m") for r in rc])'),
 ("P8 gap 0.05 -> 0.03", 'n_gap = sum(g > 0.05 for g in gaps)', 'n_gap = sum(g > 0.03 for g in gaps)'),
 ("P8 gap 0.05 -> 0.15", 'n_gap = sum(g > 0.05 for g in gaps)', 'n_gap = sum(g > 0.15 for g in gaps)'),
 ("P8 holds at 1/2", '"holds" if n_gap == len(gaps)', '"holds" if n_gap >= 1'),
 ("P8 falsified at 1/2", '("falsified" if n_gap == 0 else "undecided")', '("falsified" if n_gap <= 1 else "undecided")'),
 ("P8 top verdict = unmasked", 'p8 = {**out, "verdict": vm, "verdict_unmasked": vu', 'p8 = {**out, "verdict": vu, "verdict_unmasked": vm'),
 ("P8 RT median -> max", 'm = _median([rate(tb(r), rd) for r in rc])', 'm = max([rate(tb(r), rd) for r in rc])'),
]
surv = []
for label, old, new in M:
    assert SRC.count(old) == 1, (label, SRC.count(old))
    f = run(SRC.replace(old, new))
    print(("CAUGHT   " if f else "SURVIVES ") + label + (f"  [{', '.join(f)}]" if f else ""))
    if not f: surv.append(label)
print(f"\n{len(M) - len(surv)} of {len(M)} mutants caught; survivors: {len(surv)}")
