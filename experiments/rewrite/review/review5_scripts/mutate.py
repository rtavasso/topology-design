"""Would the planted tests catch an off-by-one count, a moved threshold, a changed statistic or a swapped reading?
Each mutant changes one expression of analyze.verdicts() in memory; the planted-verdict tests are run against it.
(Review 5 wrote the first version of this and found 42 of 60 mutants of the first tests alive; the list below is that one
brought up to the code as amended after the review, with mutants for what the review added.)

    PYTHONPATH=src python3 experiments/rewrite/review/review5_scripts/mutate.py
"""
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))
SRC = open(os.path.join(ROOT, "experiments", "rewrite", "analyze.py")).read()
import test_rewrite as TR  # noqa: E402

TESTS = [TR.test_p2_verdict_on_planted_outcomes, TR.test_p3_verdict_on_planted_outcomes, TR.test_p5_to_p8_verdicts_on_planted_outcomes]


def run(src):
    m = types.ModuleType("analyze_mut")
    exec(compile(src, "analyze_mut.py", "exec"), m.__dict__)
    TR._analyze = lambda: m
    fails = []
    for t in TESTS:
        try:
            t()
        except Exception:
            fails.append(t.__name__)
    return fails


M = [
 ("P2 pairs count 6/10 -> 5/10", '_count(n_pair, len(rc), 0.6)', '_count(n_pair, len(rc), 0.5)'),
 ("P2 pairs count 6/10 -> 7/10", '_count(n_pair, len(rc), 0.6)', '_count(n_pair, len(rc), 0.7)'),
 ("P2 triples count 6/10 -> 5/10", '_count(n_tri, len(rc), 0.6)', '_count(n_tri, len(rc), 0.5)'),
 ("P2 triples count 6/10 -> 7/10", '_count(n_tri, len(rc), 0.6)', '_count(n_tri, len(rc), 0.7)'),
 ("P2 falsifier <=3/10 -> <=4/10", 'math.floor(0.3 * m["n_seeds"] + 1e-9)', 'math.floor(0.4 * m["n_seeds"] + 1e-9)'),
 ("P2 falsifier <=3/10 -> <=2/10", 'math.floor(0.3 * m["n_seeds"] + 1e-9)', 'math.floor(0.2 * m["n_seeds"] + 1e-9)'),
 ("P2 bar >=0.7 -> >=0.69", '_ge(rate_, 0.7)', '_ge(rate_, 0.69)'),
 ("P2 bar >=0.7 -> >=0.71", '_ge(rate_, 0.7)', '_ge(rate_, 0.71)'),
 ("P2 margin 0.2 -> 0.21", '_ge(rate_ - g, 0.2)', '_ge(rate_ - g, 0.21)'),
 ("P2 margin 0.2 -> 0.19", '_ge(rate_ - g, 0.2)', '_ge(rate_ - g, 0.19)'),
 ("P2 triples > floor -> >=", '_gt(x, rt_["floor"])', '_ge(x, rt_["floor"])'),
 ("P2 pairs > floor -> >=", '_gt(x, rp["floor"])', '_ge(x, rp["floor"])'),
 ("P2 floor max -> greedy only", '"floor": max(ladder_rate(rows, 1), tw) if rows else None', '"floor": ladder_rate(rows, 1) if rows else None'),
 ("P2 floor max -> trainw only", '"floor": max(ladder_rate(rows, 1), tw) if rows else None', '"floor": tw if rows else None'),
 ("P2 RT-all read unmasked always", 'allp = [rate(comb(r, PAIRS_HELD), rd) for r in ra]', 'allp = [rate(comb(r, PAIRS_HELD), "ok_u") for r in ra]'),
 ("P2 RT-all median -> max", '"rt_all_reaches": pair_ok(_median(allp), rp["greedy"])', '"rt_all_reaches": pair_ok(max(allp) if allp else None, rp["greedy"])'),
 ("P2 RT-all gate ignores +0.2", '"rt_all_reaches": pair_ok(_median(allp), rp["greedy"])', '"rt_all_reaches": (_median(allp) or 0) >= 0.7'),
 ("P2 verdict from the unmasked reading", 'm = p2["ok_m"]', 'm = p2["ok_u"]'),
 ("P2 cue from the masked reading", '("composes under the stated cue" if p2["ok_u"]["passes"]', '("composes under the stated cue" if p2["ok_m"]["passes"]'),
 ("P2 comb filter dropped", 'if (flags or {}).get(pi["world"], {}).get(_key(pi), {}).get("comb")]', 'if (flags or {}).get(pi["world"], {}).get(_key(pi), {}) is not None]'),
 ("P2 pass without the triples clause", ' and (rt_["n"] == 0 or _count(n_tri, len(rc), 0.6))}', '}'),
 ("P2 §6 ceiling 0.9 -> 0.75 (pairs)", '_ge(_median(allp), 0.9)', '_ge(_median(allp), 0.75)'),
 ("P2 §6 ceiling ignores the triples", ' and (rt_["n"] == 0 or (_median(allt) is not None and _ge(_median(allt), 0.9))),', ','),
 ("P2 beyond-glue = all comb", 'return [pi for pi in comb(run, worlds) if not flags[pi["world"]][_key(pi)].get("glue")]', 'return comb(run, worlds)'),
 ("P3c1 count 7/10 -> 6/10", '_count(ok1, len(c1), 0.7)', '_count(ok1, len(c1), 0.6)'),
 ("P3c1 count 7/10 -> 8/10", '_count(ok1, len(c1), 0.7)', '_count(ok1, len(c1), 0.8)'),
 ("P3c1 falsifier <=3 -> <=4", 'ok1 <= math.floor(0.3 * len(c1) + 1e-9)', 'ok1 <= math.floor(0.4 * len(c1) + 1e-9)'),
 ("P3c1 falsifier <=3 -> <=2", 'ok1 <= math.floor(0.3 * len(c1) + 1e-9)', 'ok1 <= math.floor(0.2 * len(c1) + 1e-9)'),
 ("P3c2 count 4/5 -> 3/5", '_count(ok2, len(per), 0.8)', '_count(ok2, len(per), 0.6)'),
 ("P3c2 count 4/5 -> 5/5", '_count(ok2, len(per), 0.8)', '_count(ok2, len(per), 1.0)'),
 ("P3c2 falsifier <=1/5 -> <=2/5", 'ok2 <= math.floor(0.2 * len(per) + 1e-9)', 'ok2 <= math.floor(0.4 * len(per) + 1e-9)'),
 ("P3c2 falsifier <=1/5 -> 0/5", 'ok2 <= math.floor(0.2 * len(per) + 1e-9)', 'ok2 <= 0'),
 ("P3c2 rest read masked", 'nr = normalised_ratio(rest, instances(r, "eval5", world=TRAIN_WORLDS), "ok_u")', 'nr = normalised_ratio(rest, instances(r, "eval5", world=TRAIN_WORLDS), "ok_m")'),
 ("P3c2 must-fail read masked", '"must_fail_rate": rate(must, "ok_u")', '"must_fail_rate": rate(must, "ok_m")'),
 ("P3c2 rest includes the detours", '[pi for pi in e7 if g(pi) == "shortest"]', '[pi for pi in e7 if g(pi) in ("shortest", "detour")]'),
 ("P3c2 rest includes the must-fail", '[pi for pi in e7 if g(pi) == "shortest"]', '[pi for pi in e7 if g(pi) in ("shortest", "mustfail")]'),
 ("P3c2 artefact tolerance 0.05 -> 0.1", '_ge(0.05, x["must_fail_rate"])', '_ge(0.1, x["must_fail_rate"])'),
 ("P3c2 artefact tolerance 0.05 -> 0.04", '_ge(0.05, x["must_fail_rate"])', '_ge(0.04, x["must_fail_rate"])'),
 ("P3 factor 0.9 -> 0.89", '"ok": _ge(R, 0.9 * rho)', '"ok": _ge(R, 0.89 * rho)'),
 ("P3 factor 0.9 -> 0.91", '"ok": _ge(R, 0.9 * rho)', '"ok": _ge(R, 0.91 * rho)'),
 ("P3 rho not capped at 1/base", 'return min(rho, 1.0 / base) if base > 0 else rho', 'return rho'),
 ("P3 rho = top rung's always", '    rho = rho_for(base, usable)\n', '    rho = max(usable)[1]\n'),
 ("P3 rho = nearest rung (no interpolation)", 'rho = r0 + (r1 - r0) * (base - b0) / (b1 - b0)', 'rho = r0 if base - b0 <= b1 - base else r1'),
 ("P3 summary = clause 1 only", 'p3["verdict"] = "pass" if (v1 == "pass" and v2 == "holds") else', 'p3["verdict"] = "pass" if (v1 == "pass") else'),
 ("P5 tolerance 0.1 -> 0.11", '_ge(0.1, abs(m_ei - m_rt))', '_ge(0.11, abs(m_ei - m_rt))'),
 ("P5 tolerance 0.1 -> 0.09", '_ge(0.1, abs(m_ei - m_rt))', '_ge(0.09, abs(m_ei - m_rt))'),
 ("P5 falsifier ignores RT above greedy", '(_gt(g, m_ei) and _gt(m_rt, g))', '(_gt(g, m_ei))'),
 ("P5 falsifier at the greedy rung", '(_gt(g, m_ei) and _gt(m_rt, g))', '(_ge(g, m_ei) and _gt(m_rt, g))'),
 ("P5 read unmasked", 'm_rt, m_ei = _median([rate(sel(r)) for r in rc]), _median([rate(sel(r)) for r in rei])', 'm_rt, m_ei = _median([rate(sel(r), "ok_u") for r in rc]), _median([rate(sel(r), "ok_u") for r in rei])'),
 ("P5 EI median -> max", 'm_rt, m_ei = _median([rate(sel(r)) for r in rc]), _median([rate(sel(r)) for r in rei])', 'm_rt, m_ei = _median([rate(sel(r)) for r in rc]), max([rate(sel(r)) for r in rei])'),
 ("P6 post count 7/10 -> 6/10", '_count(okp, len(sp), 0.7)', '_count(okp, len(sp), 0.6)'),
 ("P6 post count 7/10 -> 8/10", '_count(okp, len(sp), 0.7)', '_count(okp, len(sp), 0.8)'),
 ("P6 canon count 7/10 -> 6/10", '_count(okc, len(sc), 0.7)', '_count(okc, len(sc), 0.6)'),
 ("P6 canon share 0.9 -> 0.89", '_ge(x["canon"], 0.9)', '_ge(x["canon"], 0.89)'),
 ("P6 canon share 0.9 -> 0.91", '_ge(x["canon"], 0.9)', '_ge(x["canon"], 0.91)'),
 ("P6 TV below uniform -> below 1.2x", '_gt(x["tv_uniform"], x["tv"])', '_gt(1.2 * x["tv_uniform"], x["tv"])'),
 ("P6 TV below uniform -> below 0.9x", '_gt(x["tv_uniform"], x["tv"])', '_gt(0.9 * x["tv_uniform"], x["tv"])'),
 ("P7 band 0.1 -> 0.11", 'within = sum(_ge(0.1, abs(x - target)) for x in ex_t)', 'within = sum(_ge(0.11, abs(x - target)) for x in ex_t)'),
 ("P7 band 0.1 -> 0.09", 'within = sum(_ge(0.1, abs(x - target)) for x in ex_t)', 'within = sum(_ge(0.09, abs(x - target)) for x in ex_t)'),
 ("P7 c1 3/3 -> 2/3", 'c1 = "holds" if within == len(ex_t)', 'c1 = "holds" if within * 3 >= 2 * len(ex_t)'),
 ("P7 c1 above at 1/3", '"above: W is read from more than the examples" if above * 3 >= 2 * len(ex_t)', '"above: W is read from more than the examples" if above >= 1'),
 ("P7 c2 margin 0.05 -> 0.06", '_gt(m_h - 0.05, x)', '_gt(m_h - 0.06, x)'),
 ("P7 c2 margin 0.05 -> 0.04", '_gt(m_h - 0.05, x)', '_gt(m_h - 0.04, x)'),
 ("P7 c2 3/3 -> 2/3", 'c2 = "holds" if below_h == len(ex_h)', 'c2 = "holds" if below_h * 3 >= 2 * len(ex_h)'),
 ("P7 falsifier 2/3 -> 3/3", '"falsified" if ge_h * 3 >= 2 * len(ex_h)', '"falsified" if ge_h == len(ex_h)'),
 ("P7 falsifier 2/3 -> 1/3", '"falsified" if ge_h * 3 >= 2 * len(ex_h)', '"falsified" if ge_h >= 1'),
 ("P7 falsifier >= median -> >= median - 0.01", 'ge_h = sum(_ge(x, m_h) for x in ex_h)', 'ge_h = sum(_ge(x, m_h - 0.01) for x in ex_h)'),
 ("P7 no ceiling factor", 'target = ceil * m_t', 'target = m_t'),
 ("P7 held-out RT read masked", '_median([rate(hb(r), "ok_u") for r in rc])', '_median([rate(hb(r), "ok_m") for r in rc])'),
 ("P7 examples read masked", 'ex_t, ex_h = [rate(tb(r), "ok_u") for r in rex], [rate(hb(r), "ok_u") for r in rex]', 'ex_t, ex_h = [rate(tb(r), "ok_m") for r in rex], [rate(hb(r), "ok_m") for r in rex]'),
 ("P8 gap 0.05 -> 0.04", 'n_gap = sum(_gt(g, 0.05) for g in gaps)', 'n_gap = sum(_gt(g, 0.04) for g in gaps)'),
 ("P8 gap 0.05 -> 0.06", 'n_gap = sum(_gt(g, 0.05) for g in gaps)', 'n_gap = sum(_gt(g, 0.06) for g in gaps)'),
 ("P8 holds at 1/2", '"holds" if n_gap == len(gaps)', '"holds" if n_gap >= 1'),
 ("P8 falsified at 1/2", '("falsified" if n_gap == 0 else "undecided")', '("falsified" if n_gap <= 1 else "undecided")'),
 ("P8 top verdict = unmasked", 'p8 = {**out, "verdict": vm, "verdict_unmasked": vu', 'p8 = {**out, "verdict": vu, "verdict_unmasked": vm'),
 ("P8 RT median -> max", 'm = _median([rate(tb(r), rd) for r in rc])', 'm = max([rate(tb(r), rd) for r in rc])'),
]

if __name__ == "__main__":
    assert run(SRC) == [], run(SRC)
    alive, missing = [], []
    for name, old, new in M:
        if SRC.count(old) != 1:
            missing.append((name, SRC.count(old)))
            continue
        if not run(SRC.replace(old, new)):
            alive.append(name)
    print(f"{len(M)} mutants: {len(M) - len(alive) - len(missing)} killed, {len(alive)} alive, {len(missing)} not applicable")
    for a in alive:
        print("  ALIVE", a)
    for a in missing:
        print("  NOT FOUND", a)
