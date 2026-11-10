"""Unscored companion to main_report.md: every arm on P2's strata (held-out binding instances that need the unseen combination),
and on their complement (held-out binding instances provable inside a training world contained in W). Post hoc: written after the
main grid's verdicts were read. Usage: PYTHONPATH=src python3 experiments/rewrite/results_data/comb_by_arm.py <results dir> <derived.json>"""
import json, sys
sys.path.insert(0, "experiments/rewrite"); sys.path.insert(0, "src")
import analyze as A

runs = A.load_results(sys.argv[1])
flags = json.load(open(sys.argv[2]))["p2_comb"]
PAIRS, TRIPLES = A.PAIRS_HELD, A.TRIPLES
ARMS = ["RT-canon", "RT-post", "RT-all", "EI16", "PF", "RT-examples", "RT-shuffled", "RT-absolute"]

def sel(run, worlds, want):
    out = []
    for pi in A.instances(run, "eval5", world=worlds, binding=True):
        f = flags.get(pi["world"], {}).get(A._key(pi), {})
        if bool(f.get("comb")) == want:
            out.append(pi)
    return out

def cell(arm, worlds, want, key):
    rs = A.by_seed(runs, arm)
    vals = [A.rate(sel(r, worlds, want), key) for r in rs]
    return A.summarize_seeds(vals), len(sel(rs[0], worlds, want))

rc0 = A.by_seed(runs, "RT-canon")[0]
print("| stratum | n | greedy | lookahead-3 | training-world ref | glued ref |")
print("|---|---|---|---|---|---|")
for nm, worlds in (("pairs", PAIRS), ("triples", TRIPLES)):
    for want in (True, False):
        rows = sel(rc0, worlds, want)
        tw = sum(bool(flags[p["world"]][A._key(p)]["trainw"]) for p in rows) / len(rows)
        gl = sum(bool(flags[p["world"]][A._key(p)].get("glue")) for p in rows) / len(rows)
        print(f"| {nm}, {'needs the combination' if want else 'provable in a contained training world'} | {len(rows)} | {A.ladder_rate(rows, 1):.2f} | {A.ladder_rate(rows, 3):.2f} | {tw:.2f} | {gl:.2f} |")
print()
print("| arm | pairs needs-comb masked | unmasked | pairs complement masked | unmasked | triples needs-comb masked | unmasked | triples complement masked | unmasked |")
print("|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    cells = []
    for worlds in (PAIRS, TRIPLES):
        for want in (True, False):
            for key in ("ok_m", "ok_u"):
                cells.append(cell(arm, worlds, want, key)[0])
    print(f"| {arm} | " + " | ".join(cells) + " |")
