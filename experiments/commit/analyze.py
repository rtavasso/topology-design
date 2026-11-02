"""Analysis of rung-3 results (experiments/commit/DESIGN.md §§4, 7).

    PYTHONPATH=src python experiments/commit/analyze.py results/kaggle_route-cm_stage1 [results/kaggle_route-cm_stage2] \
        [--gate results/kaggle_route-cm_stage1/gate.json] [--md out.md]

Stage 1: the commitment gate per phase-1 run (from the gate report when given) and the core-B learning table. Stage 2:
movement under each shift (shortest-correct on P-words at in4 / in8 under the new data, time to >= 0.95 from the end of
the phase-2 warm-up, program trajectory), per-seed coverage -> movement, the dose, filling in (pp8 by set placement),
and the stability controls. Verdicts are descriptive; DESIGN.md §7 fixes the thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "route"))
import analyze as A  # noqa: E402  (rung 1 helpers; routefind's analyze is loaded by path below, not by name)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("rf", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "routefind", "analyze.py"))
RF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RF)   # rung 2 helpers: learned, crit_step, prog_label, plus_one

MOVE_KEY = "min_acc_P"
FILL_THR = 0.90          # PA'1 filling-in endpoint on PP words (pp8 shortest-correct; the one-lookup policy reaches 0.90)
PP_TYPES = {"2:2-0-0-0-0": "E-double", "2:1-1-0-0-0": "E-E", "2:1-0-1-0-0": "E-L", "2:0-2-0-0-0": "L-double", "2:0-1-1-0-0": "L-L"}


def clean_eval(r):
    """The standard-configuration evaluation when the run recorded one (training stream != standard), else the evaluation."""
    ev = r["evals"][-1]
    return ev.get("clean") or ev["eval"]


def cleaned(r):
    """A view of the record whose final evaluation is the clean one, for the rung-1/2 helpers that read A.final(r)."""
    if not r["evals"][-1].get("clean"):
        return r
    return {**r, "evals": r["evals"][:-1] + [{**r["evals"][-1], "eval": r["evals"][-1]["clean"]}]}


def moved(r, thr=0.95):
    """Phase-2 outcome: shortest-correct committed accuracy on P-words >= thr at in4 and in8 at the end (clean eval when present)."""
    e = clean_eval(r)
    return all((e.get(st, {}).get(MOVE_KEY) or 0) >= thr for st in ("in4", "in8"))


def first_dev_mass(r, pred):
    """Fraction of the final in8 queries whose first deviation from the oracle route ("oracle>taken") satisfies
    pred(oracle, taken); the denominator is the stratum's query count (deviating or not), so a router that is right
    on 93 % of words and halts on the rest scores 0.07, not 1."""
    e8 = clean_eval(r).get("in8", {})
    fd = e8.get("first_dev") or {}
    if not isinstance(fd, dict):
        return None
    tot = e8.get("n") or sum(fd.values())
    return sum(v for k, v in fd.items() if ">" in k and pred(*k.split(">", 1))) / tot if tot else None


def collapsed(r):
    """Stage-2 review B3: the router left its committed solution for halting (EX seed 0 did so spontaneously at phase 1
    step ~2,000 at the constant lr): at least half of the first deviations at in8 are halts where the oracle continues, or
    P-word answer accuracy at in8 is below 0.5 with the committed route shorter than half the oracle's. Neither lock-in nor
    blindness; reported separately and excluded from the mover counts."""
    hm = first_dev_mass(r, lambda o, t: t == "HALT")
    e8 = clean_eval(r).get("in8", {})
    short = (e8.get("acc_P") or 0) < 0.5 and (e8.get("q_steps_P") or 0) < 0.5 * (e8.get("oracle_steps_P") or 1)
    return bool((hm is not None and hm >= 0.5) or short)


def locked(r):
    """Not moved, not collapsed, and the first deviation is the relabeled decision itself (oracle PREV, router HOP) on at
    least half the in8 P-words: the committed program, intact, under the new data."""
    if moved(r) or collapsed(r):
        return False
    pm = first_dev_mass(r, lambda o, t: o == "PREV" and t == "HOP")
    return bool(pm is not None and pm >= 0.5)


def outcome(r):
    return "moved" if moved(r) else ("collapsed" if collapsed(r) else ("locked" if locked(r) else "other"))


def filled(r, thr=FILL_THR):
    """Shift-B endpoint (PA'1): shortest-correct committed accuracy on PP words at pp8 >= thr (clean evaluation)."""
    return (clean_eval(r).get("pp8", {}).get("min_acc") or 0) >= thr


_PFRAC = {}


def p_word_frac(dc):
    """Realized fraction of training queries that are P-words under a data config (for relabel events; cached per config)."""
    key = json.dumps(dc, sort_keys=True)
    if key not in _PFRAC:
        from topology_design.route.data import RouteConfig, generate
        ep = generate(RouteConfig(**dc), 400, np.random.default_rng(0))
        _PFRAC[key] = float((ep.q_word == 1).any(-1).mean())
    return _PFRAC[key]


def relabel_events(r, steps):
    """Steps -> P-word queries seen (batch x Qn x realized P-word fraction): the dose in relabel events, not steps."""
    if steps is None:
        return None
    return int(round(steps * r["train"]["batch"] * r["data"]["Qn"] * p_word_frac(r["data"])))


def move_step(r, thr=0.95):
    """First held-out step (from the end of the phase-2 warm-up) with every k <= 8 at shortest-correct >= thr, confirmed."""
    w = r["train"].get("warmup", 0)
    if r["curve"] and "min_acc_P" in r["curve"][-1]:            # rung 3: shortest-correct on P-words of the criterion set
        pts = [(c["step"], c["min_acc_P"] if c["min_acc_P"] is not None else 0.0) for c in r["curve"]]
    else:
        pts = [(c["step"], min((c.get("min_acc_by_k") or {}).values(), default=0.0)) for c in r["curve"] if c.get("min_acc_by_k")]
    for i in range(len(pts) - 1):
        if pts[i][1] >= thr and pts[i + 1][1] >= thr:
            return max(0, pts[i][0] - w)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--gate", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    recs = A.load(args.folders)
    groups = {}
    for r in recs:
        groups.setdefault(r.get("arm") or "?", []).append(r)
    gate = json.load(open(args.gate)) if args.gate else None
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"# Rung-3 results: {len(recs)} runs\n")

    p1 = {g: rs for g, rs in groups.items() if g.endswith("_p1")}
    if p1:
        out("## Phase 1: commitment (clean evaluation; the gate of DESIGN §3)\n")
        out("| arm | n | learned | E / L / mixed-or-other / R | collapsed (learned, then left for halting) | criterion | deterministic | gate passes |")
        out("|---|---|---|---|---|---|---|---|")
        for g, rs in sorted(p1.items()):
            lab = [RF.prog_label(r) for r in rs if RF.learned(r)]
            nE, nL = lab.count("E"), lab.count("L"); nX = len(lab) - nE - nL
            nR = sum(1 for r in rs if not RF.learned(r) and RF.plus_one(r))
            col = [f"s{r['train']['seed']} (criterion at {(lambda m_: None if m_ is None else m_ + r['train'].get('warmup', 0))(move_step(r))}, collapsed by {next((c['step'] for c in r['curve'] if c['step'] > 500 and (c.get('min_acc_P') or 0) < 0.5 and any((cc.get('min_acc_P') or 0) >= 0.95 for cc in r['curve'] if cc['step'] < c['step'])), '–')})"
                   for r in rs if not RF.learned(r) and collapsed(r)]
            det = sum(1 for r in rs if (A.final(r).get("in8", {}).get("samp_min_acc_P") or 0) >= 0.95)
            gp = sum(1 for r in rs if gate and gate["gate"].get(r["name"], {}).get("passes")) if gate else "–"
            out(f"| {g} | {len(rs)} | {sum(RF.learned(r) for r in rs)} | {nE} / {nL} / {nX} / {nR} | {', '.join(col) or '0'} | {A.fmt_t95([RF.crit_step(r) for r in rs], rs[0]['train']['steps'])} | {det} | {gp} |")
        if gate:
            out("\nWhat each committed router's phase-2 targets can see (DESIGN §5.3): shift A = the δ0 oracle route on P-words; shift B = a shortest correct completion on PP words. "
                "Coverage on the P-words the committed route does not already solve (stage-2 review B1), with the all-words value in brackets where it differs; n = words / unsolved.\n")
            out("| router | start | n | beam ε0.1 keeps oracle / oracle among completions / kept shortest (set_late) | beam ε0.3 keeps oracle | 16 samples | own posterior | margin HOP−PREV; p(PREV) | margin HOP−SET; p(SET) | B: n, beam ε0.1 / samples |")
            out("|---|---|---|---|---|---|---|---|---|---|")
            for name, sees in sorted(gate["sees"].items()):
                a, b = sees["A"], sees["B"]
                f = lambda v: "–" if v is None else f"{v:.2f}"
                def fu(d, k):
                    v, u = d.get(k), d.get(k + "_unsolved")
                    return f(v) if u is None else (f(u) if f(u) == f(v) else f"{f(u)} [{f(v)}]")
                mg = lambda d: "–" if not d else f"{d['mean']:+.1f} [{d['min']:+.1f}, {d['max']:+.1f}]"
                out(f"| {name} | {gate['gate'][name]['start']} | {a.get('n_words')} / {a.get('n_unsolved', '–')} | {fu(a, 'beam_eps0.1')} / {fu(a, 'beam_eps0.1_oracle_among_completions')} / {fu(a, 'beam_eps0.1_kept_shortest')} ({f(a.get('beam_eps0.1_kept_set_late'))}) | "
                    f"{fu(a, 'beam_eps0.3')} | {fu(a, 'samples_K16')} | {fu(a, 'own_posterior_on_oracle')} | {mg(a.get('margin_hop_minus_prev'))}; {f(a.get('p_prev_at_class'))} | {mg(a.get('margin_hop_minus_set'))}; {f(a.get('p_set_at_class'))} | {b.get('n_words')} / {b.get('n_unsolved', '–')}, {fu(b, 'beam_eps0.1')} / {fu(b, 'samples_K16')} |")

    cb = {g: rs for g, rs in groups.items() if g.endswith(("_d1", "_d0", "_d1p4")) and not g.startswith(("EX_", "EI16_", "PF_", "PFIS16_d0"))}
    cb.update({g: rs for g, rs in groups.items() if g in ("PFIS16_d1p4",)})
    if cb:
        out("\n## Core B: exploration for the reward signals (from scratch, constant lr, 5k)\n")
        out("| arm | n | learned | program | criterion | distinct routes / beam hit (last 1k) | ESS / explained variance |")
        out("|---|---|---|---|---|---|---|")
        for g, rs in sorted(cb.items()):
            lab = [RF.prog_label(r) for r in rs if RF.learned(r)]
            last = lambda r, k: np.mean([c["diag"][k] for c in r["curve"] if c.get("diag") and k in c["diag"] and c["step"] > r["train"]["steps"] - 1000] or [np.nan])
            d1 = [last(r, "distinct_P") for r in rs]; bh = [last(r, "beam_hit") for r in rs]
            ess = [last(r, "ess") for r in rs]; ev = [last(r, "ev") for r in rs]
            fr = lambda v: "–" if np.all(np.isnan(v)) else f"{np.nanmin(v):.2f}…{np.nanmax(v):.2f}"
            out(f"| {g} | {len(rs)} | {sum(RF.learned(r) for r in rs)} | {'/'.join(lab) or '–'} | {A.fmt_t95([RF.crit_step(r) for r in rs], rs[0]['train']['steps'])} | {fr(d1)} / {fr(bh)} | {fr(ess)} / {fr(ev)} |")

    p2 = {g: rs for g, rs in groups.items() if any(tag in g for tag in ("_EA", "_LA", "_ES", "_LS", "_LB")) or g.endswith("_d0") and g.startswith(("EX_", "EI16_", "PF_", "PFIS16_"))}
    if p2:
        out("\n## Phase 2: movement after commitment (shortest-correct on P-words under the new data, standard-configuration evaluation; time from the end of warm-up)\n")
        out("Outcomes per run: moved (≥ 0.95 at in4 and in8) / locked (first deviation is the relabeled decision, oracle PREV → router HOP) / collapsed (halting where the oracle continues; stage-2 review B3) / other. "
            "Mover counts are over runs that did not collapse; collapses are reported separately. Time to move also in relabel events (P-word queries seen = steps × batch × Qn × realized P-word fraction).\n")
        out("| arm | n | moved / not-collapsed | locked | collapsed | time to move (steps) | (relabel events) | final in4 / in8 (mean) | program at end (E/L/mix/other) | pp8 at end | init routers |")
        out("|---|---|---|---|---|---|---|---|---|---|---|")
        for g, rs in sorted(p2.items()):
            fin = lambda r, st: clean_eval(r).get(st, {})
            i4 = np.mean([fin(r, "in4").get(MOVE_KEY) or 0 for r in rs]); i8 = np.mean([fin(r, "in8").get(MOVE_KEY) or 0 for r in rs])
            lab = [RF.prog_label(cleaned(r)) for r in rs]
            pp8 = np.mean([(fin(r, "pp8").get("min_acc") or 0) for r in rs])
            inits = sorted({r["train"].get("init_from") or "scratch" for r in rs})
            oc = [outcome(r) for r in rs]
            nc = sum(o != "collapsed" for o in oc)
            ms = [move_step(r) for r in rs if not collapsed(r)]
            ev_ = [relabel_events(r, m_) for r, m_ in zip([r for r in rs if not collapsed(r)], ms)]
            budget = rs[0]["train"]["steps"] - rs[0]["train"].get("warmup", 0)
            out(f"| {g} | {len(rs)} | {oc.count('moved')}/{nc} | {oc.count('locked')} | {oc.count('collapsed')} | {A.fmt_t95(ms, budget)} | {A.fmt_t95(ev_, relabel_events(rs[0], budget))} | {i4:.2f} / {i8:.2f} | "
                f"{lab.count('E')}/{lab.count('L')}/{lab.count('mix')}/{lab.count('other') + lab.count('–')} | {pp8:.2f} | {len(inits)} |")
        out("\nFinal first deviation at in8 per run (oracle>taken: counts), with the run's outcome:\n")
        for g, rs in sorted(p2.items()):
            for r in sorted(rs, key=lambda r: r["train"]["seed"]):
                fd = clean_eval(r).get("in8", {}).get("first_dev")
                fd = dict(sorted(fd.items(), key=lambda kv: -kv[1])[:4]) if isinstance(fd, dict) else fd
                out(f"- {g} s{r['train']['seed']} ← {r['train'].get('init_from') or 'scratch'}: {outcome(r)}; {fd}")
        if gate:
            out("\n### Per-seed: what the target saw at the checkpoint against whether the run moved (shift A; coverage on the P-words the committed route does not already solve, stage-2 review B1)\n")
            out("Gate coverage per target: EI16 = beam ε0.1 keeps the oracle route; EI16e3 = beam ε0.3; IW16 / PF-IS16 = 16 samples contain it; EX = own posterior on it; PF / RT = 1 by construction. "
                "The in-run number is the same quantity on the run's own criterion set at step 0 (`search_init`), as a consistency check.\n")
            rows = []
            for g, rs in sorted(p2.items()):
                if "_EA" not in g and "_LA" not in g:
                    continue
                sig = g.split("_")[0]
                for r in rs:
                    ck = r["train"].get("init_from")
                    name = next((n for n, gg in gate["gate"].items() if gg["ckpt"] == ck), None)
                    if not name or name not in gate["sees"]:
                        continue
                    a = gate["sees"][name]["A"]
                    u = lambda k: a.get(k + "_unsolved", a.get(k))
                    cov = {"EI16": u("beam_eps0.1"), "EI16e3": u("beam_eps0.3"), "PFIS16": u("samples_K16"), "IW16": u("samples_K16"),
                           "EX": u("own_posterior_on_oracle"), "PF": 1.0, "RT": 1.0}.get(sig)
                    si = r.get("search_init") or {}
                    inrun = {"EI16": si.get("oracle_kept_P"), "EI16e3": si.get("oracle_kept_P"), "PFIS16": si.get("oracle_any_P"), "IW16": si.get("oracle_any_P")}.get(sig)
                    rows.append((g, r["train"]["seed"], sig, cov, inrun, si.get("hit_P"), outcome(r), move_step(r)))
            f2 = lambda v: "–" if v is None else f"{v:.2f}"
            for g, sd, sig, cov, inrun, hit, oc, ms in rows:
                out(f"- {g} s{sd}: gate coverage {f2(cov)}" + (f" (in-run {f2(inrun)}, label-reaching {f2(hit)})" if inrun is not None else "") + f" → {oc} ({ms})")

            def auroc(pts):
                pos = [c for c, m in pts if m]; neg = [c for c, m in pts if not m]
                if len(pts) < 6 or not pos or not neg:
                    return None
                return float(np.mean([[1.0 if p > n else (0.5 if p == n else 0.0) for n in neg] for p in pos]))
            live = [(c, oc == "moved") for _, _, sig, c, _, _, oc, _ in rows if c is not None and oc != "collapsed"]
            within = [(c, oc == "moved") for _, _, sig, c, _, _, oc, _ in rows if c is not None and oc != "collapsed" and sig in ("EI16", "EI16e3")]
            nopf = [(c, oc == "moved") for _, _, sig, c, _, _, oc, _ in rows if c is not None and oc != "collapsed" and sig not in ("PF", "RT")]
            for lab_, pts in (("within EI16 (both ε, both starts, the doses)", within), ("pooled without PF / RT", nopf), ("pooled, all signals", live)):
                a_ = auroc(pts)
                out(f"\nAUROC of coverage for movement, {lab_}: {('–' if a_ is None else f'{a_:.2f}')} over {len(pts)} non-collapsed runs")
        pp = {g: rs for g, rs in p2.items() if "_LB" in g}
        if pp:
            out(f"\n### Filling in (shift B): PA'1 endpoint pp8 shortest-correct ≥ {FILL_THR} (clean evaluation), and set placement on the shortest-correct PP routes\n")
            out("Route types for a run of two P's from the placement counts (predecessor-first-second-third-elsewhere): E-double 2-0-0-0-0, E-E 1-1-0-0-0, E-L 1-0-1-0-0, L-double 0-2-0-0-0, L-L 0-1-1-0-0; longer runs and other patterns are left as recorded.\n")
            for g, rs in sorted(pp.items()):
                hist = {}
                for r in rs:
                    for k, v in (clean_eval(r).get("pp8", {}).get("pp_set_placement") or {}).items():
                        hist[PP_TYPES.get(k, k)] = hist.get(PP_TYPES.get(k, k), 0) + v
                pp8s = [clean_eval(r).get("pp8", {}).get("min_acc") for r in rs]
                out(f"- {g}: filled {sum(filled(r) for r in rs)}/{len(rs)} (pp8 {', '.join('–' if v is None else f'{v:.2f}' for v in pp8s)}); placement {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
    if args.md:
        open(args.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
