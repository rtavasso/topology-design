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


def first_dev(r):
    """Final in8 first-deviation histogram ("oracle>taken": queries) of the standard-configuration evaluation, and n."""
    e8 = clean_eval(r).get("in8", {})
    fd = e8.get("first_dev") or {}
    return (fd if isinstance(fd, dict) else {}), (e8.get("n") or sum(fd.values()) or 1)


def first_dev_mass(r, pred):
    """Fraction of the final in8 QUERIES (deviating or not) whose first deviation satisfies pred(oracle, taken); a router
    right on 93 % of words that halts on the rest scores 0.07, not 1."""
    fd, n = first_dev(r)
    return sum(v for k, v in fd.items() if ">" in k and pred(*k.split(">", 1))) / n


def start_of(r):
    """E (from an EX-committed router), L (from an EI16-committed one) or scratch."""
    ck = r["train"].get("init_from") or ""
    return "E" if "_EX_" in ck else ("L" if "_EI16_" in ck else "scratch")


def plus_one(r):
    """Program R on the clean evaluation: answer accuracy >= 0.95 at in4, in8 and none8 with the committed route one step
    longer than the oracle's on every stratum (RESULTS rung 2 §3)."""
    e = clean_eval(r)
    return all((e.get(st, {}).get("acc") or 0) >= 0.95 and abs((e.get(st, {}).get("q_steps") or 0) - (e.get(st, {}).get("oracle_steps") or 0) - 1.0) <= 0.15
               for st in ("in4", "in8", "none8"))


def outcome(r):
    """Phase-2 outcome under shift A (delta 0; the oracle route SETs at the flagged state and PREVs at (own flag, 1, 1)),
    read from the final standard-configuration evaluation and the first-deviation histogram (stage-2 review B3, and the
    stage-2 records themselves: the same key means different things from the two starts):
      moved      shortest-correct on P-words >= 0.95 at in4 and in8;
      moved~     reached >= 0.95 on the criterion curve (two consecutive points) and ended between 0.85 and 0.95;
      moved-R    program R (correct, one step longer everywhere): the relabeled decision was learned through R;
      halted     at least half the queries halt where the oracle continues (`*>HALT`): the halting collapse;
      locked     the committed program intact: from E the first deviation is the relabeled decision itself (PREV>HOP);
                 from L it is the missing SET at the flag (SET>HOP) with PREV still in place at the class (p >= 0.5);
      drifted    left the committed program without reaching the oracle's: from E, lost the SET at the flag (SET>HOP,
                 the all-H route); from L, SET>HOP with PREV lost; from scratch, SET>HOP is the all-H stall;
      other      none of the above (a run near the threshold, or a mixture)."""
    st = start_of(r)
    if r.get("data", {}).get("deferral", 1) == 1 and st in ("E", "L"):
        # unchanged-stream controls and shift B (delta 1, oracle E): did the committed program survive, switch, degrade or go
        prog = RF.prog_label(cleaned(r))
        if moved(r):
            return f"held-{prog}" if prog == st else f"to-{prog}"
        if first_dev_mass(r, lambda o, t: t == "HALT") >= 0.5:
            return "halted"
        e8 = clean_eval(r).get("in8", {})
        return "degraded" if (e8.get(MOVE_KEY) or 0) >= 0.5 else "lost"
    if moved(r):
        return "moved"
    if plus_one(r):
        return "moved-R"
    if move_step(r) is not None and (clean_eval(r).get("in8", {}).get(MOVE_KEY) or 0) >= 0.85:
        return "moved~"                      # reached the criterion on the curve (confirmed), ended at 0.85-0.95
    if first_dev_mass(r, lambda o, t: t == "HALT") >= 0.5:
        return "halted"
    prev_hop = first_dev_mass(r, lambda o, t: o == "PREV" and t == "HOP")
    set_hop = first_dev_mass(r, lambda o, t: o == "SET" and t == "HOP")
    p_prev = (r["curve"][-1].get("p_prev_class") if r["curve"] else None)
    if st == "E":
        if prev_hop >= 0.5:
            return "locked"
        if set_hop >= 0.5:
            return "drifted"
    elif st == "L":
        if set_hop >= 0.5:
            return "locked" if (p_prev is not None and p_prev >= 0.5) else "drifted"
    else:
        if set_hop >= 0.5:
            return "drifted"
    return "other"


def collapsed(r):
    """Kept for the phase-1 table: the halting collapse (EX seed 0)."""
    return first_dev_mass(r, lambda o, t: t == "HALT") >= 0.5


def locked(r):
    return outcome(r) == "locked"


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
        out("| arm | n | moved | moved~ | moved-R | locked | drifted | halted | other | time to move (steps) | (relabel events) | final in4 / in8 (mean) | program at end (E/L/mix/other) | pp8 at end | init routers |")
        out("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        pA = {g: rs for g, rs in p2.items() if rs[0].get("data", {}).get("deferral", 1) == 0}
        pB = {g: rs for g, rs in p2.items() if g not in pA}
        for g, rs in sorted(pA.items()):
            fin = lambda r, st: clean_eval(r).get(st, {})
            i4 = np.mean([fin(r, "in4").get(MOVE_KEY) or 0 for r in rs]); i8 = np.mean([fin(r, "in8").get(MOVE_KEY) or 0 for r in rs])
            lab = [RF.prog_label(cleaned(r)) for r in rs]
            pp8 = np.mean([(fin(r, "pp8").get("min_acc") or 0) for r in rs])
            inits = sorted({r["train"].get("init_from") or "scratch" for r in rs})
            oc = [outcome(r) for r in rs]
            ms = [move_step(r) for r in rs]
            ev_ = [relabel_events(r, m_) for r, m_ in zip(rs, ms)]
            budget = rs[0]["train"]["steps"] - rs[0]["train"].get("warmup", 0)
            cnt = lambda k: oc.count(k)
            out(f"| {g} | {len(rs)} | {cnt('moved')} | {cnt('moved~')} | {cnt('moved-R')} | {cnt('locked')} | {cnt('drifted')} | {cnt('halted')} | {cnt('other')} | {A.fmt_t95(ms, budget)} | {A.fmt_t95(ev_, relabel_events(rs[0], budget))} | {i4:.2f} / {i8:.2f} | "
                f"{lab.count('E')}/{lab.count('L')}/{lab.count('mix')}/{lab.count('other') + lab.count('–')} | {pp8:.2f} | {len(inits)} |")
        out("\n**Unchanged-stream controls and shift B (δ = 1; the oracle is E).** Outcomes: held-L / held-E (the committed program, still ≥ 0.95 on clean strata), to-E / to-L (switched program, ≥ 0.95), degraded (0.5–0.95), lost (< 0.5, not halting), halted.\n")
        out("| arm | n | held | switched | degraded | lost | halted | final in4 / in8 on clean strata (mean) | program at end (E/L/mix/other) | pp8 on clean strata | init routers |")
        out("|---|---|---|---|---|---|---|---|---|---|---|")
        for g, rs in sorted(pB.items()):
            fin = lambda r, st: clean_eval(r).get(st, {})
            i4 = np.mean([fin(r, "in4").get(MOVE_KEY) or 0 for r in rs]); i8 = np.mean([fin(r, "in8").get(MOVE_KEY) or 0 for r in rs])
            lab = [RF.prog_label(cleaned(r)) for r in rs]
            pp8 = np.mean([(fin(r, "pp8").get("min_acc") or 0) for r in rs])
            oc = [outcome(r) for r in rs]
            out(f"| {g} | {len(rs)} | {sum(o.startswith('held') for o in oc)} | {sum(o.startswith('to-') for o in oc)} | {oc.count('degraded')} | {oc.count('lost')} | {oc.count('halted')} | {i4:.2f} / {i8:.2f} | "
                f"{lab.count('E')}/{lab.count('L')}/{lab.count('mix')}/{lab.count('other') + lab.count('–')} | {pp8:.2f} | {len({r['train'].get('init_from') for r in rs})} |")
        out("\nPer run: outcome; final first deviation at in8 (oracle>taken: queries of 250); the criterion curve's end state at the relabeled classes "
            "(p(PREV) at (own flag, 1, 1), p(SET) at (own flag, 0, 0), margins HOP−PREV and HOP−SET) and the curve's maximum of shortest-correct on P-words:\n")
        for g, rs in sorted(p2.items()):
            for r in sorted(rs, key=lambda r: r["train"]["seed"]):
                fd, _ = first_dev(r)
                fd = dict(sorted(fd.items(), key=lambda kv: -kv[1])[:3])
                c = r["curve"][-1] if r["curve"] else {}
                f2 = lambda v: "–" if v is None else f"{v:.2f}"
                mx = max((x.get("min_acc_P") or 0) for x in r["curve"]) if r["curve"] else 0
                out(f"- {g} s{r['train']['seed']} ← {r['train'].get('init_from') or 'scratch'}: **{outcome(r)}**; {fd}; p(PREV) {f2(c.get('p_prev_class'))}, p(SET) {f2(c.get('p_set_class'))}, "
                    f"HOP−PREV {f2(c.get('margin_hop_prev'))}, HOP−SET {f2(c.get('margin_hop_set'))}; max shortest-correct {mx:.2f}")
        if gate:
            def gate_row(r):
                ck = r["train"].get("init_from")
                name = next((n for n, gg in gate["gate"].items() if gg["ckpt"] == ck), None)
                return gate["sees"].get(name, {}).get("A") if name else None
            def flip_step(r, thr=0.5):
                """First curve step at which the run's own beam keeps the oracle route on >= thr of criterion P-words."""
                for c in r["curve"]:
                    if (c.get("search") or {}).get("oracle_kept_P", 0) >= thr:
                        return c["step"] - r["train"].get("warmup", 0)
                return None
            out("\n### EI16: coverage at the checkpoint against movement (shift A; coverage = the beam keeps the δ0 oracle route, on P-words the committed route does not already solve)\n")
            out("The in-run column is the same quantity on the run's own criterion set at step 0; `flip` is the first step at which the run's beam keeps the oracle route on ≥ 0.5 of criterion P-words (the merge tie reversing); `move` is the first confirmed step with shortest-correct ≥ 0.95.\n")
            out("| cell | seed | start | gate coverage | in-run (step 0) | flip (step) | move (step) | outcome | HOP−PREV at checkpoint |")
            out("|---|---|---|---|---|---|---|---|---|")
            cov_rows = []
            for g, rs in sorted(p2.items()):
                if not g.startswith("EI16") or "_EA" not in g and "_LA" not in g:
                    continue
                for r in sorted(rs, key=lambda r: r["train"]["seed"]):
                    a = gate_row(r) or {}
                    key = "beam_eps0.3" if g.startswith("EI16e3") else "beam_eps0.1"
                    cov = a.get(key + "_unsolved", a.get(key))
                    si = r.get("search_init") or {}
                    mg = (a.get("margin_hop_minus_prev") or {}).get("mean")
                    f2 = lambda v: "–" if v is None else f"{v:.2f}"
                    out(f"| {g} | s{r['train']['seed']} | {start_of(r)} | {f2(cov)} | {f2(si.get('oracle_kept_P'))} | {flip_step(r)} | {move_step(r)} | {outcome(r)} | {('–' if mg is None else f'{mg:+.0f}')} |")
                    cov_rows.append((cov, outcome(r)))
            z = [o for c, o in cov_rows if c is not None and c < 0.005]; nz = [o for c, o in cov_rows if c is not None and c >= 0.005]
            out(f"\nCoverage exactly zero (< 0.005): {sum(o == 'moved' for o in z)}/{len(z)} moved; coverage ≥ 0.005: {sum(o == 'moved' for o in nz)}/{len(nz)} moved "
                f"(the preregistered rule's thresholds were 0.05 and 0.3).")
            out("\n### The other signals from E: what the target saw and what happened\n")
            out("| cell | seed | 16-sample coverage | label-reaching samples (step 0) | own posterior | outcome | move (step) |")
            out("|---|---|---|---|---|---|---|")
            for g, rs in sorted(p2.items()):
                if g.startswith(("EI16", "RT")) or ("_EA" not in g and "_LA" not in g):
                    continue
                for r in sorted(rs, key=lambda r: r["train"]["seed"]):
                    a = gate_row(r) or {}
                    si = r.get("search_init") or {}
                    f2 = lambda v: "–" if v is None else f"{v:.2f}"
                    out(f"| {g} | s{r['train']['seed']} | {f2(a.get('samples_K16_unsolved', a.get('samples_K16')))} | {f2(si.get('hit_P'))} | {f2(a.get('own_posterior_on_oracle_unsolved', a.get('own_posterior_on_oracle')))} | {outcome(r)} | {move_step(r)} |")
        out("\n### PF from E: the margin at the relabeled class drops to ≈ +1 within 100 steps and crosses later, or not\n")
        out("| cell | seed | HOP−PREV at checkpoint | at step 100 | at the end | p(PREV) at the end | crossing (first step with margin < 0) | move (step) | outcome |")
        out("|---|---|---|---|---|---|---|---|---|")
        for g, rs in sorted(p2.items()):
            if not g.startswith("PF_") or ("_EA" not in g and "_LA" not in g):
                continue
            for r in sorted(rs, key=lambda r: r["train"]["seed"]):
                a = (gate_row(r) if gate else None) or {}
                mg0 = (a.get("margin_hop_minus_prev") or {}).get("mean")
                c = r["curve"]; w = r["train"].get("warmup", 0)
                m100 = next((x.get("margin_hop_prev") for x in c if x["step"] >= w), None)
                cross = next((x["step"] - w for x in c if x.get("margin_hop_prev") is not None and x["margin_hop_prev"] < 0), None)
                f2 = lambda v: "–" if v is None else f"{v:+.1f}"
                out(f"| {g} | s{r['train']['seed']} | {f2(mg0)} | {f2(m100)} | {f2(c[-1].get('margin_hop_prev'))} | {('–' if c[-1].get('p_prev_class') is None else f'{c[-1]['p_prev_class']:.2f}')} | {cross} | {move_step(r)} | {outcome(r)} |")
        out("\n### EX from E: the halting collapse by dose, against the unchanged-stream controls\n")
        for g in ("EX_EA30", "EX_EA15", "EX_EA08", "EX_ES", "EX_d0", "IW16_EA30", "IW16_EA15", "IW16_EA08"):
            rs = p2.get(g)
            if rs:
                oc = [outcome(r) for r in rs]
                out(f"- {g}: halted {oc.count('halted')}/{len(rs)}; moved {oc.count('moved')}; moved-R {oc.count('moved-R')}; locked {oc.count('locked')}; drifted {oc.count('drifted')}; other {oc.count('other')}")
        out("\n### From the L routers: outcome by router and signal (shift A)\n")
        sigs = ["RT", "PF", "EX", "EI16", "PFIS16", "IW16"]
        seeds_L = sorted({r["train"]["seed"] for g, rs in p2.items() if "_LA" in g for r in rs})
        out("| router | " + " | ".join(sigs) + " | HOP−SET at checkpoint; p(PREV) at class |")
        out("|---|" + "---|" * (len(sigs) + 1))
        for sd in seeds_L:
            cells = []
            a = {}
            for sig in sigs:
                rs = [r for r in p2.get(f"{sig}_LA30", []) if r["train"]["seed"] == sd]
                if rs:
                    r = rs[0]; a = (gate_row(r) if gate else None) or a
                    cells.append(f"{outcome(r)} ({move_step(r)})" if move_step(r) is not None else outcome(r))
                else:
                    cells.append("–")
            ms = (a.get("margin_hop_minus_set") or {}).get("mean")
            out(f"| L s{sd} | " + " | ".join(cells) + f" | {('–' if ms is None else f'{ms:+.0f}')}; {('–' if a.get('p_prev_at_class') is None else f'{a['p_prev_at_class']:.2f}')} |")
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
                i8s = [clean_eval(r).get("in8", {}).get("min_acc_P") for r in rs]
                labs = [RF.prog_label(cleaned(r)) for r in rs]
                out(f"- {g}: filled {sum(filled(r) for r in rs)}/{len(rs)} (pp8 {', '.join('–' if v is None else f'{v:.2f}' for v in pp8s)}); ordinary P-words at in8 on clean strata {', '.join('–' if v is None else f'{v:.2f}' for v in i8s)}; program {'/'.join(labs)}; placement {dict(sorted(hist.items(), key=lambda kv: -kv[1]))}")
    if args.md:
        open(args.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
