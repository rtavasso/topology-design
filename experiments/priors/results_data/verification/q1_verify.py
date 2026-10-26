"""Independent recomputation of Q1 (priors round) from raw results, following PREREGISTRATION.md.

usage: python q1_verify.py /home/claude/topology-design
"""
import gzip, json, glob, sys, math, statistics as stt
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/topology-design"
NP = 59 * 59
TH, HOLD = 0.95, 2000


def load(sub):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/results/{sub}/results/*.json.gz")):
        with gzip.open(f) as fh:
            out.append(json.load(fh))
    return out


def npairs(f):
    return int(round(f * NP))


def cell_of(tag):  # pQ1u-sym-base -> (sym, base)
    _, g, w = tag.split("-")
    return g, w


def sustained(traj, th=TH, hold=HOLD, budget=None):
    """first t with acc>=th at every eval in [t,t+hold], requiring observation to t+hold (and t+hold<=budget)."""
    steps = [r["step"] for r in traj]
    acc = [r["test_acc"] for r in traj]
    for i, t in enumerate(steps):
        if acc[i] < th or steps[-1] < t + hold:
            continue
        if budget is not None and t + hold > budget:
            return None
        if all(a >= th for s, a in zip(steps[i:], acc[i:]) if s <= t + hold):
            return t
    return None


def first_cross(traj, th=TH, budget=None):
    for r in traj:
        if budget is not None and r["step"] > budget:
            return None
        if r["test_acc"] >= th:
            return r["step"]
    return None


def thresh(levels_ok, frac_needed=None, k_needed=None):
    """levels_ok: dict level -> list[bool]. smallest level L such that at every level >= L the count of True
    meets the requirement. returns (L, censor) with censor in {None,'below','above'}"""
    levels = sorted(levels_ok)
    good = {}
    for L in levels:
        xs = levels_ok[L]
        k = sum(xs)
        if k_needed is not None:
            good[L] = k >= k_needed
        else:
            good[L] = len(xs) > 0 and k >= frac_needed * len(xs)
    best = None
    for L in reversed(levels):
        if good[L]:
            best = L
        else:
            break
    if best is None:
        return None, "above"
    if best == levels[0]:
        return best, "below"
    return best, None


# ------------------------------------------------------------------ ungrokking
def ungrok_rows(runs):
    rows = []
    for r in runs:
        c, s, tj = r["config"], r["summary"], r["trajectory"]
        sh = s.get("shrink")
        g, w = cell_of(c["tag"])
        if sh is None or sh.get("step") is None:
            rows.append(dict(g=g, w=w, seed=c["model_seed"], f=c["shrink_frac"], excluded=True))
            continue
        t0, L = sh["step"], sh["steps"]
        end = t0 + L
        post = [x for x in tj if x["step"] > t0]
        diverged = s.get("diverged_at") is not None
        last_step = tj[-1]["step"]
        last = [x for x in tj if end - HOLD <= x["step"] <= end]
        keep = (not diverged) and last_step >= end and len(last) > 0 and all(x["test_acc"] >= TH for x in last)
        # convergence
        def win(a, b):
            xs = [x["test_acc"] for x in tj if a < x["step"] <= b]
            return stt.mean(xs) if xs else float("nan")
        m_end, m_early = win(end - 2500, end), win(end - 17500, end - 15000)
        unconv = abs(m_end - m_early) > 0.1
        # median final over last 5000
        med5 = stt.median([x["test_acc"] for x in tj if end - 5000 <= x["step"] <= end])
        at_shrink = max((x for x in tj if x["step"] <= t0), key=lambda x: x["step"])
        after = [x for x in tj if x["step"] > t0]
        rows.append(dict(
            g=g, w=w, seed=c["model_seed"], f=c["shrink_frac"], n=sh["n_kept"], excluded=False, keep=keep,
            diverged=diverged, unconv=unconv, m_end=m_end, m_early=m_early, med5=med5,
            final=tj[-1]["test_acc"], t_shrink=t0, t_grok1=s.get("t_grok"),
            rf_shrink=at_shrink.get("rule_frac_test"), wn_shrink=at_shrink.get("weight_norms"),
            test_shrink=at_shrink["test_acc"],
            min_post=min(x["test_acc"] for x in after), traj=tj, end=end,
            # any >=0.95 window of 2000 steps in last 2000, with a threshold variant
            keep90=(not diverged) and all(x["test_acc"] >= 0.90 for x in last),
            keep_sus_any=sustained([x for x in tj if x["step"] > t0]) is not None,
        ))
    return rows


def scratch_rows(runs):
    rows = []
    for r in runs:
        c, s, tj = r["config"], r["summary"], r["trajectory"]
        g, w = cell_of(c["tag"])
        B = c["steps"]
        t_s = sustained(tj, budget=B)
        t_s_half = sustained(tj, budget=B // 2)
        t_fc = first_cross(tj, budget=B)
        t_fc_half = first_cross(tj, budget=B // 2)
        rows.append(dict(g=g, w=w, seed=c["model_seed"], f=c["train_frac"], n=npairs(c["train_frac"]), lr=c["lr"],
                         B=B, succ=t_s is not None, succ_half=t_s_half is not None, t_s=t_s,
                         fc=t_fc is not None, fc_half=t_fc_half is not None, t_fc=t_fc,
                         final=tj[-1]["test_acc"], steps_run=s["steps_run"], diverged=s.get("diverged_at"),
                         # did it lose the rule after first crossing (within the 5000 extra steps)?
                         lost_after=(t_fc is not None and any(x["test_acc"] < 0.95 for x in tj if x["step"] > (t_s or t_fc) + HOLD)),
                         min_after=(min([x["test_acc"] for x in tj if t_fc is not None and x["step"] >= t_fc] or [None]) if t_fc is not None else None)))
    return rows


def n_crit(rows, g, w, key="keep", **kw):
    lv = defaultdict(list)
    for x in rows:
        if x["g"] == g and x["w"] == w and not x["excluded"]:
            lv[x["n"]].append(x[key])
    if not kw:
        kw = dict(frac_needed=0.5)
    return thresh(lv, **kw), {L: (sum(v), len(v)) for L, v in sorted(lv.items())}


def n_floor(rows, g, w, key="succ", k_needed=2):
    lrs = sorted({x["lr"] for x in rows if x["g"] == g and x["w"] == w})
    per = {}
    counts = {}
    for lr in lrs:
        lv = defaultdict(list)
        for x in rows:
            if x["g"] == g and x["w"] == w and x["lr"] == lr:
                lv[x["n"]].append(x[key])
        per[lr] = thresh(lv, k_needed=k_needed)
        counts[lr] = {L: sum(v) for L, v in sorted(lv.items())}
    vals = [(v[0], lr) for lr, v in per.items() if v[0] is not None]
    if not vals:
        return (None, "above"), per, counts, None
    m = min(v for v, _ in vals)
    best = [lr for v, lr in vals if v == m]
    cen = per[best[0]][1]
    return (m, cen), per, counts, best


def fmtc(t):
    v, c = t
    if v is None:
        return f"censored {c}"
    return f"{v}" + (f" (censored {c})" if c else "")


if __name__ == "__main__":
    U = ungrok_rows(load("kaggle_q1u"))
    Sx = scratch_rows(load("kaggle_q1s"))
    print("excluded ungrok runs:", sum(x["excluded"] for x in U))
    cells = [("sym", "narrow"), ("sym", "base"), ("sym", "wide"), ("asym", "base")]
    res = {}
    for g, w in cells:
        (nc, cnt) = n_crit(U, g, w)
        nf, per, counts, best = n_floor(Sx, g, w)
        nfh, perh, countsh, besth = n_floor(Sx, g, w, key="succ_half")
        lrs = sorted(per)
        flag_edge = best is not None and lrs[-1] in best
        flag_half = nfh[0] != nf[0]
        S = nf[0] / nc[0] if (nf[0] and nc[0]) else None
        unconv = defaultdict(lambda: [0, 0])
        for x in U:
            if x["g"] == g and x["w"] == w:
                unconv[x["n"]][0] += x["unconv"]; unconv[x["n"]][1] += 1
        res[(g, w)] = dict(nc=nc, nf=nf, nfh=nfh, S=S, best=best, flag_edge=flag_edge, flag_half=flag_half)
        print(f"\n=== {g} {w} ===")
        print(" keep counts (level: kept/n):", cnt)
        print(" unconverged (level: k/n):", {k: tuple(v) for k, v in sorted(unconv.items()) if v[0]})
        print(" n_crit:", fmtc(nc))
        for lr in lrs:
            print(f"  lr {lr:g}: success {counts[lr]}  -> floor {fmtc(per[lr])} | half-budget {countsh[lr]} -> {fmtc(perh[lr])}")
        print(" n_floor:", fmtc(nf), "best lr", best, "| half budget n_floor:", fmtc(nfh), "best", besth)
        print(" flags: best lr at top edge:", flag_edge, "; floor moves at half budget:", flag_half)
        print(" S =", S, (f"search share = {math.log(S)/math.log(nf[0]/3):.3f}" if S else ""))
    # verdicts
    sb = res[("sym", "base")]
    c174 = [x["keep"] for x in U if x["g"] == "sym" and x["w"] == "base" and x["n"] == 174]
    c1218 = [x["keep"] for x in U if x["g"] == "sym" and x["w"] == "base" and x["n"] == 1218]
    p1 = sum(c174) < 0.5 * len(c174) and sum(c1218) >= 0.75 * len(c1218)
    print(f"\nP1: 174 kept {sum(c174)}/{len(c174)}, 1218 kept {sum(c1218)}/{len(c1218)} -> {'supported' if p1 else 'FALSIFIED'}")
    S = sb["S"]
    cen = sb["nc"][1] or sb["nf"][1]
    print(f"P2: S={S} censored={cen} -> " + ("untestable" if cen else ("supported" if S <= 3 else ("falsified" if S >= 4 else "inconclusive"))),
          "| replication check S>=1.5:", S is not None and S >= 1.5, "| provisional:", sb["flag_edge"] or sb["flag_half"])
    ab = res[("asym", "base")]
    if ab["nf"][0] and sb["nf"][0] and ab["nc"][0] and sb["nc"][0]:
        rc, rf = ab["nc"][0] / sb["nc"][0], ab["nf"][0] / sb["nf"][0]
        testable = ab["nf"][0] > sb["nf"][0] and not (ab["nc"][1] or sb["nc"][1] or ab["nf"][1] or sb["nf"][1])
        print(f"P3: floor asym {ab['nf'][0]} vs sym {sb['nf'][0]}; ratio n_crit {rc:.3f} vs ratio n_floor {rf:.3f} -> " +
              ("untestable" if not testable else ("supported" if rc < rf else ("inconclusive (tie)" if rc == rf else "falsified"))),
              "| provisional:", ab["flag_edge"] or ab["flag_half"] or sb["flag_edge"] or sb["flag_half"])
    else:
        print("P3: untestable (censored)", ab, sb)
    nw, nb, nn = res[("sym", "wide")]["nc"][0], res[("sym", "base")]["nc"][0], res[("sym", "narrow")]["nc"][0]
    p4 = nw <= nb <= nn and (nw < nb or nb < nn)
    print(f"P4: n_crit narrow {nn}, base {nb}, wide {nw} -> {'supported' if p4 else 'falsified'}",
          "(opposite ordering)" if nn <= nb <= nw and (nn < nb or nb < nw) else "")
    fl = {w: res[("sym", w)]["nf"] for w in ("narrow", "base", "wide")}
    p4b = all(v[0] is None or v[0] > 348 for v in fl.values())
    print(f"P4b: floors {fl} -> {'supported' if p4b else 'falsified'}")
    json.dump({"%s-%s" % k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
              open(__file__.replace(".py", "_out.json"), "w"), indent=1, default=str)
