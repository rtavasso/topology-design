"""Derived before measuring, revision 3 (experiments/routefind/DESIGN.md §5): what the fixed-prior posterior target
(PF exactly, TB16 by sampling) contains, at 16 answers and at the spurious-density dose (4 answers).

For each training query: the posterior probability of the oracle route (program A), of program B's route (SET at the
P-leaving state, PREV at once; equal to A under deferral 0), of any other route ending on the label (spurious), and of
routes ending on a wrong answer; and, at A's SET state on words with a P (the flagged predecessor, (0, 0)), how the
target splits between SET (A) and HOP (B). Flat prior: uniform at keys, HALT at answers; step cost lam; the answer
likelihood (1 - eps on the label, eps / (A - 1) on another answer, 0 on a key).

    PYTHONPATH=src python experiments/routefind/derive/signals.py
"""
import math, sys
import numpy as np, torch
sys.path.insert(0, "src")
torch.set_num_threads(2)
from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, oracle, committed, sid, PER_SYM, NH, HOP, PREV, SET, HALT
from topology_design.route.estimators import posterior_counts, flat_prior

T, LAM, EPS = 14, 0.02, 0.02


def program_b_policy(tb, oa, ost, cfg):
    """Program B as a deterministic policy: SET at a P-leaving state (h = 0, c = 0), PREV with a pending count, HALT at
    answers / dead, HOP otherwise. P-leaving states are read off the oracle (where it PREVs)."""
    B, S = tb["flag"].shape; NS = S * PER_SYM
    ids = torch.arange(NS); sym, c, h = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
    pol = torch.zeros(B, NS, 4)
    for b in range(B):
        pl = torch.zeros(S, dtype=torch.bool)
        m = oa[b] == PREV
        pl[(ost[b, :, :-1][m] // PER_SYM).unique()] = True
        a = torch.full((NS,), HOP)
        a[pl[sym] & (h == 0) & (c == 0)] = SET
        a[c >= 1] = PREV
        a[sym >= cfg.n_keys] = HALT
        pol[b, ids, a] = 1
    return pol


def route_prob(pi0, acts, states, cfg, lam):
    """prior x step cost along given routes (B, Q, T): prod pi0(a | x) exp(-lam [move])."""
    B = pi0.shape[0]
    st = states[..., :acts.shape[-1]]
    p = pi0[torch.arange(B).view(B, 1, 1).expand_as(st), st].gather(-1, acts.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    f = torch.where(acts < 0, torch.ones_like(p), p * torch.where(acts == HALT, 1.0, math.exp(-lam)))
    return f.prod(-1)


for n_ans in (16, 4):
    for deferral in (1, 0):
        cfg = RouteConfig(deferral=deferral, n_ans=n_ans)
        ep = generate(cfg, 300, np.random.default_rng(0)); tb = tables(ep, cfg, "cpu")
        oa, osteps, ost = oracle(tb, tb["q_start"], T, cfg)
        pi0 = flat_prior(tb, cfg)
        counts, Z = posterior_counts(pi0, tb, tb["q_start"], tb["q_ans"], T, cfg, LAM, EPS)
        counts0, Z0 = posterior_counts(pi0, tb, tb["q_start"], tb["q_ans"], T, cfg, LAM, 0.0)   # label-only mass
        p_right = (1 - EPS) * Z0 / Z
        pA = route_prob(pi0, oa, ost, cfg, LAM) * (1 - EPS) / Z
        sB, stB, aB, xB = committed(program_b_policy(tb, oa, ost, cfg), tb, tb["q_start"], T)
        okB = (sB == tb["q_ans"]) & (stB == osteps)
        isA = (aB == oa).all(-1)
        pB = torch.where(okB & ~isA, route_prob(pi0, aB, xB, cfg, LAM) * (1 - EPS) / Z, torch.zeros_like(pA))
        kk = tb["q_k"]; hasP = (tb["q_word"] == 1).any(-1)
        print(f"\n=== n_ans {n_ans}, deferral {deferral}: posterior mass by k (P-words / all-H words)")
        print("   k  | P-words: A    B    spurious-right  wrong-answer | all-H: A    spurious-right  wrong")
        for k in range(1, 9):
            for lab, m in (("P", (kk == k) & hasP), ("H", (kk == k) & ~hasP)):
                pass
            mP, mH = (kk == k) & hasP, (kk == k) & ~hasP
            f = lambda x, m: float(x[m].double().mean()) if m.any() else float("nan")
            print(f"   {k}  |  {f(pA, mP):.3f} {f(pB, mP):.3f}  {f(p_right - pA - pB, mP):.3f}           {f(1 - p_right, mP):.3f}"
                  f"        |  {f(pA, mH):.3f}  {f(p_right - pA, mH):.3f}          {f(1 - p_right, mH):.3f}")
        # the target at A's SET state (deferral 1: the flagged predecessor; deferral 0: the P-leaving state)
        B_, Q_ = oa.shape[:2]
        rows = []
        for b in range(B_):
            for q in range(Q_):
                for t in range(T):
                    if oa[b, q, t] == SET:
                        x = int(ost[b, q, t]); c = counts[b, q, x]
                        rows.append((float(c[SET] / c.sum()), float(c[HOP] / c.sum()), int(kk[b, q])))
        r = np.array(rows)
        print(f"   target at A's SET states (n = {len(r)}): P(SET) mean {r[:, 0].mean():.3f}, P(HOP) {r[:, 1].mean():.3f}; "
              f"by k: " + " ".join(f"k={k}:{r[r[:, 2] == k, 0].mean():.2f}/{r[r[:, 2] == k, 1].mean():.2f}" for k in range(3, 9) if (r[:, 2] == k).any()))


# ---- what a router that fits the PF target exactly would do: argmax of the episode's posterior action counts ----
print("\n=== committed route of the target's argmax policy (a router that fits the PF target exactly), per stratum")
from topology_design.route.exec import route_ok
for n_ans in (16, 4):
    for deferral in (1, 0):
        cfg = RouteConfig(deferral=deferral, n_ans=n_ans)
        ep = generate(cfg, 300, np.random.default_rng(1)); tb = tables(ep, cfg, "cpu")
        oa, osteps, ost = oracle(tb, tb["q_start"], T, cfg)
        counts, Z = posterior_counts(flat_prior(tb, cfg), tb, tb["q_start"], tb["q_ans"], T, cfg, LAM, EPS)
        c = counts.sum(1)                                                                  # (B, NS, 4): the episode's target
        unvisited = c.sum(-1) <= 0
        pol = torch.nn.functional.one_hot(c.argmax(-1), 4).float()
        pol[unvisited] = torch.tensor([1.0, 0, 0, 0])                                       # states the target never visits: HOP
        s, st, a, x = committed(pol, tb, tb["q_start"], T)
        ok = route_ok(a, st, oa, osteps); okm = (s == tb["q_ans"]) & (st == osteps)
        kk = tb["q_k"]; hasP = (tb["q_word"] == 1).any(-1)
        f = lambda v, m: float(v[m].float().mean()) if m.any() else float("nan")
        print(f"n_ans {n_ans}, deferral {deferral}: oracle-route acc P-words k=3/5/8 "
              + "/".join(f"{f(ok, (kk == k) & hasP):.2f}" for k in (3, 5, 8))
              + "; shortest-correct P-words " + "/".join(f"{f(okm, (kk == k) & hasP):.2f}" for k in (3, 5, 8))
              + "; all-H k=1/4/8 " + "/".join(f"{f(ok, (kk == k) & ~hasP):.2f}" for k in (1, 4, 8)))
