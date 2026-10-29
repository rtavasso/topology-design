"""Derived before measuring, revision 4 (experiments/routefind/DESIGN.md §5; from design review 3, B1, whose script
this is): what a router that fits the PF target *pooled by the features it can read in one lookup* does.

    PYTHONPATH=src python experiments/routefind/derive/pooled.py [episodes per cell, default 150]


Feature class of a product state (symbol s, c, h) in an episode:
  kind: answer / dead / key;  own = flag of pair(s);  pred = any pair whose value is s is flagged (value-slot read,
  max over matches);  pdec = s is pointed to by >1 pair (ambiguous value-slot read);  c, h.
PF's fixed point under forward KL with a per-class tabular router = per-class normalized posterior counts.
Also: the same pooled target under the rung-1 readout (1/A on non-answers: EX's posterior at a uniform policy),
and the TB reward's floor mass.
"""
import math, sys, itertools
import numpy as np, torch
sys.path.insert(0, "src")
torch.set_num_threads(2)
from topology_design.route.data import RouteConfig, generate
from topology_design.route.exec import tables, oracle, committed, route_ok, PER_SYM, NH, HOP, PREV, SET, HALT
import topology_design.route.estimators as E

T, LAM, EPS = 14, 0.02, 0.02
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 150
ANAMES = "HOP PREV SET HALT".split()


def features(tb, cfg):
    B, S = tb["flag"].shape
    NS = S * PER_SYM
    ids = torch.arange(NS); s, c, h = ids // PER_SYM, (ids % PER_SYM) // NH, ids % NH
    own = tb["flag"][:, s]                                                    # (B, NS)
    pv, pf = tb["pair_val"], tb["pair_flag"]
    pred = torch.zeros(B, S, dtype=torch.long); npt = torch.zeros(B, S, dtype=torch.long)
    for b in range(B):
        for v, f in zip(pv[b].tolist(), pf[b].tolist()):
            npt[b, v] += 1
            if f: pred[b, v] = 1
    pred = pred[:, s]; amb = (npt[:, s] > 1).long()
    kind = torch.where(s < cfg.n_keys, 0, torch.where(s < cfg.DEAD, 1, 2)).unsqueeze(0).expand(B, NS)
    # class id
    cls = (((kind * 2 + own) * 2 + pred) * 3 + c.unsqueeze(0)) * 2 + h.unsqueeze(0)
    return cls, amb


def decode(k):
    h = k % 2; k //= 2; c = k % 3; k //= 3; pred = k % 2; k //= 2; own = k % 2; kind = k // 2
    return ["key", "ans", "dead"][kind], own, pred, c, h


def pooled_counts(cfg, rng_seed, Ly_fn=None, lam=LAM):
    ep = generate(cfg, NB, np.random.default_rng(rng_seed)); tb = tables(ep, cfg, "cpu")
    old = E.answer_likelihood
    if Ly_fn is not None: E.answer_likelihood = Ly_fn
    try:
        counts, Z = E.posterior_counts(E.flat_prior(tb, cfg), tb, tb["q_start"], tb["q_ans"], T, cfg, lam, EPS)
    finally:
        E.answer_likelihood = old
    cls, amb = features(tb, cfg)
    c = counts.sum(1)                                                          # (B, NS, 4)
    K = 3 * 2 * 2 * 3 * 2
    tot = torch.zeros(K, 4, dtype=torch.float64)
    tot.index_add_(0, cls.reshape(-1), c.reshape(-1, 4))
    return tot, tb, cls


def rung1_readout_Ly(sym, q_ans, cfg, eps):
    A = cfg.n_ans
    is_ans = (sym >= cfg.n_keys) & (sym < cfg.DEAD)
    return torch.where(sym == q_ans, torch.full(sym.shape, 1 - eps, dtype=torch.float64),
                       torch.where(is_ans, torch.full(sym.shape, eps / (A - 1), dtype=torch.float64),
                                   torch.full(sym.shape, 1.0 / A, dtype=torch.float64)))


def floor_Ly(floor):
    def f(sym, q_ans, cfg, eps):
        L = E.__dict__["_orig_al"](sym, q_ans, cfg, eps)
        return torch.clamp(L, min=floor)
    return f
E._orig_al = E.answer_likelihood


def show(tot, label, keys_only=True):
    print(f"\n--- pooled target by class: {label}")
    for k in range(tot.shape[0]):
        n = tot[k].sum()
        if n <= 1e-9: continue
        kind, own, pred, c, h = decode(k)
        if keys_only and kind != "key": continue
        p = (tot[k] / n).tolist()
        print(f"  own={own} pred={pred} c={c} h={h}  mass {float(n):9.2f}  " + "  ".join(f"{a} {x:.2f}" for a, x in zip(ANAMES, p))
              + f"   argmax {ANAMES[int(np.argmax(p))]}")


def eval_policy(tot, cfg, label, stoch=False):
    """Apply the per-class argmax (or soft) policy to fresh training and PP episodes."""
    res = []
    for kindw, ks in (("train", None), ("pp", (4, 6, 8))):
        for kk in ([None] if ks is None else ks):
            if kindw == "train":
                ep = generate(cfg, NB, np.random.default_rng(99))
            else:
                ep = generate(cfg, NB, np.random.default_rng(100 + kk), forced_k=kk, forced_kind="pp")
            tb = tables(ep, cfg, "cpu"); cls, _ = features(tb, cfg)
            n = tot.sum(-1, keepdim=True)
            soft = torch.where(n > 0, tot / n.clamp_min(1e-300), torch.tensor([1.0, 0, 0, 0], dtype=torch.float64))
            pol = soft[cls].float()
            if not stoch:
                pol = torch.nn.functional.one_hot(pol.argmax(-1), 4).float()
            oa, osteps, ost = oracle(tb, tb["q_start"], T if kindw == "train" else 30, cfg)
            TT = T if kindw == "train" else 30
            g = torch.Generator().manual_seed(0)
            s, st, a, x = committed(pol, tb, tb["q_start"], TT, sample=stoch, generator=g)
            okm = (s == tb["q_ans"]) & (st == osteps); ok = route_ok(a, st, oa, osteps); oka = s == tb["q_ans"]
            hasP = (tb["q_word"] == 1).any(-1); qk = tb["q_k"]
            if kindw == "train":
                for k in (3, 4, 5, 8):
                    m = hasP & (qk == k)
                    res.append(f"in{k}P min {okm[m].float().mean():.2f} orc {ok[m].float().mean():.2f} ans {oka[m].float().mean():.2f}")
                m = ~hasP
                res.append(f"noP min {okm[m].float().mean():.2f}")
            else:
                m = torch.zeros_like(hasP); m[:, 0] = True                     # forced query
                res.append(f"pp{kk} min {okm[m].float().mean():.2f} ans {oka[m].float().mean():.2f}")
    print(f"  [{label}{' sampled' if stoch else ' committed'}] " + " | ".join(res))


if __name__ == "__main__":
 for pool in (0, 4):
        n_ans = 16 if pool == 0 else f"16, pool {pool}"
        for d in (1, 0):
            cfg = RouteConfig(deferral=d, ans_pool=pool)
            tot, tb, cls = pooled_counts(cfg, 0)
            show(tot, f"PF target, deferral {d}, n_ans {n_ans}")
            eval_policy(tot, cfg, f"PF d{d} a{n_ans}")
            eval_policy(tot, cfg, f"PF d{d} a{n_ans}", stoch=True)
            if pool == 0:
                tot_r, _, _ = pooled_counts(cfg, 0, Ly_fn=rung1_readout_Ly)
                show(tot_r, f"EX-style posterior at the FLAT policy (rung-1 readout 1/A on non-answers), deferral {d}")
                eval_policy(tot_r, cfg, f"EXflat d{d}")
