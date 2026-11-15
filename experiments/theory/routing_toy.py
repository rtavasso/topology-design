"""EXPLORATORY. A toy for docs/routing_signal_stability.md: which training signal learns a program of learned blocks.

The toy (the "conjunctive toy" of the note, section 4). n passes, K = n blocks. Block k has a competence vector
q_k = softmax(theta_k) over the n operations (competences sum to 1: a block cannot be good at everything, so the modular
solution is necessary). A route z = (z_1..z_n) names the block used at each pass; it succeeds with probability
s(z) = prod_t q[z_t, t]. The router is one softmax per pass (it knows the pass index, nothing else). p = E_pi[s(z)].
Modular solution: p = 1.  Unspecialised start: p = n^-n.

Signals (all with plain gradient steps of the same size, same initialisation per seed):
  EXACT-R  exact gradient of the expected loss 1 - p
  EXACT-L  exact gradient of -log p   (the route marginal; its router target is the router's own posterior)
  RF       REINFORCE on the loss 1 - s(z), M sampled routes, no baseline; blocks by the pathwise gradient on the samples
  RF-B     the same with the batch-mean baseline
  OWN-S    own posterior by self-normalised importance weights over the M samples (weights proportional to s(z))
  FIX      exact posterior under a fixed uniform prior over routes; router and blocks regress to it

Predictions written before the run (13-15 Nov 2026):
  P-a  EXACT-L and FIX reach p > 0.9 at every n in a number of steps that grows about linearly with K, not with n^n.
  P-b  EXACT-R's steps to p > 0.9 grow about as 1/p0 = n^n (the note's Hessian identity).
  P-c  RF fails at n >= 3 within the budget; RF-B does better than RF but worse than OWN-S.
  P-d  OWN-S learns at n = 2, 3 and is slow or fails at n = 4 (16 samples rarely contain a success at p0 = 1/256).

Second batch, added after the first run (15 Nov 2026) and predicted before it was run. The first run used the graded
success probability s(z) as the reward, which every sample carries; a checker returns a verdict, so:
  RF-01    as RF-B, but the reward is one Bernoulli draw with probability s(z), and blocks learn only from successes
           (gradient R * dlog s)
  OWN-01   as OWN-S, with weights proportional to the Bernoulli outcome (no success in the batch: no update)
  SEARCH   hard EM: the best route under the current blocks (here the per-pass argmax of q); router and blocks regress to it
  P-e  SEARCH learns in about n!/n^n of the seeds (0.5, 0.22, 0.09: the start's argmax route must use n different
       blocks) and otherwise locks in a route that reuses a block, with a deterministic router.
  P-f  OWN-01 learns at every n, slower than OWN-S by about the wait for a first success (1 / (M p0) steps);
       RF-01 is slower than OWN-01 and loses some runs at n = 4.

Usage: python3 experiments/theory/routing_toy.py > experiments/theory/routing_toy.md
"""
import math, sys
import numpy as np

LR, M, SEEDS = 0.5, 16, 10
BUDGET = {2: 4000, 3: 20000, 4: 60000}


def softmax(x, axis):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def exact_parts(S, Th):
    pi, q = softmax(S, 1), softmax(Th, 1)            # pi[t,k], q[k,o]
    a = (pi * q.T).sum(1)                            # a[t]
    p = a.prod()
    dp_da = p / a
    gS = dp_da[:, None] * pi * (q.T - a[:, None])    # dp/dS[t,k]
    w = dp_da[None, :] * pi.T                        # w[k,t] = dp/dq[k,t]
    gTh = q * (w - (w * q).sum(1, keepdims=True))    # dp/dTh[k,o]
    return p, gS, gTh, pi, q, a


def step(sig, S, Th, rng):
    n, K = S.shape
    p, gS, gTh, pi, q, a = exact_parts(S, Th)
    if sig == "EXACT-R":
        return S + LR * gS, Th + LR * gTh
    if sig == "EXACT-L":
        return S + LR * gS / p, Th + LR * gTh / p
    if sig == "FIX":
        tgt = q.T / q.T.sum(1, keepdims=True)        # posterior over blocks at each pass under a uniform prior
        dS = tgt - pi
        r = tgt.T                                    # r[k,t]: responsibility of block k for pass t
        dTh = r - q * r.sum(1, keepdims=True)        # gradient of sum_t r[k,t] log q[k,t]
        return S + LR * dS, Th + LR * dTh
    if sig == "SEARCH":
        best = q.argmax(0)                           # best[t]: the block with the highest competence for pass t
        tgt = np.zeros((n, K)); tgt[np.arange(n), best] = 1
        used1 = tgt.T                                # used1[k,t]
        return S + LR * (tgt - pi), Th + LR * (used1 - q * used1.sum(1, keepdims=True))
    z = np.stack([rng.choice(K, size=M, p=pi[t]) for t in range(n)], 1)        # [M,n]
    s = np.prod(q[z, np.arange(n)[None, :]], 1)                                  # [M]
    onehot = np.zeros((M, n, K)); onehot[np.arange(M)[:, None], np.arange(n)[None, :], z] = 1
    score = onehot - pi[None]                                                    # d log pi(z) / dS
    used = np.zeros((M, K, n)); used[np.arange(M)[:, None], z, np.arange(n)[None, :]] = 1
    dlogs = used - q[None] * used.sum(2, keepdims=True)                          # d log s(z) / dTh
    if sig in ("RF", "RF-B"):
        loss = 1 - s
        b = loss.mean() if sig == "RF-B" else 0.0
        dS = -((loss - b)[:, None, None] * score).mean(0)
        dTh = (s[:, None, None] * dlogs).mean(0)                                 # pathwise: d s / dTh = s dlog s
        return S + LR * dS, Th + LR * dTh
    if sig in ("RF-01", "OWN-01"):
        R = (rng.random(M) < s).astype(float)
        if sig == "RF-01":
            dS = ((R - R.mean())[:, None, None] * score).mean(0)
            return S + LR * dS, Th + LR * (R[:, None, None] * dlogs).mean(0)
        if R.sum() == 0:
            return S, Th
        w = R / R.sum()
        return S + LR * (w[:, None, None] * score).sum(0), Th + LR * (w[:, None, None] * dlogs).sum(0)
    if sig == "OWN-S":
        w = s / s.sum()
        return S + LR * (w[:, None, None] * score).sum(0), Th + LR * (w[:, None, None] * dlogs).sum(0)
    raise ValueError(sig)


def run(sig, n, seed):
    rng = np.random.default_rng(seed)
    S, Th = 0.1 * rng.standard_normal((n, n)), 0.1 * rng.standard_normal((n, n))
    for it in range(BUDGET[n] + 1):
        p, _gS, _gTh, pi, q, _a = exact_parts(S, Th)
        if p > 0.9:
            return it, p, pi.max(1).min()
        S, Th = step(sig, S, Th, rng)
    return None, p, pi.max(1).min()


if __name__ == "__main__":
    sigs = ("EXACT-R", "EXACT-L", "FIX", "RF", "RF-B", "OWN-S", "RF-01", "OWN-01", "SEARCH")
    print("# Which signal learns a program of learned blocks — a toy (EXPLORATORY)\n")
    print(f"Definitions, and the predictions written before the run, in `routing_toy.py`. K = n blocks, step size {LR}, {M} sampled routes per step for the sampled signals, {SEEDS} seeds, "
          f"budgets {BUDGET} steps. A run has learned when the exact success probability p exceeds 0.9.\n")
    print("| passes n | p at the start (≈ n⁻ⁿ) | signal | runs that learned | steps to learn, median [min, max] | among runs that did not: final p, median | … and the router's smallest top probability over passes, median |")
    print("|---|---|---|---|---|---|---|")
    for n in (2, 3, 4):
        for sig in sigs:
            res = [run(sig, n, seed) for seed in range(SEEDS)]
            ok = [r[0] for r in res if r[0] is not None]
            bad = [r for r in res if r[0] is None]
            steps = f"{int(np.median(ok))} [{min(ok)}, {max(ok)}]" if ok else "–"
            fp = f"{np.median([r[1] for r in bad]):.3f}" if bad else "–"
            fd = f"{np.median([r[2] for r in bad]):.2f}" if bad else "–"
            print(f"| {n} | {n ** -n:.4f} | {sig} | {len(ok)}/{SEEDS} | {steps} | {fp} | {fd} |", flush=True)
