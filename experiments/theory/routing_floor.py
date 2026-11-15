"""EXPLORATORY. Follow-up toy for docs/routing_signal_stability.md, section 7 (streams recombined by a gated sum).

In the conjunctive toy the gated sum over streams is the same quantity as the expectation over routes, so "soft gates"
is the exact signal already run (EXACT-L under a likelihood loss), which ended in the reused-block state in 11 of 200
starts at four passes. That state holds because the idle stream's gate is near zero, so it is not trained.

Variant: a gate floor. Every stream always receives at least eps / K of the gate: g = (1 - eps) softmax(s) + eps / K.

Prediction written before the run (15 Nov 2026):
  P-g  With eps = 0.05 the exact likelihood ends in a program (each pass picks a different stream, each picked stream
       above 0.8 competence at its job) from at least 198 of 200 starts at four passes, against 189 of 200 with no floor:
       the idle stream is trained up to parity with the stream doing two jobs, and parity is a saddle, not a dead end.
       I do not predict the dependence on eps beyond that.

Usage: python3 experiments/theory/routing_floor.py > experiments/theory/routing_floor.md
"""
import numpy as np

LR, STARTS, BUDGET, N = 0.5, 200, 6000, 4


def softmax(x, axis):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def run(eps, seed, n=N):
    rng = np.random.default_rng(1000 + seed)
    S, Th = 0.1 * rng.standard_normal((n, n)), 0.1 * rng.standard_normal((n, n))
    for it in range(BUDGET):
        pi, q = softmax(S, 1), softmax(Th, 1)
        g = (1 - eps) * pi + eps / n
        a = (g * q.T).sum(1)
        gS = (1 - eps) * pi * (q.T - (pi * q.T).sum(1, keepdims=True)) / a[:, None]       # d log p / dS
        w = g.T / a[None, :]                                                             # d log p / dq[k,t]
        gTh = q * (w - (w * q).sum(1, keepdims=True))
        S, Th = S + LR * gS, Th + LR * gTh
    pi, q = softmax(S, 1), softmax(Th, 1)
    pick = pi.argmax(1)
    program = len(set(pick.tolist())) == n and all(q[pick[t], t] > 0.8 for t in range(n))
    hard_p = float(np.prod([q[pick[t], t] for t in range(n)]))                           # success of the argmax route
    return program, hard_p


if __name__ == "__main__":
    print("# A gate floor in the conjunctive toy (EXPLORATORY)\n")
    print(f"Exact likelihood, {N} passes, {STARTS} starts, {BUDGET} steps, step size {LR}. Prediction in `routing_floor.py`.\n")
    print("| gate floor eps | starts that end in a program | success of the router's top route, median over the others |")
    print("|---|---|---|")
    for eps in (0.0, 0.01, 0.05, 0.2):
        res = [run(eps, s) for s in range(STARTS)]
        ok = sum(r[0] for r in res)
        rest = [r[1] for r in res if not r[0]]
        print(f"| {eps} | {ok}/{STARTS} | {np.median(rest):.2f} |" if rest else f"| {eps} | {ok}/{STARTS} | – |", flush=True)
