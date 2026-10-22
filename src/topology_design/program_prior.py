"""Reference learner for experiment E1 (experiments/walk/PREREGISTRATION.md): a Bayesian mixture over short
arithmetic programs, scored by prequential code length on the same stream of examples the network sees.

Hypotheses: every function of (a, b) mod p computed by an expression tree with leaves {a, b, 1, 2} and binary
operations {+, -, *} mod p, with at most `max_size` nodes. Prior: each expression contributes 2^-size to the
function it computes (equivalent expressions add up, Solomonoff style). The functions take total mass 1 - mu;
the remaining mu goes to a lookup table that predicts uniformly on unseen inputs, so the mixture never does much
worse than uniform when the true rule is outside the class.

Likelihood: P(y | x, h) = 1 - eps if h(x) = y, else eps / (p - 1).

Prequential, block-wise: each block is predicted with the posterior from all earlier blocks, then absorbed. This
matches the network's protocol, where new pairs are scored at the step they arrive, before training on them.

This is ONE reference learner with ONE prior. "Better than gradient descent" is always relative to a prior (no free
lunch); the result is reported as such.
"""

from __future__ import annotations

import math

import numpy as np


def enumerate_functions(p: int, max_size: int = 5, leaves=("a", "b", 1, 2)):
    """Unique functions as an (M, p*p) int array with prior log-masses (M,), plus one readable expression each."""
    a = np.repeat(np.arange(p), p)
    b = np.tile(np.arange(p), p)
    by_size: dict[int, list[tuple[np.ndarray, str]]] = {}
    leaf_tabs = []
    for lf in leaves:
        tab = a if lf == "a" else b if lf == "b" else np.full(p * p, lf % p)
        leaf_tabs.append((tab.astype(np.int64), str(lf)))
    by_size[1] = leaf_tabs
    for size in range(3, max_size + 1, 2):
        out = []
        for s1 in range(1, size - 1, 2):
            s2 = size - 1 - s1
            for t1, e1 in by_size[s1]:
                for t2, e2 in by_size[s2]:
                    out.append(((t1 + t2) % p, f"({e1}+{e2})"))
                    out.append(((t1 - t2) % p, f"({e1}-{e2})"))
                    out.append(((t1 * t2) % p, f"({e1}*{e2})"))
        by_size[size] = out
    mass: dict[bytes, float] = {}
    tabs: dict[bytes, np.ndarray] = {}
    expr: dict[bytes, str] = {}
    for size, items in by_size.items():
        for tab, e in items:
            key = tab.astype(np.int16).tobytes()
            mass[key] = mass.get(key, 0.0) + 2.0 ** (-size)
            if key not in tabs:
                tabs[key], expr[key] = tab, e
    keys = list(tabs)
    T = np.stack([tabs[k] for k in keys])
    logm = np.log(np.array([mass[k] for k in keys]))
    return T, logm, [expr[k] for k in keys]


def prequential(T: np.ndarray, logm: np.ndarray, x_blocks, y_blocks, p: int, eps: float = 1e-3,
                mu: float = 0.5, true_fn: np.ndarray | None = None) -> dict:
    """Block-wise prequential code length (nats) of the mixture.
    x_blocks / y_blocks: lists of int arrays (grid indices, labels), in stream order.
    Returns nats per block, accuracy per block (argmax of the predictive), and the posterior mass of true_fn's
    hypothesis after each block (if given)."""
    logw = np.log(1 - mu) + logm - np.logaddexp.reduce(logm)  # programs
    logw_lookup = math.log(mu)
    l_hit, l_miss = math.log(1 - eps), math.log(eps / (p - 1))
    true_idx = None
    if true_fn is not None:
        hits = np.where((T == true_fn[None, :]).all(1))[0]
        true_idx = int(hits[0]) if len(hits) else None
    nats, acc, post_true = [], [], []
    for xb, yb in zip(x_blocks, y_blocks):
        pred = T[:, xb]  # (M, nb)
        lz = np.logaddexp(np.logaddexp.reduce(logw), logw_lookup)
        wn = np.exp(logw - lz)  # normalised program weights
        wl = math.exp(logw_lookup - lz)
        # predictive probability of each label for each example in the block
        probs = np.full((len(xb), p), wl / p)
        for j in range(len(xb)):
            col = pred[:, j]
            mass_by_label = np.bincount(col, weights=wn, minlength=p)
            probs[j] += (1 - eps) * mass_by_label + (eps / (p - 1)) * (wn.sum() - mass_by_label)
        nats.append(float(-np.log(probs[np.arange(len(xb)), yb]).sum()))
        acc.append(float((probs.argmax(1) == yb).mean()))
        # absorb the block
        match = pred == yb[None, :]
        logw = logw + np.where(match, l_hit, l_miss).sum(1)
        logw_lookup = logw_lookup + len(xb) * math.log(1.0 / p)
        if true_idx is not None:
            lz = np.logaddexp(np.logaddexp.reduce(logw), logw_lookup)
            post_true.append(float(math.exp(logw[true_idx] - lz)))
    return {"nats": nats, "acc": acc, "post_true": post_true}
