"""Episode pools for rung 1: the numpy generator is ~0.4 ms per episode, too slow to call every training step for
a stack of runs, so each run draws its batches from a pool generated once (with replacement). Evaluation strata are
generated fresh per stratum. Tensors are returned on the device as a dict with the keys the models use.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import RouteConfig, Episodes, generate

KEYS = ("pair_key", "pair_val", "pair_prev", "pair_flag", "q_start", "q_ans", "q_k", "q_word")


def to_dict(ep: Episodes, dev) -> dict:
    return {k: torch.as_tensor(getattr(ep, k), device=dev) for k in KEYS}


def _cat(eps: list[Episodes]) -> Episodes:
    kw = max(e.q_word.shape[-1] for e in eps)
    pad = lambda e: np.concatenate([e.q_word, np.full((e.B, e.q_word.shape[1], kw - e.q_word.shape[-1]), -1, np.int64)], -1)
    return Episodes(*[np.concatenate([getattr(e, k) if k != "q_word" else pad(e) for e in eps], 0) for k in KEYS])


class EpisodePool:
    """n training episodes for one run, generated from the run's own seed; sample(B) draws with replacement."""

    def __init__(self, cfg: RouteConfig, n: int, seed: int, dev="cpu", chunk: int = 2000):
        self.cfg, self.dev = cfg, dev
        rng = np.random.default_rng(seed)
        parts = [generate(cfg, min(chunk, n - i), rng) for i in range(0, n, chunk)]
        ep = _cat(parts)
        self.data = {k: torch.as_tensor(getattr(ep, k)) for k in KEYS}      # on the CPU; moved per batch
        self.n = ep.B
        self.g = torch.Generator().manual_seed(seed + 7)

    def sample(self, B: int) -> dict:
        idx = torch.randint(0, self.n, (B,), generator=self.g)
        return {k: v[idx].to(self.dev, non_blocking=True) for k, v in self.data.items()}


def stack_batches(batches: list[dict]) -> dict:
    """R per-run batches -> one dict of (R, B, ...) tensors (q_word padded to a common width)."""
    kw = max(b["q_word"].shape[-1] for b in batches)
    out = {}
    for k in KEYS:
        if k == "q_word":
            out[k] = torch.stack([torch.nn.functional.pad(b[k], (0, kw - b[k].shape[-1]), value=-1) for b in batches], 0)
        else:
            out[k] = torch.stack([b[k] for b in batches], 0)
    return out


# evaluation strata (DESIGN.md §2): name -> (k, kind)
# in-range (training words), all-H controls, length, and the new orders: PP runs (pending count 2, unseen in
# training), alternation (training words, a frequency stratum), a deep P (training words, rare), long new orders
STRATA = ([(f"in{k}", (k, "train")) for k in range(1, 9)]
          + [(f"none{k}", (k, "none")) for k in (3, 4, 8)]
          + [(f"len{k}", (k, "train")) for k in (10, 12, 16, 20, 24, 32)]
          + [(f"pp{k}", (k, "pp")) for k in (4, 6, 8)] + [(f"alt{k}", (k, "alt")) for k in (4, 6, 8)]
          + [("deepP8", (8, "deepP"))] + [(f"pp{k}", (k, "pp")) for k in (16, 32)] + [(f"alt{k}", (k, "alt")) for k in (16, 32)])
NEW_ORDER = ("pp4", "pp6", "pp8", "pp16", "pp32")          # the only strata whose words are outside the training support


def strata_episodes(cfg: RouteConfig, n: int, seed: int, Qn: int = 3) -> dict:
    """name -> Episodes with query 0 forced to the stratum's (k, kind); fresh per call."""
    rng = np.random.default_rng(seed)
    return {name: generate(cfg, n, rng, forced_k=k, forced_kind=kind, Qn=Qn, kmax_eval=40) for name, (k, kind) in STRATA}
