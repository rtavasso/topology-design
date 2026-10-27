"""Vectorized episode generator on the GPU (the same distribution as data.generate, built with tensor ops).

Episodes come out as tensors with a leading run dimension R, so a stack of runs (each with its own tail factor and
seed) gets its own data in one call. Distractor chains fill the pair budget: lengths are drawn from the base pmf,
laid end to end, and the last one is truncated to fit.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import PtrConfig, Episodes, zipf_pmf


class TorchGen:
    def __init__(self, dc: PtrConfig, device, seed: int = 0):
        self.dc, self.dev = dc, torch.device(device)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.base_pmf = torch.as_tensor(zipf_pmf(dc.alpha, dc.kmax), device=self.dev, dtype=torch.float)

    def pmf(self, tail_factor):
        """(R, kmax) pmf per run with the deep tail scaled."""
        tf = torch.as_tensor(tail_factor, device=self.dev, dtype=torch.float).view(-1, 1)
        k = torch.arange(1, self.dc.kmax + 1, device=self.dev)
        p = self.base_pmf.unsqueeze(0).expand(tf.shape[0], -1).clone()
        p = torch.where(k >= self.dc.k_deep, p * tf, p)
        return p / p.sum(1, keepdim=True)

    def sample(self, R: int, B: int, tail_factor=None, forced_k: int | None = None, Qn: int | None = None,
               key_pool: int | None = None, ans_pool: int | None = None) -> dict:
        dc, dev, g = self.dc, self.dev, self.g
        Qn = dc.Qn if Qn is None else Qn
        P = dc.P
        key_pool = dc.n_keys_train if key_pool is None else key_pool
        ans_pool = dc.n_ans_train if ans_pool is None else ans_pool
        n = R * B
        pmf = self.pmf(tail_factor if tail_factor is not None else torch.ones(R)).repeat_interleave(B, 0)  # (n, kmax)
        ks = torch.multinomial(pmf, Qn, replacement=True, generator=g) + 1                              # (n, Qn)
        if forced_k is not None:
            ks[:, 0] = forced_k
        assert int(ks.sum(1).max()) <= P, "queried chains must fit in P pairs"
        # distractor chains: up to P lengths from the base pmf, truncated to the remaining budget
        rem = P - ks.sum(1, keepdim=True)                                                                # (n, 1)
        dl = torch.multinomial(self.base_pmf.expand(n, -1), P, replacement=True, generator=g) + 1        # (n, P)
        start = torch.cumsum(dl, 1) - dl
        dl = torch.clamp(torch.minimum(dl, rem - start), min=0)
        lengths = torch.cat([ks, dl], 1)                                                                 # (n, C)
        C = lengths.shape[1]
        ends = torch.cumsum(lengths, 1)                                                                  # (n, C)
        starts = ends - lengths
        # position p (0..P-1) belongs to the chain c with starts[c] <= p < ends[c]
        pos = torch.arange(P, device=dev)
        chain_of = (pos.view(1, P, 1) >= starts.unsqueeze(1)).sum(2) - 1                                 # (n, P)
        last = (pos.unsqueeze(0) + 1) == ends.gather(1, chain_of)                                        # (n, P)
        perm = torch.argsort(torch.rand(n, key_pool, device=dev, generator=g), 1)[:, :P]                 # distinct keys
        ans = dc.n_keys + torch.randint(0, ans_pool, (n, C), device=dev, generator=g)
        nxt = torch.roll(perm, -1, 1)
        pair_key = perm
        pair_val = torch.where(last, ans.gather(1, chain_of), nxt)
        shuf = torch.argsort(torch.rand(n, P, device=dev, generator=g), 1)
        pair_key, pair_val = pair_key.gather(1, shuf), pair_val.gather(1, shuf)
        q_start = perm.gather(1, starts[:, :Qn])
        q_ans = ans[:, :Qn]
        q_k = ks
        qperm = torch.argsort(torch.rand(n, Qn, device=dev, generator=g), 1)
        q_start, q_ans, q_k = q_start.gather(1, qperm), q_ans.gather(1, qperm), q_k.gather(1, qperm)
        v = lambda t: t.view(R, B, -1)
        return {"pair_key": v(pair_key), "pair_val": v(pair_val), "q_start": v(q_start), "q_ans": v(q_ans),
                "q_k": v(q_k)}

    @staticmethod
    def to_episodes(batch: dict, r: int = 0) -> Episodes:
        c = lambda t: t[r].cpu().numpy().astype(np.int64)
        return Episodes(c(batch["pair_key"]), c(batch["pair_val"]), c(batch["q_start"]), c(batch["q_ans"]),
                        c(batch["q_k"]))
