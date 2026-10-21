"""Composition testbed (Study 3): a rule whose generalizing solution needs one computation built on another.

Task 'addmul':  y = ((a + b) mod p) * c  mod p,   a, b, c in Z_p.
  - Addition is linear in the additive Fourier basis of Z_p; multiplication is linear in the Fourier basis of the
    multiplicative group (discrete log). No single basis linearizes both, so a generalizing solution must compute the
    intermediate s = a + b, re-encode it, then combine with c: a 2-step composition. A shallow lookup can still
    memorize, so the two kinds of solution compete.
Task 'add3' (control): y = (a + b + c) mod p, same input/output format, generalizing solution needs no re-encoding.

Model: identical to grokking.RoutedMLP except for one embedding table per input:
    h_0 = E_a[a] + E_b[b] + E_c[c];  h_i = beta h_{i-1} + alpha_i F_i(h_{k_i});  logits = U h_L.
Routing arms reuse grokking.arm_spec ('pk012' = sequential, 'pk000' = parallel, ...).

Measurements extend grokking.measure with the composition-specific decomposition:
  - rule_frac:   share of centered-logit energy that is a function of y only  (the finished rule)
  - inter_frac:  share that is a function of (s, c) only, s = (a+b) mod p     (has formed the intermediate)
  - per-block linear decodability of s from each accumulated state h_i (where the intermediate is computed)
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import grokking as gk


@dataclass
class CompConfig:
    arm: str = "pk012"
    task: str = "addmul"
    p: int = 13
    train_frac: float = 0.5
    d: int = 128
    hidden: int = 256
    beta: float = 1.0
    out_scale: float = 1.0
    lr: float = 3e-3
    weight_decay: float = 1.0
    steps: int = 20000
    eval_every: int = 50
    model_seed: int = 0
    data_seed: int = 0
    init_scale: float = 1.0
    # stabilizers (applied identically to every arm; defaults = off, so earlier runs are unchanged)
    grad_clip: float = 0.0  # global grad-norm clip (0 = off)
    block_out_init: float = 1.0  # multiplies each block's output weight w2 at init (ReZero/Fixup-style when small)
    prenorm: bool = False  # LayerNorm (no affine) on each block's input
    stop_after_grok: int = 1000
    grok_thresh: float = 0.95
    tag: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        stab = (f"_gc{self.grad_clip:g}" if self.grad_clip else "") + \
               (f"_bo{self.block_out_init:g}" if self.block_out_init != 1.0 else "") + ("_ln" if self.prenorm else "")
        return (f"{self.task}_{self.arm}_p{self.p}_wd{self.weight_decay:g}_lr{self.lr:g}_tf{self.train_frac:g}"
                f"_os{self.out_scale:g}_b{self.beta:g}{stab}_m{self.model_seed}_d{self.data_seed}{('_' + self.tag) if self.tag else ''}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def full_grid(p: int, task: str):
    idx = torch.arange(p ** 3)
    a, b, c = idx // (p * p), (idx // p) % p, idx % p
    s = (a + b) % p
    if task == "addmul":
        y = (s * c) % p
    elif task == "add3":
        y = (s + c) % p
    else:
        raise ValueError(task)
    return a, b, c, s, y


def split(p: int, train_frac: float, data_seed: int):
    g = torch.Generator().manual_seed(20_000 + data_seed)
    perm = torch.randperm(p ** 3, generator=g)
    n = int(round(train_frac * p ** 3))
    return perm[:n].sort().values, perm[n:].sort().values


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RoutedMLP3(nn.Module):
    def __init__(self, p, d, hidden, parents, alphas, beta=1.0, out_scale=1.0, prenorm=False):
        super().__init__()
        for i, k in enumerate(parents, start=1):
            if not 0 <= k < i:
                raise ValueError(f"parent k_{i}={k} must satisfy 0 <= k < {i}")
        self.p, self.L, self.parents = p, len(parents), tuple(parents)
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.beta, self.out_scale, self.prenorm = float(beta), float(out_scale), bool(prenorm)
        self.embed = nn.ModuleList([nn.Embedding(p, d) for _ in range(3)])
        self.w1 = nn.ModuleList([nn.Linear(d, hidden, bias=False) for _ in parents])
        self.w2 = nn.ModuleList([nn.Linear(hidden, d, bias=False) for _ in parents])
        self.unembed = nn.Linear(d, p, bias=False)
        with torch.no_grad():
            for e in self.embed:
                e.weight.normal_(0.0, 1.0 / math.sqrt(d))

    def forward(self, a, b, c, alphas=None, return_states=False, return_parts=False):
        alphas = self.alphas if alphas is None else alphas
        hs = [self.embed[0](a) + self.embed[1](b) + self.embed[2](c)]
        writes = []
        for i, k in enumerate(self.parents):
            x = F.layer_norm(hs[k], hs[k].shape[-1:]) if self.prenorm else hs[k]
            w = alphas[i] * self.w2[i](F.relu(self.w1[i](x)))
            writes.append(w)
            hs.append(self.beta * hs[-1] + w)
        logits = self.out_scale * self.unembed(hs[-1])
        if return_parts:
            L = self.L
            parts = [self.out_scale * self.beta ** L * self.unembed(hs[0])]
            parts += [self.out_scale * self.beta ** (L - 1 - i) * self.unembed(w) for i, w in enumerate(writes)]
            return logits, torch.stack(parts), hs
        return (logits, hs) if return_states else logits

    def branch_param_groups(self):
        g = {"embed": [e.weight for e in self.embed], "unembed": [self.unembed.weight]}
        for i in range(self.L):
            g[f"block{i + 1}"] = [self.w1[i].weight, self.w2[i].weight]
        return g


def build_model(cfg: CompConfig) -> RoutedMLP3:
    parents, alphas = gk.arm_spec(cfg.arm)
    torch.manual_seed(cfg.model_seed)
    m = RoutedMLP3(cfg.p, cfg.d, cfg.hidden, parents, alphas, cfg.beta, cfg.out_scale, cfg.prenorm)
    with torch.no_grad():
        if cfg.init_scale != 1.0:
            for q in m.parameters():
                q.mul_(cfg.init_scale)
        if cfg.block_out_init != 1.0:
            for w2 in m.w2:
                w2.weight.mul_(cfg.block_out_init)
    return m


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def group_projection(Z: torch.Tensor, key: torch.Tensor, n_keys: int) -> torch.Tensor:
    """Orthogonal projection (uniform measure) of rows of Z onto functions of `key`: replace each row by its key-group mean."""
    sums = torch.zeros(n_keys, Z.shape[-1], dtype=Z.dtype).index_add_(0, key, Z)
    counts = torch.bincount(key, minlength=n_keys).clamp_min(1).to(Z.dtype)[:, None]
    return (sums / counts)[key]


def probe_split(p: int, seed: int = 12345):
    """Fixed random halves of the FULL grid for fitting/evaluating linear probes (independent of the train/test split;
    probe targets such as s are never training labels, so using all p^3 points is legitimate and keeps the probe
    well-powered: fitting on a half of the test split alone left ~110 points for 129 features at train_frac 0.9)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p ** 3, generator=g)
    return perm[: p ** 3 // 2], perm[p ** 3 // 2 :]


def s_energy_frac(Hc: torch.Tensor, s: torch.Tensor, c: torch.Tensor, p: int) -> float:
    """Share of the (centered) energy of Hc that depends on s = a+b jointly with c beyond c alone:
    (||P_(s,c) H||^2 - ||P_c H||^2) / ||H||^2.  Exactly 0 for the embedding sum h_0 = E_a[a] + E_b[b] + E_c[c] on the
    full grid (the group mean over a with a+b fixed is constant), so a positive value means the sum has been computed."""
    tot = Hc.pow(2).sum().clamp_min(1e-12)
    Psc = group_projection(Hc, s * p + c, p * p)
    Pc = group_projection(Hc, c, p)
    return ((Psc.pow(2).sum() - Pc.pow(2).sum()) / tot).item()


def ridge_decode_acc(H: torch.Tensor, target: torch.Tensor, n_cls: int, tr: torch.Tensor, te: torch.Tensor, lam: float = 1e-3) -> float:
    """Held-out accuracy of a closed-form ridge one-vs-all linear readout of `target` from features H."""
    H = H.double()
    mu, sd = H[tr].mean(0), H[tr].std(0).clamp_min(1e-8)  # standardize with fit-split stats: scale-free ridge
    X = torch.cat([(H - mu) / sd, torch.ones(len(H), 1, dtype=torch.float64)], 1)
    Y = F.one_hot(target, n_cls).double()
    A = X[tr].T @ X[tr] + lam * len(tr) * torch.eye(X.shape[1], dtype=torch.float64)
    Wt = torch.linalg.solve(A, X[tr].T @ Y[tr])
    return ((X[te] @ Wt).argmax(1) == target[te]).float().mean().item()


@torch.no_grad()
def measure(model: RoutedMLP3, tr, te, task: str) -> dict:
    p = model.p
    a, b, c, s, y = full_grid(p, task)
    logits, parts, hs = model(a, b, c, return_parts=True)
    out = {
        "train_loss": F.cross_entropy(logits[tr], y[tr]).item(),
        "test_loss": F.cross_entropy(logits[te], y[te]).item(),
        "train_acc": gk.acc(logits[tr], y[tr]),
        "test_acc": gk.acc(logits[te], y[te]),
    }
    Zc = gk.centered(logits)
    tot = Zc.pow(2).sum().clamp_min(1e-12)
    R = group_projection(Zc, y, p)
    key_sc = s * p + c
    I = group_projection(Zc, key_sc, p * p)
    M = Zc - I  # not even a function of (s, c): depends on a, b beyond their sum -> memorization/noise
    out["rule_frac"] = (R.pow(2).sum() / tot).item()
    out["inter_frac"] = (I.pow(2).sum() / tot).item()
    out["restricted_test_acc"] = gk.acc(R[te], y[te])
    out["excluded_train_acc"] = gk.acc(M[tr], y[tr])
    br = []
    for j in range(parts.shape[0]):
        Pj = gk.centered(parts[j])
        Rj, Ij = group_projection(Pj, y, p), group_projection(Pj, key_sc, p * p)
        br.append({"rule_energy": Rj.pow(2).mean().item(), "inter_energy": Ij.pow(2).mean().item(),
                   "rem_energy": (Pj - Ij).pow(2).mean().item()})
    out["branches"] = br
    # where is the intermediate s available? (linear probe on fixed halves of the full grid; and the share of each
    # state's energy that depends on s beyond c alone, which is exactly 0 for the embedding sum)
    half = probe_split(p)
    out["decode_s"] = [ridge_decode_acc(h, s, p, half[0], half[1]) for h in hs]
    out["state_s_frac"] = [s_energy_frac(h.double() - h.double().mean(0), s, c, p) for h in hs]
    out["logit_s_frac"] = s_energy_frac(Zc.double(), s, c, p)
    abl = []
    for i in range(model.L):
        al = model.alphas.clone()
        al[i] = 0.0
        lg = model(a, b, c, alphas=al)
        abl.append({"train_acc": gk.acc(lg[tr], y[tr]), "test_acc": gk.acc(lg[te], y[te])})
    out["ablate"] = abl
    out["norms"] = {k: math.sqrt(sum(float(q.pow(2).sum()) for q in v)) for k, v in model.branch_param_groups().items()}
    return out


def train_run(cfg: CompConfig, verbose: bool = False) -> dict:
    torch.set_num_threads(1)
    model = build_model(cfg)
    tr, te = split(cfg.p, cfg.train_frac, cfg.data_seed)
    a, b, c, s, y = full_grid(cfg.p, cfg.task)
    A, B, C, Y = a[tr], b[tr], c[tr], y[tr]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    traj, grok_step, diverged_at, t0 = [], None, None, time.time()
    for step in range(cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(A, B, C)
        loss = F.cross_entropy(logits, Y)
        if not torch.isfinite(loss):
            diverged_at = step
            break
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip if cfg.grad_clip else float("inf"))
        if step % cfg.eval_every == 0:
            rec = {"step": step, "grad_norm": float(gnorm), **measure(model, tr, te, cfg.task)}
            traj.append(rec)
            if verbose:
                print(f"{cfg.run_id} {step}: train {rec['train_acc']:.2f} test {rec['test_acc']:.2f} "
                      f"rule {rec['rule_frac']:.2f} inter {rec['inter_frac']:.2f} decode_s {[round(x, 2) for x in rec['decode_s']]}",
                      flush=True)
            if grok_step is None and rec["test_acc"] >= cfg.grok_thresh:
                grok_step = step
            if cfg.stop_after_grok and grok_step is not None and step - grok_step >= cfg.stop_after_grok:
                break
        opt.step()
    summ = {**gk.summarize(traj, cfg), **gk.stability_summary(traj, diverged_at)}
    summ["final_decode_s"] = traj[-1]["decode_s"] if traj else None
    summ["final_state_s_frac"] = traj[-1]["state_s_frac"] if traj else None
    summ["peak_inter_frac"] = max((r["inter_frac"] for r in traj), default=None)
    return {"config": asdict(cfg), "run_id": cfg.run_id, "n_params": sum(q.numel() for q in model.parameters()),
            "seconds": time.time() - t0, "trajectory": traj, "summary": summ}


RunConfig = CompConfig  # lets the generic sweep runner use this module
