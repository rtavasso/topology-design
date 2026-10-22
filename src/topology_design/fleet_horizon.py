"""Experiment E2 of experiments/walk/PREREGISTRATION.md: is the answer visible before the network produces it, and
can a learner act on it?

  refit_readout   re-solve the network's own linear readout on frozen features: L2-regularised multinomial logistic
                  regression (the network's own loss) of train labels on the final residual stream, no bias, as in
                  the network; a convex problem, solved by full-batch Adam. The L2 strength is chosen on a fixed
                  20% holdout of the current training set. Test labels are used only to score. A closed-form ridge
                  (least-squares) refit is also reported; it is a weaker classifier and is secondary.
  extrapolated_acc  test accuracy after jumping H steps ahead, two operators:
                  linear (primary): theta + H * v, v = bias-corrected EMA (~500 steps) of delta_theta, with a guard
                    that stops a coordinate at zero instead of pushing it through zero;
                  decay-aware (secondary): theta' = -lam*theta + g, lam = lr*wd exact, g = EMA of the gradient-driven
                    step (delta_theta + lam*theta_prev), solved exactly over H steps.
                  On a held-out pilot seed, the linear operator predicted future test accuracy in sym (e.g. 0.54 now,
                  predicted 0.98 vs actual 1.00 at +1000 steps) but overshoots in parameter space early and in asym;
                  the decay-aware operator relaxes toward the recent past and predicted nothing. Both are recorded.
  swap_readout    intervention: replace the readout by the refit readout (rescaled to minimise train CE) and reset
                  that tensor's Adam moments, then keep training.
"""

from __future__ import annotations

import torch

LAM_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # ridge: x trace(Gram)/d
LOGIT_LAM_GRID = (1e-5, 1e-4, 1e-3)  # logistic: L2 coefficient on standardised features
LOGIT_STEPS = 300
DRIFT_BETA = 1.0 - 1.0 / 500.0


@torch.no_grad()
def stream_features(fleet, grid: dict, chunk: int = 1024) -> torch.Tensor:
    """(R, N, d) final residual stream on the full grid."""
    N = grid["a"].shape[0]
    out = []
    for lo in range(0, N, chunk):
        sl = slice(lo, min(N, lo + chunk))
        _, h = fleet.forward(grid["a"][sl], grid["b"][sl], stream=True)
        out.append(h)
    return torch.cat(out, 1)


def _ridge(X: torch.Tensor, T: torch.Tensor, mask: torch.Tensor, lam_c: torch.Tensor) -> torch.Tensor:
    """Per-run ridge W (R, d, p) of T on X over rows in mask; lam = lam_c * trace(G)/d. float64 solve."""
    Xm = X * mask.unsqueeze(-1)
    G = torch.einsum("rnd,rne->rde", Xm, X).double()
    Bm = torch.einsum("rnd,rnp->rdp", Xm, T).double()
    d = G.shape[-1]
    tr = G.diagonal(dim1=1, dim2=2).sum(-1) / d
    eye = torch.eye(d, dtype=G.dtype, device=G.device)
    W = torch.linalg.solve(G + (lam_c.double() * tr.clamp_min(1e-12)).view(-1, 1, 1) * eye, Bm)
    return W.to(X.dtype)


def _acc(X, W, y, mask) -> torch.Tensor:
    pred = torch.einsum("rnd,rdp->rnp", X, W).argmax(-1)
    m = mask.to(X.dtype)
    return ((pred == y).to(X.dtype) * m).sum(1) / m.sum(1).clamp_min(1.0)


def _logistic(X: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, lam: torch.Tensor, p: int,
              W0: torch.Tensor | None = None, steps: int = LOGIT_STEPS) -> torch.Tensor:
    """Per-run L2 multinomial logistic regression W (R, d, p) on rows in mask (convex; full-batch Adam)."""
    W = (torch.zeros(X.shape[0], X.shape[-1], p, dtype=X.dtype, device=X.device) if W0 is None else W0.clone())
    W.requires_grad_(True)
    opt = torch.optim.Adam([W], lr=0.05)
    m = mask.to(X.dtype)
    cnt = m.sum(1).clamp_min(1.0)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            lg = torch.einsum("rnd,rdp->rnp", X, W)
            ce = torch.nn.functional.cross_entropy(lg.reshape(-1, p), y.reshape(-1), reduction="none").view(y.shape)
            loss = ((ce * m).sum(1) / cnt + lam * W.pow(2).sum((1, 2))).sum()
            loss.backward()
            opt.step()
    return W.detach()


@torch.no_grad()
def refit_readout(fleet, grid: dict, val_mask: torch.Tensor, X: torch.Tensor | None = None, ridge: bool = True) -> dict:
    """Readout refit on frozen features. val_mask: (R, N) fixed random 20% of grid points; the holdout is the part
    of the current training set inside it. Returns logistic-refit accuracies (R,) (primary), ridge accuracies
    (secondary), the chosen L2 strength, and the logistic W (R, d, p) in raw feature units."""
    X = stream_features(fleet, grid) if X is None else X
    y, train, test = grid["y"], grid["train"], grid["test"]
    p = fleet.p
    R = X.shape[0]
    rms = (X.pow(2).mean((1, 2), keepdim=True)).sqrt().clamp_min(1e-12)  # one scalar per run: W stays bias-free
    Xs = X / rms
    fit, val = train & ~val_mask, train & val_mask
    best_acc = torch.full((R,), -1.0, dtype=X.dtype, device=X.device)
    best_lam = torch.full((R,), LOGIT_LAM_GRID[0], dtype=X.dtype, device=X.device)
    best_W = None
    for lam in LOGIT_LAM_GRID:
        lt = torch.full((R,), lam, dtype=X.dtype, device=X.device)
        Wl = _logistic(Xs, y, fit, lt, p)
        a = _acc(Xs, Wl, y, val)
        better = a > best_acc
        best_acc, best_lam = torch.where(better, a, best_acc), torch.where(better, lt, best_lam)
        best_W = Wl if best_W is None else torch.where(better.view(-1, 1, 1), Wl, best_W)
    W = _logistic(Xs, y, train, best_lam, p, W0=best_W, steps=LOGIT_STEPS // 2)  # warm-started refit on all train
    out = {"refit_test_acc": _acc(Xs, W, y, test), "refit_train_acc": _acc(Xs, W, y, train),
           "refit_lam": best_lam, "W": W / rms.view(-1, 1, 1), "X": X}
    if ridge:
        out.update({f"ridge_{k}": v for k, v in _ridge_refit(X, grid, val_mask, p).items()})
    return out


@torch.no_grad()
def _ridge_refit(X, grid, val_mask, p) -> dict:
    """Closed-form ridge refit of centered one-hot labels (secondary)."""
    y, train, test = grid["y"], grid["train"], grid["test"]
    T = torch.nn.functional.one_hot(y, p).to(X.dtype) - 1.0 / p
    fit, val = train & ~val_mask, train & val_mask
    R = X.shape[0]
    best_acc = torch.full((R,), -1.0, dtype=X.dtype, device=X.device)
    best_c = torch.full((R,), LAM_GRID[0], dtype=X.dtype, device=X.device)
    for c in LAM_GRID:
        cc = torch.full((R,), c, dtype=X.dtype, device=X.device)
        a = _acc(X, _ridge(X, T, fit, cc), y, val)
        better = a > best_acc
        best_acc, best_c = torch.where(better, a, best_acc), torch.where(better, cc, best_c)
    W = _ridge(X, T, train, best_c)
    return {"test_acc": _acc(X, W, y, test), "train_acc": _acc(X, W, y, train), "lam_c": best_c, "W": W}


@torch.no_grad()
def test_acc(fleet, grid: dict, chunk: int = 1024) -> torch.Tensor:
    N = grid["a"].shape[0]
    hit = torch.zeros(fleet.R, dtype=fleet.dtype, device=fleet.device)
    for lo in range(0, N, chunk):
        sl = slice(lo, min(N, lo + chunk))
        lg = fleet.forward(grid["a"][sl], grid["b"][sl])
        hit += ((lg.argmax(-1) == grid["y"][:, sl]).to(fleet.dtype) * grid["test"][:, sl].to(fleet.dtype)).sum(1)
    return hit / grid["test"].to(fleet.dtype).sum(1).clamp_min(1.0)


def drift_increment(fleet, k: str, new: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """The gradient-driven part of one AdamW step: delta_theta + lam * theta_prev (removes the decoupled decay)."""
    sh = (-1,) + (1,) * (new.dim() - 1)
    lam = (fleet.lr_t[k] * fleet.wd_t[k]).view(sh)
    return (new - prev) + lam * prev


def extrapolate_linear(fleet, vlin: dict, t_vel: int, H) -> dict:
    """theta + H * v per tensor (H scalar or (R,)), with coordinates stopped at zero rather than pushed through it."""
    corr = 1.0 - DRIFT_BETA ** t_vel
    out = {}
    for k, th in fleet.params.items():
        sh = (-1,) + (1,) * (th.dim() - 1)
        Ht = torch.as_tensor(H, dtype=th.dtype, device=th.device)
        Ht = Ht.view(sh) if Ht.dim() == 1 else Ht
        v = vlin[k] / corr
        x = th.detach()
        nxt = x + Ht * v
        out[k] = torch.where((nxt * x < 0) & (v * x < 0), torch.zeros_like(x), nxt)
    return out


def extrapolate(fleet, vel: dict, t_vel: int, H) -> dict:
    """Decay-aware extrapolation of every tensor by H steps (H scalar or (R,) tensor)."""
    corr = 1.0 - DRIFT_BETA ** t_vel
    out = {}
    for k, th in fleet.params.items():
        sh = (-1,) + (1,) * (th.dim() - 1)
        lam = (fleet.lr_t[k] * fleet.wd_t[k]).view(sh)
        Ht = torch.as_tensor(H, dtype=th.dtype, device=th.device)
        Ht = Ht.view(sh) if Ht.dim() == 1 else Ht
        g = vel[k] / corr
        decay = torch.exp(-lam * Ht)
        gain = torch.where(lam > 0, (1 - decay) / lam.clamp_min(1e-30), Ht * torch.ones_like(lam))
        out[k] = decay * th.detach() + gain * g
    return out


@torch.no_grad()
def extrapolated_acc(fleet, grid: dict, vlin: dict, vel: dict, t_vel: int, horizons) -> dict:
    """Test accuracy after each horizon H: extrap_acc_H (linear, primary) and extrapd_acc_H (decay-aware)."""
    out = {}
    if t_vel == 0:
        cur = test_acc(fleet, grid)
        return {**{f"extrap_acc_{H}": cur.clone() for H in horizons}, **{f"extrapd_acc_{H}": cur.clone() for H in horizons}}
    P0 = fleet.params
    try:
        for H in horizons:
            fleet.params = extrapolate_linear(fleet, vlin, t_vel, float(H))
            out[f"extrap_acc_{H}"] = test_acc(fleet, grid)
            fleet.params = extrapolate(fleet, vel, t_vel, float(H))
            out[f"extrapd_acc_{H}"] = test_acc(fleet, grid)
    finally:
        fleet.params = P0
    return out


@torch.no_grad()
def swap_readout(fleet, opt, grid: dict, val_mask: torch.Tensor, runs: torch.Tensor, A, B, Y, w) -> dict:
    """For runs (R,) bool: unembed := (s * W_refit)^T / out_scale, with s minimising train CE on the current
    training batch, and that tensor's Adam moments reset to zero. Returns refit accuracy and scale (R,)."""
    rf = refit_readout(fleet, grid, val_mask, ridge=False)
    W = rf["W"]  # (R, d, p), logistic refit in raw feature units
    _, h = fleet.forward(A, B, stream=True)  # (R, n, d) training batch
    raw = torch.einsum("rnd,rdp->rnp", h, W)
    wf = torch.ones(A.shape, dtype=fleet.dtype, device=fleet.device) if w is None else w
    best_s = torch.ones(fleet.R, dtype=fleet.dtype, device=fleet.device)
    best_l = torch.full((fleet.R,), float("inf"), dtype=fleet.dtype, device=fleet.device)
    for s_ in torch.logspace(-2, 2, 41).tolist():  # includes s = 1 (the logistic solution itself)
        lg = s_ * raw
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y.reshape(-1),
                                               reduction="none").view(Y.shape)
        l = (ce * wf).sum(1) / wf.sum(1).clamp_min(1e-12)
        better = l < best_l
        best_l, best_s = torch.where(better, l, best_l), torch.where(better, torch.full_like(best_s, s_), best_s)
    newU = (best_s.view(-1, 1, 1) * W).transpose(1, 2) / fleet.out_scale.view(-1, 1, 1)  # (R, p, d)
    m = runs.view(-1, 1, 1)
    U = fleet.params["unembed"]
    U.copy_(torch.where(m, newU.to(U.dtype), U))
    opt.m["unembed"] = torch.where(m, torch.zeros_like(opt.m["unembed"]), opt.m["unembed"])
    opt.v["unembed"] = torch.where(m, torch.zeros_like(opt.v["unembed"]), opt.v["unembed"])
    return {"refit_test_acc": rf["refit_test_acc"], "scale": best_s, "train_ce": best_l}


def holdout_mask(cfgs, N: int, device) -> torch.Tensor:
    """(R, N): a fixed random 20% of grid points per run (seed 70000 + data_seed)."""
    rows = []
    for c in cfgs:
        g = torch.Generator().manual_seed(70_000 + c.data_seed)
        rows.append(torch.rand(N, generator=g) < 0.2)
    return torch.stack(rows).to(device)
