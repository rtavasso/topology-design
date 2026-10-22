"""Phase 5A measurements and 5E first-order probes for the fleet trainer (experiments/fastslow/DESIGN.md).

All gradients of test-set quantities use torch.autograd.grad, never .backward(), so they never touch p.grad
(the training gradient). Quantities are computed per run and per *unit*: each configured branch (or each block if
no branches are configured) plus the shared parameters (embeddings and readout).

Separation statistic for a split part G of a vector x (x = training gradient g, or preconditioned update u):
    S(G) = < -G_perp , gen_hat - memperp_hat > / ||x||,    G_perp = G - (<G,x>/||x||^2) x
with gen = grad of the test-label support and memperp = grad of the train-specific support with its gen component
removed, all restricted to the unit. Computed from inner products only (no vectors materialised beyond the splits).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def unit_names(fleet) -> tuple[list[list[str]], list[str]]:
    from .fleet import branch_blocks
    c0 = fleet.cfgs[0]
    units = branch_blocks(c0) or [[i] for i in range(fleet.L)]
    names = [[f"w1_{i}" for i in u] + [f"w2_{i}" for i in u] for u in units]
    return names, ["embed_a", "embed_b", "unembed"]


def _dot(a: dict, b: dict, names: list[str]) -> torch.Tensor:
    return sum((a[k].flatten(1) * b[k].flatten(1)).sum(1) for k in names)


def _support_grads(fleet, grid: dict, chunk: int = 1024) -> tuple[dict, dict]:
    """Per-run gradients of gen (mean test-label centered logit) and of the mean train-label centered logit."""
    P = fleet.params
    keys = list(P)
    Ggen = {k: torch.zeros_like(P[k]) for k in keys}
    Gtr = {k: torch.zeros_like(P[k]) for k in keys}
    N = grid["a"].shape[0]
    dt = fleet.dtype
    n_te = grid["test"].to(dt).sum(1).clamp_min(1.0)
    n_tr = grid["train"].to(dt).sum(1).clamp_min(1.0)
    with torch.enable_grad():
        for lo in range(0, N, chunk):
            sl = slice(lo, min(N, lo + chunk))
            lg = fleet.forward(grid["a"][sl], grid["b"][sl])
            y = grid["y"][:, sl]
            lab = lg.gather(-1, y.unsqueeze(-1)).squeeze(-1) - lg.mean(-1)
            gen_part = ((lab * grid["test"][:, sl].to(dt)).sum(1) / n_te).sum()
            tr_part = ((lab * grid["train"][:, sl].to(dt)).sum(1) / n_tr).sum()
            g1 = torch.autograd.grad(gen_part, [P[k] for k in keys], retain_graph=True)
            g2 = torch.autograd.grad(tr_part, [P[k] for k in keys])
            for k, a, b in zip(keys, g1, g2):
                Ggen[k] += a
                Gtr[k] += b
    Gmem = {k: Gtr[k] - Ggen[k] for k in keys}
    return Ggen, Gmem


def _group_sign_sum(fleet, batch: dict, active: torch.Tensor, m: int, gen: torch.Generator | None = None) -> dict:
    """Sum over m random groups of training examples of sign(group gradient), per parameter (resampled per call)."""
    A, B, Y, w = batch["A"], batch["B"], batch["Y"], batch["w"]
    R, n = A.shape
    P = fleet.params
    keys = list(P)
    order = torch.argsort(torch.rand(R, n, device=A.device), 1)
    S = {k: torch.zeros_like(P[k]) for k in keys}
    with torch.enable_grad():
        for idx in torch.tensor_split(order, m, dim=1):
            Ak, Bk, Yk, wk = A.gather(1, idx), B.gather(1, idx), Y.gather(1, idx), w.gather(1, idx)
            lg = fleet.forward(Ak, Bk)
            ce = F.cross_entropy(lg.transpose(1, 2), Yk, reduction="none")
            loss = (ce * wk).sum(1) / wk.sum(1).clamp_min(1e-12)
            loss = torch.where(active & torch.isfinite(loss), loss, torch.zeros_like(loss)).sum()
            gs = torch.autograd.grad(loss, [P[k] for k in keys])
            for k, g in zip(keys, gs):
                S[k] += torch.sign(g)
    return S


def _sep_stats(x: dict, G: dict, Ggen: dict, Gmem: dict, names: list[str]) -> torch.Tensor:
    """S(G) for vector x and part G, restricted to `names`."""
    gg = _dot(Ggen, Ggen, names)
    mg = _dot(Gmem, Ggen, names)
    c = mg / gg.clamp_min(1e-30)
    mperp_sq = (_dot(Gmem, Gmem, names) - c * mg).clamp_min(1e-30)

    def d_dot(v):  # <v, gen_hat - memperp_hat>
        vg, vm = _dot(v, Ggen, names), _dot(v, Gmem, names)
        return vg / gg.clamp_min(1e-30).sqrt() - (vm - c * vg) / mperp_sq.sqrt()

    xx = _dot(x, x, names).clamp_min(1e-30)
    Gx = _dot(G, x, names)
    return (-d_dot(G) + (Gx / xx) * d_dot(x)) / xx.sqrt()


def _rot_stat(x: dict, E: dict, Ggen: dict, Gmem: dict, names: list[str]) -> torch.Tensor:
    """Matched-angle rotation null: G = ||E|| (cos(th) x_hat + sin(th) r_perp_hat) with th = angle(E, x) and r random,
    orthogonal to x. Then G_perp = ||E|| sin(th) r_perp_hat."""
    xx = _dot(x, x, names).clamp_min(1e-30)
    ee = _dot(E, E, names).clamp_min(1e-30)
    cos_t = (_dot(E, x, names) / (xx * ee).sqrt()).clamp(-1, 1)
    sin_t = (1 - cos_t ** 2).clamp_min(0).sqrt()
    Rn = {k: torch.randn_like(x[k]) for k in names}
    rx = _dot(Rn, x, names)
    Rp = {k: Rn[k] - (rx / xx).view((-1,) + (1,) * (x[k].dim() - 1)) * x[k] for k in names}
    rp_norm = _dot(Rp, Rp, names).sqrt().clamp_min(1e-30)
    Gp = {k: (ee.sqrt() * sin_t / rp_norm).view((-1,) + (1,) * (x[k].dim() - 1)) * Rp[k] for k in names}
    # Gp is already orthogonal to x, so S = -<Gp, d>/||x|| = _sep_stats with G = Gp (its x-component is 0)
    return _sep_stats(x, Gp, Ggen, Gmem, names)


def _masked(x: dict, mask: dict, names) -> dict:
    return {k: x[k] * mask[k] for k in names}


def _random_mask_like(mask: dict, names) -> dict:
    count = sum(mask[k].flatten(1).sum(1).float() for k in names)
    total = sum(float(mask[k][0].numel()) for k in names)
    dens = count / total
    return {k: torch.rand_like(mask[k], dtype=torch.float32) < dens.view((-1,) + (1,) * (mask[k].dim() - 1)) for k in names}


def _topk_mask(x: dict, mask: dict, names) -> dict:
    count = sum(mask[k].flatten(1).sum(1) for k in names)
    allabs = torch.cat([x[k].flatten(1).abs() for k in names], 1)
    srt = allabs.sort(1, descending=True).values
    kk = count.clamp(min=1, max=srt.shape[1]) - 1
    thr = srt.gather(1, kk.view(-1, 1)).squeeze(1)
    thr = torch.where(count > 0, thr, torch.full_like(thr, float("inf")))
    return {k: x[k].abs() >= thr.view((-1,) + (1,) * (x[k].dim() - 1)) for k in names}


@torch.no_grad()
def signal_measure(fleet, opt, grid: dict, batch: dict, active: torch.Tensor) -> dict:
    """5A quantities per run and unit. Call after the step's gradient is in p.grad and before opt.step()."""
    units, shared = unit_names(fleet)
    allu = units + [shared]
    c0 = fleet.cfgs[0]
    P = fleet.params
    g = {k: P[k].grad.detach().clone() for k in P}
    Ggen, Gmem = _support_grads(fleet, grid)
    signs = _group_sign_sum(fleet, batch, active, c0.groups)
    tau = c0.route_tau
    amask = {k: (signs[k].abs() / c0.groups) >= tau for k in P}
    t = opt.t
    ell = {a: {k: opt.ema_g[a][k] / (1 - a ** max(t, 1)) for k in P} for a in opt.ema_g}
    u = opt.preconditioned(g)
    au = c0.route_alpha
    eu = {k: opt.ema_u[k] / (1 - au ** max(t, 1)) for k in P} if t > 0 else {k: u[k].clone() for k in P}
    out = {k: [] for k in ("S_time98", "S_time998", "S_agree", "S_rand", "S_topk", "S_rot98", "S_rot998",
                           "S_u_time", "S_u_agree", "S_u_rand", "S_u_rot", "agree_density",
                           "proj_grad", "proj_decay", "proj_update", "gen_sq")}
    gen_all = sum(_dot(Ggen, Ggen, n) for n in allu).clamp_min(1e-30)
    decay = {k: -(fleet.lr_t[k] * fleet.wd_t[k]).view((-1,) + (1,) * (P[k].dim() - 1)) * P[k].detach() for k in P}
    upd = {k: -fleet.lr_t[k].view((-1,) + (1,) * (P[k].dim() - 1)) * u[k] for k in P}
    for names in allu:
        out["S_time98"].append(_sep_stats(g, ell[0.98], Ggen, Gmem, names))
        out["S_time998"].append(_sep_stats(g, ell[0.998], Ggen, Gmem, names))
        out["S_agree"].append(_sep_stats(g, _masked(g, amask, names), Ggen, Gmem, names))
        out["S_rand"].append(_sep_stats(g, _masked(g, _random_mask_like(amask, names), names), Ggen, Gmem, names))
        out["S_topk"].append(_sep_stats(g, _masked(g, _topk_mask(g, amask, names), names), Ggen, Gmem, names))
        out["S_rot98"].append(_rot_stat(g, ell[0.98], Ggen, Gmem, names))
        out["S_rot998"].append(_rot_stat(g, ell[0.998], Ggen, Gmem, names))
        out["S_u_time"].append(_sep_stats(u, eu, Ggen, Gmem, names))
        out["S_u_agree"].append(_sep_stats(u, _masked(u, amask, names), Ggen, Gmem, names))
        out["S_u_rand"].append(_sep_stats(u, _masked(u, _random_mask_like(amask, names), names), Ggen, Gmem, names))
        out["S_u_rot"].append(_rot_stat(u, eu, Ggen, Gmem, names))
        cnt = sum(amask[k].flatten(1).sum(1).float() for k in names)
        out["agree_density"].append(cnt / sum(float(amask[k][0].numel()) for k in names))
        gn = _dot(Ggen, Ggen, names).clamp_min(1e-30).sqrt()
        out["proj_grad"].append(-_dot(g, Ggen, names) / gn)
        out["proj_decay"].append(_dot(decay, Ggen, names) / gn)
        out["proj_update"].append(_dot(upd, Ggen, names) / gn)
        out["gen_sq"].append(_dot(Ggen, Ggen, names) / gen_all)
    res = {k: torch.stack(v, 1) for k, v in out.items()}  # (R, n_units + 1): units..., shared
    # signal masking: each branch's training gradient with the other branches present vs ablated
    from .fleet import branch_blocks
    br = branch_blocks(c0)
    if len(br) >= 2:
        M, pg, pa = [], [], []
        A, B, Y, w = batch["A"], batch["B"], batch["Y"], batch["w"]
        for bi, blocks in enumerate(br):
            others = [i for j, b in enumerate(br) if j != bi for i in b]
            al = fleet.alphas.clone()
            al[:, others] = 0.0
            names = units[bi]
            with torch.enable_grad():
                lg = fleet.forward(A, B, alphas=al)
                ce = F.cross_entropy(lg.transpose(1, 2), Y, reduction="none")
                loss = (ce * w).sum(1) / w.sum(1).clamp_min(1e-12)
                loss = torch.where(active & torch.isfinite(loss), loss, torch.zeros_like(loss)).sum()
                ga = dict(zip(names, torch.autograd.grad(loss, [P[k] for k in names])))
            gx = {k: g[k] for k in names}
            n_present = _dot(gx, gx, names).sqrt()
            n_abl = _dot(ga, ga, names).sqrt().clamp_min(1e-30)
            gn = _dot(Ggen, Ggen, names).clamp_min(1e-30).sqrt()
            M.append(1 - n_present / n_abl)
            pg.append(-_dot(gx, Ggen, names) / gn)
            pa.append(-_dot(ga, Ggen, names) / gn)
        res["mask_M"], res["mask_proj_present"], res["mask_proj_ablated"] = torch.stack(M, 1), torch.stack(pg, 1), torch.stack(pa, 1)
    return res


@torch.no_grad()
def probe(fleet, grid: dict, pairs: torch.Tensor) -> dict:
    """First-order probes on held-out pairs (R, P grid indices, test points). Per run, pair and unit:
    imprint = <grad lab_x, -grad ce_x>   (how much one plain gradient step on x raises x's own true-label logit)
    transfer = <grad gen_excl_x, -grad ce_x>  (how much it raises the test support, with x excluded from the test set)
    Both are exact additive decompositions over units (inner products split over parameter blocks)."""
    units, shared = unit_names(fleet)
    allu = units + [shared]
    P = fleet.params
    keys = list(P)
    Ggen, _ = _support_grads(fleet, grid)
    n_te = grid["test"].to(fleet.dtype).sum(1)
    imp = torch.zeros(fleet.R, pairs.shape[1], len(allu), device=fleet.device, dtype=fleet.dtype)
    tra = torch.zeros_like(imp)
    for j in range(pairs.shape[1]):
        idx = pairs[:, j]
        A, B = grid["a"][idx].unsqueeze(1), grid["b"][idx].unsqueeze(1)
        y = grid["y"].gather(1, idx.unsqueeze(1))
        with torch.enable_grad():
            lg = fleet.forward(A, B)
            ce = F.cross_entropy(lg.transpose(1, 2), y, reduction="none").squeeze(1)
            lab = (lg.gather(-1, y.unsqueeze(-1)).squeeze(-1) - lg.mean(-1)).squeeze(1)
            gce = dict(zip(keys, torch.autograd.grad(ce.sum(), [P[k] for k in keys], retain_graph=True)))
            glab = dict(zip(keys, torch.autograd.grad(lab.sum(), [P[k] for k in keys])))
        # gen with x excluded: (n_te * gen - lab_x) / (n_te - 1)
        nt = n_te.view(-1)
        gex = {k: (nt.view((-1,) + (1,) * (Ggen[k].dim() - 1)) * Ggen[k] - glab[k]) /
                  (nt - 1).clamp_min(1.0).view((-1,) + (1,) * (Ggen[k].dim() - 1)) for k in keys}
        neg = {k: -gce[k] for k in keys}
        for q, names in enumerate(allu):
            imp[:, j, q] = _dot(glab, neg, names)
            tra[:, j, q] = _dot(gex, neg, names)
    return {"imprint": imp, "transfer": tra}
