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

import copy
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


def _group_sign_sum(fleet, batch: dict, active: torch.Tensor, m: int, gen: torch.Generator) -> dict:
    """Sum over m random groups of present training examples of sign(group gradient), per parameter."""
    from .fleet import seen_groups
    A, B, Y, w = batch["A"], batch["B"], batch["Y"], batch["w"]
    P = fleet.params
    keys = list(P)
    S = {k: torch.zeros_like(P[k]) for k in keys}
    with torch.enable_grad():
        for idx in seen_groups(w, m, gen):
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


def _rot_stat(x: dict, E: dict, Ggen: dict, Gmem: dict, names: list[str], gen: torch.Generator) -> torch.Tensor:
    """Matched-angle rotation null: G = ||E|| (cos(th) x_hat + sin(th) r_perp_hat) with th = angle(E, x) and r random,
    orthogonal to x. Then G_perp = ||E|| sin(th) r_perp_hat."""
    xx = _dot(x, x, names).clamp_min(1e-30)
    ee = _dot(E, E, names).clamp_min(1e-30)
    cos_t = (_dot(E, x, names) / (xx * ee).sqrt()).clamp(-1, 1)
    sin_t = (1 - cos_t ** 2).clamp_min(0).sqrt()
    from .fleet import rand_like_gen
    Rn = {k: rand_like_gen(x[k], gen, normal=True) for k in names}
    rx = _dot(Rn, x, names)
    Rp = {k: Rn[k] - (rx / xx).view((-1,) + (1,) * (x[k].dim() - 1)) * x[k] for k in names}
    rp_norm = _dot(Rp, Rp, names).sqrt().clamp_min(1e-30)
    Gp = {k: (ee.sqrt() * sin_t / rp_norm).view((-1,) + (1,) * (x[k].dim() - 1)) * Rp[k] for k in names}
    # Gp is already orthogonal to x, so S = -<Gp, d>/||x|| = _sep_stats with G = Gp (its x-component is 0)
    return _sep_stats(x, Gp, Ggen, Gmem, names)


def _masked(x: dict, mask: dict, names) -> dict:
    return {k: x[k] * mask[k] for k in names}


def _random_mask_like(mask: dict, names, gen: torch.Generator) -> dict:
    from .fleet import rand_like_gen
    count = sum(mask[k].flatten(1).sum(1).float() for k in names)
    total = sum(float(mask[k][0].numel()) for k in names)
    dens = count / total
    return {k: rand_like_gen(mask[k], gen) < dens.view((-1,) + (1,) * (mask[k].dim() - 1)) for k in names}


def _topk_mask(x: dict, mask: dict, names) -> dict:
    count = sum(mask[k].flatten(1).sum(1) for k in names)
    allabs = torch.cat([x[k].flatten(1).abs() for k in names], 1)
    srt = allabs.sort(1, descending=True).values
    kk = count.clamp(min=1, max=srt.shape[1]) - 1
    thr = srt.gather(1, kk.view(-1, 1)).squeeze(1)
    thr = torch.where(count > 0, thr, torch.full_like(thr, float("inf")))
    return {k: x[k].abs() >= thr.view((-1,) + (1,) * (x[k].dim() - 1)) for k in names}


@torch.no_grad()
def signal_measure(fleet, opt, grid: dict, batch: dict, active: torch.Tensor, st: dict | None = None) -> dict:
    """5A quantities per run and unit. Call after the step's gradient is in p.grad and before opt.step().

    g is the gradient the optimizer will use (p.grad: CE, plus any consolidation / unmasked-arm changes). The time
    splits use the EMAs as they will be after this step (including the current g / u), i.e. exactly what 5C time
    routing would use. Random draws come from a measurement-only generator, so measuring never changes training."""
    from .fleet import make_generator
    units, shared = unit_names(fleet)
    allu = units + [shared]
    c0 = fleet.cfgs[0]
    P = fleet.params
    gen = make_generator(fleet.device, 90_000 + opt.t)
    g = {k: P[k].grad.detach().clone() for k in P}
    Ggen, Gmem = _support_grads(fleet, grid)
    signs = _group_sign_sum(fleet, batch, active, c0.groups, gen)
    tau = c0.route_tau
    amask = {k: (signs[k].abs() / c0.groups) >= tau for k in P}
    t1 = opt.t + 1
    ell = {a: {k: (opt.ema_g[a][k] * a + (1 - a) * g[k]) / (1 - a ** t1) for k in P} for a in opt.ema_g}
    u = opt.preconditioned(g)
    au = c0.route_alpha
    eu = {k: (opt.ema_u[k] * au + (1 - au) * u[k]) / (1 - au ** t1) for k in P}
    out = {k: [] for k in ("S_time98", "S_time998", "S_agree", "S_rand", "S_topk", "S_rot98", "S_rot998",
                           "S_u_time", "S_u_agree", "S_u_rand", "S_u_topk", "S_u_rot", "agree_density",
                           "proj_grad", "proj_decay", "proj_update", "gen_sq")}
    gen_all = sum(_dot(Ggen, Ggen, n) for n in allu).clamp_min(1e-30)
    bs = lambda k: (-1,) + (1,) * (P[k].dim() - 1)
    # the three motions of one AdamW step, in parameter units: -lr*g (plain gradient step), decay, update -lr*u
    gstep = {k: -fleet.lr_t[k].view(bs(k)) * g[k] for k in P}
    decay = {k: -(fleet.lr_t[k] * fleet.wd_t[k]).view(bs(k)) * P[k].detach() for k in P}
    upd = {k: -fleet.lr_t[k].view(bs(k)) * u[k] for k in P}
    for names in allu:
        out["S_time98"].append(_sep_stats(g, ell[0.98], Ggen, Gmem, names))
        out["S_time998"].append(_sep_stats(g, ell[0.998], Ggen, Gmem, names))
        out["S_agree"].append(_sep_stats(g, _masked(g, amask, names), Ggen, Gmem, names))
        out["S_rand"].append(_sep_stats(g, _masked(g, _random_mask_like(amask, names, gen), names), Ggen, Gmem, names))
        out["S_topk"].append(_sep_stats(g, _masked(g, _topk_mask(g, amask, names), names), Ggen, Gmem, names))
        out["S_rot98"].append(_rot_stat(g, ell[0.98], Ggen, Gmem, names, gen))
        out["S_rot998"].append(_rot_stat(g, ell[0.998], Ggen, Gmem, names, gen))
        out["S_u_time"].append(_sep_stats(u, eu, Ggen, Gmem, names))
        out["S_u_agree"].append(_sep_stats(u, _masked(u, amask, names), Ggen, Gmem, names))
        out["S_u_rand"].append(_sep_stats(u, _masked(u, _random_mask_like(amask, names, gen), names), Ggen, Gmem, names))
        out["S_u_topk"].append(_sep_stats(u, _masked(u, _topk_mask(u, amask, names), names), Ggen, Gmem, names))
        out["S_u_rot"].append(_rot_stat(u, eu, Ggen, Gmem, names, gen))
        cnt = sum(amask[k].flatten(1).sum(1).float() for k in names)
        out["agree_density"].append(cnt / sum(float(amask[k][0].numel()) for k in names))
        gn = _dot(Ggen, Ggen, names).clamp_min(1e-30).sqrt()
        out["proj_grad"].append(_dot(gstep, Ggen, names) / gn)
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
        with torch.enable_grad():  # CE-only gradient with all branches present (p.grad may include auxiliary terms)
            lg = fleet.forward(A, B)
            ce = F.cross_entropy(lg.transpose(1, 2), Y, reduction="none")
            loss = (ce * w).sum(1) / w.sum(1).clamp_min(1e-12)
            loss = torch.where(active & torch.isfinite(loss), loss, torch.zeros_like(loss)).sum()
            g_ce = dict(zip(list(P), torch.autograd.grad(loss, list(P.values()))))
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
            gx = {k: g_ce[k] for k in names}
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


def _component_groups(fleet) -> list[list[int]]:
    """Function-space units for the matched probes: each branch's blocks (component index i+1), then the embedding
    path (component 0). Without branches: each block, then the embedding path."""
    from .fleet import branch_blocks
    units = branch_blocks(fleet.cfgs[0]) or [[i] for i in range(fleet.L)]
    return [[i + 1 for i in u] for u in units] + [[0]]


@torch.no_grad()
def _pair_and_test_support(fleet, grid: dict, run_idx: torch.Tensor, pair: torch.Tensor, chunk: int = 256):
    """For copies whose labels/test set are those of runs `run_idx` (Rc,): per component group, the centered
    true-label logit of `pair` (Rc,) and the test-label support with that pair excluded from the test set."""
    groups = _component_groups(fleet)
    dt = fleet.dtype
    N = grid["a"].shape[0]
    y = grid["y"][run_idx]
    test = grid["test"][run_idx].to(dt)
    Rc = run_idx.shape[0]
    S = torch.zeros(Rc, len(groups), device=fleet.device, dtype=dt)
    for lo in range(0, N, chunk):
        sl = slice(lo, min(N, lo + chunk))
        _, comps = fleet.forward(grid["a"][sl], grid["b"][sl], parts=True)
        yi = y[:, sl].unsqueeze(-1)
        for q, grp in enumerate(groups):
            Z = sum(comps[i] for i in grp)
            lab = Z.gather(-1, yi).squeeze(-1) - Z.mean(-1)
            S[:, q] += (lab * test[:, sl]).sum(1)
        del comps
    _, comps = fleet.forward(grid["a"][pair].unsqueeze(1), grid["b"][pair].unsqueeze(1), parts=True)
    yp = y.gather(1, pair.unsqueeze(1)).unsqueeze(-1)
    lab_x = torch.stack([(lambda Z: (Z.gather(-1, yp).squeeze(-1) - Z.mean(-1)).squeeze(1))(sum(comps[i] for i in grp))
                         for grp in groups], 1)
    in_test = test.gather(1, pair.unsqueeze(1))  # (Rc, 1): 1 if the pair is a test point
    n_ex = (test.sum(1, keepdim=True) - in_test).clamp_min(1.0)
    return lab_x, (S - in_test * lab_x) / n_ex


def _clone(fleet, opt, idx: torch.Tensor, seed: int):
    """Independent copies of runs `idx` (repeats allowed): parameters and full optimizer state."""
    from .fleet import make_generator
    fc = copy.copy(fleet)
    fc.params, fc.lr_t, fc.wd_t = dict(fleet.params), dict(fleet.lr_t), dict(fleet.wd_t)
    fc.subset(idx)
    oc = copy.copy(opt)
    oc.f = fc
    oc.m, oc.v = dict(opt.m), dict(opt.v)
    if opt.ema_u is not None:
        oc.ema_u = dict(opt.ema_u)
    if opt.ema_g is not None:
        oc.ema_g = {a: dict(d) for a, d in opt.ema_g.items()}
    if opt.gf_h is not None:
        oc.gf_h = dict(opt.gf_h)
    if opt.path_eff:
        oc.pe_sum = dict(opt.pe_sum)
    oc.subset(idx)
    oc.gen = make_generator(fleet.device, seed)
    oc.sign_sum = None
    return fc, oc


def matched_probe(fleet, opt, grid: dict, st: dict, A, B, Y, w, step: int, n_steps: int,
                  max_copies: int = 512) -> dict:
    """5E protocol-matched probes. For each run and probe pair: a copy of the run (parameters and optimizer state)
    trains `n_steps` steps with the pair added to its current training set; a control copy trains the same steps
    without it. Recorded per function-space unit (branches, then the embedding path; `_component_groups`), as
    treated minus control after the steps:
      matched_logit = the pair's own centered true-label logit;
      matched_gen   = test support with the pair excluded from the test set.
    The added pair does not enter the consolidation loss. Returns tensors (R, n_pairs, n_units)."""
    from .fleet import _loss_and_grads
    R, n = A.shape
    pairs = st["probe_pairs"]
    Pn = pairs.shape[1]
    dev, dt = fleet.device, fleet.dtype
    wf = torch.ones(R, n, device=dev, dtype=dt) if w is None else w

    def run_copies(idx, add_pair):
        Rc = idx.shape[0]
        fc, oc = _clone(fleet, opt, idx, 80_000 + step + int(idx.sum()))
        Ac, Bc, Yc, wc = A[idx], B[idx], Y[idx], wf[idx]
        wcons = wc
        stc = {k: (v[idx] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == R else v) for k, v in st.items()}
        if add_pair is not None:
            Ac = torch.cat([Ac, grid["a"][add_pair].unsqueeze(1)], 1)
            Bc = torch.cat([Bc, grid["b"][add_pair].unsqueeze(1)], 1)
            Yc = torch.cat([Yc, grid["y"][idx].gather(1, add_pair.unsqueeze(1))], 1)
            wc = torch.cat([wc, torch.ones(Rc, 1, device=dev, dtype=dt)], 1)
            wcons = torch.cat([wcons, torch.zeros(Rc, 1, device=dev, dtype=dt)], 1)
            if "cons_perm" in stc:
                stc["cons_perm"] = torch.cat([stc["cons_perm"], torch.full((Rc, 1), n, dtype=torch.long, device=dev)], 1)
                if stc.get("cons_ema") is not None:
                    e = stc["cons_ema"]
                    stc["cons_ema"] = torch.cat([e, torch.zeros(Rc, 1, e.shape[-1], device=dev, dtype=e.dtype)], 1)
        act = st["active"][idx].clone()
        for _ in range(n_steps):
            oc.zero_grad()
            per_run, _, _ = _loss_and_grads(fc, oc, Ac, Bc, Yc, wc, act, stc, wc=wcons)
            act = act & torch.isfinite(per_run)
            oc.step(act)
        return fc

    ar = torch.arange(R, device=dev)
    ctrl = run_copies(ar, None)
    ctrl_lab, ctrl_gen = [], []
    for j in range(Pn):
        lx, gx = _pair_and_test_support(ctrl, grid, ar, pairs[:, j])
        ctrl_lab.append(lx)
        ctrl_gen.append(gx)
    ctrl_lab, ctrl_gen = torch.stack(ctrl_lab, 1), torch.stack(ctrl_gen, 1)  # (R, Pn, G)
    lab = torch.zeros_like(ctrl_lab)
    gen = torch.zeros_like(ctrl_gen)
    per = max(1, max_copies // R)
    for lo in range(0, Pn, per):
        js = torch.arange(lo, min(Pn, lo + per), device=dev)
        idx = ar.repeat_interleave(len(js))
        pj = js.repeat(R)
        pr = pairs[idx, pj]
        fc = run_copies(idx, pr)
        lx, gx = _pair_and_test_support(fc, grid, idx, pr)
        lab[idx, pj] = lx
        gen[idx, pj] = gx
        del fc
    return {"matched_logit": lab - ctrl_lab, "matched_gen": gen - ctrl_gen}
