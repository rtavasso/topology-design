"""Review experiment 2: what the gate knows when the dense phase ends.
A copy of loopmoe.run's training loop (same generators, same order of draws) with probes at chosen steps.
Probes use their own generator so the training data stream is untouched.
  mode "branch": the candidate on the branch task (arch_e G5); probe at the switch step and at the end.
  mode "reg0":   round B's network (reg0, softmax, dense), stopped at 300 steps; probes at steps 90 and 300.
"""
import sys, time, math
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/streams")
import torch
from loopmoe import DEF, Teacher, Student, make_programs, err


def probe(S, T, cfg, te, tag):
    """On held-out programs: mean top probability (all passes / IF passes / other passes), whether attention points at
    position t, the token -> top-stream map, and on IF passes how often the top stream is the one the student uses for
    the primitive the teacher's rule picks (given the student's own state). Also top-1 and dense held-out error."""
    g = torch.Generator().manual_seed(999)
    L = list(te.keys())[0]; pr = te[L]
    x = torch.randn(len(pr), 64, cfg["d"], generator=g)
    with torch.no_grad():
        k_save = S.c["topk"]
        S.c["topk"] = 0
        hd, aux = S.forward(pr, x, record=True)                      # dense forward: gates are the full softmax
        S.c["topk"] = 1
        h1, _ = S.forward(pr, x)
        S.c["topk"] = k_save
        tgt = T(pr, x)
        e_dense, e_top1 = err(hd, tgt, x).item(), err(h1, tgt, x).item()
        P = torch.stack(aux["gates"], 1)                             # [M, n, B, K]
        top = P.max(-1).values; arg = P.argmax(-1)                   # [M, n, B]
        tok = pr[:, :, None].expand_as(arg)
        att = torch.stack(aux["attn"], 1)                            # [M, n, B, n]
        att_ok = (att.argmax(-1) == torch.arange(L)[None, :, None]).double().mean().item()
        ntok = int(pr.max()) + 1
        tab = torch.stack([torch.bincount(arg[tok == u], minlength=cfg["K"]) for u in range(ntok)]).double()
        mode = tab.argmax(1).tolist()
        out = dict(tag=tag, conf=top.mean().item(), att_ok=att_ok, e_dense=e_dense, e_top1=e_top1, token_to_stream=mode,
                   injective=len(set(mode[:cfg["P"]])) == cfg["P"])
        if cfg["task"] == "branch":
            isif = tok == cfg["P"]
            out["conf_if"], out["conf_other"] = top[isif].mean().item(), top[~isif].mean().item()
            states = torch.stack(aux["states"], 1)                   # [M, n, B, d]
            want = torch.where(states @ T.w > 0, mode[0], mode[1])   # stream the student uses for the teacher's choice
            out["branch_agree"] = (arg[isif] == want[isif]).double().mean().item()
            out["if_frac_on_prim0_stream"] = (arg[isif] == mode[0]).double().mean().item()
    return out


def run_probed(over, seed, probes=(), stop=None, tseed=10):
    cfg = dict(DEF); cfg.update(over)
    T = Teacher(cfg, seed=tseed); S = Student(cfg, seed)
    tr, te = make_programs(cfg, seed)
    opt = torch.optim.Adam(S.params(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(5000 + seed)
    lens = list(tr.keys()); switch = None; res = []
    M0 = S.p["M"].detach().clone() if "M" in S.p else None
    for it in range(stop or cfg["steps"]):
        if it in probes:
            res.append(probe(S, T, cfg, te, f"step {it}"))
        pr = tr[lens[it % len(lens)]]
        x = torch.randn(len(pr), cfg["B"], cfg["d"], generator=g)
        hard = it >= cfg["anneal"] * cfg["steps"]
        if cfg["auto"]:
            hard = switch is not None
        h, aux = S.forward(pr, x, train=True, hard=hard)
        loss = err(h, T(pr, x), x)
        if cfg["auto"] and switch is None:
            with torch.no_grad():
                conf = torch.softmax(torch.stack(aux["logits"]), -1).max(-1).values.mean().item()
            if conf > cfg["auto"] or it >= cfg["steps"] // 2:
                switch = it + 1
                opt.zero_grad(); loss.backward(); opt.step()
                res.append(probe(S, T, cfg, te, f"switch (step {switch})"))
                continue
        opt.zero_grad(); loss.backward(); opt.step()
    res.append(probe(S, T, cfg, te, "end"))
    if M0 is not None:
        res[-1]["M_drift"] = ((S.p["M"].detach() - M0).norm() / M0.norm().clamp(min=1e-12)).item()
    return res


def show(label, seed, res):
    for r in res:
        print(f"{label} | seed {seed} | " + " | ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)


if __name__ == "__main__":
    mode, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    C = dict(progress="regsin", keys="sinus", score="softmax")
    BR = dict(task="branch", ntrain=64, gate_h=True, ln_h=False)
    for s in seeds:
        t0 = time.time()
        if mode == "branch":
            show("G5 candidate, branch", s, run_probed(dict(C, topk=1, auto=0.8, **BR), s))
        elif mode == "reg0":
            show("reg0 dense softmax (round B network)", s, run_probed(dict(progress="reg0", score="softmax"), s, probes=(90, 300), stop=301))
        elif mode == "reg0_8":
            show("reg0 dense softmax, 8 streams", s, run_probed(dict(progress="reg0", score="softmax", K=8, P=5, ntrain=80, B=8), s, probes=(90, 300), stop=301))
        elif mode == "teacher":                                       # s is the TEACHER seed here; student seed 0
            show(f"G5 candidate, branch, teacher seed {s}", 0, run_probed(dict(C, topk=1, auto=0.8, **BR), 0, tseed=s))
        elif mode == "b2":                                            # round B's B2: reg0, softmax, top-1 from scratch
            show("B2 reg0 top-1 from scratch", s, run_probed(dict(progress="reg0", score="softmax", topk=1), s))
        elif mode == "prog":
            show("G1 candidate, program", s, run_probed(dict(C, topk=1, auto=0.8), s))
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
