"""Probe: is 0.999 (held-out, forced) against 0.98 (seen) a margin selection effect? One tok_mono run (stage-2 config),
split given on the command line. Decision accuracy binned by the true trigger margin |<w_a - w_b, x>|."""
import sys, time, json, torch
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/worlds")
from stage0 import cfg
from stage2 import best_cfg
from worlds import run
arm, s = sys.argv[1], int(sys.argv[2])
a, c = best_cfg("select", arm)
t0 = time.time()
out, S, T, tr, te = run(cfg("select", a, 9, s, **c), s, ret_model=True)
dt = time.time() - t0
T.trained = torch.zeros(6, 6)
for u, v in tr.tolist(): T.trained[u, v] = T.trained[v, u] = 1.0
res = {}
with torch.no_grad():
    for name, ws in (("seen", tr), ("held", te)):
        x = torch.randn(len(ws), 4096, 6, generator=torch.Generator().manual_seed(31 + s))
        S.forward(ws, x, 1, T, record=True)
        pool = S.pools[0]
        right = (pool.argmax(-1) == T.choice(ws, x))
        sc = torch.einsum("mbj,mkj->mbk", x, T.w[ws]); margin = (sc[..., 1] - sc[..., 0])
        logit = torch.log(pool[..., 1] / pool[..., 0])      # learned bid difference
        det = T.determined(ws, x) if name == "held" else torch.ones_like(right)
        res[name] = (right, margin, det, logit)
qs = torch.quantile(res["seen"][1].abs().flatten(), torch.tensor([0.1, 0.25, 0.5, 0.75]).double())
edges = [0.0] + qs.tolist() + [1e9]
print(f"{arm} split {s}: train time {dt:.1f}s seen {out['seen']:.4f} held {out['held']:.4f}")
r, m, d, lg = res["seen"]; rh, mh, dh, lh = res["held"]
print("overall acc: seen %.3f  held forced %.3f  held unforced %.3f  share forced %.2f" % (r.double().mean(), rh[dh].double().mean(), rh[~dh].double().mean(), dh.double().mean()))
print("mean |margin|: seen %.2f  held forced %.2f  held unforced %.2f" % (m.abs().mean(), mh[dh].abs().mean(), mh[~dh].abs().mean()))
print("| |margin| bin | seen acc (n) | held forced acc (n) | held unforced acc (n) |")
for lo, hi in zip(edges[:-1], edges[1:]):
    f = lambda rr, mm, mask: (lambda k: "%.3f (%d)" % (rr[k].double().mean().item(), k.sum().item()) if k.sum() > 0 else "-")(mask & (mm.abs() >= lo) & (mm.abs() < hi))
    print(f"| {lo:.2f}-{min(hi,9.99):.2f} | {f(r, m, torch.ones_like(r))} | {f(rh, mh, dh)} | {f(rh, mh, ~dh)} |")
# margin-matched: reweight held-out decisions to the seen margin distribution (20 quantile bins)
qq = torch.quantile(m.abs().flatten(), torch.linspace(0, 1, 21).double()); qq[-1] = 1e9; qq[0] = 0
def matched(rr, mm, mask):
    acc = 0.0; w = 0.0
    for lo, hi in zip(qq[:-1], qq[1:]):
        k = mask & (mm.abs() >= lo) & (mm.abs() < hi)
        if k.sum() > 0: acc += rr[k].double().mean().item(); w += 1
    return acc / w
print("margin-matched to the seen distribution: seen %.3f  held forced %.3f  held unforced %.3f  held all %.3f" % (
    matched(r, m, torch.ones_like(r)), matched(rh, mh, dh), matched(rh, mh, ~dh), matched(rh, mh, torch.ones_like(rh))))
corr = lambda a_, b_: torch.corrcoef(torch.stack([a_.flatten(), b_.flatten()]))[0, 1].item()
print("corr(learned bid difference, true margin): seen %.3f held %.3f ; slope seen %.2f held %.2f" % (corr(lg, m), corr(lh, mh),
      (lg * m).sum() / (m * m).sum(), (lh * mh).sum() / (mh * mh).sum()))
