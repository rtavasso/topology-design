"""Review probe 3: (a) a world-IRRELEVANT task for the monolith (all six primitives identical, so the update is the same
in every world and nothing has to be composed): does an unseen r alone break it? (b) the per-token monolith on a
cover-7 split with a cut vertex, per held-out world; (c) the per-token monolith at a second seed."""
import sys, time
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/worlds")
import torch, worlds
from probe import train, diag, err
t0 = time.time()
_init = worlds.Teacher.__init__
def init(self, cfg):
    _init(self, cfg)
    if cfg.get("same"):
        self.A = self.A[:1].expand(cfg["P"], -1, -1).clone()
worlds.Teacher.__init__ = init

def perworld(c, S, T, ws):
    with torch.no_grad():
        x = torch.randn(len(ws), 512, c["d"], generator=torch.Generator().manual_seed(11))
        return [err(S.forward(ws[m:m+1], x[m:m+1], c["n"], T), T.run(ws[m:m+1], x[m:m+1], c["n"]), x[m:m+1]).item() for m in range(len(ws))]

for job, seed in [(dict(arm="mono", task="sum", cover=9, same=True), 0),
                  (dict(arm="tok", task="select", cover=7), 2),
                  (dict(arm="tok", task="select", cover=9), 1),
                  (dict(arm="tok", task="sum", cover=9), 1)]:
    t = time.time()
    c, out, S, T, tr, te = train(job, seed)
    pw = perworld(c, S, T, te); d = diag(c, S, T, te)
    print(job, "seed", seed, f"seen {out['seen']:.4f} held {out['held']:.4f} ref {out.get('ref')} chosen/other {d['chosen']:.2f}/{d['other']:.2f} in/out {d['cin']:.2f}/{d['cout']:.2f} R2 {d['r2']:.2f}  {time.time()-t:.0f}s")
    print("   per held-out world:", {tuple(w): round(e, 3) for w, e in zip(te.tolist(), pw)}, flush=True)
    if job.get("cover") == 7:
        side = {0, 3}
        cr = [e for w, e in zip(te.tolist(), pw) if (w[0] in side) != (w[1] in side) and 2 not in w]
        nc = [e for w, e in zip(te.tolist(), pw) if not ((w[0] in side) != (w[1] in side) and 2 not in w)]
        print(f"   crossing the cut vertex: mean {sum(cr)/len(cr):.3f} (n={len(cr)}); others: mean {sum(nc)/max(len(nc),1):.3f} (n={len(nc)})")
print("total", round(time.time() - t0, 1))
