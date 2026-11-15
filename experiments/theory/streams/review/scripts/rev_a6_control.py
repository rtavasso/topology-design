"""Review check: positive control for test A6 (does the running mean move at all when the batch is NOT symmetric?),
and how large the stand-in is on an ordinary (unsymmetrised) training batch."""
import sys
sys.path.insert(0, "/home/claude/topology-design/experiments/theory/streams")
import torch
from loopmoe import DEF, Student
torch.set_default_dtype(torch.float64)
cfg = dict(DEF); cfg.update(progress="clock", score="softmax", topk=1, fill="ema", n=1)
S = Student(cfg, 5); S.p["V"].data.mul_(10.0)
x = torch.randn(4, 16, 6, generator=torch.Generator().manual_seed(6))
progs = torch.tensor([[0], [1], [2], [0]])
for _ in range(5):
    S.forward(progs, x, train=True)
f = torch.einsum("kiw,mbkw->mbki", S.p["V"], torch.tanh(torch.einsum("kwj,mbj->mbkw", S.p["B"], x)))
print("ema max abs on an ordinary batch:", S.ema.abs().max().item(), "| typical |f| entry:", f.abs().mean().item())
