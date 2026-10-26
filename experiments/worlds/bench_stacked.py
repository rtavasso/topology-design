"""Throughput of stacked (run-batched) PFNs vs separate models (experiments/worlds/DESIGN.md §8).

    PYTHONPATH=src python experiments/worlds/bench_stacked.py --device auto --d 128 --layers 4 --R 1 9 27 54 --amp

Reports run-steps/s (one training step of one run = one run-step) for a stack of R runs, and for R separate models
trained one after another (the baseline the stacking is meant to beat).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from topology_design.stacked import StackedPFN  # noqa: E402


def step_fn(model, opt, R, B, K, Q, p, dev, amp, scaler):
    cx, cy, cf = (torch.randint(0, p, (R, B, K), device=dev) for _ in range(3))
    qx, qy, qt = (torch.randint(0, p, (R, B, Q), device=dev) for _ in range(3))
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
        out = model(cx, cy, cf, qx, qy)
        loss = F.cross_entropy(out.reshape(-1, p).float(), qt.reshape(-1), reduction="sum") / (B * Q)
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else:
        loss.backward(); opt.step()


def time_it(R, args, dev, separate=False, steps=None):
    steps = steps or args.steps
    nmod = R if separate else 1
    Rm = 1 if separate else R
    models = [StackedPFN(Rm, args.p, args.d, args.layers, args.heads, 4 * args.d).to(dev) for _ in range(nmod)]
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in models]
    scaler = torch.cuda.amp.GradScaler() if (args.amp and dev.type == "cuda") else None
    for m, o in zip(models, opts):  # warm-up
        step_fn(m, o, Rm, args.B, args.K, args.Q, args.p, dev, args.amp and dev.type == "cuda", scaler)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        for m, o in zip(models, opts):
            step_fn(m, o, Rm, args.B, args.K, args.Q, args.p, dev, args.amp and dev.type == "cuda", scaler)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t
    return R * steps / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--R", type=int, nargs="+", default=[1, 9, 27])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--B", type=int, default=64, help="episodes per run per step")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--Q", type=int, default=8)
    ap.add_argument("--p", type=int, default=7)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-separate", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if args.device == "auto" else args.device))
    print(f"device {dev}, d {args.d}, layers {args.layers}, B {args.B}, K {args.K}, Q {args.Q}, amp {args.amp}")
    for R in args.R:
        st = time_it(R, args, dev)
        line = f"R={R:3d}  stacked {st:8.1f} run-steps/s"
        if not args.no_separate:
            se = time_it(R, args, dev, separate=True, steps=max(2, args.steps // max(1, R // 3)))
            line += f"   separate {se:8.1f} run-steps/s   speed-up {st / se:5.1f}x"
        print(line, flush=True)


if __name__ == "__main__":
    main()
