"""Measure training throughput (ms per run-step) for the unbatched CPU path vs batched CPU / MPS.

Usage (native macOS, from repo root):  .venv/bin/python experiments/grokking/bench_device.py
Safe to run while a CPU sweep is going: it uses 1 CPU thread, and the GPU is otherwise idle.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import torch
from topology_design import grokking as gk, grokking_batched as gb

torch.set_num_threads(1)


def bench(R, device, steps=40):
    cfgs = [gk.RunConfig(arm="sink_1", p=59, model_seed=s, data_seed=s) for s in range(R)]
    bm = gb.BatchedRoutedMLP(cfgs, device)
    a, b, y = gk.full_grid(59)
    tr = torch.stack([gk.split(59, 0.4, s)[0] for s in range(R)])
    A, B, Y = a[tr].to(device), b[tr].to(device), y[tr].to(device)
    opt = gb.BatchedAdamW(bm.params, torch.full((R,), 1e-3, device=device), torch.ones(R, device=device))
    n = A.shape[1]

    def step():
        opt.zero_grad()
        lg = bm.forward(A, B)
        torch.nn.functional.cross_entropy(lg.reshape(R * n, -1), Y.reshape(-1), reduction="none").view(R, n).mean(1).sum().backward()
        opt.step()

    for _ in range(5):
        step()
    if device == "mps":
        torch.mps.synchronize()
    t = time.time()
    for _ in range(steps):
        step()
    if device == "mps":
        torch.mps.synchronize()
    return (time.time() - t) / steps


base = bench(1, "cpu")
print(f"cpu  R=1 : {base * 1e3:6.1f} ms/step  ({base * 1e3:5.2f} ms per run-step)  [current sweep path]")
for R in (16,):
    t = bench(R, "cpu")
    print(f"cpu  R={R}: {t * 1e3:6.1f} ms/step  ({t / R * 1e3:5.2f} ms per run-step)")
if torch.backends.mps.is_available():
    for R in (1, 8, 16, 32, 64, 128):
        t = bench(R, "mps")
        print(f"mps  R={R:<3d}: {t * 1e3:6.1f} ms/step  ({t / R * 1e3:5.2f} ms per run-step; "
              f"{base / (t / R):5.1f}x one CPU core)")
else:
    print("MPS not available")
