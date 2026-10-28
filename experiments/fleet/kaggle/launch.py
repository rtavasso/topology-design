"""Launch a fleet job on a Kaggle GPU via the Kaggle API, wait for it, and download logs and results.

Jobs:
  validation (default):  python experiments/fleet/kaggle/launch.py --user NAME
  a grid:                python experiments/fleet/kaggle/launch.py --user NAME --grid experiments/fastslow/grids/pilot.json
  rung 0, staged:        python experiments/fleet/kaggle/launch.py --user NAME --ptr-gate experiments/ptr/grids/pilot3.json \
                             --ptr-grid experiments/ptr/grids/rev3.json
  rung 1:                python experiments/fleet/kaggle/launch.py --user NAME --route-grid experiments/route/grids/pilot.json
      One session runs the gate grid, applies the preregistered rule (experiments/ptr/gate.py) and runs the gated grid;
      without --no-wait the command waits and downloads everything (run it under `caffeinate -i` so the laptop stays
      awake), otherwise `--fetch` with the same arguments later.
      Results arrive in --out (default results/kaggle_<grid name>)/results/*.json.gz. Results already there are
      uploaded with the code, so relaunching the same command resumes an interrupted grid.
  --no-wait: push and exit (e.g. before closing the laptop); later `--fetch` with the same arguments waits and downloads.

Prerequisites:
  - Network access to www.kaggle.com and storage.googleapis.com (Kaggle stores uploads and outputs there).
  - A Kaggle API token (Kaggle -> Settings -> API -> Create New Token). Newer tokens (KGAT_...) go in
    ~/.kaggle/access_token or the KAGGLE_API_TOKEN environment variable; legacy tokens are a kaggle.json. Never commit it.
  - Your Kaggle username: --user, or KAGGLE_USERNAME, or read from a legacy kaggle.json.
  - kaggle CLI >= 1.7 (pip install -U kaggle) for KGAT tokens.
  - A phone-verified Kaggle account (required for GPU and internet in kernels).


The code is uploaded as a private Kaggle dataset (so the GitHub repo can stay private), and a private GPU script kernel
runs experiments/fleet/kaggle/run_on_kaggle.py against it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INCLUDE = ["src", "tests/test_fleet.py", "tests/test_fleet_p5.py", "tests/test_priors.py", "experiments/fleet",
           "pyproject.toml"]
PTR_INCLUDE = ["tests/test_ptr.py", "experiments/ptr/run_ptr.py", "experiments/ptr/gate.py"]
ROUTE_INCLUDE = ["tests/test_route.py", "tests/test_ptr.py", "experiments/route/run_route.py"]
VALIDATE_EXTRA = ["experiments/paths/results_data/compact.json.gz"]


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")


def kaggle_exe() -> str:
    """The kaggle CLI next to this Python (e.g. .venv/bin/kaggle), else in ./.venv, else whatever is on PATH."""
    for here in (os.path.join(os.path.dirname(sys.executable), "kaggle"), os.path.join(ROOT, ".venv", "bin", "kaggle")):
        if os.path.exists(here):
            return here
    found = shutil.which("kaggle")
    if not found:
        sys.exit("kaggle CLI not found. Run this with the Python that has it installed (e.g. .venv/bin/python ...), "
                 "or install it here: python -m pip install -U kaggle")
    return found


def kaggle(*args, check=True):
    cmd = [kaggle_exe(), *args]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout.strip()[-2000:], flush=True)
    if check and p.returncode != 0:
        sys.exit(p.returncode)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=None, help="grid JSON (repo-relative) to run instead of the validation")
    ap.add_argument("--bench-worlds", action="store_true",
                    help="throughput test of stacked models for the worlds round (experiments/worlds/DESIGN.md §8)")
    ap.add_argument("--ptr-grid", default=None, help="rung-0 grid JSON (experiments/ptr/grids/*.json) to run")
    ap.add_argument("--route-grid", default=None, help="rung-1 grid JSON (experiments/route/grids/*.json) to run")
    ap.add_argument("--ptr-gate", default=None, help="rung-0: a gate grid run first in the same session; gate.py then sets "
                    "LH-Δ's head multiplier in --ptr-grid before it runs (staged, unattended)")
    ap.add_argument("--per-gpu", type=int, default=1, help="ptr: shard processes per GPU (runs are stacked per process)")
    ap.add_argument("--max-stack", type=int, default=10, help="ptr: runs per stacked model (10 at T = 10 ≈ 7 GB on a T4; a T = 24 stack must stay ≤ 5; stacks split on OOM)")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--no-tests", action="store_true", help="grid jobs: skip the fleet tests before the grid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--no-wait", action="store_true", help="push the job and exit")
    ap.add_argument("--fetch", action="store_true", help="do not launch; wait for the job and download its output")
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"))
    args = ap.parse_args()
    name = "validate" if args.grid is None else os.path.splitext(os.path.basename(args.grid))[0]
    if args.bench_worlds:
        name = "bench-worlds"
    if args.route_grid:
        name = "route-" + os.path.splitext(os.path.basename(args.route_grid))[0]
    if args.ptr_grid:
        name = "ptr-" + os.path.splitext(os.path.basename(args.ptr_grid))[0]
        if args.ptr_gate:
            name = "ptr-" + os.path.splitext(os.path.basename(args.ptr_gate))[0] + "-" + os.path.splitext(os.path.basename(args.ptr_grid))[0]
    args.out = args.out or os.path.join(ROOT, "results", f"kaggle_{name}")
    special = args.grid is not None or args.bench_worlds or args.ptr_grid is not None or args.route_grid is not None
    args.kernel = args.kernel or ("topology-fleet-validate" if not special else f"topology-fleet-{slug(name)}")
    args.dataset = args.dataset or ("topology-design-code" if not special else f"{args.kernel}-code")
    user = args.user
    if not user:
        cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
        legacy = os.path.join(cfg_dir, "kaggle.json")
        if not os.path.exists(legacy):
            sys.exit("Kaggle username needed: pass --user NAME or set KAGGLE_USERNAME")
        user = json.load(open(legacy))["username"]
    if args.fetch:
        return wait_and_fetch(args, user)
    tmp = tempfile.mkdtemp()
    # 1. code snapshot (plus job description, grid and any prior results) as a private dataset
    ds = os.path.join(tmp, "dataset")
    os.makedirs(ds)
    job = {"job": "validate"}
    include = INCLUDE + VALIDATE_EXTRA
    if args.grid is not None:
        grid_rel = os.path.relpath(os.path.abspath(args.grid), ROOT)
        if not os.path.exists(os.path.join(ROOT, grid_rel)):
            sys.exit(f"grid not found: {args.grid}")
        job = {"job": "grid", "grid": grid_rel, "max_batch": args.max_batch, "tests": not args.no_tests}
        include = INCLUDE + [grid_rel]
    if args.bench_worlds:
        job = {"job": "bench_worlds"}
        include = INCLUDE + ["experiments/worlds/bench_stacked.py", "tests/test_stacked.py"]
    if args.ptr_grid:
        grid_rel = os.path.relpath(os.path.abspath(args.ptr_grid), ROOT)
        if not os.path.exists(os.path.join(ROOT, grid_rel)):
            sys.exit(f"grid not found: {args.ptr_grid}")
        job = {"job": "ptr", "grid": grid_rel, "per_gpu": args.per_gpu, "max_stack": args.max_stack}
        include = INCLUDE + PTR_INCLUDE + [grid_rel]
        if args.ptr_gate:
            gate_rel = os.path.relpath(os.path.abspath(args.ptr_gate), ROOT)
            if not os.path.exists(os.path.join(ROOT, gate_rel)):
                sys.exit(f"gate grid not found: {args.ptr_gate}")
            job["gate"] = gate_rel
            include.append(gate_rel)
    if args.route_grid:
        grid_rel = os.path.relpath(os.path.abspath(args.route_grid), ROOT)
        if not os.path.exists(os.path.join(ROOT, grid_rel)):
            sys.exit(f"grid not found: {args.route_grid}")
        job = {"job": "route", "grid": grid_rel, "per_gpu": args.per_gpu, "max_stack": args.max_stack}
        include = INCLUDE + ROUTE_INCLUDE + [grid_rel]
    jp = os.path.join(tmp, "job.json")
    with open(jp, "w") as f:
        json.dump(job, f)
    prior_dir = os.path.join(args.out, "results")
    prior = sorted(f for f in os.listdir(prior_dir) if f.endswith(".json.gz")) if os.path.isdir(prior_dir) else []
    with tarfile.open(os.path.join(ds, "code.tar.gz"), "w:gz") as tf:
        for rel in include:
            tf.add(os.path.join(ROOT, rel), arcname=os.path.join("code", rel))
        tf.add(jp, arcname="code/job.json")
        for f in prior:
            tf.add(os.path.join(prior_dir, f), arcname=os.path.join("code", "prior", f))
    print(f"job {job}; {len(prior)} prior results uploaded for resume", flush=True)
    json.dump({"title": args.dataset, "id": f"{user}/{args.dataset}", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(ds, "dataset-metadata.json"), "w"))
    # the status call can return 403 for a dataset that does not exist yet, so try a new version first and fall back
    v = kaggle("datasets", "version", "-p", ds, "-m", f"code {time.strftime('%Y-%m-%d %H:%M')}", check=False)
    if v.returncode != 0 or "error" in v.stdout.lower() or "not found" in v.stdout.lower():
        kaggle("datasets", "create", "-p", ds)
    for _ in range(30):  # wait until the new dataset version is ready before the kernel mounts it
        st = kaggle("datasets", "status", f"{user}/{args.dataset}", check=False).stdout.lower()
        if "ready" in st:
            break
        time.sleep(10)
    # 2. GPU script kernel
    kd = os.path.join(tmp, "kernel")
    os.makedirs(kd)
    shutil.copy(os.path.join(os.path.dirname(__file__), "run_on_kaggle.py"), kd)
    json.dump({"id": f"{user}/{args.kernel}", "title": args.kernel, "code_file": "run_on_kaggle.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
               "enable_internet": True, "dataset_sources": [f"{user}/{args.dataset}"], "competition_sources": [],
               "kernel_sources": []}, open(os.path.join(kd, "kernel-metadata.json"), "w"))
    pr = kaggle("kernels", "push", "-p", kd, check=False)
    if pr.returncode != 0 or "error" in pr.stdout.lower():
        msg = pr.stdout.strip().splitlines()[-1] if pr.stdout.strip() else "unknown error"
        hint = (" Kaggle allows 2 GPU sessions at once: relaunch this same command when one of your running jobs "
                "finishes (check with: kaggle kernels list --mine).") if "session count" in pr.stdout.lower() else ""
        sys.exit(f"NOT launched: {msg}.{hint}")
    if args.no_wait:
        print(f"pushed. Later: the same command with --fetch downloads into {args.out}")
        return
    wait_and_fetch(args, user)


def wait_and_fetch(args, user):
    while True:
        st = kaggle("kernels", "status", f"{user}/{args.kernel}", check=False).stdout.lower()
        if any(s in st for s in ("complete", "error", "cancel")):
            break
        time.sleep(args.poll)
    os.makedirs(args.out, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{args.kernel}", "-p", args.out)
    res = os.path.join(args.out, "results")
    loose = [f for f in os.listdir(args.out) if f.endswith(".json.gz")]
    if loose:  # the CLI may flatten subdirectories
        os.makedirs(res, exist_ok=True)
        for f in loose:
            os.replace(os.path.join(args.out, f), os.path.join(res, f))
    n = len([f for f in os.listdir(res) if f.endswith(".json.gz")]) if os.path.isdir(res) else 0
    print(f"logs in {args.out}" + (f"; {n} results in {res}" if n else ""))


if __name__ == "__main__":
    main()
