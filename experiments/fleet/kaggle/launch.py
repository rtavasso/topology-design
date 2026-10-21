"""Launch the fleet validation on a Kaggle GPU via the Kaggle API, wait for it, and download the logs.

Prerequisites:
  - Network access to www.kaggle.com and storage.googleapis.com (Kaggle stores uploads and outputs there).
  - A Kaggle API token (Kaggle -> Settings -> API -> Create New Token). Newer tokens (KGAT_...) go in
    ~/.kaggle/access_token or the KAGGLE_API_TOKEN environment variable; legacy tokens are a kaggle.json. Never commit it.
  - Your Kaggle username: --user, or KAGGLE_USERNAME, or read from a legacy kaggle.json.
  - kaggle CLI >= 1.7 (pip install -U kaggle) for KGAT tokens.
  - A phone-verified Kaggle account (required for GPU and internet in kernels).

Usage (from the repo root):  python experiments/fleet/kaggle/launch.py [--out results/kaggle_validate]

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
INCLUDE = ["src", "tests/test_fleet.py", "experiments/fleet", "experiments/paths/results_data/compact.json.gz",
           "pyproject.toml"]


def kaggle(*args, check=True):
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout.strip()[-2000:], flush=True)
    if check and p.returncode != 0:
        sys.exit(p.returncode)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kaggle_validate"))
    ap.add_argument("--dataset", default="topology-design-code")
    ap.add_argument("--kernel", default="topology-fleet-validate")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"))
    args = ap.parse_args()
    user = args.user
    if not user:
        cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
        legacy = os.path.join(cfg_dir, "kaggle.json")
        if not os.path.exists(legacy):
            sys.exit("Kaggle username needed: pass --user NAME or set KAGGLE_USERNAME")
        user = json.load(open(legacy))["username"]
    tmp = tempfile.mkdtemp()
    # 1. code snapshot as a private dataset
    ds = os.path.join(tmp, "dataset")
    os.makedirs(ds)
    with tarfile.open(os.path.join(ds, "code.tar.gz"), "w:gz") as tf:
        for rel in INCLUDE:
            tf.add(os.path.join(ROOT, rel), arcname=os.path.join("code", rel))
    json.dump({"title": args.dataset, "id": f"{user}/{args.dataset}", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(ds, "dataset-metadata.json"), "w"))
    exists = kaggle("datasets", "status", f"{user}/{args.dataset}", check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", ds, "-m", f"code {time.strftime('%Y-%m-%d %H:%M')}")
    else:
        kaggle("datasets", "create", "-p", ds)
    time.sleep(30)  # let the dataset version finish processing
    # 2. GPU script kernel
    kd = os.path.join(tmp, "kernel")
    os.makedirs(kd)
    shutil.copy(os.path.join(os.path.dirname(__file__), "run_on_kaggle.py"), kd)
    json.dump({"id": f"{user}/{args.kernel}", "title": args.kernel, "code_file": "run_on_kaggle.py",
               "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
               "enable_internet": True, "dataset_sources": [f"{user}/{args.dataset}"], "competition_sources": [],
               "kernel_sources": []}, open(os.path.join(kd, "kernel-metadata.json"), "w"))
    kaggle("kernels", "push", "-p", kd)
    # 3. wait and fetch
    while True:
        st = kaggle("kernels", "status", f"{user}/{args.kernel}", check=False).stdout.lower()
        if any(s in st for s in ("complete", "error", "cancel")):
            break
        time.sleep(args.poll)
    os.makedirs(args.out, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{args.kernel}", "-p", args.out)
    print(f"logs in {args.out}")


if __name__ == "__main__":
    main()
