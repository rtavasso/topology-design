"""Executed inside the Kaggle kernel (GPU). Finds the uploaded code and runs the job described in code/job.json:

  {"job": "validate"}                      fleet tests, benchmarks and the seed-paired GPU validation (default)
  {"job": "grid", "grid": "path/in/repo.json", "max_batch": 64, "tests": true}
                                           run a grid with run_fleet.py, one process per visible GPU; results go to
                                           /kaggle/working/results as .json.gz (prior results uploaded with the code
                                           in code/prior/ are copied in first, so an interrupted grid resumes)

Logs and results are written to /kaggle/working and come back with `kaggle kernels output`.
KAGGLE_WORKING / KAGGLE_INPUT override the paths (local testing)."""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile

W = os.environ.get("KAGGLE_WORKING", "/kaggle/working")
INP = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
TMP = os.environ.get("KAGGLE_TMP", "/tmp")
code = None
for t in glob.glob(f"{INP}/**/code.tar.gz", recursive=True):
    with tarfile.open(t) as tf:
        tf.extractall(TMP)
    code = os.path.join(TMP, "code")
if code is None:  # Kaggle may have auto-extracted the archive
    hits = glob.glob(f"{INP}/**/src/topology_design/fleet.py", recursive=True)
    if not hits:
        print("code not found under", INP, glob.glob(f"{INP}/**", recursive=True)[:50])
        sys.exit(1)
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
    code = os.path.join(TMP, "code")  # writable copy outside the working dir, so outputs stay logs + results
    shutil.copytree(src_root, code, dirs_exist_ok=True)
os.chdir(code)
job = json.load(open("job.json")) if os.path.exists("job.json") else {"job": "validate"}
print("code at", code, "job:", job, flush=True)
env = dict(os.environ, PYTHONPATH=os.path.join(code, "src"))
os.makedirs(W, exist_ok=True)


def run(name, cmd, extra_env=None):
    print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(W, f"{name}.txt"), "w") as f:
        p = subprocess.run(cmd, env={**env, **(extra_env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        f.write(p.stdout)
        print(p.stdout[-4000:], flush=True)
    return p.returncode


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=False)
run("env", [sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.is_available(), "
            "[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"])

if job["job"] == "validate":
    rc = run("tests", [sys.executable, "-m", "pytest", "-q", "tests/test_fleet.py", "tests/test_fleet_p5.py"])
    run("bench_p5", [sys.executable, "experiments/fleet/bench_p5.py", "--R", "64", "--steps", "300"])
    run("bench", [sys.executable, "experiments/fleet/bench_fleet.py", "--R", "16", "64", "256", "512", "1024"])
    run("bench_mixed", [sys.executable, "experiments/fleet/bench_fleet.py", "--R", "64", "256", "--mixed"])
    rv = run("validate", [sys.executable, "experiments/fleet/validate_gpu.py"])
    print(f"\nSUMMARY tests_rc={rc} validate_rc={rv}", flush=True)
elif job["job"] == "grid":
    out = os.path.join(W, "results")
    os.makedirs(out, exist_ok=True)
    if os.path.isdir("prior"):
        for f in os.listdir("prior"):
            shutil.copy(os.path.join("prior", f), out)
        print(f"resumed with {len(os.listdir(out))} prior results", flush=True)
    if job.get("tests", True):
        rc = run("tests", [sys.executable, "-m", "pytest", "-q", "-x", "tests/test_fleet.py", "tests/test_fleet_p5.py",
                           "tests/test_priors.py"])
        if rc != 0:
            print("TESTS FAILED: grid not run", flush=True)
            sys.exit(1)
    cmd = [sys.executable, "experiments/fleet/run_fleet.py", "--grid", job["grid"], "--out", out, "--gz",
           "--max-batch", str(job.get("max_batch", 64))]
    k = max(1, n_gpus())
    procs = []
    for i in range(k):  # one process per GPU, each taking every k-th batch
        log = open(os.path.join(W, f"grid_shard{i}.txt"), "w")
        e = {**env, "CUDA_VISIBLE_DEVICES": str(i)} if k > 1 else env
        procs.append((subprocess.Popen(cmd + ["--shard", f"{i}/{k}"], env=e, stdout=log, stderr=subprocess.STDOUT), log))
    rcs = []
    for p, log in procs:
        rcs.append(p.wait())
        log.close()
    for i in range(k):
        print(open(os.path.join(W, f"grid_shard{i}.txt")).read()[-3000:], flush=True)
    n = len([f for f in os.listdir(out) if f.endswith(".json.gz")])
    print(f"\nSUMMARY grid={job['grid']} shards={k} rcs={rcs} results={n}", flush=True)
else:
    sys.exit(f"unknown job {job}")
