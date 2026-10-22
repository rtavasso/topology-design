"""Executed inside the Kaggle kernel (GPU). Finds the uploaded code, runs the fleet tests, benchmark and validation,
and writes logs to /kaggle/working so they come back with `kaggle kernels output`."""

import glob
import os
import subprocess
import sys
import tarfile

W = "/kaggle/working"
code = None
for t in glob.glob("/kaggle/input/**/code.tar.gz", recursive=True):
    with tarfile.open(t) as tf:
        tf.extractall("/tmp")
    code = "/tmp/code"
if code is None:  # Kaggle may have auto-extracted the archive
    hits = glob.glob("/kaggle/input/**/src/topology_design/fleet.py", recursive=True)
    if not hits:
        print("code not found under /kaggle/input:", glob.glob("/kaggle/input/**", recursive=True)[:50])
        sys.exit(1)
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
    import shutil
    code = "/tmp/code"  # writable copy outside /kaggle/working, so outputs stay just the logs
    shutil.copytree(src_root, code, dirs_exist_ok=True)
os.chdir(code)
print("code at", code, "results_data:", os.listdir(os.path.join(code, "experiments", "paths", "results_data")), flush=True)
env = dict(os.environ, PYTHONPATH=os.path.join(code, "src"))


def run(name, cmd):
    print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(W, f"{name}.txt"), "w") as f:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(p.stdout)
        print(p.stdout[-4000:], flush=True)
    return p.returncode


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=False)
run("env", [sys.executable, "-c", "import torch;print(torch.__version__, torch.cuda.is_available(), "
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"])
rc = run("tests", [sys.executable, "-m", "pytest", "-q", "tests/test_fleet.py"])
run("bench", [sys.executable, "experiments/fleet/bench_fleet.py", "--R", "16", "64", "256", "512", "1024"])
run("bench_mixed", [sys.executable, "experiments/fleet/bench_fleet.py", "--R", "64", "256", "--mixed"])
rv = run("validate", [sys.executable, "experiments/fleet/validate_gpu.py"])
print(f"\nSUMMARY tests_rc={rc} validate_rc={rv}", flush=True)
