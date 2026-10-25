"""Run a list of grids on Kaggle unattended: launch as many as the GPU-session limit allows, poll, download each
finished job, relaunch a job whose results came back incomplete (it resumes), start the next, and run each
experiment's analysis. Start it once and leave it.

    caffeinate -i .venv/bin/python experiments/fleet/kaggle/kaggle_queue.py --user riley9 \
        experiments/walk/grids/e1.json experiments/walk/grids/e2.json \
        experiments/walk/grids/e3.json experiments/walk/grids/e4.json

Jobs already running on Kaggle (e.g. launched earlier by hand) are detected and waited for, not relaunched.
State is kept in results/kaggle_queue.json, so stopping and restarting the queue is safe. Log: results/kaggle_queue.log.
On macOS, `caffeinate -i` keeps the laptop from idle-sleeping while the queue runs (closing the lid still sleeps it).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "fleet"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import launch  # noqa: E402

STATE = os.path.join(ROOT, "results", "kaggle_queue.json")
LOG = os.path.join(ROOT, "results", "kaggle_queue.log")
MAX_RELAUNCH = 2


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notify(msg: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "Kaggle queue"'], check=False)


def name_of(grid: str) -> str:
    return os.path.splitext(os.path.basename(grid))[0]


def kernel_of(grid: str) -> str:
    return f"topology-fleet-{launch.slug(name_of(grid))}"


def status(user: str, grid: str) -> str:
    """'running' | 'queued' | 'complete' | 'error' | 'cancel' | 'missing' | 'unknown'."""
    p = subprocess.run([launch.kaggle_exe(), "kernels", "status", f"{user}/{kernel_of(grid)}"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout.lower()
    if "404" in out or "not found" in out:
        return "missing"
    for s in ("running", "queued", "complete", "error", "cancel"):
        if s in out:
            return s
    # The CLI's wording for a kernel that was never pushed varies by version (2.2.4 matched none of the above, so
    # never-launched jobs were taken as running and the queue waited forever). Ask the kernel list directly.
    if not kernel_listed(user, grid):
        return "missing"
    log(f"  status {name_of(grid)}: unrecognized output: {p.stdout.strip()[-300:]!r}")
    return "unknown"


def kernel_listed(user: str, grid: str) -> bool:
    """True if the kernel appears in the user's own kernel list (so it has been pushed at least once)."""
    slug = kernel_of(grid)
    p = subprocess.run([launch.kaggle_exe(), "kernels", "list", "--mine", "--search", slug, "--page-size", "100"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return f"{user}/{slug}".lower() in p.stdout.lower()


def expected_runs(grid: str) -> int:
    from run_fleet import expand
    return len(expand(json.load(open(os.path.join(ROOT, grid)))))


def have_runs(grid: str) -> int:
    d = os.path.join(ROOT, "results", f"kaggle_{name_of(grid)}", "results")
    return len([f for f in os.listdir(d) if f.endswith((".json", ".json.gz"))]) if os.path.isdir(d) else 0


def run_launch(user: str, grid: str, fetch: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, os.path.join(HERE, "launch.py"), "--user", user, "--grid", grid,
           "--fetch" if fetch else "--no-wait"]
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join(p.stdout.strip().splitlines()[-3:])
    log(f"  {'fetch' if fetch else 'launch'} {name_of(grid)} -> rc {p.returncode}: {tail}")
    return p


def analyze(grid: str) -> None:
    script = os.path.join(os.path.dirname(os.path.join(ROOT, grid)), "..", "analyze.py")
    script = os.path.abspath(script)
    if not os.path.exists(script):
        return
    res = os.path.join(ROOT, "results", f"kaggle_{name_of(grid)}", "results")
    p = subprocess.run([sys.executable, script, name_of(grid), res], cwd=ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    log(f"  analysis {name_of(grid)} -> rc {p.returncode} (summary in results/kaggle_{name_of(grid)}/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grids", nargs="+", help="grid JSON paths (repo-relative), in launch order")
    ap.add_argument("--user", required=True)
    ap.add_argument("--max-sessions", type=int, default=2)
    ap.add_argument("--poll", type=int, default=300, help="seconds between status checks")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args()
    grids = [os.path.relpath(os.path.abspath(g), ROOT) for g in args.grids]
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for g in grids:  # re-check jobs recorded as running that were never actually pushed (the 2.2.4 status bug)
        if g in state and state[g]["phase"] == "running" and have_runs(g) == 0 and status(args.user, g) == "missing":
            state[g]["phase"] = "pending"
            log(f"{name_of(g)}: recorded as running but not on Kaggle -> pending")
    for g in grids:
        if g not in state:
            st = status(args.user, g)
            if st == "missing":
                phase = "pending"  # never launched
            elif st in ("complete", "error", "cancel") and have_runs(g) >= expected_runs(g):
                phase = "done"  # finished and already downloaded
            else:
                phase = "running"  # running, or finished on Kaggle but not downloaded yet: fetch it, don't relaunch
            state[g] = {"phase": phase, "relaunches": 0}
            log(f"{name_of(g)}: initial Kaggle status '{st}' -> {phase}")
    json.dump(state, open(STATE, "w"), indent=1)
    log(f"queue: {[name_of(g) for g in grids]}, max {args.max_sessions} sessions, poll every {args.poll}s")
    while True:
        # 1. collect finished jobs
        for g in grids:
            s = state[g]
            if s["phase"] != "running":
                continue
            st = status(args.user, g)
            if st in ("running", "queued", "unknown"):
                continue
            log(f"{name_of(g)}: Kaggle status '{st}'")
            run_launch(args.user, g, fetch=True)
            n, want = have_runs(g), expected_runs(g)
            log(f"{name_of(g)}: {n}/{want} results downloaded")
            if n >= want:
                s["phase"] = "done"
                notify(f"{name_of(g)} finished ({n} runs)")
                if not args.no_analyze:
                    analyze(g)
            elif s["relaunches"] < MAX_RELAUNCH:
                s["relaunches"] += 1
                s["phase"] = "pending"  # relaunch resumes from the downloaded results
                log(f"{name_of(g)}: incomplete, will relaunch to resume (attempt {s['relaunches']})")
            else:
                s["phase"] = "failed"
                notify(f"{name_of(g)} failed after {MAX_RELAUNCH} relaunches")
                log(f"{name_of(g)}: still incomplete after {MAX_RELAUNCH} relaunches; giving up")
            json.dump(state, open(STATE, "w"), indent=1)
        # 2. launch pending jobs while sessions are free
        for g in grids:
            running = sum(state[x]["phase"] == "running" for x in grids)
            if running >= args.max_sessions:
                break
            if state[g]["phase"] != "pending":
                continue
            p = run_launch(args.user, g, fetch=False)
            if p.returncode == 0:
                state[g]["phase"] = "running"
                log(f"{name_of(g)}: launched")
            elif "session count" in p.stdout.lower():
                log(f"{name_of(g)}: Kaggle session limit reached (another job is using a session); will retry")
                break
            else:
                log(f"{name_of(g)}: launch failed; will retry next poll")
            json.dump(state, open(STATE, "w"), indent=1)
        phases = {name_of(g): state[g]["phase"] for g in grids}
        if all(p in ("done", "failed") for p in phases.values()):
            log(f"all finished: {phases}")
            notify("all jobs finished")
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
