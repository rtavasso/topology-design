"""Probe runs on main splits (seed = split, as stage2.py main). Usage: res_probe_runs.py <label> <arm> <json overrides> <split> ...
Appends to res_probe_runs.json. Compares with the stored stage-2 runs of tok_mono and tok_streams (select, coverage 9)."""
import sys, time, json, os
W = "/home/claude/topology-design/experiments/theory/worlds"
sys.path.insert(0, W)
from stage0 import cfg, regime
from worlds import run
label, arm, over = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
cover = over.pop("cover", 9)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_probe_runs.json")
store = json.load(open(OUT)) if os.path.exists(OUT) else {}
tm = json.load(open(f"{W}/stage2_select_{cover}_tok_mono.json"))["runs"]; ts = json.load(open(f"{W}/stage2_select_{cover}_tok_streams.json"))["runs"]
for s in map(int, sys.argv[4:]):
    t0 = time.time(); r = run(cfg("select", arm, cover, s, **over), s); dt = time.time() - t0
    store.setdefault(label, {})[str(s)] = dict(seen=r["seen"], held=r["held"], nparams=r["nparams"], regime=regime(r, "select"), secs=dt, cover=cover)
    json.dump(store, open(OUT, "w"))
    print(f"{label} split {s} ({dt:.0f}s, {r['nparams']} params): seen {r['seen']:.4f} held {r['held']:.4f} excess {r['held']-r['seen']:.4f} {regime(r,'select')} | stored tok_mono {tm[s]['seen']:.4f}/{tm[s]['held']:.4f} tok_streams {ts[s]['seen']:.4f}/{ts[s]['held']:.4f}", flush=True)
