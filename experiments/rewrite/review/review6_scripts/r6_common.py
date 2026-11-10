"""Review 6 — independent loader for the rung-4 main-grid records. Does NOT import experiments/rewrite/analyze.py.

    python3 experiments/rewrite/review/review6_scripts/r6_common.py        # builds the compact cache (about 2 min)

Reads results/kaggle_rewrite-main/results/*.json.gz one file at a time and writes a compact cache
(path in the environment variable R6_CACHE; default <tempdir>/r6_cache.pkl, about 150 MB): instance metadata once per eval set (checked identical across the 41 runs)
and per-run outcome arrays. Every other r6_*.py script loads this cache through load()."""
import glob, gzip, json, os, pickle, sys, tempfile
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "results", "kaggle_rewrite-main", "results")
DERIVED = os.path.join(ROOT, "experiments", "rewrite", "derive", "derived.json")
CACHE = os.environ.get("R6_CACHE", os.path.join(tempfile.gettempdir(), "r6_cache.pkl"))

TRAIN = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
PAIRS = ["AI", "CI"]
TRIPLES = ["ACI", "ACD", "AID", "CID"]
FULL = ["ACID"]
HELD = PAIRS + TRIPLES + FULL
ARMS = ["RT-canon", "RT-post", "RT-all", "RT-absolute", "EI16", "PF", "RT-examples", "RT-shuffled"]
CATS = ["ok", "horizon", "wrong halt", "no halt at goal", "invalid: no match", "invalid: out of world"]
META_KEYS = ["world", "t", "s", "k", "binding", "GP1", "Dany", "size", "ladder", "target_first", "n_classes"]


def build():
    meta, runs = {}, {}
    for p in sorted(glob.glob(os.path.join(RES, "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            r = json.load(f)
        name = r["cfg"]["name"]
        run = {"cfg": r["cfg"], "params": r["params"], "router": r["router"], "seconds": r["seconds"], "ei_found": r.get("ei_found"),
               "losses": r["losses"], "shard": r.get("shard"), "curve": r["curve"], "final_cells": {t: v["cells"] for t, v in r["final"].items()}, "out": {}}
        for tag, v in r["final"].items():
            pis = v["per_instance"]
            m = [{k: pi[k] for k in META_KEYS} for pi in pis]
            if tag not in meta:
                meta[tag] = m
            else:
                assert len(meta[tag]) == len(m), (name, tag)
                for a, b in zip(meta[tag], m):
                    assert a == b, (name, tag, a, b)
            o = {"ok_m": np.array([bool(pi["ok_m"]) for pi in pis]), "ok_u": np.array([bool(pi["ok_u"]) for pi in pis]),
                 "cat_m": np.array([CATS.index(pi["cat_m"]) for pi in pis], dtype=np.int8),
                 "cat_u": np.array([CATS.index(pi["cat_u"]) for pi in pis], dtype=np.int8),
                 "steps_m": np.array([pi["steps_m"] for pi in pis], dtype=np.int16),
                 "first_ok": np.array([bool(pi["first_ok"]) for pi in pis]),
                 "first_canon": np.array([bool(pi.get("first_canon")) for pi in pis]),
                 "vis_m": np.array([bool(pi["visited_goal_m"]) for pi in pis]), "vis_u": np.array([bool(pi["visited_goal_u"]) for pi in pis]),
                 "p_first": [pi.get("p_first") for pi in pis] if tag == "eval5" else None}
            run["out"][tag] = o
        runs[name] = run
        print("loaded", name, file=sys.stderr)
        del r
    with open(CACHE, "wb") as f:
        pickle.dump({"meta": meta, "runs": runs}, f, protocol=4)
    return {"meta": meta, "runs": runs}


def load():
    if not os.path.exists(CACHE):
        return build()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def derived():
    return json.load(open(DERIVED))


def seeds(D, arm):
    return sorted([r for r in D["runs"].values() if r["cfg"]["arm"] == arm], key=lambda r: r["cfg"]["seed"])


def idx(meta, pred):
    return np.array([i for i, m in enumerate(meta) if pred(m)], dtype=np.int64)


def rate(run, tag, ix, key="ok_m"):
    return float(run["out"][tag][key][ix].mean()) if len(ix) else None


def lad(meta, ix, d):
    return float(np.mean([meta[i]["ladder"][str(d)] for i in ix])) if len(ix) else None


def med(v):
    v = sorted(x for x in v if x is not None)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def mmm(v, nd=3):
    v = [x for x in v if x is not None]
    return f"{med(v):.{nd}f} [{min(v):.{nd}f}, {max(v):.{nd}f}] n={len(v)}" if v else "-"


def tdepth(s):
    d = m = 0
    for ch in s:
        if ch == "(":
            d += 1; m = max(m, d)
        elif ch == ")":
            d -= 1
    return m


if __name__ == "__main__":
    D = build()
    print({t: len(m) for t, m in D["meta"].items()}, len(D["runs"]))
