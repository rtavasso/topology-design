"""Independent rewriting engine (reviewer's own; does not import topology_design)."""
import gzip, json, os
DATA = "REPO_ROOT/data/rewrite"
AX = {"Af": "A", "Ab": "A", "C": "C", "If": "I", "IbR": "I", "IbL": "I", "Df": "D", "Db": "D"}
RULES = ["Af", "Ab", "C", "If", "IbR", "IbL", "Df", "Db"]
TRAIN = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
HELD_B = ["AI", "CI", "ACI", "ACD", "AID", "CID"]

def parse(s):
    pos = 0
    def rec():
        nonlocal pos
        if s[pos] == "(":
            pos += 1; l = rec(); assert s[pos] == "+"; pos += 1; r = rec(); assert s[pos] == ")"; pos += 1
            return (l, r)
        c = s[pos]; pos += 1
        return c
    t = rec(); assert pos == len(s)
    return t

def leaves(t):
    return 1 if isinstance(t, str) else leaves(t[0]) + leaves(t[1])

def local(rule, u):
    leaf = isinstance(u, str)
    if rule == "Af":
        return None if leaf or isinstance(u[0], str) else (u[0][0], (u[0][1], u[1]))
    if rule == "Ab":
        return None if leaf or isinstance(u[1], str) else ((u[0], u[1][0]), u[1][1])
    if rule == "C":
        return None if leaf else (u[1], u[0])
    if rule == "If":
        if leaf: return None
        if u[1] == "0": return u[0]
        if u[0] == "0": return u[1]
        return None
    if rule == "IbR": return (u, "0")
    if rule == "IbL": return ("0", u)
    if rule == "Df":
        return None if leaf or u[0] != u[1] else u[0]
    if rule == "Db": return (u, u)

_mv = {}
def moves(t, cap):
    """All state-changing rewrites of t in the full world under the cap: [(rule, prefix_index, new)]."""
    key = (t, cap)
    r = _mv.get(key)
    if r is not None:
        return r
    out = []
    n = leaves(t)
    # enumerate nodes in prefix order with a rebuild closure
    idx = [0]
    def walk(u, rebuild):
        i = idx[0]; idx[0] += 1
        for rule in RULES:
            v = local(rule, u)
            if v is None: continue
            if n - leaves(u) + leaves(v) > cap: continue
            new = rebuild(v)
            if new != t:
                out.append((rule, i, new))
        if not isinstance(u, str):
            l, rr = u
            walk(l, lambda x, rr=rr, rebuild=rebuild: rebuild((x, rr)))
            walk(rr, lambda x, l=l, rebuild=rebuild: rebuild((l, x)))
    walk(t, lambda x: x)
    _mv[key] = out
    return out

def read(tag, w):
    with gzip.open(os.path.join(DATA, f"{tag}_{w}.jsonl.gz"), "rt") as f:
        out = []
        for l in f:
            r = json.loads(l)
            r.pop("post", None); r.pop("pf", None)
            out.append(r)
        return out
