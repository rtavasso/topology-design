import gzip, json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
DATA = os.path.join(REPO, "data", "rewrite")
TRAIN_WORLDS = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
PAIRS = ["AI", "CI"]
TRIPLES = ["ACI", "ACD", "AID", "CID"]
ALL = ["A", "C", "I", "D", "AC", "AI", "AD", "CI", "CD", "ID", "ACI", "ACD", "AID", "CID", "ACID"]
def load(tag, worlds=ALL):
    out = {}
    for w in worlds:
        p = os.path.join(DATA, f"{tag}_{w}.jsonl.gz")
        if os.path.exists(p):
            with gzip.open(p, "rt") as f:
                out[w] = [json.loads(l) for l in f]
    return out
def lad(rs, d):
    return sum(bool(r["ladder"][str(d)]) for r in rs) / len(rs) if rs else float("nan")
DER = json.load(open(os.path.join(REPO, "experiments/rewrite/derive/derived.json")))
