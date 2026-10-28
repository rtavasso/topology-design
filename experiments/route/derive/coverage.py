"""The distinguishing quantity (DESIGN.md §5): what a state-blind policy can achieve. A fixed program is one word
w_1..w_L applied to every query (with HALT when the register holds an answer, which a step-indexed policy cannot
do, so we grant it the more generous reading: the program succeeds iff the query's word is a prefix of it); the
best fixed program of length L covers the largest fraction of the training distribution. Also the per-length
coverage of the most common word, and the number of distinct training words.

    PYTHONPATH=src python experiments/route/derive/coverage.py
"""
import collections, itertools, sys
import numpy as np
sys.path.insert(0, "src")
from topology_design.route.data import RouteConfig, generate, H, P

cfg = RouteConfig(); rng = np.random.default_rng(0)
ep = generate(cfg, 4000, rng)
words = ["".join("HP"[a] for a in ep.q_word[b, j, :ep.q_k[b, j]]) for b in range(4000) for j in range(cfg.Qn)]
cnt = collections.Counter(words); n = len(words)
print(f"{n} training queries, {len(cnt)} distinct words")
by_k = collections.defaultdict(collections.Counter)
for w in words:
    by_k[len(w)][w] += 1
print("per length: most common word and its coverage of that length")
for k in sorted(by_k):
    w, c = by_k[k].most_common(1)[0]
    print(f"  k={k}: n={sum(by_k[k].values()):5d}  {w:10s} covers {c / sum(by_k[k].values()):.3f}")
# best fixed program of length L: a query succeeds iff its word is a prefix of the program
for L in (4, 8, 12):
    best = max(itertools.product("HP", repeat=L), key=lambda prog: sum(c for w, c in cnt.items() if "".join(prog).startswith(w)))
    cov = sum(c for w, c in cnt.items() if "".join(best).startswith(w)) / n
    print(f"best fixed program of length {L}: {''.join(best)} covers {cov:.3f} of training queries "
          f"(all-H covers {sum(c for w, c in cnt.items() if set(w) == {'H'}) / n:.3f})")
