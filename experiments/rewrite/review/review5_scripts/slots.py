"""Which (prefix position, rule) slots are ever valid at cap 5 (training: size <= 5, cap 5)? Brute force over every term."""
import json
from eng import *
by = {1: ["a", "b", "c", "d", "0"]}
for n in range(2, 6):
    by[n] = [(l, r) for i in range(1, n) for l in by[i] for r in by[n - i]]
valid = {}
nterms = 0
for n in range(1, 6):
    for t in by[n]:
        nterms += 1
        for rule, i, v in moves(t, 5):
            valid[(i, rule)] = valid.get((i, rule), 0) + 1
        if nterms % 20000 == 0:
            import eng; eng._mv.clear()
print("terms", nterms)
for pos in range(0, 13):
    print(pos, {r: valid.get((pos, r), 0) for r in RULES})
json.dump(sorted([list(k) for k in valid]), open("valid_slots_cap5.json", "w"))
# canonical proofs of eval5 (cap 5): max position actually used, per training world
mx = {}
for w in TRAIN:
    m = -1; cnt = {}
    for r in read("eval5", w):
        assert r["cap"] == 5
        u = parse(r["t"])
        for a in r["canon"][:-1]:
            rule, path = a.split("@")
            # prefix index of path
            idx = 0; node = u
            for c in path:
                idx += 1 + (0 if c == "L" else (2 * leaves(node[0]) - 1))
                node = node[0] if c == "L" else node[1]
            m = max(m, idx); cnt[idx] = cnt.get(idx, 0) + 1
            nxt = [v for (rl, i, v) in moves(u, 5) if rl == rule and i == idx]
            assert len(nxt) == 1, (r["t"], a)
            u = nxt[0]
        assert u == parse(r["s"])
    mx[w] = (m, dict(sorted(cnt.items())))
    print(w, "max canonical position in eval5:", m, mx[w][1])
