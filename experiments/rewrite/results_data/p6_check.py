import sys, statistics as st
sys.path.insert(0, "experiments/rewrite"); sys.path.insert(0, "src")
import analyze as A
runs = A.load_results(sys.argv[1] if len(sys.argv) > 1 else "results/kaggle_rewrite-main/results")
def stats(r):
    sel = [pi for pi in A.instances(r, "eval5", world=A.TRAIN_WORLDS) if pi.get("p_first") and pi.get("target_first") and pi["n_classes"] > 1 and len(pi["p_first"]) > 1]
    keymis = sum(set(pi["p_first"]) != set(pi["target_first"]) for pi in sel)
    tv, tvu, am, raw, byn = [], [], [], [], {}
    devs_m, devs_t = [], []
    for pi in sel:
        p, tg = pi["p_first"], pi["target_first"]
        tot = sum(p.values()); raw.append(tot)
        q = {k: v / tot for k, v in p.items()}
        n = len(p); u = 1.0 / n
        t = 0.5 * sum(abs(q.get(k, 0) - tg.get(k, 0)) for k in set(q) | set(tg))
        tu = 0.5 * sum(abs((u if k in p else 0) - tg.get(k, 0)) for k in set(p) | set(tg))
        tv.append(t); tvu.append(tu)
        am.append(max(q, key=q.get) == max(tg, key=tg.get))
        byn.setdefault(n, []).append((t, tu))
        for k in p:
            devs_m.append(q[k] - u); devs_t.append(tg.get(k, 0) - u)
    # correlation of deviations from uniform
    mm, mt = st.mean(devs_m), st.mean(devs_t)
    cov = sum((a - mm) * (b - mt) for a, b in zip(devs_m, devs_t)); 
    corr = cov / (sum((a - mm) ** 2 for a in devs_m) ** .5 * sum((b - mt) ** 2 for b in devs_t) ** .5 + 1e-12)
    uniform_targets = sum(tu < 1e-9 for tu in tvu)
    return dict(n=len(sel), keymis=keymis, tv=st.mean(tv), tvu=st.mean(tvu), argmax_agree=st.mean(am), raw_mass=st.mean(raw), corr=corr,
                uniform_targets=uniform_targets,
                tv_on_uniform_targets=st.mean([t for t, tu in zip(tv, tvu) if tu < 1e-9] or [0]),
                tv_on_nonuniform=st.mean([t for t, tu in zip(tv, tvu) if tu >= 1e-9] or [0]),
                tvu_on_nonuniform=st.mean([tu for tu in tvu if tu >= 1e-9] or [0]),
                byn={n: (len(v), round(st.mean(a for a, _ in v), 3), round(st.mean(b for _, b in v), 3)) for n, v in sorted(byn.items())})
for name in ["RT-post_s0", "RT-post_s6", "PF_s0", "RT-canon_s0"]:
    r = [x for x in runs if x.get("name") == name or x.get("cfg", {}).get("name") == name]
    if not r:
        print("missing", name, list(runs[0].keys())[:8]); continue
    s = stats(r[0])
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
print("--- all RT-post / PF seeds: non-uniform-target instances (n=1210), uniform-target instances (n=1003)")
for r in sorted(runs, key=lambda x: x.get("name", "")):
    nm = r.get("name") or r.get("cfg", {}).get("name")
    if nm.startswith(("RT-post", "PF")):
        s = stats(r)
        print(nm, round(s["tv_on_nonuniform"], 3), "vs", round(s["tvu_on_nonuniform"], 3), "| uniform-target", round(s["tv_on_uniform_targets"], 3), "| corr", round(s["corr"], 2), "| argmax", round(s["argmax_agree"], 2))
