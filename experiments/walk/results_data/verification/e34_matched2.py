import numpy as np, sys
sys.path.insert(0, '.')
from e3 import B0, B1, RHOS, build, lg, cancels
rows = build()
def comp(r, i, win):
    st = r['step']; w = np.abs(st - st[i]) <= win
    g = r['gen'][w].mean(0); m = r['mem'][w].mean(0)
    return g[B0].sum(), g[B1].sum(), m[B0].sum(), m[B1].sum()
fr = [0.0, 0.25, 0.5, 0.75, 1.0]
win = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f'per-run d = logit s_gen - logit s_mem at matched times (window +/-{win}); median d [n valid]; frac f of plateau t_mem->t_grok; also share medians gen/mem')
for g, arm in [('asym','main'),('asym','lr'),('asym','wd'),('sym','main'),('sym','lr'),('sym','wd')]:
    for rho in RHOS:
        el = [q for q in rows if q['g']==g and q['arm']==arm and q['rho']==rho and q['el']]
        if not el: continue
        cells = []
        negG = 0
        for f in fr:
            ds, sgs, sms = [], [], []
            for q in el:
                r = q['r']; st = r['step']; tm = q['t_mem']; tg = r['summary']['t_grok']
                i = int(np.argmin(np.abs(st - (tm + f * (tg - tm)))))
                G0, G1, M0, M1 = comp(r, i, win)
                if f == 0 and G0 + G1 < 0: negG += 1
                if G0 + G1 <= 0 or cancels(G0, G1) or M0 + M1 <= 0 or cancels(M0, M1): continue
                sg, sm = G0/(G0+G1), M0/(M0+M1)
                ds.append(lg(sg) - lg(sm)); sgs.append(sg); sms.append(sm)
            cells.append(f"f{f:.2f}: d={np.median(ds) if ds else np.nan:+.2f} ({np.median(sgs) if ds else np.nan:.2f}/{np.median(sms) if ds else np.nan:.2f}) [{len(ds)}]")
        endd = np.median([lg(q['sg']) - lg(q['sm']) for q in el])
        print(f"{g:4s} {arm:4s} rho{rho:<4}: gen total<0 at t_mem in {negG}/{len(el)}; " + ' | '.join(cells) + f" | prereg d={endd:+.2f}")
