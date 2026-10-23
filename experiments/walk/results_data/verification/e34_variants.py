import numpy as np, sys
sys.path.insert(0, '.')
import e3
from e3 import *
def gvar(r, mode):
    st, T, gen = r['step'], r['steps'], r['gen']
    w = st > T - 2000 if mode.endswith('strict') else st >= T - 2000
    g0, g1 = gen[w][:, B0].sum(1), gen[w][:, B1].sum(1)
    if mode.startswith('rom'): return g0.mean() / (g0.mean() + g1.mean())
    if mode.startswith('meanshare'): return np.mean(g0 / (g0 + g1))
    if mode.startswith('medshare'): return np.median(g0 / (g0 + g1))
    if mode.startswith('meanlogit'):
        return 1/(1+np.exp(-np.mean(lg(g0/(g0+g1)))))
rows = build()
for mode in ['rom', 'rom_strict', 'meanshare', 'medshare', 'meanlogit']:
    s = []
    for g, arm in [('asym','main'),('sym','main'),('sym','lr'),('sym','wd')]:
        for rho in RHOS:
            el = [q for q in rows if q['g']==g and q['arm']==arm and q['rho']==rho and q['el']]
            if el: s.append(f'{np.median([gvar(q["r"],mode) for q in el]):.3f}')
    print(mode, ' '.join(s))
# sym P2 shifts with variants
for mode in ['rom', 'meanshare', 'medshare','meanlogit']:
  for g in ['sym','asym']:
    med = {}
    for rho in RHOS:
        el = [q for q in rows if q['g']==g and q['arm']=='main' and q['rho']==rho and q['el']]
        med[rho] = np.median([lg(gvar(q['r'],mode)) - lg(q['sm']) for q in el])
    print(mode, g, {k: round(v - med[1], 3) for k, v in med.items()})
# mem total variant
rows2 = build(mem_total='blocks')
print('mem blocks-only total: differs in t_mem for', sum(a['t_mem']!=b['t_mem'] for a,b in zip(rows,rows2)), 'runs; elig diff', sum(a['el']!=b['el'] for a,b in zip(rows,rows2)))
rows3 = build(grok='sustained')
print('sustained grok elig diff', [(a['g'],a['arm'],a['rho'],a['seed'],a['el'],b['el']) for a,b in zip(rows,rows3) if a['el']!=b['el']])
