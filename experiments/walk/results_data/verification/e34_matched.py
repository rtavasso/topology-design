"""Matched-time check: shares of gen and mem at the same eval (t_mem, plateau midpoint, end)."""
import numpy as np, sys
sys.path.insert(0, '.')
import e3
from e3 import B0, B1, RHOS, build, lg, cancels, perm_tau
rows = build()
def at(r, i):
    g, m = r['gen'][i], r['mem'][i]
    G0, G1, M0, M1 = g[B0].sum(), g[B1].sum(), m[B0].sum(), m[B1].sum()
    return dict(sg=G0/(G0+G1), sm=M0/(M0+M1), Gtot=G0+G1, Mtot=M0+M1, cg=cancels(G0,G1), cm=cancels(M0,M1),
                te=r['test_acc'][i], G0=G0, G1=G1, M0=M0, M1=M1)
def idx(st, t): return int(np.argmin(np.abs(st - t)))
def summarize(g, arm, rho, win=0):
    el = [q for q in rows if q['g']==g and q['arm']==arm and q['rho']==rho and q['el']]
    out = {}
    for name in ('t_mem', 'mid', 'end'):
        vals = []
        for q in el:
            r = q['r']; st = r['step']; T = r['steps']
            Gfin = q['G0'] + q['G1']
            tm = q['t_mem']; tg = r['summary']['t_grok']
            if name == 't_mem': i = idx(st, tm)
            elif name == 'mid': i = idx(st, (tm + tg) / 2)
            else: i = len(st) - 1
            if win:  # average raw components over +/- win steps before computing shares
                w = np.abs(st - st[i]) <= win
                g = r['gen'][w].mean(0); m = r['mem'][w].mean(0)
                G0, G1, M0, M1 = g[B0].sum(), g[B1].sum(), m[B0].sum(), m[B1].sum()
                d = dict(sg=G0/(G0+G1), sm=M0/(M0+M1), Gtot=G0+G1, Mtot=M0+M1, cg=cancels(G0,G1), cm=cancels(M0,M1), te=r['test_acc'][i])
            else:
                d = at(r, i)
            d['Grel'] = d['Gtot'] / Gfin; d['Mrel'] = d['Mtot'] / (q['M0'] + q['M1']); d['t'] = st[i]; d['tm']=tm; d['tg']=tg
            d['gnorm'] = (abs(d.get('G0',0))+abs(d.get('G1',0)))
            vals.append(d)
        med = lambda k: np.median([v[k] for v in vals])
        out[name] = (f"gen {med('sg'):+.2f} mem {med('sm'):+.2f} | gen/final {med('Grel'):.3f} mem/peak {med('Mrel'):.2f} "
                     f"test {med('te'):.2f} cancel g{sum(v['cg'] for v in vals)} m{sum(v['cm'] for v in vals)} /{len(vals)}")
        out[name+'_raw'] = vals
    return el, out
if __name__ == '__main__':
    win = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for g, arm in [('asym','main'),('asym','lr'),('asym','wd'),('sym','main'),('sym','lr'),('sym','wd')]:
        for rho in RHOS:
            el, out = summarize(g, arm, rho, win)
            if not el: continue
            tms = [q['t_mem'] for q in el]; tgs = [q['r']['summary']['t_grok'] for q in el]
            print(f"{g} {arm} rho{rho} (n={len(el)}; median t_mem {np.median(tms):.0f}, t_grok {np.median(tgs):.0f})")
            for k in ('t_mem','mid','end'): print(f"   {k:5s} {out[k]}")
