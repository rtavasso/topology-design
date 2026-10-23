import numpy as np, sys
sys.path.insert(0,'.')
from e3 import *
rows=build()
for g in ('asym','sym'):
  for f in (0.5,0.75,1.0):
    rr=[]
    for q in rows:
        if q['g']!=g or q['arm']!='main' or not q['el']: continue
        r=q['r'];st=r['step'];tm=q['t_mem'];tg=r['summary']['t_grok']
        i=int(np.argmin(np.abs(st-(tm+f*(tg-tm)))))
        G=r['gen'][i];M=r['mem'][i];G0,G1,M0,M1=G[B0].sum(),G[B1].sum(),M[B0].sum(),M[B1].sum()
        if G0+G1<=0 or cancels(G0,G1) or cancels(M0,M1): continue
        rr.append(dict(rho=q['rho'],seed=q['seed'],d=lg(G0/(G0+G1))-lg(M0/(M0+M1)),sg=G0/(G0+G1),sm=M0/(M0+M1)))
    td,pd=perm_tau(rr,lambda q:q['d'],5000,two_sided=True)
    tg_,pg=perm_tau(rr,lambda q:lg(q['sg']),5000); tm_,pm=perm_tau(rr,lambda q:lg(q['sm']),5000)
    med={rho:np.median([q['d'] for q in rr if q['rho']==rho]) for rho in RHOS}
    print(g,f'f={f} n={len(rr)} tau_d={td:+.3f} p={pd:.4f}; tau_gen={tg_:.2f} p={pg:.4f} tau_mem={tm_:.2f} p={pm:.4f}; med d',{k:round(v,2) for k,v in med.items()})
