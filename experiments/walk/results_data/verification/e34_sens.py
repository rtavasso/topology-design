import numpy as np, sys
sys.path.insert(0, '.')
import e3
from e3 import *
rows = build()
# clipping at boundaries
for g in ('sym','asym'):
    for rho in RHOS:
        el=[q for q in rows if q['g']==g and q['arm']=='main' and q['rho']==rho and q['el']]
        cl=lambda s: 'lo' if s<=0.02 else ('hi' if s>=0.98 else 'in')
        both=sum(cl(q['sg'])==cl(q['sm'])!='in' for q in el)
        print(g,rho,'both clipped same side:',both,'/',len(el),' mem clipped:',sum(cl(q['sm'])!='in' for q in el),' mem range',round(min(q['sm'] for q in el),3),round(max(q['sm'] for q in el),3))
for c in (0.005, 0.001):
    e3.lg = lambda s, c=c: np.log(np.clip(s,c,1-c)/(1-np.clip(s,c,1-c)))
    import e3 as m
    print(f'--- clip [{c},{1-c}]')
    for g in ('asym','sym'):
        el=[q for q in rows if q['g']==g and q['arm']=='main' and q['el']]
        tg,pg=m.perm_tau(el,lambda q:m.lg(q['sg']),2000); tm,pm=m.perm_tau(el,lambda q:m.lg(q['sm']),2000)
        td,pd=m.perm_tau(el,lambda q:m.lg(q['sg'])-m.lg(q['sm']),2000,two_sided=True)
        med={rho:np.median([m.lg(q['sg'])-m.lg(q['sm']) for q in el if q['rho']==rho]) for rho in RHOS}
        print(g,f'P1 gen {tg:.3f} p{pg:.4f} mem {tm:.3f} p{pm:.4f}; P2 tau {td:.3f} p {pd:.4f} shifts',{k:round(v-med[1],2) for k,v in med.items()})
e3.lg = lambda s: np.log(np.clip(s,0.02,0.98)/(1-np.clip(s,0.02,0.98)))
print('--- sustained-grok eligibility (asym)')
rows3=build(grok='sustained')
for g in ('asym',):
    cells={rho:[q for q in rows3 if q['g']==g and q['arm']=='main' and q['rho']==rho and q['el']] for rho in RHOS}
    print({k:len(v) for k,v in cells.items()})
    test=[r for r in RHOS if len(cells[r])>=6]; el=[q for r in test for q in cells[r]]
    tg,pg=perm_tau(el,lambda q:lg(q['sg']),5000); tm,pm=perm_tau(el,lambda q:lg(q['sm']),5000)
    td,pd=perm_tau(el,lambda q:lg(q['sg'])-lg(q['sm']),5000,two_sided=True)
    med={rho:np.median([lg(q['sg'])-lg(q['sm']) for q in cells[rho]]) for rho in test}
    print('testable',test,f'P1 gen {tg:.3f} p{pg:.4f} mem {tm:.3f} p{pm:.4f}; P2 tau {td:.3f} p {pd:.4f} shifts',{k:round(v-med[1],2) for k,v in med.items()})
