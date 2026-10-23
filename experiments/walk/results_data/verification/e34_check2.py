import numpy as np, sys
sys.path.insert(0, '.')
from e3 import *
from variants import gvar
rows = build(); rows2 = build(mem_total='blocks')
for g, arm in [('asym','main'),('asym','lr'),('asym','wd'),('sym','main'),('sym','lr'),('sym','wd')]:
    s1=[];s2=[]
    for rho in RHOS:
        el=[(a,b) for a,b in zip(rows,rows2) if a['g']==g and a['arm']==arm and a['rho']==rho and a['el']]
        if el: s1.append(f"{np.median([a['sm'] for a,b in el]):.3f}"); s2.append(f"{np.median([b['sm'] for a,b in el]):.3f}")
    print(g,arm,'mem all7:',s1,'blocks:',s2)
for g in ['sym','asym']:
    med={}
    for rho in RHOS:
        el=[q for q in rows if q['g']==g and q['arm']=='main' and q['rho']==rho and q['el']]
        med[rho]=np.median([lg(gvar(q['r'],'rom_strict'))-lg(q['sm']) for q in el])
    print('strict',g,{k:round(v-med[1],3) for k,v in med.items()})
# runs flagged by sustained
for a in rows:
    r=a['r']
    if (a['g'],a['arm'],a['rho'],a['seed']) in [('asym','lr',4.0,621),('asym','main',4.0,624),('asym','main',2.0,627),('asym','lr',0.25,628)]:
        st,ta=r['step'],r['test_acc']
        print(a['g'],a['arm'],a['rho'],a['seed'],'t_grok',r['summary']['t_grok'],'final test',ta[-1].round(3),'min test after tgrok',ta[st>=r['summary']['t_grok']].min().round(3),
              'frac evals>=.95 after tgrok', (ta[st>=r['summary']['t_grok']]>=0.95).mean().round(2), 'sustained', sustained(st,ta))
print('eligible runs final test acc min:', min(q['r']['test_acc'][-1] for q in rows if q['el']))
low=[(q['g'],q['arm'],q['rho'],q['seed'],round(q['r']['test_acc'][q['r']['step']>q['r']['steps']-2000].min(),3)) for q in rows if q['el'] and q['r']['test_acc'][q['r']['step']>q['r']['steps']-2000].min()<0.95]
print('eligible runs with test acc <0.95 somewhere in last 2000:', len(low), low)
