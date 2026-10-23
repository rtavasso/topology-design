import numpy as np, sys
sys.path.insert(0,'.')
from e3 import *
from scipy.stats import binomtest
rows=build()
for g,arm in [('sym','lr'),('sym','wd'),('sym','main')]:
  for f in (0.5,1.0):
    a={};b={}
    for q in rows:
        if q['g']!=g or q['arm']!=arm or not q['el']: continue
        r=q['r'];st=r['step'];tm=q['t_mem'];tg=r['summary']['t_grok']
        i=int(np.argmin(np.abs(st-(tm+f*(tg-tm)))))
        G=r['gen'][i];M=r['mem'][i]
        sg=G[B0].sum()/G[1:].sum(); sm=M[B0].sum()/M[1:].sum()
        (a if q['rho']==0.25 else b)[q['seed']]=(lg(sg),lg(sm))
    ks=sorted(set(a)&set(b))
    for j,n in ((0,'gen'),(1,'mem')):
        d=[b[k][j]-a[k][j] for k in ks]; npos=sum(x>0 for x in d); nneg=sum(x<0 for x in d)
        print(g,arm,f'f={f}',n,f'+{npos}/-{nneg}',f'median diff {np.median(d):+.2f}', f'p2={binomtest(npos,npos+nneg).pvalue:.3g}' if npos+nneg else '')
