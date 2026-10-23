import numpy as np, sys
sys.path.insert(0,'.')
from e4 import *
def cause(r):
    st,ta=r['step'],r['train_acc']; tf=r['summary']['t_fit']
    if tf is None: return 'N'   # never fits
    w=(st>=tf)&(st<=tf+6000)
    if (ta[w]>=0.9).all(): return 'S' if tf+6000<=12000 else 's'
    # dip: when, and does it recover
    i=np.where(w&(ta<0.9))[0][0]
    return 'D' if ta[-1]>=0.9 else 'X'   # D: dips then recovers by end; X: dips and ends <0.9
for p in PAR:
  print('==',p,' cells: per seed cause; S stable, N never fits, D dip<0.9 in window but final>=0.9, X dip and final<0.9')
  for wd in (1.0,):
    for arm in ARMS:
        cells=[]
        for lr in LRS:
            rs=sorted([r for r in R if r['tag'].endswith(p) and r['arm']==arm and r['wd']==wd and r['lr']==lr],key=lambda r:r['seed'])
            cells.append(''.join(cause(r) for r in rs))
        print(f'  wd={wd} {arm}:',' '.join(cells))
# overall cause counts by lr for non-stable runs
import collections
c=collections.Counter((cause(r), r['lr']) for r in R)
print({lr:{k:c[(k,lr)] for k in 'SsNDX'} for lr in LRS})
# min train acc in window, and t_fit, for low-lr C3/C4 failures (std, wd1)
for r in R:
    if r['tag'].endswith('std') and r['arm'] in ('pk0120','pk0123') and r['wd']==1.0 and r['lr']<=0.002:
        st,ta=r['step'],r['train_acc'];tf=r['summary']['t_fit']
        if tf is None: print(r['arm'],r['lr'],r['seed'],'nofit, max train',ta.max().round(3)); continue
        w=(st>=tf)&(st<=tf+6000)
        print(r['arm'],r['lr'],r['seed'],'t_fit',tf,'min in window',ta[w].min().round(3),'at',st[w][np.argmin(ta[w])],'final',ta[-1].round(3),'final test',r['test_acc'][-1].round(3), 'n dips', int(((ta[1:]<0.9)&(ta[:-1]>=0.9)&(st[1:]>tf)).sum()))
