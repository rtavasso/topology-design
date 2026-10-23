import numpy as np, sys
sys.path.insert(0,'.')
from e4 import *
def frac_ok(r):
    st,ta=r['step'],r['train_acc']; tf=r['summary']['t_fit']
    if tf is None: return 0.0
    w=(st>=tf)&(st<=tf+6000)
    # time-weighted fraction of window with train acc >= 0.9 (evals are unevenly spaced)
    s=st[w]; ok=(ta[w]>=0.9).astype(float)
    dt=np.diff(np.append(s, tf+6000)); return float((ok*dt).sum()/dt.sum())
def mono_lrstar(M,p,arm,wd):
    best=None
    for lr in LRS:
        if sum(M[(p,arm,wd,lr)])>=2: best=lr
        else: break
    return best
M=maps()
F={}
for r in R: F.setdefault((r['tag'].split('-')[1],r['arm'],r['wd'],r['lr']),[]).append(frac_ok(r))
fin={}
for r in R: fin.setdefault((r['tag'].split('-')[1],r['arm'],r['wd'],r['lr']),[]).append(r['train_acc'][r['step']>=10000].min()>=0.9 if r['summary']['t_fit'] else False)
for wd in (0.3,1.0,3.0):
  print(f'--- wd={wd}')
  for p in PAR:
    out=[]
    for arm in ARMS:
        ml=mono_lrstar(M,p,arm,wd)
        tol=[lr for lr in LRS if sum(f>=0.95 for f in F[(p,arm,wd,lr)])>=2]
        tol99=[lr for lr in LRS if sum(f>=0.99 for f in F[(p,arm,wd,lr)])>=2]
        late=[lr for lr in LRS if sum(fin[(p,arm,wd,lr)])>=2]
        meanfrac=np.mean([np.mean(F[(p,arm,wd,lr)]) for lr in LRS])
        out.append(f"C{ARMS.index(arm)+1}: mono={ml} tol95={max(tol) if tol else None}({len(tol)}) tol99={max(tol99) if tol99 else None}({len(tol99)}) late-stable={max(late) if late else None}({len(late)}) meanfrac={meanfrac:.2f}")
    print(p,' | '.join(out))
