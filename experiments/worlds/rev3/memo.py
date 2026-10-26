# Revision-3 design computation (DESIGN.md §5). Run from a folder containing f6.npz and TT.npy (made by enum6.py).
# Feasibility of the memorization weight: on unseen-in-range functions (l=4,5), how often is the pool version space
# non-empty and do Bayes(a) (pool prior) and Bayes(b) (generator prior) differ at the next query?
import numpy as np
d=np.load("f6.npz"); T=d["T"].astype(np.int8); L=d["L"]
rng=np.random.default_rng(0)
nonid=L>=1
held=np.zeros(len(T),bool)
for l in range(1,6):
    ix=np.where(L==l)[0]; held[rng.choice(ix,size=max(1,round(0.1*len(ix))),replace=False)]=True
wb=np.zeros(len(T)); 
for l in range(1,7): m=(L==l); wb[m]=1/(6*m.sum())
def pool(n45):
    p=(L>=1)&(L<=3)&~held
    for l in (4,5):
        ix=np.where((L==l)&~held)[0]
        if n45 is not None: ix=rng.choice(ix,size=n45,replace=False)
        p[ix]=True
    w=np.zeros(len(T))
    for l in range(1,6):
        m=p&(L==l); w[m]=1/(5*m.sum())
    return w
def hist(w,cons,col):
    h=np.bincount(T[cons,col],weights=w[cons],minlength=7); s=h.sum(); return h/s if s>0 else None
unseen=np.where(held&((L==4)|(L==5)))[0]
for name,n45 in (("N100",100),("N400",400),("Nall",None)):
    wa=pool(n45); ne=np.zeros(9); usable=np.zeros(9); tvs=[[] for _ in range(9)]; E=300
    for e in range(E):
        f=rng.choice(unseen); idx=rng.choice(49,size=16,replace=False); z=T[f,idx]
        consA=wa>0; consB=wb>0
        for k in range(9):
            qa=hist(wa,consA,idx[k]); qb=hist(wb,consB,idx[k])
            if qa is not None:
                ne[k]+=1; tv=0.5*np.abs(qa-qb).sum(); tvs[k].append(tv); usable[k]+= tv>0.1
            consA&=T[:,idx[k]]==z[k]; consB&=T[:,idx[k]]==z[k]
    print(name, "k:", " ".join(f"{k}:{ne[k]/E:.2f}/{usable[k]/E:.2f}" for k in range(9)), "| median TV k=2,4:",
          f"{np.median(tvs[2]):.2f},{np.median(tvs[4]) if tvs[4] else float('nan'):.2f}")
print("format = share of contexts with non-empty pool version space / share also with TV(qa,qb) > 0.1")
