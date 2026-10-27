# Review 3: is the memorization-weight window driven by pool memorization or by Bayes(b)'s l=6 mass?
# Compare TV(Bayes(a), Bayes(b)) with TV(Bayes(a), Bayes(b5)), where b5 = generator prior restricted to l<=5 (incl. held-out).
import numpy as np
d=np.load("f6.npz"); T=d["T"].astype(np.int8); L=d["L"]
rng=np.random.default_rng(0)
held=np.zeros(len(T),bool)
for l in range(1,6):
    ix=np.where(L==l)[0]; held[rng.choice(ix,size=max(1,round(0.1*len(ix))),replace=False)]=True
wb=np.zeros(len(T)); wb5=np.zeros(len(T))
for l in range(1,7): m=(L==l); wb[m]=1/(6*m.sum())
for l in range(1,6): m=(L==l); wb5[m]=1/(5*m.sum())
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
    wa=pool(n45); E=300; ub=np.zeros(9); ub5=np.zeros(9); mass6=[[] for _ in range(9)]
    for e in range(E):
        f=rng.choice(unseen); idx=rng.choice(49,size=16,replace=False); z=T[f,idx]
        cA=wa>0; cB=wb>0
        for k in range(9):
            qa=hist(wa,cA,idx[k])
            if qa is not None:
                qb=hist(wb,cB,idx[k]); qb5=hist(wb5,cB&(wb5>0),idx[k])
                ub[k]+=0.5*np.abs(qa-qb).sum()>0.1; ub5[k]+=0.5*np.abs(qa-qb5).sum()>0.1
            mass6[k].append(wb[cB&(L==6)].sum()/wb[cB].sum())
            cA&=T[:,idx[k]]==z[k]; cB&=T[:,idx[k]]==z[k]
    print(name,"TV>0.1 vs b / vs b5:"," ".join(f"{k}:{ub[k]/E:.2f}/{ub5[k]/E:.2f}" for k in range(9)))
    print("   Bayes(b) posterior mass on l=6:"," ".join(f"{k}:{np.mean(mass6[k]):.2f}" for k in range(9)))
