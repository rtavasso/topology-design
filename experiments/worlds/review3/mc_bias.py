# Review 3: bias of the M=1024 Monte Carlo predictive (with the 1e-3 floor) vs the exact predictive of the same model.
# Model = a Bayes posterior over functions (exact arm analogue). Evaluated on the next query of each prefix.
import numpy as np, sys
d=np.load("f6.npz"); T=d["T"].astype(np.int8); L=d["L"]
rng=np.random.default_rng(5)
held=np.zeros(len(T),bool)
for l in range(1,6):
    ix=np.where(L==l)[0]; held[rng.choice(ix,size=max(1,round(0.1*len(ix))),replace=False)]=True
def prior_pool(n45):
    p=(L>=1)&(L<=3)&~held
    for l in (4,5):
        ix=np.where((L==l)&~held)[0]
        if n45: ix=rng.choice(ix,size=n45,replace=False)
        p[ix]=True
    w=np.zeros(len(T))
    for l in range(1,6): m=p&(L==l); w[m]=1/(5*m.sum())
    return w
wb=np.zeros(len(T))
for l in range(1,7): m=(L==l); wb[m]=1/(6*m.sum())
EPS=float(sys.argv[1]) if len(sys.argv)>1 else 1e-3
fl=lambda q:(1-EPS)*q+EPS/7
M=1024; R=8   # R MC replicates per point
def run(model_w, targets, name, E=150, ks=(0,1,2,3,4,6,8,12)):
    ex={k:[] for k in ks}; mc={k:[] for k in ks}; zero={k:[] for k in ks}
    for e in range(E):
        f=rng.choice(targets); idx=rng.choice(49,size=16,replace=False); z=T[f,idx]
        cons=model_w>0
        for k in range(13):
            if k in ks:
                pw=model_w*cons; s=pw.sum()
                if s==0: q=0.0; qhat=np.zeros(R)
                else:
                    pw=pw/s; x=idx[k]; q=pw[T[:,x]==z[k]].sum()
                    sup=np.where(pw>0)[0]; samp=rng.choice(sup,size=(R,M),p=pw[sup])
                    qhat=(T[samp,x]==z[k]).mean(1)
                ex[k].append(-np.log(fl(q))); mc[k].append(np.mean(-np.log(fl(qhat)))); zero[k].append(np.mean(qhat==0))
            cons&=T[:,idx[k]]==z[k]
    print(f"{name}: k: exact | MC-1024 | bias | share q̂=0")
    for k in ks: print(f"   k={k:2d}: {np.mean(ex[k]):.3f} | {np.mean(mc[k]):.3f} | {np.mean(mc[k])-np.mean(ex[k]):+.3f} | {np.mean(zero[k]):.2f}")
wa=prior_pool(400)
run(wa, np.where(wa>0)[0], "Bayes(a,N400) on pool functions")
run(wa, np.where(held&((L==4)|(L==5)))[0], "Bayes(a,N400) on unseen l=4-5")
wa5=prior_pool(None)
run(wa5, np.where(L==6)[0], "Bayes(a,Nall) on length l=6")
run(wb, np.where(L==6)[0], "Bayes(b) on length l=6")
