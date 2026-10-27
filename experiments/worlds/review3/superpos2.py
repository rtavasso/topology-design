# Review 3: re-derivation of DESIGN §5.1 with (i) batched restarts, (ii) KL on all 49 inputs (as superpos.py) vs only on the
# non-context inputs (what the training objective actually scores), (iii) more restarts/steps to separate optimization
# from expressivity. Same episode generator and prior as superpos.py (rng seed 0, equal weight per length l<=5).
import numpy as np, torch, sys, time
torch.set_num_threads(1)
R=int(sys.argv[1]); STEPS=int(sys.argv[2]); E=int(sys.argv[3]); PRIOR=sys.argv[4] if len(sys.argv)>4 else "all"
torch.manual_seed(0); rng=np.random.default_rng(0)
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy")
m=(L>=1)&(L<=5); F=T[m]; FL=L[m]
if PRIOR=="n400":   # N400-like pool: all l<=3 plus 400 random of each l=4,5
    r2=np.random.default_rng(7); keep=FL<=3
    for l in (4,5): ix=np.where(FL==l)[0]; keep[r2.choice(ix,400,replace=False)]=True
    F=F[keep]; FL=FL[keep]
w=np.zeros(len(F))
for l in range(1,6): w[FL==l]=1/(5*(FL==l).sum())
Tm=torch.zeros(33,49,49,dtype=torch.float64)
for a in range(32): Tm[a,torch.arange(49),torch.as_tensor(TT[a].astype(np.int64))]=1
Tm[32]=torch.eye(49)
R0=torch.zeros(49,7,dtype=torch.float64); R0[torch.arange(49),torch.arange(49)//7]=1
def fit(target, mask):
    th=(torch.randn(R,7,33,dtype=torch.float64)*0.5).requires_grad_(True); opt=torch.optim.Adam([th],lr=0.1)
    tl=target.clamp_min(1e-12).log(); mk=mask/mask.sum()
    for s in range(STEPS):
        W=torch.softmax(th,2); b=torch.eye(49,dtype=torch.float64).expand(R,49,49)
        for t in range(7): b=b@torch.einsum('ra,aij->rij',W[:,t],Tm)
        q=(b@R0).clamp_min(1e-12)
        kl=((target*(tl-q.log())).sum(2)*mk).sum(1)      # (R,) mean KL over masked inputs
        opt.zero_grad(); kl.sum().backward(); opt.step()
    with torch.no_grad():
        W=torch.softmax(th,2); b=torch.eye(49,dtype=torch.float64).expand(R,49,49)
        for t in range(7): b=b@torch.einsum('ra,aij->rij',W[:,t],Tm)
        q=(b@R0).clamp_min(1e-12); kl=((target*(tl-q.log())).sum(2)*mk).sum(1)
        klnc=((target*(tl-q.log())).sum(2)*mask).sum(1)/mask.sum()
    i=int(kl.argmin()); return kl[i].item(), (kl<kl.min()+1e-3).double().mean().item()
import os
ks=[int(x) for x in os.environ.get("KS","0,1,2,3,4,6,8,12").split(",")]; MODES=os.environ.get("MODES","all,nonctx").split(","); res={(k,mo):[] for k in ks for mo in ("all","nonctx")}; neff={k:[] for k in ks}; t0=time.time()
for e in range(E):
    f=rng.choice(len(F),p=w); idx=rng.choice(49,size=16,replace=False)
    for k in ks:
        cons=np.ones(len(F),bool)
        for j in idx[:k]: cons&=F[:,j]==F[f,j]
        pw=w*cons; pw/=pw.sum()
        tgt=np.zeros((49,7))
        for v in range(7): tgt[:,v]=(pw[:,None]*(F==v)).sum(0)
        tgt=torch.tensor(tgt)
        nonctx=torch.ones(49,dtype=torch.float64); nonctx[torch.as_tensor(idx[:k])]=0
        for mo,mask in [(a,b) for a,b in (("all",torch.ones(49,dtype=torch.float64)),("nonctx",nonctx)) if a in MODES]:
            res[(k,mo)].append(fit(tgt,mask)[0])
        neff[k].append(np.exp(-(pw[pw>0]*np.log(pw[pw>0])).sum()))
    print("episode",e,f"{time.time()-t0:.0f}s",flush=True)
print(f"R={R} steps={STEPS} E={E} prior={PRIOR}")
for k in ks:
    a=np.array(res[(k,"all")] or [np.nan]); n=np.array(res[(k,"nonctx")] or [np.nan])
    print(f"k={k:2d} neff {np.median(neff[k]):8.1f} | KL all-49: med {np.median(a):.3f} mean {a.mean():.3f} max {a.max():.3f} >0.05 {np.mean(a>0.05):.2f}"
          f" | KL non-context only: med {np.median(n):.3f} mean {n.mean():.3f} max {n.max():.3f} >0.05 {np.mean(n>0.05):.2f}")
