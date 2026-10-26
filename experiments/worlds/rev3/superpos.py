# Revision-3 design computation (DESIGN.md §5). Run from a folder containing f6.npz and TT.npy (made by enum6.py).
# Derive-before-measure: best superposed (factorized per-step) program vs exact Bayes predictive, by context size.
# Bayes prior: equal weight per length 1..5, uniform within length (all l<=5 functions; stands in for Bayes(a) at Nall).
# The superposed model: T=7 steps, each a softmax over 33 tokens (32 instructions + NOP), applied to a belief over 49
# register states for every input; output = marginal of r0. Optimized directly per context (best case, restarts).
import numpy as np, torch, sys
torch.manual_seed(0); rng=np.random.default_rng(0)
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy")
m=(L>=1)&(L<=5); F=T[m]; FL=L[m]; w=np.zeros(len(F))
for l in range(1,6): w[FL==l]=1/(5*(FL==l).sum())
Tm=torch.zeros(33,49,49)
for a in range(32): Tm[a,torch.arange(49),torch.as_tensor(TT[a].astype(np.int64))]=1
Tm[32]=torch.eye(49)
R0=torch.zeros(49,7); R0[torch.arange(49),torch.arange(49)//7]=1   # state -> r0
def fit(target, restarts=3, steps=400):
    best=1e9
    for r in range(restarts):
        th=torch.randn(7,33)*0.5; th.requires_grad_(True); opt=torch.optim.Adam([th],lr=0.1)
        for s in range(steps):
            W=torch.softmax(th,1); b=torch.eye(49)   # belief: input x -> state distribution (start state = x)
            for t in range(7): b=b@torch.einsum('a,aij->ij',W[t],Tm)
            q=(b@R0).clamp_min(1e-9)
            kl=(target*(target.clamp_min(1e-12).log()-q.log())).sum(1).mean()
            opt.zero_grad(); kl.backward(); opt.step()
        best=min(best,kl.item())
    return best
ks=[0,1,2,3,4,6,8,12]; E=int(sys.argv[1]) if len(sys.argv)>1 else 12
res={k:[] for k in ks}; neff={k:[] for k in ks}
for e in range(E):
    f=rng.choice(len(F),p=w); idx=rng.choice(49,size=16,replace=False)
    for k in ks:
        cons=np.ones(len(F),bool)
        for j in idx[:k]: cons&=F[:,j]==F[f,j]
        pw=w*cons; pw/=pw.sum()
        tgt=np.zeros((49,7))
        for v in range(7): tgt[:,v]=(pw[:,None]*(F==v)).sum(0)
        res[k].append(fit(torch.tensor(tgt,dtype=torch.float32))); neff[k].append(np.exp(-(pw[pw>0]*np.log(pw[pw>0])).sum()))
    print("episode",e,flush=True)
for k in ks: print(f"k={k:2d}: effective #functions {np.median(neff[k]):9.1f}   best superposed KL to Bayes predictive (nats/input): median {np.median(res[k]):.3f}, max {np.max(res[k]):.3f}, share > 0.05: {np.mean(np.array(res[k])>0.05):.2f}")
