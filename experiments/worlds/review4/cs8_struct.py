# Review 4: DESIGN §6.2 reports CS-8's best per-context fit as a per-input (marginal) KL. H2 is stated on E2, a
# FUNCTION-level quantity. Here: fit the static 8-chain mixture as review3/superpos_mix.py does (non-context mask,
# equal weight per length l<=5), then sample M paths from the fitted mixture, execute on all 49 inputs, and measure
# (i) mass on functions consistent with the context, (ii) TV between the sampled function distribution and the exact
# posterior, (iii) for comparison the TV of M samples from the exact posterior itself (estimator floor).
import numpy as np, torch, sys
torch.set_num_threads(2); torch.manual_seed(0); rng=np.random.default_rng(0)
R,STEPS,E,C,M=4,400,int(sys.argv[1]) if len(sys.argv)>1 else 8,8,8192
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy")
m=(L>=1)&(L<=5); F=T[m]; FL=L[m]
w=np.zeros(len(F))
for l in range(1,6): w[FL==l]=1/(5*(FL==l).sum())
key={F[i].tobytes():i for i in range(len(F))}
Tm=torch.zeros(33,49,49,dtype=torch.float64)
for a in range(32): Tm[a,torch.arange(49),torch.as_tensor(TT[a].astype(np.int64))]=1
Tm[32]=torch.eye(49)
R0=torch.zeros(49,7,dtype=torch.float64); R0[torch.arange(49),torch.arange(49)//7]=1
TTx=np.vstack([TT,np.arange(49,dtype=TT.dtype)[None]])  # action 32 = NOP
def fit(target,mask):
    th=(torch.randn(R*C,7,33,dtype=torch.float64)*0.5).requires_grad_(True); mx=torch.zeros(R,C,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([th,mx],lr=0.1); tl=target.clamp_min(1e-12).log(); mk=mask/mask.sum()
    def fwd():
        W=torch.softmax(th,2); b=torch.eye(49,dtype=torch.float64).expand(R*C,49,49)
        for t in range(7): b=b@torch.einsum('ra,aij->rij',W[:,t],Tm)
        q=((b@R0).reshape(R,C,49,7)*torch.softmax(mx,1)[:,:,None,None]).sum(1).clamp_min(1e-12)
        return W,((target*(tl-q.log())).sum(2)*mk).sum(1)
    for s in range(STEPS):
        _,kl=fwd(); opt.zero_grad(); kl.sum().backward(); opt.step()
    with torch.no_grad():
        W,kl=fwd(); i=int(kl.argmin())
        return kl[i].item(), W.reshape(R,C,7,33)[i].numpy(), torch.softmax(mx,1)[i].numpy()
def sample_fns(W,pi):
    comp=rng.choice(C,size=M,p=pi); s=np.tile(np.arange(49),(M,1))
    for t in range(7):
        P=W[comp,t]; u=rng.random((M,1)); a=(P.cumsum(1)<u).sum(1).clip(0,32)
        s=TTx[a[:,None],s]
    return s//7
print("k  neff  marginalKL  CS8:mass-consistent  CS8:mass-in-posterior-support  TV(CS8,post)  TV(post-samples,post)")
for k in (1,2,3):
    out=[]
    for e in range(E):
        f=rng.choice(len(F),p=w); idx=rng.choice(49,size=16,replace=False); ctx=idx[:k]
        cons=(F[:,ctx]==F[f,ctx]).all(1); pw=w*cons; pw/=pw.sum()
        tgt=np.zeros((49,7))
        for v in range(7): tgt[:,v]=(pw[:,None]*(F==v)).sum(0)
        mask=torch.ones(49,dtype=torch.float64); mask[torch.as_tensor(ctx)]=0
        kl,W,pi=fit(torch.tensor(tgt),mask)
        O=sample_fns(W,pi).astype(np.uint8)
        mc=(O[:,ctx]==F[f,ctx]).all(1).mean()
        ids=np.array([key.get(o.tobytes(),-1) for o in O]); ins=np.isin(ids,np.where(cons)[0]).mean()
        qh=np.bincount(ids[ids>=0],minlength=len(F))/M; tv=0.5*(np.abs(qh-pw).sum()+(ids<0).mean())
        ps=np.bincount(rng.choice(len(F),size=M,p=pw),minlength=len(F))/M; tv0=0.5*np.abs(ps-pw).sum()
        out.append([np.exp(-(pw[pw>0]*np.log(pw[pw>0])).sum()),kl,mc,ins,tv,tv0])
    o=np.median(out,0); print(f"{k}  {o[0]:6.1f}  {o[1]:.3f}       {o[2]:.2f}                 {o[3]:.2f}                          {o[4]:.2f}          {o[5]:.2f}",flush=True)
