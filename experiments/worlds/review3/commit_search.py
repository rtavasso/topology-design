# Review 3: per-context trainability of a COMMITTED path policy under the design's sampled per-point MML estimator
# (the committed analogue of §5.1's per-context superposed fit, which DESIGN has no counterpart for).
# Tabular autoregressive policy over prefixes (the most expressive per-context policy), K paths per iteration,
# w_kj = hit_kj e^{-lam len_k} / sum_k' hit_k'j e^{-lam len_k'}, points without a hit skipped, SGD on logits.
# Target: one function, scored on n random inputs (an identified-world context). Success = policy mass on paths whose
# function equals the target on all 49 inputs (sampled), plus held-out-input log loss with the design's floor.
import numpy as np, sys, time
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy").astype(np.int64)
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 1200; K=16; LAM=0.01; LR=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
HB=float(sys.argv[4]) if len(sys.argv)>4 else np.log(0.22*32/0.78)   # HALT bias -> expected length ~3
NPTS=int(sys.argv[5]) if len(sys.argv)>5 else 8
MODE=sys.argv[7] if len(sys.argv)>7 else "marg"; K=int(sys.argv[8]) if len(sys.argv)>8 else 16
def run(target):
    logits={}
    def node(p):
        if p not in logits: z=np.zeros(33); z[32]=HB; logits[p]=z
        return logits[p]
    def sample():
        p=(); s=np.arange(49)
        while True:
            z=node(p); pr=np.exp(z-z.max()); pr/=pr.sum()
            a=32 if len(p)==7 else rng.choice(33,p=pr)
            if a==32: return p,s//7
            s=TT[a][s]; p=p+(a,)
    pts=rng.choice(49,size=NPTS,replace=False); hold=np.setdiff1d(np.arange(49),pts)
    for it in range(ITERS):
        paths=[sample() for _ in range(K)]
        O=np.stack([o for _,o in paths]); ln=np.array([len(p) for p,_ in paths])
        hb=(O[:,pts]==target[pts])
        if MODE=="marg":
            hit=hb*np.exp(-LAM*ln)[:,None]; H=hit.sum(0)
            if (H>0).sum()==0: continue
            w=(hit[:,H>0]/H[H>0]).sum(1)/NPTS
        else:
            eps=float(MODE[2:]); ll=hb.sum(1)*np.log(1-eps)+(~hb).sum(1)*np.log(eps/6)-LAM*ln
            w=np.exp(ll-ll.max()); w/=w.sum()
        for (p,_),wk in zip(paths,w):
            if wk==0: continue
            full=p+(32,) if len(p)<7 else p
            for t in range(len(full)):
                pre=full[:t]; z=node(pre); pr=np.exp(z-z.max()); pr/=pr.sum()
                if t==7: break
                g=-pr; g[full[t]]+=1; logits[pre]=z+LR*wk*g
    S=[sample() for _ in range(1024)]; O=np.stack([o for _,o in S])
    exact=(O==target).all(1).mean(); cons=(O[:,pts]==target[pts]).all(1).mean(); const=(O.min(1)==O.max(1)).mean()
    q=(O[:,hold]==target[hold]).mean(0); ll=-np.log((1-1e-3)*q+1e-3/7).mean()
    return exact,cons,const,ll,np.mean([len(p) for p,_ in S])
fs=[]
for l in (2,3,4,5):
    ix=np.where(L==l)[0]
    for _ in range(int(sys.argv[6]) if len(sys.argv)>6 else 10): fs.append((l,ix[rng.integers(len(ix))]))
t0=time.time(); R=[]
for l,f in fs:
    r=run(T[f]); R.append((l,)+r); print(l, " ".join(f"{x:.3f}" for x in r), f"{time.time()-t0:.0f}s", flush=True)
R=np.array(R)
print(f"MODE={MODE} ITERS={ITERS} K={K} LR={LR} HB={HB:.2f} NPTS={NPTS}")
for l in (2,3,4,5):
    m=R[:,0]==l; X=R[m]
    print(f"l={l}: mass on exact target {X[:,1].mean():.2f} (share of targets >0.5: {(X[:,1]>0.5).mean():.2f}), consistent-on-context {X[:,2].mean():.2f},"
          f" constant-output {X[:,3].mean():.2f}, held-out loss {X[:,4].mean():.2f} nats (share >0.05: {(X[:,4]>0.05).mean():.2f}), mean len {X[:,5].mean():.1f}")
