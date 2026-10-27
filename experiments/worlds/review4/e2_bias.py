# Review 4: is E2 (router's distribution over functions vs exact posterior, from M sampled compositions) estimable?
# Planted router = the exact posterior (should score 0). Prior: l<=5, equal weight per length. M = 8192 samples.
# Reports plug-in estimates for a perfect router: KL(p||q_hat) with q_hat floored at 1/(10M) or add-1/2 smoothing,
# KL(q_hat||p) (p exact), and TV(q_hat,p). Also the share of posterior mass on functions never sampled.
import numpy as np
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; keep=(L>=1)&(L<=5); T=T[keep].astype(np.int8); L=L[keep]
cnt=np.bincount(L,minlength=6); prior=(1/5)/cnt[L]
rng=np.random.default_rng(2); M=8192
print("k  effN  KL(p||q,floor) KL(p||q,add1/2) KL(q||p)  TV    unsampled-mass")
for k in (1,2,3,4,6):
    r=[]
    for e in range(60):
        i=rng.choice(len(T),p=prior/prior.sum()); ctx=rng.choice(49,size=k,replace=False)
        idx=np.where((T[:,ctx]==T[i,ctx]).all(1))[0]; p=prior[idx]/prior[idx].sum()
        c=np.bincount(rng.choice(len(idx),size=M,p=p),minlength=len(idx)); q=c/M
        a=-(p*np.log(p)).sum()
        k1=(p*np.log(p/np.maximum(q,1/(10*M)))).sum(); qa=(c+0.5)/(M+0.5*len(idx)); k2=(p*np.log(p/qa)).sum()
        nz=q>0; k3=(q[nz]*np.log(q[nz]/p[nz])).sum(); tv=0.5*np.abs(p-q).sum(); um=p[c==0].sum()
        r.append([np.exp(a),k1,k2,k3,tv,um])
    r=np.mean(r,0); print(f"{k}  {r[0]:6.1f}  {r[1]:.3f}          {r[2]:.3f}           {r[3]:.3f}    {r[4]:.3f}  {r[5]:.3f}")
