# Review 4: how ambiguous are contexts under the generator prior (l<=5, equal weight per length)?
# For k context pairs (distinct inputs), next query at a fresh input. Reports, averaged over the prior:
#  Bayes log loss (nats), effective #functions exp(H(post)), and log loss of predictors that ignore the
#  prior's 7x-per-length tilt: (u) uniform over consistent functions (l<=5), (s) uniform over the shortest
#  consistent functions, (m) committed MAP function with mass 1-eps, eps spread (floor 1e-2 as in DESIGN 6.4).
# Also: share of targets of length l whose k=8 context admits a strictly shorter consistent function (G2 spuriousness).
import numpy as np, sys
d=np.load("f6.npz"); T=d["T"]; L=d["L"]
keep=(L>=1)&(L<=5); T=T[keep].astype(np.int8); L=L[keep]
cnt=np.bincount(L,minlength=6); prior=(1/5)/cnt[L]; lp=np.log(prior)
rng=np.random.default_rng(1); NS=int(sys.argv[1]) if len(sys.argv)>1 else 1500
ks=[0,1,2,3,4,5,6,8,10,12]
acc={k:np.zeros(6) for k in ks}
tl=rng.choice(len(T),size=NS,p=prior/prior.sum())
for n,i in enumerate(tl):
    f=T[i]; order=rng.permutation(49)
    m=np.ones(len(T),bool)
    for k in range(13):
        if k in ks:
            q=order[k]; idx=np.where(m)[0]; w=prior[idx]; w=w/w.sum()
            out=T[idx,q]; pz=np.bincount(out,weights=w,minlength=7); true=f[q]
            bayes=-np.log(pz[true]); H=-(w*np.log(w)).sum()
            pu=np.bincount(out,minlength=7)/len(idx); uni=-np.log(pu[true])
            ml=L[idx].min(); sh=idx[L[idx]==ml]; ps=np.bincount(T[sh,q],minlength=7)/len(sh); sho=-np.log(max(ps[true],1e-2))
            mp=idx[np.argmax(w)]; mapl=-np.log(0.99 if T[mp,q]==true else 0.01/6)
            acc[k]+=[bayes,np.exp(H),uni,sho,mapl,(len(idx))]
        m&=(T[:,order[k]]==f[order[k]])
print(f"prior l<=5 equal per length, {NS} episodes; per z position, nats")
print("k  bayes  effN   uniform  shortest  MAP(floor)  |VS|")
for k in ks:
    a=acc[k]/NS; print(f"{k:2d} {a[0]:.3f} {a[1]:7.1f} {a[2]:.3f}   {a[3]:.3f}    {a[4]:.3f}   {a[5]:.0f}")
# G2 spuriousness at k=8
for l in (2,3,4,5):
    ix=rng.choice(np.where(L==l)[0],size=min(300,(L==l).sum()),replace=False); sp=0; vs=[]
    for i in ix:
        ctx=rng.choice(49,size=8,replace=False); mm=(T[:,ctx]==T[i,ctx]).all(1)
        sp+= (L[mm]<l).any(); vs.append(mm.sum())
    print(f"l={l}, k=8: share with a strictly shorter consistent fn {sp/len(ix):.2f}; median |VS(l<=5)| {np.median(vs):.0f}")
