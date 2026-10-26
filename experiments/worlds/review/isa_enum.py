# Enumerate 2-register mod-7 ISA by BFS layers; per state: #minimal paths and the SET of flag-masks over minimal paths.
import numpy as np, time, sys, pickle
p=7; LMAX=int(sys.argv[1]) if len(sys.argv)>1 else 6
X,Y=np.meshgrid(np.arange(p),np.arange(p),indexing="ij"); X,Y=X.ravel(),Y.ravel()
start=np.stack([X,Y]).astype(np.int8)[None]  # (1,2,49)
ins=[]
for d in (0,1):
    for o in ("add","sub","mul"):
        for s1 in (0,1):
            for s2 in (0,1): ins.append((o,d,s1,s2))
    for o in ("neg","inc"):
        for s in (0,1): ins.append((o,d,s,None))
assert len(ins)==32
# flag bits: 0 A additive used; 1 M mul used; 2 N neg used; 3 Isrc inc reads r1; 4 Idst inc writes r1; 5 O mul after additive
def fmap(m,i):
    o,d,s1,s2=i
    if o=="mul":
        m2=m|2
        if m&1: m2|=32
        return m2
    m2=m|1
    if o=="neg": m2|=4
    if o=="inc":
        if s1==1: m2|=8
        if d==1: m2|=16
    return m2
FM=np.array([[fmap(m,i) for m in range(64)] for i in ins])  # (32,64)
def apply(S,i):
    o,d,s1,s2=i; out=S.copy(); a=S[:,s1].astype(np.int16)
    if o=="add": v=(a+S[:,s2])%p
    elif o=="sub": v=(a-S[:,s2])%p
    elif o=="mul": v=(a*S[:,s2])%p
    elif o=="neg": v=(-a)%p
    else: v=(a+1)%p
    out[:,d]=v.astype(np.int8); return out
rng=np.random.default_rng(1); W1=rng.integers(1,2**62,size=98,dtype=np.uint64); W2=rng.integers(1,2**62,size=98,dtype=np.uint64)
def skey(S):
    out=np.empty((len(S),2),np.uint64)
    for a in range(0,len(S),500000):
        f=S[a:a+500000].reshape(-1,98).astype(np.uint64)
        out[a:a+500000,0]=(f*W1).sum(1); out[a:a+500000,1]=(f*W2).sum(1)
    return out.view(np.dtype((np.void,16))).ravel()
def transform(bits,i):
    out=np.zeros_like(bits)
    for m in range(64):
        out|=((bits>>np.uint64(m))&np.uint64(1))<<np.uint64(FM[i,m])
    return out
layers=[start]; counts=[np.ones(1,np.int64)]; masks=[np.ones(1,np.uint64)]  # mask-set: bit m set if some min path has flags m
seen=np.sort(skey(start)); t0=time.time()
for L in range(1,LMAX+1):
    F=layers[-1]; n=len(F)
    ch=np.concatenate([apply(F,i) for i in ins]); k=skey(ch)
    par=np.tile(np.arange(n),32); iid=np.repeat(np.arange(32),n)
    pos=np.searchsorted(seen,k); pos[pos>=len(seen)]=0; old=seen[pos]==k
    keep=~old; k,ch,par,iid=k[keep],ch[keep],par[keep],iid[keep]
    uk,first,inv=np.unique(k,return_index=True,return_inverse=True)
    newS=ch[first]; m=len(uk)
    c=np.zeros(m,np.int64); np.add.at(c,inv,counts[-1][par])
    mk=np.zeros(m,np.uint64)
    pm=masks[-1]
    for i in range(32):
        sel=iid==i
        tb=transform(pm[par[sel]],i)
        np.bitwise_or.at(mk,inv[sel],tb)
    layers.append(newS); counts.append(c); masks.append(mk)
    seen=np.sort(np.concatenate([seen,uk]))
    print(f"L={L} states {m} paths {c.sum()} ({time.time()-t0:.0f}s)",flush=True)
# function level
fn={}  # r0 bytes -> [L, nprog, maskset]
for L in range(LMAX+1):
    S=layers[L]; r0=S[:,0]
    keys=[r.tobytes() for r in r0]
    for j,kk in enumerate(keys):
        e=fn.get(kk)
        if e is None: fn[kk]=[L,int(counts[L][j]),int(masks[L][j])]
        elif e[0]==L: e[1]+=int(counts[L][j]); e[2]|=int(masks[L][j])
pickle.dump(fn,open(sys.argv[2] if len(sys.argv)>2 else "fn.pkl","wb"))
print("functions",len(fn))
