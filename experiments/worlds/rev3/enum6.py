# Revision-3 design computation: enumerates all functions of minimal length <= 6 with their tables -> f6.npz (~30 s).
# Enumerate all output functions of minimal length <= 6 (mod 7, 2 registers), store r0 tables and lengths.
import numpy as np, time
N=49
INS=[(o,d,a,b) for d in (0,1) for o in ("add","sub","mul") for a in (0,1) for b in (0,1)]+[(o,d,a,None) for d in (0,1) for o in ("neg","inc") for a in (0,1)]
TT=np.zeros((32,N),np.uint8)
for i,(o,d,a,b) in enumerate(INS):
    for p in range(N):
        r=[p//7,p%7]; x=r[a]; y=0 if b is None else r[b]
        v={"add":x+y,"sub":x-y,"mul":x*y,"neg":-x,"inc":x+1}[o]%7; r[d]=v; TT[i,p]=7*r[0]+r[1]
np.save("TT.npy",TT)
rng=np.random.default_rng(3); W=rng.integers(1,2**62,size=(2,N),dtype=np.uint64)
def key(M):
    out=np.empty((len(M),2),np.uint64)
    for s in range(0,len(M),200000):
        m=M[s:s+200000].astype(np.uint64); out[s:s+200000,0]=(m*W[0]).sum(1); out[s:s+200000,1]=(m*W[1]).sum(1)
    return out.view(np.dtype((np.void,16))).ravel()
S=np.arange(N,dtype=np.uint8)[None]; seen=np.sort(key(S))
tabs=[S//7]; lens=[np.zeros(1,np.int8)]; t0=time.time()
for L in range(1,7):
    chs,ks=[],[]
    for i in range(32):
        c=TT[i][S]; kk=key(c); pos=np.searchsorted(seen,kk); pos[pos>=len(seen)]=0; new=seen[pos]!=kk
        chs.append(c[new]); ks.append(kk[new])
    ch=np.concatenate(chs); k=np.concatenate(ks); del chs,ks
    u,first=np.unique(k,return_index=True); S=ch[first]; del ch; seen=np.sort(np.concatenate([seen,u]))
    if L==6: seen=None
    tabs.append(S//7); lens.append(np.full(len(S),L,np.int8)); print(L,len(S),f"{time.time()-t0:.0f}s",flush=True)
T=np.concatenate(tabs).astype(np.uint8); Ls=np.concatenate(lens)
k=key(T); order=np.lexsort((Ls,k)); k,T,Ls=k[order],T[order],Ls[order]; keep=np.r_[True,k[1:]!=k[:-1]]
T,Ls=T[keep],Ls[keep]; print("functions",len(T),np.bincount(Ls))
np.savez_compressed("f6.npz",T=T,L=Ls)
