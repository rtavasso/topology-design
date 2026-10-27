# Review 3: in initsig.py's setting, how many "exact on every remaining point" paths are the true function (all 49 inputs)?
import numpy as np
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy"); rng=np.random.default_rng(1)
targets=np.where((L==4)|(L==5))[0]; K=16; E=20000; ex_pts=np.zeros(17); n_pts=np.zeros(17); true=0; anyex=0
for e in range(E):
    f=T[rng.choice(targets)]; idx=rng.choice(49,size=16,replace=False); i=rng.integers(1,17); pts=idx[i-1:]
    O=np.empty((K,49),np.uint8)
    for k in range(K):
        s=np.arange(49,dtype=np.uint8)
        for t in range(7):
            a=rng.integers(33)
            if a==32: break
            s=TT[a][s]
        O[k]=s//7
    ex=(O[:,pts]==f[pts]).all(1); anyex+=ex.any(); true+=(O==f).all(1).any(); n_pts[len(pts)]+=1; ex_pts[len(pts)]+=ex.any()
print(f"episodes with a path exact on remaining points {anyex/E:.4f}; with a path equal to the target on all 49 inputs {true/E:.5f}")
print("by number of scored points:", " ".join(f"{m}:{ex_pts[m]/max(n_pts[m],1):.3f}" for m in range(1,17)))
