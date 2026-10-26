# Revision-3 design computation (DESIGN.md §5). Run from a folder containing f6.npz and TT.npy (made by enum6.py).
# Training signal at initialization (uniform policy over 32 instructions + HALT, max 7 steps) under three scoring rules,
# for target functions of minimal length 4-5. Each sampled path is executed on all 49 inputs (paths ignore the query).
import numpy as np
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy")
rng=np.random.default_rng(1)
def sample_paths(K):
    outs=np.empty((K,49),np.uint8)
    for k in range(K):
        s=np.arange(49,dtype=np.uint8)
        for t in range(7):
            a=rng.integers(33)
            if a==32: break
            s=TT[a][s]
        outs[k]=s//7
    return outs
targets=np.where((L==4)|(L==5))[0]
K=16; E=3000
acc={r:{"const":0.,"agree":0.,"exact":0.,"n":0} for r in ("marg","nc0.1","nc0.3")}
unif_agree=0.; any_joint=0; const_frac=0.
for e in range(E):
    f=T[rng.choice(targets)]; idx=rng.choice(49,size=16,replace=False); i=rng.integers(1,17)  # context = first i-1 pairs
    pts=idx[i-1:]; O=sample_paths(K); hit=(O[:,pts]==f[pts])  # (K, m)
    agree=hit.mean(1); const=(O.min(1)==O.max(1)); exact=hit.all(1)
    unif_agree+=agree.mean(); const_frac+=const.mean(); any_joint+=exact.any()
    H=hit.sum(0); wm=(hit[:,H>0]/H[H>0]).sum(1)
    if wm.sum()>0:
        wm=wm/wm.sum(); a=acc["marg"]; a["const"]+=wm@const; a["agree"]+=wm@agree; a["exact"]+=wm@exact; a["n"]+=1
    for eps in (0.1,0.3):
        ll=hit.sum(1)*np.log(1-eps)+(~hit).sum(1)*np.log(eps/6); w=np.exp(ll-ll.max()); w/=w.sum()
        a=acc[f"nc{eps}"]; a["const"]+=w@const; a["agree"]+=w@agree; a["exact"]+=w@exact; a["n"]+=1
print(f"uniform: mean agreement {unif_agree/E:.3f}, constant-output paths {const_frac/E:.3f}, episodes with any exact path (K={K}) {any_joint/E:.4f}")
for r,a in acc.items():
    n=a["n"]; print(f"{r:7s}: weighted agreement {a['agree']/n:.3f}, weight on constant paths {a['const']/n:.3f}, weight on exact paths {a['exact']/n:.4f}")
