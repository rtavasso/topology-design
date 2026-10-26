# Run: python3 experiments/worlds/branch_classify.py 25   (sample size; ~8 min on CPU). 26 Oct result in docs/history.md Entry 31.
"""Classify first-step decisions for target functions of minimal length 4 (budget 4): optimal / lateral / away /
budget-fatal, plus a sufficient (not necessary) unreachability test (two inputs merged by the state but separated by the target)."""
import numpy as np, time, sys
p=7
X,Y=np.meshgrid(np.arange(p),np.arange(p),indexing="ij"); X,Y=X.ravel(),Y.ravel()
S0=np.stack([X,Y]).astype(np.int8)
B={"add":lambda a,b:(a+b)%p,"sub":lambda a,b:(a-b)%p,"mul":lambda a,b:(a*b)%p}
U={"neg":lambda a:(-a)%p,"inc":lambda a:(a+1)%p}
I=[("b",o,d,a,b) for d in (0,1) for o in B for a in (0,1) for b in (0,1)]+[("u",o,d,a,None) for d in (0,1) for o in U for a in (0,1)]
def ap(s,i):
    k,o,d,a,b=i; n=s.copy(); n[d]=(B[o](s[a].astype(int),s[b].astype(int)) if k=="b" else U[o](s[a].astype(int))); return n
def dist(s,f,maxd):
    if (s[0]==f).all(): return 0
    seen={s.tobytes()}; fr=[s]
    for d in range(1,maxd+1):
        nx=[]
        for t in fr:
            for i in I:
                n=ap(t,i); k=n.tobytes()
                if k in seen: continue
                seen.add(k)
                if (n[0]==f).all(): return d
                nx.append(n)
        fr=nx
    return None  # > maxd
def dead(s,f):  # sufficient condition for unreachability at any length: two inputs with equal state, different target
    key=s[0].astype(int)*p+s[1].astype(int)
    for v in np.unique(key):
        if len(np.unique(f[key==v]))>1: return True
    return False
# targets: functions of minimal length 4
seen={S0.tobytes()}; fr=[S0]; out={S0[0].tobytes():0}
for L in range(1,5):
    nx=[]
    for s in fr:
        for i in I:
            n=ap(s,i); k=n.tobytes()
            if k not in seen:
                seen.add(k); nx.append(n)
                if n[0].tobytes() not in out: out[n[0].tobytes()]=L
    fr=nx
F4=[np.frombuffer(k,dtype=np.int8) for k,v in out.items() if v==4]
rng=np.random.default_rng(0); idx=rng.choice(len(F4),size=int(sys.argv[1]),replace=False)
cnt={"optimal":0,"lateral":0,"away/fatal":0}; ndead=0; deadfatal=0; tot=0; t0=time.time()
for j in idx:
    f=F4[j]
    for i in I:
        s1=ap(S0,i); d=dist(s1,f,3)  # remaining budget after step 1 = 3 for an optimal path
        tot+=1
        if d is not None and d<=3: cnt["optimal"]+=1   # d==3 exactly (cannot be <3 since minimal length is 4)
        else:
            d4=dist(s1,f,4)
            if d4==4: cnt["lateral"]+=1
            else: cnt["away/fatal"]+=1
        if dead(s1,f): ndead+=1
print(f"{len(idx)} targets of minimal length 4, 32 first moves each ({time.time()-t0:.0f}s)")
for k,v in cnt.items(): print(f"  {k}: {v/tot:.3f}")
print(f"  provably dead (state merges inputs the target separates): {ndead/tot:.3f}")
