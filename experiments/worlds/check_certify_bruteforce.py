# Planted check for certify_patterns.py (DESIGN.md §11): brute force over all 32^4 programs; 26 Oct: 4,825 functions, 0 mismatches.
# brute-force check of minimal compliant lengths (<=4) for pattern mul>inc, by enumerating all programs
import itertools, numpy as np, sys
sys.argv=["x","3"]  # LCERT irrelevant; we import bfs only
src=open("/home/claude/topology-design/experiments/worlds/certify_patterns.py").read().split("t0 = time.time()")[0]
exec(src)
P=("mul","inc"); LM=4
def uses(prog):
    fresh=[0,0]
    for (o,d,a,b) in prog:
        if o==P[1] and (fresh[a] or (b is not None and fresh[b])): return True
        fresh[d]=1 if o==P[0] else 0
    return False
bf={}  # r0 bytes -> (min len, min compliant len)
S=S0[0]
for L in range(0,LM+1):
    for prog in itertools.product(range(32),repeat=L):
        st=S0
        for i in prog: st=apply(st,INS[i])
        k=st[0,0].tobytes(); c=not uses([INS[i] for i in prog])
        e=bf.setdefault(k,[99,99]); e[0]=min(e[0],L)
        if c: e[1]=min(e[1],L)
fk,fl,_=bfs(P,LM)
fkall,flall,_=bfs(None,LM)
mism=0
for k,(l,lc) in bf.items():
    h=hkey(np.frombuffer(k,dtype=np.int8)[None])[0]
    j=np.searchsorted(fk,h); got=fl[j] if j<len(fk) and fk[j]==h else 99
    j2=np.searchsorted(fkall,h); gl=flall[j2]
    mism+= (got!=lc) or (gl!=l)
print("functions",len(bf),"mismatches",mism, "with lc>l:",sum(1 for l,lc in bf.values() if lc>l))
