# Design tool: like count_isa.py with R registers (extra registers start at 0). Usage: python count_isa_registers.py P LMAX R
import sys, time, numpy as np
p=int(sys.argv[1]); LMAX=int(sys.argv[2]); R=int(sys.argv[3])
X,Y=np.meshgrid(np.arange(p),np.arange(p),indexing="ij"); X,Y=X.ravel(),Y.ravel()
regs=[X,Y]+[np.zeros(p*p,dtype=int)]*(R-2)
start=np.stack(regs).astype(np.int8)
B={"add":lambda a,b:(a+b)%p,"sub":lambda a,b:(a-b)%p,"mul":lambda a,b:(a*b)%p}
U={"neg":lambda a:(-a)%p,"inc":lambda a:(a+1)%p}
ins=[("b",o,d,s1,s2) for d in range(R) for o in B for s1 in range(R) for s2 in range(R)]+[("u",o,d,s,None) for d in range(R) for o in U for s in range(R)]
print(f"p={p} R={R} |I|={len(ins)}")
seen={start.tobytes()}; fr=[start]; out={start[0].tobytes():0}; t0=time.time()
for L in range(1,LMAX+1):
    nx=[]
    for st in fr:
        for k,o,d,s1,s2 in ins:
            n=st.copy(); n[d]=B[o](st[s1].astype(int),st[s2].astype(int)) if k=="b" else U[o](st[s1].astype(int))
            kk=n.tobytes()
            if kk not in seen:
                seen.add(kk); nx.append(n); ko=n[0].tobytes()
                if ko not in out: out[ko]=L
    fr=nx
    print(f"L={L}: new states {len(nx)}, new output functions {sum(1 for v in out.values() if v==L)}, cumulative {len(out)} ({time.time()-t0:.0f}s)",flush=True)
