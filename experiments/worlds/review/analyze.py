import pickle, numpy as np, sys
fn=pickle.load(open(sys.argv[1],"rb"))
A,M,N,IS,ID,O=1,2,4,8,16,32
def ms(bits): return [m for m in range(64) if bits>>m&1]
Ls=sorted(set(v[0] for v in fn.values()))
print("== #minimal programs per function by L")
for L in Ls:
    c=np.array([v[1] for v in fn.values() if v[0]==L])
    print(L,len(c),"total",c.sum(),"median",np.median(c),"mean %.1f"%c.mean(),"max",c.max(), "frac with 1 prog %.2f"%(c==1).mean())
def viol(m,L,inc_bits):
    return {"neg": bool(m&N) and L>1, "inc": bool(m&inc_bits), "order": bool(m&O)}
for name,ib in (("inc src or dst r1",IS|ID),("inc src=r1",IS),("inc dst=r1",ID)):
    print(f"\n== reading of 'inc applied to r1': {name}")
    rows={}
    for L in Ls:
        r=dict(n=0,single_any=0,mixed_all=0,both_kinds=0,x5pool=0,x5_mixedprogs=0)
        for j in ("neg","inc","order"): r["all_"+j]=0; r["exact1_"+j]=0; r["strict1_"+j]=0
        for v in fn.values():
            if v[0]!=L: continue
            mm=ms(v[2]); r["n"]+=1
            sc=[(m&3) in (1,2) or m==0 for m in mm]
            if any(sc): r["single_any"]+=1
            if not any(sc): r["mixed_all"]+=1
            if any(sc) and not all(sc): r["both_kinds"]+=1
            V=[viol(m,L,ib) for m in mm]
            comp=[not any(x.values()) for x in V]
            if any(comp): r["x5pool"]+=1
            if any(comp) and not all(comp): r["x5_mixedprogs"]+=1
            for j in ("neg","inc","order"):
                if all(x[j] for x in V):
                    r["all_"+j]+=1
                    if any(sum(x.values())==1 for x in V): r["exact1_"+j]+=1
                    if all(sum(x.values())==1 for x in V): r["strict1_"+j]+=1
        rows[L]=r
    keys=list(rows[Ls[-1]].keys())
    print("L".ljust(16)+"".join(f"{L:>9}" for L in Ls))
    for k in keys: print(k.ljust(16)+"".join(f"{rows[L][k]:>9}" for L in Ls))
    if name!="inc src or dst r1": break
