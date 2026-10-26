# BFS in restricted grammars (programs, not minimal), product with automaton; which test functions leak?
import pickle, numpy as np, sys
p=7
fn=pickle.load(open("fn5.pkl","rb"))
X,Y=np.meshgrid(np.arange(p),np.arange(p),indexing="ij"); X,Y=X.ravel(),Y.ravel()
ins=[]
for d in (0,1):
    for o in ("add","sub","mul"):
        for s1 in (0,1):
            for s2 in (0,1): ins.append((o,d,s1,s2))
    for o in ("neg","inc"):
        for s in (0,1): ins.append((o,d,s,None))
def ap(r,i):
    o,d,s1,s2=i; a=r[s1]
    v={"add":lambda:(a+r[s2])%p,"sub":lambda:(a-r[s2])%p,"mul":lambda:(a*r[s2])%p,"neg":lambda:(-a)%p,"inc":lambda:(a+1)%p}[o]()
    r=list(r); r[d]=v; return tuple(r)
def bfs(allowed, LMAX):
    # allowed(i, L_so_far, aut) -> (ok, aut')
    st={( (X.astype(np.int64).tobytes(),Y.astype(np.int64).tobytes()), 0)}
    start=(tuple(map(tuple,[X,Y])),0)
    fr=[(np.stack([X,Y]),0)]; seen={(fr[0][0].tobytes(),0)}; out={X.tobytes():0}
    for L in range(1,LMAX+1):
        nx=[]
        for S,aut in fr:
            for i in ins:
                ok,a2=allowed(i,L,aut)
                if not ok: continue
                o,d,s1,s2=i; a=S[s1]
                v={"add":lambda:(a+S[s2])%p,"sub":lambda:(a-S[s2])%p,"mul":lambda:(a*S[s2])%p,"neg":lambda:(-a)%p,"inc":lambda:(a+1)%p}[o]()
                S2=S.copy(); S2[d]=v; k=(S2.tobytes(),a2)
                if k in seen: continue
                seen.add(k); nx.append((S2,a2)); ko=S2[0].tobytes()
                if ko not in out: out[ko]=L
        fr=nx
    return out
def x5(i,L,aut):
    o,d,s1,s2=i
    if o=="inc" and (s1==1 or d==1): return False,aut
    if o=="neg": return (L==1 and False) , aut   # neg as part of length>=2 programs forbidden; length-1 neg handled separately
    if o=="mul": return (aut==0), aut
    return True,1
def x5_noorder(i,L,aut):  # only inc+neg restrictions
    o,d,s1,s2=i
    if o=="inc" and (s1==1 or d==1): return False,aut
    if o=="neg": return False,aut
    return True,aut
key=lambda b: np.frombuffer(b,dtype=np.int64).astype(np.int8).tobytes()
g=bfs(x5,5); g={key(k):v for k,v in g.items()}
A,M,N,IS,ID,O=1,2,4,8,16,32
def ms(bits): return [m for m in range(64) if bits>>m&1]
def V(m,L): return {"neg":bool(m&N) and L>1,"inc":bool(m&(IS|ID)),"order":bool(m&O)}
for j in ("neg","inc","order"):
    tot=leak=0; byL={}
    for k,v in fn.items():
        L=v[0]; Vs=[V(m,L) for m in ms(v[2])]
        if Vs and all(x[j] for x in Vs) and any(sum(x.values())==1 for x in Vs):
            tot+=1
            if k in g: leak+=1; byL[L]=byL.get(L,0)+1
    print(f"split {j}: test(exact1, L<=5)={tot}; computable by an X5-compliant program of length<=5: {leak} {byL}")
print("X5 grammar-reachable functions at length<=5:",len(g))
# single-concept grammars
add_only=lambda i,L,a: (i[0]!="mul",a)
mul_only=lambda i,L,a: (i[0]=="mul",a)
ga={key(k) for k in bfs(add_only,6)}; gm={key(k) for k in bfs(mul_only,8)}
print("additive closure",len(ga),"mul-only closure (len<=8)",len(gm),"union",len(ga|gm))
mixed=[k for k,v in fn.items() if v[2] and all((m&3)==3 for m in ms(v[2]))]
print("concept-mix test (L<=5)",len(mixed),"of which single-concept-computable:",sum(k in ga or k in gm for k in mixed))
