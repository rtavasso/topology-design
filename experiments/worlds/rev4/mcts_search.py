# Revision-4 derivation (DESIGN.md §5.1). Needs f6.npz and TT.npy from experiments/worlds/rev3/enum6.py in the working folder.
# Revision-4 derivation: can search ALONE (uniform prior, no learned policy/value) find a program consistent with the
# context, by target length and simulation budget? This is the foothold expert iteration would bootstrap from.
# MCTS (PUCT, uniform prior) over 32 instructions + HALT, depth <= 7, exact execution. Terminal reward = fraction of the
# k context examples matched; random rollouts (uniform, HALT prob 1/3) evaluate leaves. Success = a visited complete
# program matching all k context examples; "true" = it also equals the target on all 49 inputs.
import numpy as np, sys, math
rng=np.random.default_rng(0)
d=np.load("f6.npz"); T=d["T"]; L=d["L"]; TT=np.load("TT.npy")
K=8; C=1.5
def run(target, ctx, S):
    tree={}  # key: tuple(prefix) -> [N(33), W(33)]
    best=None
    for s in range(S):
        state=np.arange(49,dtype=np.uint8); pref=(); path=[]
        # selection / expansion
        while True:
            node=tree.get(pref)
            if node is None:
                tree[pref]=[np.zeros(33),np.zeros(33)]; break
            Nn,Wn=node; tot=Nn.sum()
            q=np.where(Nn>0,Wn/np.maximum(Nn,1),0.5); u=q+C*(1/33)*math.sqrt(tot+1)/(1+Nn)
            if len(pref)>=7: u[:32]=-1e9
            a=int(np.argmax(u+1e-9*rng.random(33))); path.append((pref,a))
            if a==32: break
            state=TT[a][state]; pref=pref+(a,)
        # rollout from current prefix unless halted
        if not(path and path[-1][1]==32):
            st=state.copy(); depth=len(pref)
            while depth<7 and rng.random()>1/3:
                st=TT[rng.integers(32)][st]; depth+=1
            out=st//7
        else:
            out=state//7
        r=(out[ctx]==target[ctx]).mean()
        if r==1.0:
            true=bool((out==target).all())
            if best is None or (true and not best[1]): best=(s+1,true)
        for pf,a in path:
            tree[pf][0][a]+=1; tree[pf][1][a]+=r
    return best
budgets=[int(x) for x in sys.argv[1].split(",")] if len(sys.argv)>1 else [64,256,1024]
NT=int(sys.argv[2]) if len(sys.argv)>2 else 15
for l in range(1,6):
    pool=np.where(L==l)[0]; ix=rng.choice(pool,size=min(NT,len(pool)),replace=False); NTl=len(ix)
    row=[]
    for S in budgets:
        found=0; true=0
        for i in ix:
            ctx=rng.choice(49,size=K,replace=False); b=run(T[i],ctx,S)
            if b: found+=1; true+=b[1]
        row.append(f"S={S}: consistent {found}/{NTl}, true {true}/{NTl}")
    print(f"l={l}: "+"; ".join(row),flush=True)
