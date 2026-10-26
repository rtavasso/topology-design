import pickle, numpy as np, sys
fn=pickle.load(open(sys.argv[1],"rb")); LM=int(sys.argv[2])
keys=list(fn.keys()); T=np.frombuffer(b"".join(keys),dtype=np.int8).reshape(len(keys),49)
Lf=np.array([fn[k][0] for k in keys])
rng=np.random.default_rng(0)
def prior(mask, Ls):
    w=np.zeros(len(T))
    for L in Ls:
        s=mask&(Lf==L)
        if s.sum(): w[s]=1/len(Ls)/s.sum()
    return w
allm=np.ones(len(T),bool)
# in-support holdout: 10% per L (L>=1)
hold=np.zeros(len(T),bool)
for L in range(1,6):
    idx=np.where(Lf==L)[0]; hold[rng.choice(idx,max(1,round(0.1*len(idx))),replace=False)]=True
pool=(~hold)&(Lf<=5)&(Lf>=1)
wa=prior(pool,range(1,6))            # cell prior (D5)
wb=prior(allm&(Lf>=1),range(1,LM+1))  # generator prior, equal per length
print("per-function training frequency under D5 prior: L1 %.2e  L5 %.2e  ratio %.0f"%(wa[(Lf==1)&pool][0],wa[(Lf==5)&pool][0],wa[(Lf==1)&pool][0]/wa[(Lf==5)&pool][0]))
def stream(fi,w,K=16):
    order=rng.permutation(49); cons=w>0; code=[]; mapok=[]
    for k in range(K):
        x=order[k]; y=T[fi,x]
        post=w*cons
        Z=post.sum()
        if Z==0: code.append(np.inf); mapok.append(False)
        else:
            py=np.bincount(T[:,x],weights=post,minlength=7)/Z
            code.append(-np.log(py[y]) if py[y]>0 else np.inf)
            mapok.append(None)
        cons=cons&(T[:,x]==y)
    # MAP correctness on remaining inputs after K
    post=w*cons
    if post.sum()==0: return np.array(code),0.0,0
    rest=order[K:]
    pred=np.array([np.argmax(np.bincount(T[:,x],weights=post,minlength=7)) for x in rest])
    return np.array(code),(pred==T[fi,rest]).mean(), int(cons.sum())
for name,sel,w in (("held-out L5 (in-support split), cell prior (a)",hold&(Lf==5),wa),
                   ("held-out L5, generator prior (b)",hold&(Lf==5),wb),
                   ("train-pool L5, cell prior (a)",pool&(Lf==5),wa),
                   ("held-out L4, cell prior (a)",hold&(Lf==4),wa))+((("L6, generator prior (b)",Lf==6,wb),) if LM==6 else ()):
    idx=rng.choice(np.where(sel)[0],200,replace=False)
    C=[];A=[]
    for fi in idx:
        c,a,_=stream(fi,w); C.append(c); A.append(a)
    C=np.array(C); fin=np.isfinite(C).all(1)
    print(f"{name}: finite code length in {fin.mean():.2f} of streams; ",end="")
    if fin.any():
        Cf=C[fin]; print("mean code (nats) total %.1f, first 4 ex %.1f, ex 5-8 %.1f, ex 9-16 %.2f; uniform=%.1f; query acc after 16 %.3f"%(Cf.sum(1).mean(),Cf[:,:4].sum(1).mean(),Cf[:,4:8].sum(1).mean(),Cf[:,8:].sum(1).mean(),16*np.log(7),np.mean(A)))
    else: print("query acc after 16: %.3f"%np.mean(A))
    if not fin.all():
        firstinf=[np.argmax(~np.isfinite(c)) for c in C[~fin]]; print("   first infinite example index (median)",np.median(firstinf))
