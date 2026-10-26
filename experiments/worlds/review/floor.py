import pickle, numpy as np
fn=pickle.load(open("fn6.pkl","rb")); keys=list(fn.keys())
T=np.frombuffer(b"".join(keys),dtype=np.int8).reshape(len(keys),49); L=np.array([fn[k][0] for k in keys])
rng=np.random.default_rng(0); P=T[L<=5]
acc_nn=[];acc_mode=[];acc_ctxmode=[]
for fi in rng.choice(np.where(L==6)[0],200,replace=False):
    o=rng.permutation(49); c,q=o[:16],o[16:]
    ag=(P[:,c]==T[fi,c]).sum(1); best=P[ag==ag.max()]
    pred=np.array([np.bincount(best[:,x],minlength=7).argmax() for x in q])
    acc_nn.append((pred==T[fi,q]).mean())
    m=np.bincount(T[fi,c],minlength=7).argmax(); acc_ctxmode.append((T[fi,q]==m).mean())
print("L6 test, per-query accuracy: nearest L<=5 function by context agreement %.3f; predict context-mode output %.3f; chance 0.143"%(np.mean(acc_nn),np.mean(acc_ctxmode)))
