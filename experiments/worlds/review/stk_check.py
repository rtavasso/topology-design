import sys; sys.path.insert(0,"/home/claude/topology-design/src")
import torch, time
from topology_design.stacked import StackedPFN
import torch.nn.functional as F
torch.manual_seed(0)
# 1) independence with AdamW over several steps (optimizer state is elementwise, so ok); check grad flows into all params
m=StackedPFN(R=2,d=32,layers=2,heads=4,mlp=64)
cx,cy,cf=(torch.randint(0,7,(2,3,16)) for _ in range(3)); qx,qy=(torch.randint(0,7,(2,3,4)) for _ in range(2))
out=m(cx,cy,cf,qx,qy); out.sum().backward()
print("params w/o grad:",[n for n,p in m.named_parameters() if p.grad is None or p.grad.abs().sum()==0])
# 2) does the context-token output depend on query tokens? (context should not see queries)
a=m(cx,cy,cf,qx,qy)
# 3) permutation invariance of context (no PE): permute context order
perm=torch.randperm(16)
b=m(cx[...,perm],cy[...,perm],cf[...,perm],qx,qy)
print("context-permutation invariant:",torch.allclose(a,b,atol=1e-5))
# 4) the embedding: x and y embeddings are summed: is (x,y) distinguishable from (y,x)? yes, separate tables. But f-embedding of query = index p (unknown)
# 5) profile memory of gather backward vs F.embedding
R,B,d=27,256,128
E=torch.randn(R,7,d,requires_grad=True); idx=torch.randint(0,7,(R,B,24))
import tracemalloc
g=lambda E,i: torch.gather(E.unsqueeze(1).expand(-1,i.shape[1],-1,-1),2,i.unsqueeze(-1).expand(-1,-1,-1,E.shape[-1]))
t=time.time(); 
for _ in range(5): g(E,idx).sum().backward()
t1=time.time()-t
off=(idx+7*torch.arange(R).view(R,1,1))
t=time.time()
for _ in range(5): F.embedding(off,E.view(R*7,d)).sum().backward()
t2=time.time()-t
print(f"gather-expand fwd+bwd {t1/5*1e3:.1f} ms vs F.embedding {t2/5*1e3:.1f} ms (CPU)")
