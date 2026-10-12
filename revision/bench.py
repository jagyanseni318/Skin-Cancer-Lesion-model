import torch,time,torch.nn as nn
from common import *
def bench(cl,bench_flag,bs=32):
    torch.backends.cudnn.benchmark=bench_flag
    m=DenseNet121_TrustXAI().to(DEVICE)
    if cl: m=m.to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(m.parameters(),1e-4); sc=torch.amp.GradScaler('cuda')
    x=torch.randn(bs,3,224,224,device=DEVICE); y=torch.randint(0,7,(bs,),device=DEVICE)
    if cl: x=x.contiguous(memory_format=torch.channels_last)
    for i in range(25):
        if i==5: torch.cuda.synchronize(); t=time.time()
        opt.zero_grad()
        with torch.autocast('cuda',dtype=torch.float16): l=nn.functional.cross_entropy(m(x).float(),y)
        sc.scale(l).backward(); sc.step(opt); sc.update()
    torch.cuda.synchronize(); return bs*20/(time.time()-t)
for cl in [False,True]:
    for b in [False,True]: print('channels_last',cl,'benchmark',b, round(bench(cl,b)),'img/s',flush=True)
