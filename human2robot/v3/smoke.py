"""Strict official-weight load and real-data finite forward/backward before allocating H100s."""
import torch,numpy as np
from v3.policy import GripperPolicy
from v3.common import ROOT,sha,dump
from v3.train import dataset,stats,forward

torch.set_num_threads(4);torch.manual_seed(17)
p=ROOT/'models/resnet18-imagenet.pth'
assert sha(p)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
m=GripperPolicy(p);d=dataset('bridge','train','cpu');st=stats(d);idx=torch.arange(2)
pred=forward(m,d,idx,st);y=(d['target'][idx]-st['tm'])/st['ts']
per=torch.nn.functional.smooth_l1_loss(pred,y,reduction='none').mean(-1);mask=d['mask'][idx]
loss=(per*mask).sum()/mask.sum();assert torch.isfinite(loss)
loss.backward();grad=sum(float(p.grad.square().sum()) for p in m.encoder.parameters() if p.grad is not None)**.5
assert grad>0 and np.isfinite(grad)
m2=GripperPolicy();m2.load_state_dict(m.state_dict(),strict=True)
m.eval();m2.eval()
with torch.no_grad():torch.testing.assert_close(forward(m,d,idx,st),forward(m2,d,idx,st))
r={'strict_load':True,'loss':float(loss.detach()),'encoder_gradient_norm':grad,'parameters':sum(p.numel() for p in m.parameters()),'samples':2,'source':'Bridge train split, real RGB and observed TCP','model_weights_sha256':sha(ROOT/'models/resnet18-imagenet.pth')}
dump(ROOT/'CPU_SMOKE.json',r);print(r,flush=True)
