"""Matched 10k-window gripper pretraining, followed by identical RoboTwin finetuning."""
import argparse,json,time,random,os
from pathlib import Path
import numpy as np
import torch
from v3.common import *
from v3.policy import GripperPolicy


def dataset(domain,split,device):
    root=ROOT/'packed'/domain/split
    return {k:torch.from_numpy(np.load(root/f'{k}.npy')).to(device) for k in ['images','state','target','side','mask']}


def stats(d):
    y=d['target'][d['mask'].bool()]
    return {'sm':d['state'].mean(0),'ss':d['state'].std(0).clamp_min(.02),'tm':y.mean(0),'ts':y.std(0).clamp_min(torch.tensor([.01]*3+[.05]*3+[.1],device=d['state'].device))}


def forward(model,d,idx,st):return model(d['images'][idx],(d['state'][idx]-st['sm'])/st['ss'],d['side'][idx])


@torch.no_grad()
def evaluate(model,d,st):
    model.eval();pred=[]
    for idx in torch.arange(len(d['images']),device=d['images'].device).split(128):
        pred.append((forward(model,d,idx,st)*st['ts']+st['tm']).float().cpu().numpy())
    pred=np.concatenate(pred);truth=d['target'].cpu().numpy();delta=pred-truth
    valid=d['mask'].cpu().numpy().astype(bool)
    trans=np.linalg.norm(delta[...,:3],axis=-1)[valid]
    from scipy.spatial.transform import Rotation
    rot=(Rotation.from_rotvec(pred[...,3:6][valid]).inv()*Rotation.from_rotvec(truth[...,3:6][valid])).magnitude()
    grip=np.abs(delta[...,6])[valid]; score=float(np.mean((delta[valid]/st['ts'].cpu().numpy())**2))
    metrics={'normalized_mse':score,'tcp_position_mean_m':float(trans.mean()),'tcp_position_p95_m':float(np.quantile(trans,.95)),
        'tcp_rotation_mean_deg':float(np.rad2deg(rot).mean()),'gripper_opening_mae_0to1':float(grip.mean()),
        'gripper_open_close_accuracy':float(np.mean((pred[...,6][valid]>.5)==(truth[...,6][valid]>.5))),'windows':len(pred),'valid_action_targets':int(valid.sum())}
    return metrics,pred,truth


def run(a):
    torch.set_num_threads(6);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True,warn_only=True)
    device=torch.device(a.device);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    audit_name='DATA_READY.json' if a.condition.startswith('human') else 'DATA_READY_REAL.json'
    assert (ROOT/audit_name).exists()
    model=GripperPolicy(ROOT/'models/resnet18-imagenet.pth').to(device)
    init={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    print('Start training',a.condition,a.seed,'parameters',sum(p.numel() for p in model.parameters()),'device',device,flush=True)
    record={'condition':a.condition,'seed':a.seed,'model':'ImageNet1K ResNet18 + spatial MLP gripper policy','is_vla':False,'parameters':sum(p.numel() for p in model.parameters()),'pretrain_steps':a.pre_steps,'finetune_steps':a.fine_steps,'batch':a.batch,'horizon':HORIZON,'optimizer':'AdamW, encoder 1e-4, head 3e-4, wd .01, clip 1','data_audit_sha256':sha(ROOT/audit_name),'weights_sha256':sha(ROOT/'models/resnet18-imagenet.pth'),'head_transfer':'all policy weights retained; optimizer and normalization reset per stage','curves':[]}
    dump(out/'protocol.json',record)
    simtrain=dataset('sim','train',device);simval=dataset('sim','validation',device);simstats=stats(simtrain)
    stages=[]
    if a.condition!='robotwin_only':stages.append(('pretrain','bridge' if a.condition=='bridge' else 'human',a.pre_steps))
    stages.append(('finetune','sim',a.fine_steps))
    for stage,domain,steps in stages:
        train=simtrain if domain=='sim' else dataset(domain,'train',device)
        val=simval if domain=='sim' else dataset(domain,'validation',device); st=simstats if domain=='sim' else stats(train)
        assert domain=='sim' or len(train['images'])==10000
        # Fixed permutation destroys image/action correspondence without changing target distribution.
        target=train['target'];target_mask=train['mask']
        if stage=='pretrain' and a.condition=='human_shuffled':
            generator=torch.Generator(device=device).manual_seed(a.seed+77)
            perm=torch.randperm(len(target),generator=generator,device=device)
            target=target[perm];target_mask=target_mask[perm]
        opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-4},{'params':model.head.parameters(),'lr':3e-4}],weight_decay=.01)
        generator=torch.Generator(device=device).manual_seed(a.seed+(17 if stage=='pretrain' else 29))
        best=float('inf');started=time.time();best_model=None
        for step in range(1,steps+1):
            # Reset augment RNG at finetune start to pair all downstream update streams.
            if step==1:torch.manual_seed(a.seed+(117 if stage=='pretrain' else 129))
            model.train();idx=torch.randint(len(train['images']),(a.batch,),generator=generator,device=device)
            pred=forward(model,train,idx,st);truth=(target[idx]-st['tm'])/st['ts']
            per=torch.nn.functional.smooth_l1_loss(pred,truth,reduction='none').mean(-1)
            weights=target_mask[idx]
            loss=(per*weights).sum()/weights.sum().clamp_min(1)
            if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step==1 or step%100==0:
                item={'stage':stage,'step':step,'loss':float(loss),'grad_norm':float(grad),'seconds':time.time()-started};record['curves'].append(item);print(json.dumps(item),flush=True)
                dump(out/'progress.json',item)
            if step%1000==0 or step==steps:
                metrics,_,_=evaluate(model,val,st);print('VALIDATION',stage,step,metrics,flush=True)
                if metrics['normalized_mse']<best:
                    best=metrics['normalized_mse'];best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=step
                torch.save({'model':model.state_dict(),'stats':{k:v.cpu() for k,v in st.items()},'step':step,'stage':stage,'seed':a.seed,'condition':a.condition},out/f'{stage}_latest.pt')
        model.load_state_dict(best_model,strict=True)
        # sim_stats uses only training split, enabling documented target-domain calibration before finetune.
        ckpt={'model':best_model,'stats':{k:v.cpu() for k,v in st.items()},'sim_stats':{k:v.cpu() for k,v in simstats.items()},'step':best_step,'stage':stage,'seed':a.seed,'condition':a.condition,'task':'place_empty_cup','action_representation':'current_tcp_relative_xyz_rotvec_opening','horizon':HORIZON}
        torch.save(ckpt,out/f'{stage}_best.pt')
        test=dataset(domain,'test',device);metrics,pred,truth=evaluate(model,test,st)
        dump(out/f'{stage}_offline.json',metrics);np.savez_compressed(out/f'{stage}_predictions.npz',prediction=pred,target=truth)
        if stage=='pretrain':
            short_val={**simval,'mask':torch.zeros_like(simval['mask'])};short_val['mask'][:,0]=1
            m,p,t=evaluate(model,short_val,simstats);m.update(target_domain_statistics_used=True,supervised_horizon=1,is_strict_zero_shot=False);dump(out/'before_finetune_sim_validation.json',m)
            del train,val,test,target
    record['completed']=True;record['note']='Human metrics use weak camera-relative pseudo-metres; only native RoboTwin metrics and closed-loop success compare conditions.'
    dump(out/'protocol.json',record);dump(out/'DONE.json',{'condition':a.condition,'seed':a.seed,'completed':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['human','bridge','robotwin_only','human_shuffled'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--pre-steps',type=int,default=20000);p.add_argument('--fine-steps',type=int,default=10000);p.add_argument('--batch',type=int,default=64);p.add_argument('--device',default='cuda');run(p.parse_args())
