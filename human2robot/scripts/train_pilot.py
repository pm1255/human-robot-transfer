"""Small RGB learning pilot, not a foundation VLA or a substitution benchmark.

Conditions share source windows, encoder initialization and optimizer steps.
Human targets are 2D hand-motion pseudo-labels; robot actions, if supplied, come
from the robot dataset itself. No retargeted synthetic robot state is fabricated.
"""
from __future__ import annotations
import argparse,hashlib,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Pilot(nn.Module):
    def __init__(self,state_dim=14,action_dim=14):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(6,24,5,2,2),nn.GELU(),nn.Conv2d(24,48,3,2,1),nn.GELU(),
            nn.Conv2d(48,64,3,2,1),nn.GELU(),nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(576,128),nn.GELU())
        self.motion=nn.Linear(128,3)
        self.future=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,3*24*24))
        self.robot=nn.Sequential(nn.Linear(128+state_dim,128),nn.GELU(),nn.Linear(128,action_dim))


def digest(module):
    h=hashlib.sha256()
    for p in module.parameters():h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def human_data(root,horizon=3):
    xs=[];ys=[];futures=[];origins=[]
    for file in sorted(Path(root).glob('*/rgb_supervision.npz')):
        data=np.load(file);meta=json.loads((file.parent/'annotation.json').read_text())
        if meta['metadata']['source_kind']!='human_video':continue
        im=data['images'];points=data['landmarks_2d'];mask=data['observed_2d'];track=data['track_id']
        center=(points[:,4]+points[:,8])/2;aperture=np.linalg.norm(points[:,4]-points[:,8],axis=1)
        motion=np.column_stack([center,aperture])
        for t in range(1,len(im)-horizon):
            if not mask[t-1:t+horizon+1].all() or len(set(track[t-1:t+horizon+1]))!=1:continue
            target=motion[t+horizon]-motion[t]
            if not np.isfinite(target).all():continue
            xs.append(np.concatenate([im[t-1],im[t]],axis=2));ys.append(target);futures.append(im[t+horizon]);origins.append((file.parent.name,t))
    if len(xs)<8:raise ValueError('Need at least 8 valid within-track human windows')
    x=torch.tensor(np.stack(xs).transpose(0,3,1,2),dtype=torch.float32)/255
    future=torch.tensor(np.stack(futures).transpose(0,3,1,2),dtype=torch.float32)/255
    future=F.interpolate(future,(24,24),mode='area')-F.interpolate(x[:,3:],(24,24),mode='area')
    y=torch.tensor(np.stack(ys),dtype=torch.float32);scale=y.std(0).clamp_min(.02)
    return x,y/scale,future,origins,scale


def human_run(root,out,condition,steps,seed,device):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    x,y,future,origins,scale=human_data(root);n=len(x)
    net=Pilot().to(device);initial_hash=digest(net.encoder)
    original=[p.detach().clone() for p in net.encoder.parameters()]
    x=x.to(device);y=y.to(device);future=future.to(device)
    generator=torch.Generator().manual_seed(seed+101)
    permutation=torch.randperm(n,generator=torch.Generator().manual_seed(seed+55)).to(device)
    training_y=y[permutation] if condition=='shuffled' else y
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    def losses(ix):
        z=net.encoder(x[ix]);motion=F.mse_loss(net.motion(z),training_y[ix]);video=F.mse_loss(net.future(z).reshape(-1,3,24,24),future[ix])
        return motion if condition in {'explicit','shuffled'} else video if condition=='implicit' else motion+video
    with torch.no_grad():initial_loss=float(losses(torch.arange(n,device=device)))
    grad_norms=[];curve=[];exposures=np.zeros(n,np.int64);start=time.time()
    for step in range(steps):
        idx=torch.randint(n,(min(32,n),),generator=generator);np.add.at(exposures,idx.numpy(),1);idx=idx.to(device)
        optimizer.zero_grad(set_to_none=True);loss=losses(idx)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(net.encoder.parameters(),5.0));grad_norms.append(norm);optimizer.step()
        if step%20==0 or step==steps-1:
            curve.append({'step':step+1,'loss':float(loss.detach())});print(json.dumps({'condition':condition,**curve[-1]}),flush=True)
    with torch.no_grad():
        final_loss=float(losses(torch.arange(n,device=device)))
        aligned=float(F.mse_loss(net.motion(net.encoder(x)),y))
        image_permutation=torch.randperm(n,device=device)
        shuffled_image=float(F.mse_loss(net.motion(net.encoder(x[image_permutation])),y))
        zero_image=float(F.mse_loss(net.motion(net.encoder(torch.zeros_like(x))),y))
    delta=float(torch.sqrt(sum(((a-b)**2).sum() for a,b in zip(net.encoder.parameters(),original))))
    result={'condition':condition,'seed':seed,'samples':n,'source_clips':len(set(a for a,_ in origins)),
        'steps':steps,'initial_loss':initial_loss,'final_loss':final_loss,'parameter_delta_l2':delta,
        'encoder_initial_sha256':initial_hash,'encoder_final_sha256':digest(net.encoder),
        'encoder_gradient_mean':float(np.mean(grad_norms)),'human_sample_exposures':int(exposures.sum()),
        'unique_windows_seen':int((exposures>0).sum()),'aligned_motion_train_mse':aligned,
        'shuffled_image_motion_train_mse':shuffled_image,'zero_image_motion_train_mse':zero_image,
        'future_frame_copy_baseline_mse':float((future**2).mean()),'wall_seconds':time.time()-start,
        'target_semantics':('future low-resolution RGB residual' if condition=='implicit' else
            'future image-plane pinch center/aperture delta + RGB residual' if condition=='hybrid' else
            'future image-plane pinch center/aperture delta; estimated 2D human motion'),
        'evaluation_scope':'training-set diagnostic, not held-out or closed-loop performance',
        'device':str(device),'torch':torch.__version__,'curve':curve}
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':net.state_dict(),'target_scale':scale,'metadata':result},out/f'{condition}.pt')
    (out/f'{condition}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if delta<=0 or not np.isfinite(grad_norms).all() or not (exposures>0).any():raise RuntimeError('Human data failed to update the encoder')
    return result


def robot_finetune(dataset,out,checkpoint,condition,steps,seed,device):
    """Offline action-fit pilot. Split is by source episode, stats from train only."""
    torch.manual_seed(seed)
    d=np.load(dataset);required={'images','previous_images','state','action','episode_id','split'}
    if not required.issubset(d.files):raise ValueError(f'Missing RoboTwin arrays: {required-set(d.files)}')
    if d['state'].shape[1]!=14 or d['action'].shape[1]!=14:raise ValueError('Expected audited Aloha 14D state/action')
    train=d['split']=='train';test=d['split']=='validation'
    if not train.any() or not test.any():raise ValueError('Need train and validation episodes')
    if set(d['episode_id'][train])&set(d['episode_id'][test]):raise ValueError('Episode leakage')
    x=torch.tensor(np.concatenate([d['previous_images'],d['images']],axis=3).transpose(0,3,1,2),dtype=torch.float32,device=device)/255
    state=torch.tensor(d['state'],dtype=torch.float32,device=device);action=torch.tensor(d['action'],dtype=torch.float32,device=device)
    mean_s=state[train].mean(0);std_s=state[train].std(0).clamp_min(.01)
    mean_a=action[train].mean(0);std_a=action[train].std(0).clamp_min(.01)
    state=(state-mean_s)/std_s;target=(action-mean_a)/std_a
    net=Pilot().to(device)
    if checkpoint:
        ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
        # Transfer the learned visual representation; action head starts identically.
        encoder={k[len('encoder.'):]:v for k,v in ckpt['state_dict'].items() if k.startswith('encoder.')}
        net.encoder.load_state_dict(encoder,strict=True)
    initial_hash=digest(net.encoder);optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4)
    train_ids=np.flatnonzero(train);validation_ids=np.flatnonzero(test);rng=np.random.default_rng(seed)
    def predict(ids):return net.robot(torch.cat([net.encoder(x[ids]),state[ids]],dim=1))
    with torch.no_grad():initial=float(F.mse_loss(predict(validation_ids),target[validation_ids]))
    losses=[]
    for step in range(steps):
        ids=rng.choice(train_ids,size=min(32,len(train_ids)),replace=True)
        optimizer.zero_grad(set_to_none=True);loss=F.mse_loss(predict(ids),target[ids]);loss.backward();optimizer.step()
        if not torch.isfinite(loss):raise RuntimeError('Non-finite robot loss')
        if step%25==0:losses.append(float(loss));print(json.dumps({'robot_condition':condition,'step':step,'loss':float(loss.detach())}),flush=True)
    with torch.no_grad():
        pred=predict(validation_ids);final=float(F.mse_loss(pred,target[validation_ids]));mae=float(((pred-target[validation_ids])*std_a).abs().mean())
        baseline=float(F.mse_loss(torch.zeros_like(pred),target[validation_ids]))
        persistence=(torch.tensor(d['state'][test],device=device)-mean_a)/std_a
        persistence_mse=float(F.mse_loss(persistence,target[validation_ids]))
        shuffled_ids=np.random.default_rng(seed+9).permutation(validation_ids)
        shuffled_pred=net.robot(torch.cat([net.encoder(x[shuffled_ids]),state[validation_ids]],dim=1))
        shuffled_mse=float(F.mse_loss(shuffled_pred,target[validation_ids]))
        zero_pred=net.robot(torch.cat([net.encoder(torch.zeros_like(x[validation_ids])),state[validation_ids]],dim=1))
        zero_mse=float(F.mse_loss(zero_pred,target[validation_ids]))
        per_dimension_mae=(((pred-target[validation_ids])*std_a).abs().mean(0)).cpu().tolist()
    report={'condition':condition,'seed':seed,'steps':steps,'train_frames':int(train.sum()),'validation_frames':int(test.sum()),
            'train_episodes':len(set(d['episode_id'][train])),'validation_episodes':len(set(d['episode_id'][test])),
            'initial_validation_normalized_mse':initial,'final_validation_normalized_mse':final,
            'validation_action_mae_mixed_units':mae,'constant_train_mean_baseline_mse':baseline,
            'state_persistence_baseline_mse':persistence_mse,'shuffled_validation_image_mse':shuffled_mse,
            'zero_validation_image_mse':zero_mse,'per_dimension_mae_native_units':per_dimension_mae,
            'encoder_before_finetune_sha256':initial_hash,'encoder_after_finetune_sha256':digest(net.encoder),
            'closed_loop_success_rate':None,'scope':'offline RoboTwin action fit only; joint radians and gripper source units are mixed'}
    out.mkdir(parents=True,exist_ok=True);(out/f'robot_{condition}.json').write_text(json.dumps(report,indent=2))
    torch.save({'state_dict':net.state_dict(),'state_mean':mean_s,'state_std':std_s,'action_mean':mean_a,'action_std':std_a,'metadata':report},out/f'robot_{condition}.pt')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=120);p.add_argument('--robot-steps',type=int,default=200);p.add_argument('--robot-data')
    p.add_argument('--conditions',default='explicit,shuffled,implicit,hybrid');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA was requested but unavailable; do not silently claim GPU training')
    if a.steps<1:raise ValueError('steps must be positive')
    out=Path(a.out);runs=[]
    for c in a.conditions.split(','):
        if c not in {'explicit','shuffled','implicit','hybrid'}:raise ValueError(c)
        runs.append(human_run(a.human_root,out,c,a.steps,a.seed,torch.device(a.device)))
    robot=[]
    if a.robot_data:
        for c in ['scratch']+a.conditions.split(','):
            robot.append(robot_finetune(a.robot_data,out,None if c=='scratch' else out/f'{c}.pt',c,a.robot_steps,a.seed,torch.device(a.device)))
    report={'status':'短程训练已运行','claim':'这是小型视觉模型的通路验证。梯度和参数变化证明数据进入了优化，不证明真机数据可被替代，也不能据此判定显式或隐式更好。',
            'runs':runs,'robot_finetuning':robot,'closed_loop_evaluation':'not_run','real_robot_substitution':'not_established'}
    (out/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
