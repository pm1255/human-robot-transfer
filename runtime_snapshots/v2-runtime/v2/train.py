"""Matched RGB-human transfer experiment; only robot rows supervise robot actions."""
import argparse,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from v2.policy import Policy

def load_robot(root,device,horizon):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());splits={}
    for split in ['train','validation','test']:
        ims=[];ss=[];aa=[];mm=[];ee=[];tt=[]
        for entry in manifest['episodes']:
            if entry['split']!=split:continue
            d=np.load(root/entry['file']);n=len(d['state']);idx=np.arange(n)[:,None]+np.arange(horizon)[None,:]
            ims.append(d['images']);ss.append(d['state']);aa.append(d['action'][np.minimum(idx,n-1)]);mm.append(idx<n);ee.extend([entry['episode']]*n);tt.extend(range(n))
        splits[split]={'images':torch.tensor(np.concatenate(ims),device=device),'state':torch.tensor(np.concatenate(ss),device=device),'action':torch.tensor(np.concatenate(aa),device=device),'mask':torch.tensor(np.concatenate(mm),device=device),'episode':np.array(ee),'frame':np.array(tt)}
    return splits,manifest

def load_human(root,device):
    ims=[];future=[];motion=[];mask=[]
    for path in sorted(Path(root).glob('*/supervision.npz')):
        d=np.load(path);n=len(d['images']);k=5
        if n<=k:continue
        ims.append(d['images'][:-k]);future.append(d['images'][k:]);motion.append((d['features'][k:]-d['features'][:-k]).reshape(-1,6))
        valid=d['valid'][k:]&d['valid'][:-k]&(d['track_id'][k:]==d['track_id'][:-k])
        # No bridging detector gaps: identity must remain observed across all intervening frames.
        for j in range(1,k):valid &= d['valid'][j:n-k+j]&(d['track_id'][j:n-k+j]==d['track_id'][:-k])
        mask.append(np.repeat(valid,3,axis=1))
    return {k:torch.tensor(np.concatenate(v),device=device) for k,v in [('images',ims),('future',future),('motion',motion),('mask',mask)]}

def metrics(pred,truth,mask):
    e=(pred-truth).cpu().numpy();m=mask.cpu().numpy();e=e[m];j=[i for i in range(14) if i not in [6,13]];g=[6,13]
    return {'joint_rmse_rad':float(np.sqrt(np.mean(e[:,j]**2))),'joint_mae_rad':float(np.abs(e[:,j]).mean()),'joint_p95_rad':float(np.quantile(np.abs(e[:,j]),.95)),'gripper_mae_native':float(np.abs(e[:,g]).mean()),'per_dimension_rmse':np.sqrt(np.mean(e**2,0)).tolist()}

def evaluate(model,data,stats,out,tag,representation):
    sm,sd,am,ad=stats;pred=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(data['state']),64):
            s=data['state'][i:i+64];p=model(data['images'][i:i+64],(s-sm)/sd)*ad+am+(s[:,None] if representation=='delta' else 0);pred.append(p)
    p=torch.cat(pred);truth=data['action'];mask=data['mask'];report=metrics(p,truth,mask)
    report['persistence']=metrics(data['state'][:,None].expand_as(truth),truth,mask)
    report['last_horizon']=metrics(p[:,-1:],truth[:,-1:],mask[:,-1:])
    # Independent held-out representative chunks for UI (same indices for all arms).
    select=np.arange(0,len(p),max(1,len(p)//40))[:40]
    np.savez_compressed(out/f'{tag}_predictions.npz',predicted=p[select].cpu(),target=truth[select].cpu(),state=data['state'][select].cpu(),mask=mask[select].cpu(),episode=data['episode'][select],frame=data['frame'][select],images=data['images'][select].cpu())
    model.train();return report

def run(a):
    torch.set_num_threads(6);device=a.device;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data,manifest=load_robot(a.robot,device,a.horizon);human=load_human(a.human,device);tr=data['train'];s=tr['state'];delta=tr['action']-(s[:,None] if a.action_representation=='delta' else 0);valid=tr['mask']
    sm=s.mean(0);sd=s.std(0).clamp_min(.05);am=delta[valid].mean(0);ad=delta[valid].std(0).clamp_min(.03);stats=(sm,sd,am,ad)
    metadata={'args':vars(a),'robot_manifest':manifest,'human_frames':len(human['images']),'human_manifest_sha256':hashlib.sha256((Path(a.human)/'manifest.json').read_bytes()).hexdigest() if (Path(a.human)/'manifest.json').exists() else 'smoke_partial','human_valid_hand_targets':int(human['mask'].sum().item()/3),'action_units':'12 radians + 2 native gripper values','objective':'robot_only: action; explicit: action + masked two-hand displacement; implicit_joint: action + future RGB residual; shuffled: action + permuted motion','primary_endpoint':'native RoboTwin adjust_bottle closed-loop success, never ranking heterogeneous auxiliary losses','human_robot_actions_used':False}
    (out/'experiment.json').write_text(json.dumps(metadata,indent=2))
    for condition in a.conditions.split(','):
        folder=out/condition;folder.mkdir(exist_ok=True)
        if (folder/'report.json').exists():continue
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        model=Policy(a.horizon).to(device);initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);rng=np.random.default_rng(a.seed);hrng=np.random.default_rng(a.seed+777)
        shuffle=torch.randperm(len(human['motion']),device=device,generator=torch.Generator(device=device).manual_seed(a.seed+99))
        log=[];start=time.time()
        for step in range(a.pre_steps+a.fine_steps):
            idx=torch.tensor(rng.integers(len(s),size=a.batch),device=device);state=s[idx];target=(tr['action'][idx]-(state[:,None] if a.action_representation=='delta' else 0)-am)/ad
            prediction=model(tr['images'][idx],(state-sm)/sd);m=tr['mask'][idx,:,None];action_loss=((prediction-target).square()*m).sum()/(m.sum()*14)
            auxiliary=action_loss*0;image_loss=action_loss*0
            if step<a.pre_steps and condition!='robot_only':
                hi=torch.tensor(hrng.integers(len(human['images']),size=a.batch//2),device=device);motion,image=model.human(human['images'][hi])
                if condition in ['explicit','shuffled']:
                    ti=shuffle[hi] if condition=='shuffled' else hi;hm=human['mask'][ti];ht=human['motion'][ti]/.1
                    auxiliary=((motion-ht).square()*hm).sum()/hm.sum().clamp_min(1)
                if condition=='implicit_joint':
                    current=F.interpolate(human['images'][hi].float().permute(0,3,1,2)/255,(32,32),mode='area');future=F.interpolate(human['future'][hi].float().permute(0,3,1,2)/255,(32,32),mode='area')
                    image_loss=F.mse_loss(image,future-current)
            loss=action_loss+.1*auxiliary+image_loss
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%200==0 or step==a.pre_steps+a.fine_steps-1:
                row={'step':step+1,'phase':'joint_pretrain' if step<a.pre_steps else 'same_robot_finetune','action_loss':action_loss.item(),'motion_loss':auxiliary.item(),'image_loss':image_loss.item(),'seconds':time.time()-start};log.append(row);print(condition,a.seed,row,flush=True)
            if step+1 in [a.pre_steps,a.pre_steps+a.fine_steps]:
                tag='pretrain' if step+1==a.pre_steps else 'final';ckpt={'model':model.state_dict(),'stats':{k:v.cpu() for k,v in zip(['state_mean','state_std','target_mean','target_std'],stats)},'action_representation':a.action_representation,'horizon':a.horizon,'condition':condition,'seed':a.seed,'step':step+1,'initial_hash':initial}
                torch.save(ckpt,folder/f'{tag}.pt')
                if condition=='implicit_joint':
                    with torch.no_grad():
                        hi=torch.arange(0,len(human['images']),max(1,len(human['images'])//12),device=device)[:12]
                        _,residual=model.human(human['images'][hi]);cur=F.interpolate(human['images'][hi].float().permute(0,3,1,2)/255,(32,32),mode='area')
                        np.savez_compressed(folder/f'{tag}_future_images.npz',current=human['images'][hi].cpu(),future=human['future'][hi].cpu(),predicted=((cur+residual).clamp(0,1)*255).byte().permute(0,2,3,1).cpu())
                report=evaluate(model,data['test'],stats,folder,tag,a.action_representation);(folder/f'{tag}_offline.json').write_text(json.dumps(report,indent=2))
        (folder/'curves.json').write_text(json.dumps(log));(folder/'report.json').write_text(json.dumps({'initial_hash':initial,'seconds':time.time()-start,'steps':a.pre_steps+a.fine_steps,'robot_frame_exposures':(a.pre_steps+a.fine_steps)*a.batch,'human_frame_exposures':a.pre_steps*(a.batch//2) if condition!='robot_only' else 0,'test':report},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--human',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=17);p.add_argument('--pre-steps',type=int,default=2000);p.add_argument('--fine-steps',type=int,default=4000);p.add_argument('--batch',type=int,default=64);p.add_argument('--horizon',type=int,default=16);p.add_argument('--action-representation',choices=['delta','absolute'],default='delta');p.add_argument('--conditions',default='robot_only,explicit,implicit_joint,shuffled');run(p.parse_args())
