import argparse, json, random, time, math
from pathlib import Path
import numpy as np
import torch
from v4.common import *
from v4.model import WanPolicy, losses

class Dataset:
    def __init__(self,domain,split):
        self.rows=json.loads((ROOT/'manifests'/f'{domain}.json').read_text())[split]
        self.text={k:torch.load(ROOT/'text'/f'{k}.pt',weights_only=True) for k in {r['text_key'] for r in self.rows}}
    def batch(self,idx):
        rows=[self.rows[int(i)] for i in idx]
        z=torch.stack([torch.load(ROOT/'latents'/f'{r["uid"]}.pt',weights_only=True) for r in rows]).cuda()
        return (z,z[:,:,:1].clone(),[self.text[r['text_key']].cuda() for r in rows],
            *[torch.tensor(np.array([r[k] for r in rows]),device='cuda',dtype=torch.long if k=='side' else torch.float32) for k in ['state','side','target','mask']])
    def stats(self):
        s=torch.tensor([r['state'] for r in self.rows],device='cuda');y=torch.tensor([r['target'] for r in self.rows],device='cuda')
        mask=torch.tensor([r['mask'] for r in self.rows],device='cuda')>0;y=y[mask]
        return {'sm':s.mean(0),'ss':s.std(0).clamp_min(.02),'tm':y.mean(0),
            'ts':y.std(0).clamp_min(torch.tensor([.01]*3+[.05]*3+[.1],device='cuda'))}

@torch.no_grad()
def evaluate(model,data,stats,limit=None):
    from scipy.spatial.transform import Rotation
    model.eval();ps=[];ys=[];ms=[]
    ids=np.arange(len(data.rows))
    if limit and len(ids)>limit: ids=np.random.default_rng(729).choice(ids,limit,replace=False)
    for ids2 in np.array_split(ids,math.ceil(len(ids)/4)):
        z,cur,text,state,side,y,mask=data.batch(ids2)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model.policy(cur,text,(state-stats['sm'])/stats['ss'],side)*stats['ts']+stats['tm']
        ps.append(pred.float().cpu().numpy());ys.append(y.cpu().numpy());ms.append(mask.cpu().numpy()>0)
    p=np.concatenate(ps);y=np.concatenate(ys);mask=np.concatenate(ms);delta=p-y
    pos=np.linalg.norm(delta[...,:3],axis=-1)[mask]
    rot=(Rotation.from_rotvec(p[...,3:6][mask]).inv()*Rotation.from_rotvec(y[...,3:6][mask])).magnitude()
    return dict(normalized_mse=float(np.mean((delta[mask]/stats['ts'].cpu().numpy())**2)),
        tcp_position_mean_m=float(pos.mean()),tcp_position_p95_m=float(np.quantile(pos,.95)),
        tcp_rotation_mean_deg=float(np.rad2deg(rot).mean()),gripper_opening_mae=float(np.abs(delta[...,6][mask]).mean()),
        windows=len(p),metric_scope='offline teacher-observation action error, not closed-loop success'),p,y

def run(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.set_float32_matmul_precision('high');out=ROOT/'training'/f'seed{a.seed}'/a.condition;out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists(): print('ALREADY_DONE',out);return
    model=WanPolicy().cuda();trainable=[p for p in model.parameters() if p.requires_grad]
    total=sum(p.numel() for p in model.parameters());tuned=sum(p.numel() for p in trainable)
    protocol=vars(a)|dict(base='Wan2.1-T2V-1.3B',umt5='original UMT5-XXL frozen',vae='original Wan2.1 frozen',
        total_parameters=total,trainable_parameters=tuned,rank=16,effective_batch=a.micro*a.accum,
        image_loss='future latent flow matching (not current-image reconstruction)',action_loss='masked smooth L1 normalized TCP xyz/rotvec/opening',
        action_branch_sees_future=False,language_input=True,base_weight_sha=sha(WEIGHTS/'diffusion_pytorch_model.safetensors'),
        data_audit_sha=sha(ROOT/'DATA_AUDIT.json'),code_sha={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    dump(out/'protocol.json',protocol); print('Start training',json.dumps(protocol),flush=True)
    stages=[]
    if a.condition!='robotwin_only': stages.append(('pretrain','bridge' if a.condition=='bridge_joint' else 'human',a.pre_steps))
    stages.append(('finetune','sim',a.fine_steps))
    for stage,domain,steps in stages:
        if (out/f'{stage}_DONE.json').exists():
            model.load_adapter(torch.load(out/f'{stage}_best.pt',map_location='cpu',weights_only=False)['adapter']);continue
        train=Dataset(domain,'train');val=Dataset(domain,'validation');stats=train.stats()
        if stage=='pretrain': assert len(train.rows)==10000
        aw=0. if stage=='pretrain' and a.condition=='human_video_only' else 1.
        opt=torch.optim.AdamW(trainable,lr=2e-4,weight_decay=.01)
        rng=np.random.default_rng(a.seed+(117 if stage=='pretrain' else 129));torch.manual_seed(a.seed+(117 if stage=='pretrain' else 129))
        start=1;best=float('inf');started=time.time()
        latest=out/f'{stage}_latest.pt'
        if latest.exists():
            ck=torch.load(latest,map_location='cpu',weights_only=False);model.load_adapter(ck['adapter']);opt.load_state_dict(ck['optimizer'])
            start=ck['step']+1;best=ck['best'];rng.bit_generator.state=ck['numpy_rng'];torch.cuda.set_rng_state(ck['cuda_rng'].cpu());torch.set_rng_state(ck['torch_rng'].cpu())
        for step in range(start,steps+1):
            model.train();opt.zero_grad(set_to_none=True);lr=2e-4*min(1,step/500)*(.05+.95*(1+math.cos(math.pi*step/steps))/2)
            for group in opt.param_groups: group['lr']=lr
            logs={'action':0.,'video':0.}
            for _ in range(a.accum):
                batch=train.batch(rng.integers(len(train.rows),size=a.micro))
                with torch.autocast('cuda',dtype=torch.bfloat16): loss,parts=losses(model,batch,stats,aw)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                (loss/a.accum).backward()
                for k in logs: logs[k]+=float(parts[k])/a.accum
            grad=torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step()
            if step==1 or step%10==0:
                p=dict(stage=stage,step=step,total_steps=steps,**logs,action_weight=aw,grad_norm=float(grad),lr=lr,
                    elapsed_seconds=time.time()-started,seconds_per_update=(time.time()-started)/(step-start+1),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                dump(out/'progress.json',p)
                with (out/'curves.jsonl').open('a') as f:f.write(json.dumps(p)+'\n')
                print(json.dumps(p),flush=True)
            if step%500==0 or step==steps:
                metrics,_,_=evaluate(model,val,stats,limit=200)
                # Video-only pretraining selects on held-out flow loss, never random action-head error.
                if aw==0:
                    model.eval();vl=[]
                    with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(917)
                        for j in range(32):
                            with torch.autocast('cuda',dtype=torch.bfloat16): l,_=losses(model,val.batch([j%len(val.rows)]),stats,0.)
                            vl.append(float(l))
                    score=float(np.mean(vl));metrics['video_flow_validation']=score
                else: score=metrics['normalized_mse']
                ck=dict(adapter=model.adapter_state(),stats={k:v.cpu() for k,v in stats.items()},sim_stats={k:v.cpu() for k,v in stats.items()} if domain=='sim' else None,
                    stage=stage,step=step,condition=a.condition,seed=a.seed,base_weight_sha=protocol['base_weight_sha'],protocol=protocol)
                if score<best: best=score;save_tensor(out/f'{stage}_best.pt',ck)
                ck.update(optimizer=opt.state_dict(),best=best,numpy_rng=rng.bit_generator.state,cuda_rng=torch.cuda.get_rng_state(),torch_rng=torch.get_rng_state())
                save_tensor(latest,ck);dump(out/f'{stage}_validation.json',dict(step=step,**metrics))
        model.load_adapter(torch.load(out/f'{stage}_best.pt',map_location='cpu',weights_only=False)['adapter'])
        if stage=='finetune':
            metrics,pred,truth=evaluate(model,Dataset(domain,'test'),stats)
            dump(out/'finetune_offline.json',metrics);np.savez_compressed(out/'finetune_predictions.npz',prediction=pred,target=truth)
        dump(out/f'{stage}_DONE.json',dict(steps=steps))
    dump(out/'DONE.json',dict(condition=a.condition,seed=a.seed,completed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=GROUPS,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--pre-steps',type=int,default=20000);p.add_argument('--fine-steps',type=int,default=10000)
    p.add_argument('--micro',type=int,default=2);p.add_argument('--accum',type=int,default=8);a=p.parse_args();run(a)
