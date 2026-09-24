"""Held-out causal action predictions + sampled future videos, with ground truth beside them."""
import argparse,json,subprocess,html
import numpy as np
import torch
from v4.common import *
from v4.train import Dataset
from v4.model import WanPolicy
from v4.native import module

@torch.no_grad()
def run(path,count):
    ck=torch.load(path,map_location='cpu',weights_only=False);model=WanPolicy().cuda().eval();model.load_adapter(ck['adapter'])
    st={k:v.cuda() for k,v in ck['stats'].items()};ds=Dataset('sim','test')
    vae=module('vae').WanVAE(vae_pth=str(WEIGHTS/'Wan2.1_VAE.pth'),dtype=torch.bfloat16,device='cuda')
    out=Path(path).parent/'visualizations';out.mkdir(exist_ok=True)
    cards=[]
    for index in np.linspace(0,len(ds.rows)-1,count).astype(int):
        row=ds.rows[index];z,cur,text,state,side,target,mask=ds.batch([index]);s=(state-st['sm'])/st['ss']
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.policy(cur,text,s,side)
        torch.manual_seed(int(index)+791);x=torch.randn_like(z);x[:,:,:1]=cur
        grid=torch.linspace(1,0,31,device='cuda')
        for t,t_next in zip(grid[:-1],grid[1:]):
            with torch.autocast('cuda',dtype=torch.bfloat16):v=model.velocity(x,t[None]*1000,text,s,side,pred,torch.ones_like(mask))
            x=x+(t_next-t)*v;x[:,:,:1]=cur
        video=vae.decode([x[0]])[0].permute(1,2,3,0).cpu().numpy()
        video=np.clip((video+1)*127.5,0,255).astype(np.uint8)
        rgb=np.load(ROOT/'rgb'/f'{row["episode"]}.npy',mmap_mode='r');truth=np.array(rgb[row['t']+np.arange(5)*row['stride']])
        paired=np.concatenate([truth,video],axis=2)
        proc=subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','512x256','-r','5','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/f'future_{index}.mp4')],input=paired.tobytes(),check=True)
        dump(out/f'prediction_{index}.json',dict(instruction=row['text'],left_video='ground truth',right_video='Wan sampled future, conditioned on predicted actions',
            action_prediction=(pred*st['ts']+st['tm'])[0].float().cpu().tolist(),action_target=target[0].cpu().tolist(),
            action_fields=['local_x_m','local_y_m','local_z_m','local_rotvec_x_rad','local_rotvec_y_rad','local_rotvec_z_rad','opening_0to1'],
            future_actions_used_as_policy_input=False,video_fps=5,action_fps=15))
        cards.append(f'<article><h2>{html.escape(row["text"])}</h2><p>左：真实未来视频　右：Wan 从当前帧和预测动作生成的未来</p><video controls loop src="future_{index}.mp4"></video><p><a href="prediction_{index}.json">预测动作与真值（位置 m、旋转 rad、开合 0–1）</a></p></article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Wan 留出集预测</title><style>body{font:17px sans-serif;margin:32px;max-width:1100px}video{width:100%;max-width:1024px}article{margin:40px 0}</style><h1>Wan 未来视频与夹爪动作</h1><p>这是留出集预测；任务成功率以独立闭环执行为准。</p>'+''.join(cards))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--count',type=int,default=4);a=p.parse_args();run(a.checkpoint,a.count)
