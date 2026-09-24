"""Eight independent GPU shards cache frozen original Wan VAE and UMT5 outputs."""
import argparse,json,gc
import numpy as np
import torch
from v4.common import *
from v4.native import module

def run(worker,workers):
    torch.set_num_threads(4); texts=json.loads((ROOT/'texts.json').read_text()); target=ROOT/'text';target.mkdir(exist_ok=True)
    keys=sorted(texts)[worker::workers]
    encoder=module('t5').T5EncoderModel(text_len=512,device=torch.device('cuda'),
        checkpoint_path=str(WEIGHTS/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(WEIGHTS/'google/umt5-xxl'))
    with torch.no_grad():
        for i,k in enumerate(keys):
            if not (target/f'{k}.pt').exists():
                z=encoder([texts[k]],torch.device('cuda'))[0]
                save_tensor(target/f'{k}.pt',z.cpu().bfloat16())
    del encoder;gc.collect();torch.cuda.empty_cache()
    vae=module('vae').WanVAE(vae_pth=str(WEIGHTS/'Wan2.1_VAE.pth'),dtype=torch.bfloat16,device='cuda')
    allrows={}
    for domain in ['human','bridge','sim']:
        for rows in json.loads((ROOT/'manifests'/f'{domain}.json').read_text()).values():
            for r in rows: allrows[r['uid']]=r
    rows=[allrows[k] for k in sorted(allrows)[worker::workers]];dest=ROOT/'latents';dest.mkdir(exist_ok=True)
    last=None;rgb=None
    with torch.no_grad():
        for i,r in enumerate(sorted(rows,key=lambda r:r['episode'])):
            path=dest/f'{r["uid"]}.pt'
            if path.exists(): continue
            if last!=r['episode']:
                rgb=np.load(ROOT/'rgb'/f'{r["episode"]}.npy',mmap_mode='r');last=r['episode']
            clip=np.array(rgb[r['t']+np.arange(FRAMES)*r['stride']])
            x=torch.from_numpy(clip).permute(3,0,1,2).cuda().float()/127.5-1
            z=vae.encode([x])[0]
            assert z.shape==(16,2,32,32) and torch.isfinite(z).all()
            # Causal encoder property verified against an independently encoded single frame.
            if i==0:
                one=vae.encode([x[:,:1]])[0]
                delta=float((one-z[:,:1]).abs().max());assert delta<.03,delta
                dump(ROOT/f'VAE_CAUSALITY_{worker}.json',{'max_difference':delta})
            save_tensor(path,z.cpu().bfloat16())
            if i%100==0: dump(ROOT/f'CACHE_PROGRESS_{worker}.json',{'done':i+1,'total':len(rows)});print('CACHE',worker,i+1,len(rows),flush=True)
    dump(ROOT/f'CACHE_DONE_{worker}.json',{'samples':len(rows),'texts':len(keys)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();run(a.worker,a.workers)
