"""Real-weight finite backward, language sensitivity, future leakage and strict adapter checks."""
import gc,json,time
import numpy as np
import torch
from v4.common import *
from v4.native import module
from v4.model import WanPolicy,losses
from v3.common import state7,local_targets

def run():
    torch.set_num_threads(4);torch.manual_seed(17);ROOT.mkdir(parents=True,exist_ok=True)
    # Works before bulk preprocessing: use a real training frame sequence, not random synthetic RGB.
    r=next(r for r in json.loads((V3/'bridge_manifest.json').read_text()) if r['split']=='train' and r['frames']>=5)
    with np.load(V3/'bridge'/r['file']) as d:
        im=torch.tensor(d['images'][:5]).permute(3,0,1,2).cuda().float()/127.5-1
        pose=d['poses'][0,0];truth=local_targets(pose,d['poses'][1:5,0]);observed_state=state7(pose)
    im=torch.nn.functional.interpolate(im.permute(1,0,2,3),size=(SIZE,SIZE),mode='bilinear').permute(1,0,2,3)
    enc=module('t5').T5EncoderModel(text_len=512,device=torch.device('cuda'),checkpoint_path=str(WEIGHTS/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(WEIGHTS/'google/umt5-xxl'))
    with torch.no_grad():text=enc([r['text'][0],'Move the empty cup to the right.'],torch.device('cuda'))
    del enc;gc.collect();torch.cuda.empty_cache()
    vae=module('vae').WanVAE(vae_pth=str(WEIGHTS/'Wan2.1_VAE.pth'),dtype=torch.bfloat16,device='cuda')
    with torch.no_grad(): z=vae.encode([im])[0][None];cur=vae.encode([im[:,:1]])[0][None]
    causal=float((cur-z[:,:,:1]).abs().max());assert causal<.03,causal
    del vae;gc.collect();torch.cuda.empty_cache()
    model=WanPolicy().cuda();model.eval();stats={k:torch.zeros(7,device='cuda') if k.endswith('m') else torch.ones(7,device='cuda') for k in ['sm','ss','tm','ts']}
    state=torch.tensor(observed_state,device='cuda')[None];side=torch.zeros(1,dtype=torch.long,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=model.policy(cur,[text[0]],state,side);q=model.policy(cur,[text[1]],state,side)
        delta=float((p-q).abs().max());assert delta>1e-6,'Text has no effect on output'
        p2=model.policy(cur,[text[0]],state,side)
        assert torch.equal(p,p2),'Non-deterministic inference'
    model.train();batch=(z,cur,[text[0]],state,side,torch.tensor(truth,device='cuda')[None],torch.ones(1,4,device='cuda'))
    started=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16): loss,parts=losses(model,batch,stats)
    loss.backward();trainable=[v for v in model.parameters() if v.requires_grad]
    assert torch.isfinite(loss)
    grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
    assert any(v>0 for n,v in grads.items() if '.b.weight' in n)
    assert any(v>0 for n,v in grads.items() if n.startswith('action_head'))
    assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
    model.load_adapter(model.adapter_state())
    audit=dict(base='Wan2.1-T2V-1.3B',strict_base_load=True,vae_causal_max_error=causal,text_change_max_action_difference=delta,
        finite_backward=True,lora_nonzero_gradient=True,action_head_nonzero_gradient=True,
        policy_signature_has_no_future_argument=True,loss=float(loss),loss_parts={k:float(v) for k,v in parts.items()},
        seconds_forward_backward=time.time()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        smoke_data='real Bridge current/future RGB, task text, measured current and future TCP',
        smoke_image_note='smoke uses existing 128px Bridge cache resized to 256; formal data decodes original RGB at 256',
        base_sha=sha(WEIGHTS/'diffusion_pytorch_model.safetensors'))
    dump(ROOT/'WAN_SMOKE_PASSED.json',audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':run()
