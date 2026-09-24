"""Wan LoRA shares language/visual weights across causal policy and future-video branches.

The policy branch receives ONLY the current-frame latent, text, state and side.
The video branch additionally receives noisy future latents and optional actions.
No teacher-forced future features are passed into the action head.
"""
import json, math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from v4.common import WEIGHTS, HORIZON
from v4.native import module

class LoRALinear(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__(); self.base = base.requires_grad_(False); self.scale = 1.
        self.a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
    def forward(self, x):
        return self.base(x) + self.b(self.a(x.float())).to(x.dtype) * self.scale

class WanPolicy(nn.Module):
    def __init__(self, weights=WEIGHTS, rank=16, tiny_config=None):
        super().__init__(); native = module('model')
        cfg = tiny_config or {k:v for k,v in json.loads((weights/'config.json').read_text()).items() if not k.startswith('_')}
        self.wan = native.WanModel(**cfg)
        if tiny_config is None:
            self.wan.load_state_dict(load_file(str(weights/'diffusion_pytorch_model.safetensors')), strict=True)
        self.wan.requires_grad_(False).to(dtype=torch.bfloat16)
        # Official Wan computes time modulation in fp32.
        self.wan.time_embedding.float(); self.wan.time_projection.float()
        # WanLayerNorm explicitly casts activations to fp32, so keep affine weights fp32 too.
        for layer in self.wan.modules():
            if isinstance(layer, nn.LayerNorm): layer.float()
        for block in self.wan.blocks:
            for attn in [block.self_attn, block.cross_attn]:
                for name in ['q','k','v','o']:
                    setattr(attn, name, LoRALinear(getattr(attn, name), rank))
        dim = self.wan.dim
        self.action_head = nn.Sequential(nn.LayerNorm(dim*16+9), nn.Linear(dim*16+9,512),
            nn.GELU(), nn.Linear(512,HORIZON*7))
        self.action_condition = nn.Linear(HORIZON*7+HORIZON+9, dim)
        nn.init.zeros_(self.action_condition.weight); nn.init.zeros_(self.action_condition.bias)
        self.gradient_checkpointing = True

    def features(self, z, timestep, text, action_context=None):
        w = self.wan; native = module('model'); b = len(z)
        if w.freqs.device != z.device: w.freqs = w.freqs.to(z.device)
        x = w.patch_embedding(z.to(w.patch_embedding.weight.dtype)); grid = x.shape[2:]
        x = x.flatten(2).transpose(1,2)
        grids = torch.tensor([grid]*b, dtype=torch.long)
        lengths = torch.full((b,),x.shape[1],dtype=torch.long)
        with torch.autocast(z.device.type, enabled=False):
            e = w.time_embedding(native.sinusoidal_embedding_1d(w.freq_dim,timestep).float())
            e0 = w.time_projection(e).unflatten(1,(6,w.dim))
        context = torch.stack([torch.cat([u,u.new_zeros(w.text_len-len(u),u.shape[-1])]) for u in text])
        context = w.text_embedding(context.to(w.patch_embedding.weight.dtype))
        if action_context is not None:
            context = context + self.action_condition(action_context.float())[:,None].to(context.dtype)
        kwargs = dict(e=e0,seq_lens=lengths,grid_sizes=grids,freqs=w.freqs,context=context,context_lens=None)
        for block in w.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False, **kwargs)
            else: x = block(x,**kwargs)
        return x,e,grids

    def policy(self, current, text, state, side):
        assert current.shape[2] == 1, 'Policy input must contain only the current image'
        x,_,grid = self.features(current,torch.zeros(len(current),device=current.device),text)
        h,w = grid[0,1:].tolist()
        spatial = x.transpose(1,2).reshape(len(x),self.wan.dim,h,w).float()
        spatial = nn.functional.adaptive_avg_pool2d(spatial,(4,4)).flatten(1)
        slot = nn.functional.one_hot(side.long(),2).float()
        return self.action_head(torch.cat([spatial,state.float(),slot],-1)).reshape(-1,HORIZON,7)

    def velocity(self, noisy, timestep, text, state, side, actions, mask):
        ac = None
        if actions is not None:
            slot = nn.functional.one_hot(side.long(),2).float()
            ac = torch.cat([(actions*mask[...,None]).flatten(1),mask,state,slot],-1)
        x,e,grids = self.features(noisy,timestep,text,ac)
        return torch.stack(self.wan.unpatchify(self.wan.head(x,e),grids)).float()

    def adapter_state(self):
        names = {n for n,p in self.named_parameters() if p.requires_grad}
        return {n:v.detach().cpu() for n,v in self.state_dict().items() if n in names}

    def load_adapter(self, state):
        expected = set(self.adapter_state())
        if set(state) != expected: raise ValueError('Adapter keys differ from model: '+str(expected.symmetric_difference(state)))
        self.load_state_dict(state,strict=False)

def losses(model, batch, stats, action_weight=1., video_weight=1.):
    z,cur,text,state,side,target,mask = batch
    s = (state-stats['sm'])/stats['ss']; y = (target-stats['tm'])/stats['ts']
    if action_weight:
        pred = model.policy(cur,text,s,side)
        la = (nn.functional.smooth_l1_loss(pred,y,reduction='none').mean(-1)*mask).sum()/mask.sum().clamp_min(1)
    else:
        la = z.new_zeros((), dtype=torch.float32)
    sigma = torch.sigmoid(torch.randn(len(z),device=z.device))
    noise = torch.randn_like(z); alpha = sigma[:,None,None,None,None]
    noisy = (1-alpha)*z+alpha*noise
    noisy[:,:,0:1] = cur  # observation conditioning, causal VAE first latent
    velocity = model.velocity(noisy,sigma*1000,text,s,side,y if action_weight else None,mask)
    lv = nn.functional.mse_loss(velocity[:,:,1:],(noise-z)[:,:,1:])
    return action_weight*la+video_weight*lv, {'action':la.detach(), 'video':lv.detach()}
