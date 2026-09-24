"""Deployment adapter uses the exact trained Wan visual/text/action weights."""
import torch
from torch import nn
from v4.common import ROOT, WEIGHTS, EVAL_TEXT, key
from v4.model import WanPolicy
from v4.native import module

class WanEvalPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.net=WanPolicy().cuda().eval()
        self.vae=module('vae').WanVAE(vae_pth=str(WEIGHTS/'Wan2.1_VAE.pth'),dtype=torch.bfloat16,device='cuda')
        self.text=torch.load(ROOT/'text'/f'{key(EVAL_TEXT)}.pt',map_location='cuda',weights_only=True)
    def load_adapter(self,adapter): self.net.load_adapter(adapter)
    @torch.no_grad()
    def forward(self,image,state,side):
        # Both arms observe the same head-camera frame; encode once then duplicate.
        assert torch.equal(image[0],image[-1])
        x=image[0].permute(2,0,1).float()[:,None]/127.5-1
        z=self.vae.encode([x])[0][None].expand(len(image),-1,-1,-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.net.policy(z,[self.text]*len(image),state,side).float()
