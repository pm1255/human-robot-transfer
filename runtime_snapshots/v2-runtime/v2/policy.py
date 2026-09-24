import torch
from torch import nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self,horizon=16):
        super().__init__();self.horizon=horizon
        self.encoder=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,48,3,2,1),nn.GroupNorm(6,48),nn.SiLU(),nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,192),nn.LayerNorm(192),nn.SiLU())
        self.action=nn.Sequential(nn.Linear(192*3+14,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),nn.Linear(512,horizon*14))
        self.motion=nn.Sequential(nn.Linear(192,192),nn.SiLU(),nn.Linear(192,6))
        self.image=nn.Sequential(nn.Linear(192,384),nn.SiLU(),nn.Linear(384,3*32*32))
    def encode(self,images):return self.encoder(images.float().permute(0,3,1,2)/255.)
    def forward(self,images,state):
        b,v=images.shape[:2];z=self.encode(images.flatten(0,1)).reshape(b,v*192)
        return self.action(torch.cat([z,state],1)).reshape(b,self.horizon,14)
    def human(self,image):
        z=self.encode(image);return self.motion(z),self.image(z).reshape(-1,3,32,32)
