"""ImageNet ResNet18 spatial visual policy, gripper-only output. This is not a VLA."""
import torch
from torch import nn
from torchvision.models import resnet18
from v3.common import HORIZON

class GripperPolicy(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__();net=resnet18(weights=None)
        if weights_path:net.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True)
        self.encoder=nn.Sequential(*list(net.children())[:-2])
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512*4*4+9,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,HORIZON*7))
        self.register_buffer('image_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('image_std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def train(self,mode=True):
        super().train(mode)
        # Fixed pretrained BN statistics avoids train/eval mismatch on small robot batches.
        for m in self.encoder.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,image,state,side):
        x=image.permute(0,3,1,2).float()/255
        if self.training:
            factor=1+(torch.rand((len(x),1,1,1),device=x.device)-.5)*.2
            x=(x*factor).clamp(0,1)
        f=self.encoder((x-self.image_mean)/self.image_std).flatten(1)
        onehot=torch.nn.functional.one_hot(side.long(),2).float()
        return self.head(torch.cat([f,state,onehot],1)).reshape(-1,HORIZON,7)
