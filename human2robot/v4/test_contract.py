"""Small CPU architecture tests; these do not substitute for the true Wan GPU smoke."""
import unittest
import torch
from v4.model import WanPolicy,losses

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(9)
        cls.model=WanPolicy(rank=2,tiny_config=dict(dim=64,ffn_dim=128,freq_dim=16,in_dim=16,out_dim=16,
            num_heads=4,num_layers=2,text_dim=32,text_len=8,model_type='t2v'))
        cls.model.gradient_checkpointing=False
        cls.z=torch.randn(1,16,2,8,8);cls.text=[torch.randn(4,32)]
        cls.state=torch.zeros(1,7);cls.side=torch.zeros(1,dtype=torch.long)
        cls.stats={k:torch.zeros(7) if k.endswith('m') else torch.ones(7) for k in ['sm','ss','tm','ts']}
    def test_language_and_current_only_contract(self):
        self.model.eval()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            p=self.model.policy(self.z[:,:,:1],self.text,self.state,self.side)
            q=self.model.policy(self.z[:,:,:1],[self.text[0]+3],self.state,self.side)
            self.assertGreater(float((p-q).abs().max()),1e-6)
            with self.assertRaises(AssertionError):self.model.policy(self.z,self.text,self.state,self.side)
    def test_masked_targets_do_not_affect_losses(self):
        self.model.eval();mask=torch.tensor([[1.,0,0,0]])
        target=torch.zeros(1,4,7);other=target.clone();other[:,1:]=1000
        results=[]
        for y in [target,other]:
            torch.manual_seed(51)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                l,_=losses(self.model,(self.z,self.z[:,:,:1],self.text,self.state,self.side,y,mask),self.stats)
            results.append(l)
        torch.testing.assert_close(*results,rtol=0,atol=0)
    def test_video_only_ignores_all_pseudo_state_and_actions(self):
        self.model.eval();values=[]
        for offset in [0.,100.]:
            torch.manual_seed(51)
            batch=(self.z,self.z[:,:,:1],self.text,self.state+offset,self.side,torch.ones(1,4,7)*offset,torch.ones(1,4))
            with torch.autocast('cpu',dtype=torch.bfloat16):l,_=losses(self.model,batch,self.stats,action_weight=0)
            values.append(l)
        torch.testing.assert_close(*values,rtol=0,atol=0)
    def test_adapter_keys_strict(self):
        d=self.model.adapter_state();self.model.load_adapter(d)
        d['unexpected']=torch.zeros(1)
        with self.assertRaises(ValueError):self.model.load_adapter(d)

if __name__=='__main__':unittest.main()
