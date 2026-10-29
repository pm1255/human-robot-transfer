"""Check the actual FoundationPose CUDA environment, not annotation accuracy."""
from pathlib import Path
import argparse,json
import numpy as np
from .common import Progress,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
 progress=Progress(root,'foundation_environment',1)
 try:
  import torch
  import open3d
  import warp
  import nvdiffrast.torch as dr
  import pytorch3d
  import mycpp
  from Utils import bilateral_filter_depth,nvdiffrast_render
  from estimater import FoundationPose
  progress.update(0,phase='actual_cuda_smoke_check')
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
  context=dr.RasterizeCudaContext()
  # Invoke a real Warp kernel used by FoundationPose. Constant plane is a fixture.
  depth=torch.ones((48,64),dtype=torch.float32,device='cuda')
  filtered=bilateral_filter_depth(depth,radius=2,device='cuda')
  torch.cuda.synchronize()
  if not torch.isfinite(filtered).all():raise RuntimeError('Nonfinite Warp output')
  report=dict(environment_ready=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
              cuda=torch.version.cuda,warp=warp.__version__,open3d=open3d.__version__,
              pytorch3d=pytorch3d.__version__,kernel_test='synthetic_plane_not_annotation',pose_inference_completed=False)
  atomic_json(root/'foundation_environment.json',report);progress.update(1,'completed')
 except Exception as exc:
  atomic_json(root/'foundation_environment.json',dict(environment_ready=False,error=str(exc),pose_inference_completed=False))
  progress.update(0,'failed',error=str(exc));raise
 finally:progress.close()
if __name__=='__main__':main()
