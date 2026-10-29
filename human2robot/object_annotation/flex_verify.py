"""Check adapted GPU hashing/neighbours against an independent Python reference."""
import argparse
import itertools
from pathlib import Path
import numpy as np
import torch
from .flex_compat import install_dtype_itemsize
from .common import Progress,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();progress=Progress(args.output,'depth_kernel_check',9)
    done=0
    try:
        install_dtype_itemsize()
        from flex_gemm.kernels.triton.hashmap import hashmap_build,hashmap_lookup,hashmap_unique
        from flex_gemm.kernels.triton.neighbor_cache.neighbor_map import build_neighbor_map_from_kernel_size_dilation
        for dtype in [torch.int8,torch.int16,torch.int32]:
            keys=torch.tensor([[0,x,y,z] for x in range(4) for y in range(3) for z in range(3)
                               if (x+y+z)%3!=0],dtype=dtype,device='cuda')
            queries=torch.cat([keys.flip(0),torch.tensor([[0,99,99,99]],dtype=dtype,device='cuda')])
            result=hashmap_lookup(hashmap_build(keys),keys,queries).cpu().tolist()
            assert result==list(reversed(range(len(keys))))+[-1]
            done+=1;progress.update(done)
            unique=hashmap_unique(keys.repeat(3,1)).cpu().tolist()
            assert set(map(tuple,unique))==set(map(tuple,keys.cpu().tolist()))
            done+=1;progress.update(done)
            coords=keys.cpu().tolist();lookup={tuple(k):i for i,k in enumerate(coords)}
            offsets=list(itertools.product([-1,0,1],repeat=3))
            expected=np.array([[lookup.get((k[0],k[1]+dx,k[2]+dy,k[3]+dz),-1)
                                for dx,dy,dz in offsets] for k in coords])
            actual=build_neighbor_map_from_kernel_size_dilation(keys,None,kernel_size=(3,3,3)).cpu().numpy()
            np.testing.assert_array_equal(actual,expected)
            done+=1;progress.update(done)
        atomic_json(args.output/'depth_kernel_validation.json',dict(checks=done,status='passed',
            evidence='GPU hash lookup, uniqueness and 3D neighbours match independent Python reference for int8/int16/int32',
            model_accuracy_verified=False))
        progress.update(done,'completed')
    except Exception as exc:
        progress.update(done,'failed',error=str(exc));raise
    finally:progress.close()


if __name__=='__main__':main()
