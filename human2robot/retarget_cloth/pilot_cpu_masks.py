"""CPU fallback pilot with SAM image detector extracted from SAM3.1 weights."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
import sam3.model_builder as builder
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.decoder import TransformerDecoder
from human2robot.object_annotation.common import Progress,atomic_json

p=argparse.ArgumentParser();p.add_argument('--anchors',action='store_true');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=4834);p.add_argument('--stride',type=int,default=30);p.add_argument('--root',type=Path,default=Path('outputs/robot_composite_20261020/cpu_pilot'));p.add_argument('--threads',type=int,default=6);a=p.parse_args()
torch.set_num_threads(a.threads)
root=a.root;root.mkdir(parents=True,exist_ok=True)
objects=Path('outputs/object_annotation_20261020');data=json.loads((objects/'objects.json').read_text())
indices=sorted(set(range(0,4834,a.stride))|{0,30,90,300,600,696,781,1200,2400,3600,3948,4800,4833}) if a.anchors else [0,30,90,300,600,696,781,1200,2400,3600,3948,4800,4833]
indices=[i for i in indices if a.start<=i<a.stop]
monitor=Progress(objects,f'human_removal_cpu_stride{a.stride}_{a.start}_{a.stop}' if a.anchors else 'human_removal_cpu_pilot_v6',len(indices))
try:
    monitor.update(0,phase='loading_image_model_cpu')
    original_position_encoding = builder._create_position_encoding
    builder._create_position_encoding = lambda precompute_resolution=None: original_position_encoding(None)
    original_coords = TransformerDecoder._get_coords
    TransformerDecoder._get_coords = staticmethod(lambda h,w,device: original_coords(h,w,'cpu'))
    # Pinned transfer is a CUDA optimization; this isolated CPU pilot never
    # transfers tensors to a GPU. No vendor files or global processes change.
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self
    model=build_sam3_image_model(device='cpu',checkpoint_path=data['meta']['checkpoint'],load_from_HF=False)
    # SAM3.1 has three detector neck levels. The SAM3 builder's fourth level
    # is unweighted but discarded by scalp=1; remove it and scalp together.
    assert model.backbone.scalp==1
    model.backbone.vision_backbone.convs=model.backbone.vision_backbone.convs[:3]
    model.backbone.scalp=0
    processor=Sam3Processor(model,device='cpu',confidence_threshold=.3)
    for step,i in enumerate(indices,1):
        if (root/'human_masks'/f'{i:06d}.npz').exists():
            monitor.update(step,phase='resume_saved_anchor');continue
        row=data['frames'][i];s=row['chunk_start'];raw=cv2.imread(str(objects/'chunks'/f'{s:06d}'/f'{i-s:06d}.jpg'))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            state=processor.set_image(Image.fromarray(raw[:,:,::-1]))
            out=processor.set_text_prompt('human arms and hands',state)
        masks=out['masks'].cpu().numpy().reshape(-1,480,640);mask=masks.any(0) if len(masks) else np.zeros((480,640),bool)
        (root/'human_masks').mkdir(exist_ok=True)
        np.savez_compressed(root/'human_masks'/f'{i:06d}.npz',mask=np.packbits(mask),frame_index=i)
        atomic_json(root/'human_masks'/f'{i:06d}.json',dict(frame_index=i,method='SAM3.1_image_detector_CPU',area=int(mask.sum()),scores=out['scores'].tolist(),accuracy_verified=False))
        view=raw.copy();view[mask]=(view[mask]*.4+[25,25,150]).clip(0,255).astype(np.uint8);cv2.imwrite(str(root/f'human_{i:06d}.jpg'),view)
        monitor.update(step,phase='cpu_inference');monitor.writer.add_image(f'mask_{i:06d}',view[:,:,::-1].copy(),i,dataformats='HWC')
    monitor.update(len(indices),'completed')
except Exception as exc:monitor.update(0,'failed',error=str(exc));raise
finally:monitor.close()
