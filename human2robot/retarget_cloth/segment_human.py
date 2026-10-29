"""SAM3.1 removal masks for source-view compositing; never edit cloth labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from human2robot.object_annotation.common import Progress, atomic_json
from human2robot.object_annotation.sam3_video import as_numpy
from human2robot.object_annotation.sam3_compat import install_session_adapter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--objects',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();a.root.mkdir(exist_ok=True,parents=True)
    data=json.loads((a.objects/'objects.json').read_text());meta=data['meta']
    indices=[0,30,90,300,600,696,781,1200,2400,3600,3948,4800,4833] if a.pilot else list(range(len(data['frames'])))
    monitor=Progress(a.objects,'human_removal_pilot' if a.pilot else 'human_removal_full',len(indices))
    out=a.root/'human_masks';out.mkdir(exist_ok=True)
    single=a.root/('single_pilot' if a.pilot else 'single_full');single.mkdir(exist_ok=True)
    done=0
    try:
        from sam3.model_builder import build_sam3_predictor
        monitor.update(0,phase='loading_sam3_1')
        predictor=build_sam3_predictor(checkpoint_path=meta['checkpoint'],version='sam3.1',compile=False,warm_up=False,use_fa3=False,async_loading_frames=False)
        install_session_adapter(predictor)
        for i in indices:
            dest=out/f'{i:06d}.npz';record=dest.with_suffix('.json')
            if dest.exists() and record.exists():done+=1;continue
            row=data['frames'][i];s=row['chunk_start'];raw=cv2.imread(str(a.objects/'chunks'/f'{s:06d}'/f'{i-s:06d}.jpg'))
            cv2.imwrite(str(single/'000000.jpg'),raw,[cv2.IMWRITE_JPEG_QUALITY,97])
            sid=predictor.handle_request(dict(type='start_session',resource_path=str(single),offload_video_to_cpu=True))['session_id']
            output=predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text='human arms and hands'))['outputs']
            for result in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=sid,start_frame_index=0,propagation_direction='forward')):output=result['outputs']
            masks=as_numpy(output['out_binary_masks']).astype(bool).reshape(-1,480,640)
            mask=masks.any(0) if len(masks) else np.zeros((480,640),bool)
            scores=as_numpy(output['out_probs']).reshape(-1).tolist()
            np.savez_compressed(dest,mask=np.packbits(mask),frame_index=i)
            atomic_json(record,dict(frame_index=i,method='SAM3.1_text_image',prompt='human arms and hands',area=int(mask.sum()),scores=scores,accuracy_verified=False))
            predictor.handle_request(dict(type='close_session',session_id=sid,run_gc_collect=False))
            if a.pilot or i%120==0:
                view=raw.copy();view[mask]=(view[mask]*.4+[25,25,150]).clip(0,255).astype(np.uint8)
                cv2.imwrite(str(a.root/f'human_{i:06d}.jpg'),view)
                monitor.writer.add_image(f'mask_{i:06d}',view[:,:,::-1].copy(),i,dataformats='HWC')
            done+=1
            if done%10==0 or a.pilot:monitor.update(done,phase='human_mask_inference',gpu_memory_gb=torch.cuda.memory_allocated()/1024**3)
        monitor.update(done,'completed')
    except Exception as exc:
        monitor.update(done,'failed',error=str(exc));raise
    finally:monitor.close()


if __name__=='__main__':main()
