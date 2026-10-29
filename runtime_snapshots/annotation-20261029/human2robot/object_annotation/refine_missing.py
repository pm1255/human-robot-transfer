"""Re-detect missing masks with SAM3.1 single-frame grounding, never interpolate."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch

from .common import Progress, atomic_json
from .sam3_compat import install_session_adapter
from .sam3_video import as_numpy, mask_box


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    path=root/'objects.json';data=json.loads(path.read_text());meta=data['meta'];rows=data['frames']
    targets=rows;progress=Progress(root,'sam3_redetect',len(targets));done=0
    try:
        if not targets:
            progress.update(0,'completed');return
        progress.update(0,phase='loading_model')
        from sam3.model_builder import build_sam3_predictor
        predictor=build_sam3_predictor(checkpoint_path=meta['checkpoint'],version='sam3.1',
            compile=False,warm_up=False,use_fa3=False,async_loading_frames=False)
        install_session_adapter(predictor)
        original=root/'objects_before_redetection.json'
        if not original.exists():shutil.copy2(path,original)
        frame_dir=root/'redetect_frame';frame_dir.mkdir(exist_ok=True)
        evidence=root/'spatial_redetections';evidence.mkdir(exist_ok=True)
        recovered=0
        for row in targets:
            i=row['frame_index'];start=row['chunk_start']
            saved=evidence/f'{i:06d}.npz';record=evidence/f'{i:06d}.json'
            if not saved.exists() or not record.exists():
                frame=cv2.imread(str(root/'chunks'/f'{start:06d}'/f'{i-start:06d}.jpg'))
                if frame is None:raise ValueError(f'Missing original chunk frame {i}')
                cv2.imwrite(str(frame_dir/'000000.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,97])
                sid=predictor.handle_request(dict(type='start_session',resource_path=str(frame_dir),offload_video_to_cpu=True))['session_id']
                output=predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text='white cloth on the table',bounding_boxes=[[0.08,0.28,0.84,0.62]],bounding_box_labels=[1]))['outputs']
                masks=as_numpy(output['out_binary_masks']).astype(bool).reshape(-1,meta['height'],meta['width'])
                ids=as_numpy(output['out_obj_ids']).reshape(-1);scores=as_numpy(output['out_probs']).reshape(-1)
                selected=[(m,int(oid),float(s)) for m,oid,s in zip(masks,ids,scores) if s>=.5 and m.any()]
                # Keep one manipulated tabletop object; never union the worn shirt.
                selected=[x for x in selected if np.nonzero(x[0])[0].mean()<meta['height']*.86]
                if selected: selected=[max(selected,key=lambda x: x[0][int(meta['height']*.30):int(meta['height']*.86),int(meta['width']*.08):int(meta['width']*.92)].sum())]
                mask=np.any([x[0] for x in selected],axis=0) if selected else np.zeros((meta['height'],meta['width']),bool)
                instances=[dict(model_id=oid,score=s,bbox=mask_box(m),area=int(m.sum())) for m,oid,s in selected]
                np.savez_compressed(saved,mask=np.packbits(mask.reshape(-1)),frame_index=i)
                atomic_json(record,dict(instances=instances,mask_area=int(mask.sum()),bbox=mask_box(mask),
                    mask_source='SAM3.1_single_frame_spatial_redetection',prompt='white cloth on the table',box_xywh_normalized=[0.08,0.28,0.84,0.62],threshold=.5))
                predictor.handle_request(dict(type='close_session',session_id=sid,run_gc_collect=False))
            item=json.loads(record.read_text());recovered+=int(item['mask_area']>0);done+=1
            if done%10==0:progress.update(done,recovered_frames=recovered,phase='single_frame_grounding')
        previous=None;new_rows=[]
        for start in sorted({r['chunk_start'] for r in rows}):
            chunk_rows=[r.copy() for r in rows if r['chunk_start']==start]
            file=root/'masks'/f'{start:06d}.npz'
            with np.load(file) as source:packed={k:source[k] for k in source.files}
            for local,row in enumerate(chunk_rows):
                i=row['frame_index'];saved=evidence/f'{i:06d}.npz'
                if saved.exists():
                    item=json.loads((evidence/f'{i:06d}.json').read_text())
                    if True:
                        with np.load(saved) as redetected:packed['masks'][local]=redetected['mask']
                        row.update(mask_area=item['mask_area'],bbox=item['bbox'],instances=item['instances'],
                                   mask_source=item['mask_source'],status='predicted_single_frame' if item['mask_area'] else 'no_detection')
                row.setdefault('mask_source','SAM3.1_video_tracking')
                mask=np.unpackbits(packed['masks'][local],count=meta['height']*meta['width']).reshape(meta['height'],meta['width']).astype(bool)
                flags=[]
                if not row['mask_area']:flags.append('no_detection')
                if len(row['instances'])>1:flags.append('multiple_instances_review')
                if row['mask_source'].endswith('redetection'):flags.append('single_frame_redetection_review')
                if previous is not None:
                    union=(previous|mask).sum();iou=float((previous&mask).sum()/union) if union else None
                    row['adjacent_mask_iou']=iou
                    if iou is not None and iou<.5:flags.append('mask_change_review')
                row['flags']=flags;previous=mask
            temp=file.with_suffix('.partial.npz');np.savez_compressed(temp,**packed);temp.replace(file)
            atomic_json(file.with_suffix('.json'),chunk_rows);new_rows.extend(chunk_rows)
        data['frames']=new_rows;data['meta']['missing_mask_refinement']='SAM3.1 single-frame spatial grounding, threshold 0.5; largest tabletop instance; no interpolation'
        atomic_json(path,data);atomic_json(root/'segmentation_manifest.json',data['meta'])
        detected=sum(r['mask_area']>0 for r in new_rows)
        atomic_json(root/'segmentation_quality.json',dict(frames=len(rows),detected_frames=detected,
            missing_frames=len(rows)-detected,redetected_frames=recovered,
            review_frames=sum(bool(r['flags']) for r in new_rows),independently_verified=False))
        progress.update(done,'completed',recovered_frames=recovered)
    except Exception as exc:
        progress.update(done,'failed',error=str(exc));raise
    finally:progress.close()


if __name__=='__main__':main()
