"""SAM3.1 predictions with visually reviewed garment localization anchors.

Interpolated boxes are prompts, never labels. Final boxes come from masks.
Samples and full inference are kept separately from failed earlier attempts.
"""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from .common import Progress, atomic_json, sha256
from .sam3_compat import install_session_adapter
from .sam3_video import as_numpy, mask_box


def guide_box(anchors, index):
    indices=np.array([a['frame_index'] for a in anchors])
    boxes=np.array([a['bbox'] for a in anchors],float)
    return np.array([np.interp(index,indices,boxes[:,j]) for j in range(4)])


def choose_mask(masks,ids,scores,box,frame):
    h,w=frame.shape[:2];x0,y0,x1,y1=np.round(box).astype(int)
    roi=np.zeros((h,w),bool);roi[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=True
    candidates=[];rejected=[]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    for mask,oid,score in zip(masks,ids,scores):
        area=int(mask.sum())
        if not area or score<.5:continue
        inside=float((mask&roi).sum()/area)
        coverage=float((mask&roi).sum()/max(1,roi.sum()))
        saturation=float(np.median(hsv[:,:,1][mask]))
        # Target identity checks reject wall/whole table/worn shirt candidates.
        reasons=[]
        if inside<.55:reasons.append('outside_target_guide')
        if saturation>80:reasons.append('brown_table_candidate')
        item=dict(model_id=int(oid),score=float(score),area=area,bbox=mask_box(mask),
                  fraction_inside_guide=inside,guide_coverage=coverage,median_saturation=saturation)
        if reasons:rejected.append(dict(**item,rejection=reasons));continue
        candidates.append((float(score)*inside*(.25+min(coverage,1)),mask,item))
    if not candidates:return np.zeros((h,w),bool),[],rejected
    _,mask,item=max(candidates,key=lambda x:x[0]);return mask,[item],rejected


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--guides',required=True)
    p.add_argument('--sample-only',action='store_true');a=p.parse_args();root=Path(a.output)
    data=json.loads((root/'objects.json').read_text());meta=data['meta'];rows=data['frames']
    guide=json.loads(Path(a.guides).read_text());anchors=guide['anchors'];digest=sha256(a.guides)
    indices=sorted(set([x['frame_index'] for x in anchors]+[x['frame_index']+60 for x in anchors if x['frame_index']+60<len(rows)]+[15,30,60,90])) if a.sample_only else list(range(len(rows)))
    stage='sam3_guided_sample' if a.sample_only else 'sam3_guided'
    progress=Progress(root,stage,len(indices));done=0
    evidence=root/'guided_predictions';evidence.mkdir(exist_ok=True)
    previews=root/'guided_previews';previews.mkdir(exist_ok=True)
    single=root/'guided_input';single.mkdir(exist_ok=True)
    try:
        from sam3.model_builder import build_sam3_predictor
        progress.update(0,phase='loading_model')
        predictor=build_sam3_predictor(checkpoint_path=meta['checkpoint'],version='sam3.1',compile=False,warm_up=False,use_fa3=False,async_loading_frames=False)
        install_session_adapter(predictor)
        detected=0
        for i in indices:
            row=rows[i];start=row['chunk_start'];box=guide_box(anchors,i)
            saved=evidence/f'{i:06d}.npz';record=evidence/f'{i:06d}.json'
            frame=cv2.imread(str(root/'chunks'/f'{start:06d}'/f'{i-start:06d}.jpg'))
            if frame is None:raise ValueError(f'Missing decoded original frame {i}')
            if not saved.exists() or not record.exists() or json.loads(record.read_text()).get('guide_sha256')!=digest:
                h,w=frame.shape[:2];x0,y0,x1,y1=box;xywh=[x0/w,y0/h,(x1-x0)/w,(y1-y0)/h]
                cv2.imwrite(str(single/'000000.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,97])
                sid=predictor.handle_request(dict(type='start_session',resource_path=str(single),offload_video_to_cpu=True))['session_id']
                out=predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text='white t-shirt',bounding_boxes=[xywh],bounding_box_labels=[1]))['outputs']
                masks=as_numpy(out['out_binary_masks']).astype(bool).reshape(-1,h,w)
                mask,instances,rejected=choose_mask(masks,as_numpy(out['out_obj_ids']).reshape(-1),as_numpy(out['out_probs']).reshape(-1),box,frame)
                predictor.handle_request(dict(type='close_session',session_id=sid,run_gc_collect=False))
                np.savez_compressed(saved,mask=np.packbits(mask.reshape(-1)),frame_index=i)
                item=dict(frame_index=i,instances=instances,rejected_candidates=rejected,mask_area=int(mask.sum()),bbox=mask_box(mask),mask_source='SAM3.1_guided_single_frame',guide_bbox=box.tolist(),guide_sha256=digest,threshold=.5)
                atomic_json(record,item)
            else:
                item=json.loads(record.read_text())
                with np.load(saved) as pack:mask=np.unpackbits(pack['mask'],count=meta['height']*meta['width']).reshape(meta['height'],meta['width']).astype(bool)
            if a.sample_only or i%120==0:
                view=frame.copy();view[mask]=(view[mask]*.55+np.array([60,210,60])*.45).astype(np.uint8)
                if item['bbox']:cv2.rectangle(view,tuple(item['bbox'][:2]),tuple(item['bbox'][2:]),(0,255,255),2)
                cv2.putText(view,f"{i} | {'MASK' if mask.any() else 'MISSING'}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,0,255),2)
                cv2.imwrite(str(previews/f'{i:06d}.jpg'),view)
            done+=1;detected+=int(item['mask_area']>0)
            if done%10==0:progress.update(done,detected_frames=detected,phase='guided_grounding')
        if a.sample_only:
            atomic_json(root/'guided_sample_summary.json',dict(frames=indices,detected=detected,guide_sha256=digest,accuracy_verified=False))
            progress.update(done,'completed',detected_frames=detected);return
        backup=root/'masks_before_guided'
        if not backup.exists():shutil.copytree(root/'masks',backup)
        if not (root/'objects_before_guided.json').exists():shutil.copy2(root/'objects.json',root/'objects_before_guided.json')
        previous=None
        for start in sorted({r['chunk_start'] for r in rows}):
            chunk_rows=[r for r in rows if r['chunk_start']==start];packed=[]
            for row in chunk_rows:
                i=row['frame_index'];item=json.loads((evidence/f'{i:06d}.json').read_text())
                with np.load(evidence/f'{i:06d}.npz') as pack:bits=pack['mask']
                mask=np.unpackbits(bits,count=meta['height']*meta['width']).reshape(meta['height'],meta['width']).astype(bool)
                flags=[]
                if not mask.any():flags.append('no_detection')
                iou=None
                if previous is not None:
                    union=(previous|mask).sum();iou=float((previous&mask).sum()/union) if union else None
                    if iou is not None and iou<.5:flags.append('mask_change_review')
                row.update({k:item[k] for k in ['mask_area','bbox','instances','mask_source','guide_bbox']})
                row.update(flags=flags,adjacent_mask_iou=iou,status='predicted' if mask.any() else 'no_detection')
                packed.append(bits);previous=mask
            file=root/'masks'/f'{start:06d}.npz';tmp=file.with_suffix('.partial.npz')
            np.savez_compressed(tmp,masks=np.stack(packed),shape=np.array([meta['height'],meta['width']]),frame_indices=np.array([r['frame_index'] for r in chunk_rows]));tmp.replace(file)
            atomic_json(file.with_suffix('.json'),chunk_rows)
        meta.update(mask_policy='single garment instance selected by SAM score and overlap with visually guided target location; no mask interpolation',guided_prompt_sha256=digest,guided_prompt_source=guide['source'],missing_mask_refinement='all frames re-detected with time-varying spatial prompts; guide boxes are not output labels')
        atomic_json(root/'objects.json',data);atomic_json(root/'segmentation_manifest.json',meta)
        atomic_json(root/'segmentation_quality.json',dict(frames=len(rows),detected_frames=detected,missing_frames=len(rows)-detected,review_frames=sum(bool(r['flags']) for r in rows),independently_verified=False))
        progress.update(done,'completed',detected_frames=detected)
    except Exception as e:
        progress.update(done,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
