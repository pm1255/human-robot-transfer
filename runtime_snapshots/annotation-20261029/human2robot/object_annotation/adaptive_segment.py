"""SAM3.1 predictions from per-frame table/garment localization prompts.

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


def garment_proposal(frame,index,anchors,expand=False):
    h,w=frame.shape[:2];hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    table=cv2.inRange(hsv,np.array([12,90,25]),np.array([29,255,255]));table[int(h*.958):]=0
    table=cv2.morphologyEx(table,cv2.MORPH_OPEN,np.ones((19,19),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(table)
    good=sorted(range(1,n),key=lambda k:stats[k,4],reverse=True)[:2];good=[k for k in good if stats[k,4]>1500]
    roi=np.zeros((h,w),np.uint8);pts=cv2.findNonZero(np.isin(lab,good).astype(np.uint8))
    if pts is not None:cv2.fillConvexPoly(roi,cv2.convexHull(pts),1)
    cloth=(hsv[:,:,1]<70)&(hsv[:,:,2]>65)&roi.astype(bool)
    n,lab,stats,_=cv2.connectedComponentsWithStats(cloth.astype(np.uint8))
    good=sorted(range(1,n),key=lambda k:stats[k,4],reverse=True)[:3]
    good=[k for k in good if stats[k,4]>max(150,stats[good[0],4]*.03)] if good else []
    proposal=np.isin(lab,good)
    if index<10:
        box=np.array(anchors[0]['bbox'],float)
        return box,roi.astype(bool),'visually_reviewed_initial_partial_garment'
    if proposal.sum()<300:return None,roi.astype(bool),'no_reliable_color_location'
    box=np.array(mask_box(proposal),float)+np.array([-4,-4,4,4]);box=np.clip(box,[0,0,0,0],[w,h,w,h])
    if expand:
        white=(hsv[:,:,1]<80)&(hsv[:,:,2]>60)
        nn,ll,ss,_=cv2.connectedComponentsWithStats(white.astype(np.uint8))
        original_box=np.round(box).astype(int);br=np.zeros((h,w),bool)
        br[original_box[1]:original_box[3],original_box[0]:original_box[2]]=True
        components=[]
        for k in range(1,nn):
            component=ll==k;area=int(component.sum());overlap=int((component&br).sum())
            if overlap>200 and overlap/max(area,1)>.35:components.append(k)
        grown=np.isin(ll,components)
        if grown.any():
            yy,xx=np.nonzero(grown)
            box=np.clip([min(box[0],xx.min()-4),min(box[1],yy.min()-4),max(box[2],xx.max()+5),max(box[3],yy.max()+5)],[0,0,0,0],[w,h,w,h])
    return box,roi.astype(bool),'table_color_geometry_proposal_only'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--guides',required=True)
    p.add_argument('--sample-only',action='store_true');p.add_argument('--expand',action='store_true');a=p.parse_args();root=Path(a.output)
    data=json.loads((root/'objects.json').read_text());meta=data['meta'];rows=data['frames'];h,w=meta['height'],meta['width']
    guide=json.loads(Path(a.guides).read_text());anchors=guide['anchors'];digest=sha256(a.guides)
    indices=sorted(set([x['frame_index'] for x in anchors]+[x['frame_index']+60 for x in anchors if x['frame_index']+60<len(rows)]+[15,30,60,90])) if a.sample_only else list(range(len(rows)))
    stage=('sam3_edge_sample' if a.sample_only else 'sam3_edge_refine') if a.expand else ('sam3_adaptive_sample' if a.sample_only else 'sam3_adaptive')
    progress=Progress(root,stage,len(indices));done=0
    evidence=root/('adaptive_predictions_v3' if a.expand else 'adaptive_predictions_v2');evidence.mkdir(exist_ok=True)
    previews=root/('adaptive_previews_v3' if a.expand else 'adaptive_previews_v2');previews.mkdir(exist_ok=True)
    single=root/'adaptive_input';single.mkdir(exist_ok=True)
    try:
        from sam3.model_builder import build_sam3_predictor
        progress.update(0,phase='loading_model')
        predictor=build_sam3_predictor(checkpoint_path=meta['checkpoint'],version='sam3.1',compile=False,warm_up=False,use_fa3=False,async_loading_frames=False)
        install_session_adapter(predictor)
        detected=0
        for i in indices:
            row=rows[i];start=row['chunk_start'];box=None
            saved=evidence/f'{i:06d}.npz';record=evidence/f'{i:06d}.json'
            frame=cv2.imread(str(root/'chunks'/f'{start:06d}'/f'{i-start:06d}.jpg'))
            if frame is None:raise ValueError(f'Missing decoded original frame {i}')
            if a.expand and not saved.exists():
                box,_,_=garment_proposal(frame,i,anchors,expand=True)
                old_record=root/'adaptive_predictions_v2'/f'{i:06d}.json'
                old=json.loads(old_record.read_text());old_box=old.get('guide_bbox')
                if (box is None and old_box is None) or (box is not None and old_box is not None and np.max(np.abs(box-old_box))<8):
                    shutil.copy2(root/'adaptive_predictions_v2'/saved.name,saved);shutil.copy2(old_record,record)
            if not saved.exists() or not record.exists() or json.loads(record.read_text()).get('guide_sha256')!=digest:
                h,w=frame.shape[:2];box,table_roi,proposal_source=garment_proposal(frame,i,anchors,expand=a.expand)
                rejected=[];instances=[];mask=np.zeros((h,w),bool)
                if box is not None:
                    cv2.imwrite(str(single/'000000.jpg'),frame,[cv2.IMWRITE_JPEG_QUALITY,97])
                    sid=predictor.handle_request(dict(type='start_session',resource_path=str(single),offload_video_to_cpu=True))['session_id']
                    out=predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,points=[[box[0]/w,box[1]/h],[box[2]/w,box[3]/h]],point_labels=[2,3],obj_id=1))['outputs']
                    for result in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=sid,start_frame_index=0,propagation_direction='forward')):
                        out=result['outputs']
                    masks=as_numpy(out['out_binary_masks']).astype(bool).reshape(-1,h,w)
                    ids=as_numpy(out['out_obj_ids']).reshape(-1)
                    selected=[m for m,oid in zip(masks,ids) if oid==1]
                    if selected:mask=selected[0]
                    if i>=10 and mask.any() and (mask&table_roi).sum()/mask.sum()<(.4 if a.expand else .65):
                        rejected.append('outside_table');mask=np.zeros((h,w),bool)
                    if mask.any():instances=[dict(model_id=1,score=None,score_semantics='interactive_instance_not_calibrated',bbox=mask_box(mask),area=int(mask.sum()))]
                    predictor.handle_request(dict(type='close_session',session_id=sid,run_gc_collect=False))
                np.savez_compressed(saved,mask=np.packbits(mask.reshape(-1)),frame_index=i)
                item=dict(frame_index=i,instances=instances,rejected_candidates=rejected,mask_area=int(mask.sum()),bbox=mask_box(mask),mask_source='SAM3.1_adaptive_instance_single_frame',guide_bbox=None if box is None else box.tolist(),guide_sha256=digest,proposal_source=proposal_source)
                atomic_json(record,item)
            else:
                item=json.loads(record.read_text())
                with np.load(saved) as pack:mask=np.unpackbits(pack['mask'],count=meta['height']*meta['width']).reshape(meta['height'],meta['width']).astype(bool)
            if a.expand:
                old_record=root/'adaptive_predictions_v2'/f'{i:06d}.json'
                old=json.loads(old_record.read_text())
                with np.load(root/'adaptive_predictions_v2'/f'{i:06d}.npz') as pack:
                    old_mask=np.unpackbits(pack['mask'],count=h*w).reshape(h,w).astype(bool)
                retained=float((old_mask&mask).sum()/max(1,old_mask.sum()))
                if old['mask_area'] and (not mask.any() or retained<.65 or mask.sum()>2*old_mask.sum()):
                    item=dict(old,edge_candidate_rejected=True,edge_candidate_retained_fraction=retained)
                    mask=old_mask
                    np.savez_compressed(saved,mask=np.packbits(mask.reshape(-1)),frame_index=i)
                    atomic_json(record,item)
            if a.sample_only or i%120==0:
                view=frame.copy();view[mask]=(view[mask]*.55+np.array([60,210,60])*.45).astype(np.uint8)
                if item['bbox']:cv2.rectangle(view,tuple(item['bbox'][:2]),tuple(item['bbox'][2:]),(0,255,255),2)
                cv2.putText(view,f"{i} | {'MASK' if mask.any() else 'MISSING'}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,0,255),2)
                cv2.imwrite(str(previews/f'{i:06d}.jpg'),view)
            done+=1;detected+=int(item['mask_area']>0)
            if done%10==0:progress.update(done,detected_frames=detected,phase='guided_grounding')
        if a.sample_only:
            atomic_json(root/('edge_sample_summary.json' if a.expand else 'adaptive_sample_summary.json'),dict(frames=indices,detected=detected,guide_sha256=digest,accuracy_verified=False))
            progress.update(done,'completed',detected_frames=detected);return
        backup=root/'masks_before_adaptive'
        if not backup.exists():shutil.copytree(root/'masks',backup)
        if not (root/'objects_before_adaptive.json').exists():shutil.copy2(root/'objects.json',root/'objects_before_adaptive.json')
        previous=None
        for start in sorted({r['chunk_start'] for r in rows}):
            chunk_rows=[r for r in rows if r['chunk_start']==start];packed=[]
            for row in chunk_rows:
                i=row['frame_index'];item=json.loads((evidence/f'{i:06d}.json').read_text())
                with np.load(evidence/f'{i:06d}.npz') as pack:bits=pack['mask']
                mask=np.unpackbits(bits,count=meta['height']*meta['width']).reshape(meta['height'],meta['width']).astype(bool)
                flags=[]
                if not mask.any():flags.append('no_detection')
                elif mask.sum()<1000:flags.append('small_partial_mask_review')
                if mask.any():
                    bbox=mask_box(mask);guide_bbox=item.get('guide_bbox')
                    if guide_bbox and any(bbox[j]>=guide_bbox[j]-1 for j in (2,3)) or (guide_bbox and any(bbox[j]<=guide_bbox[j]+1 for j in (0,1))):flags.append('prompt_boundary_review')
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
        meta.update(final_segmentation_stage=stage,spatial_prompt_method='table HSV proposal plus connected garment extension' if a.expand else 'table HSV proposal',mask_policy='SAM3.1 interactive instance mask from frame-adaptive garment location; boxes derive from masks',guided_prompt_sha256=digest,guided_prompt_source=guide['source'],missing_mask_refinement='all frames independently segmented with SAM3.1 instance box prompts; table/color heuristics locate garment only; masks are model output; no temporal interpolation')
        atomic_json(root/'objects.json',data);atomic_json(root/'segmentation_manifest.json',meta)
        atomic_json(root/'segmentation_quality.json',dict(frames=len(rows),detected_frames=detected,missing_frames=len(rows)-detected,review_frames=sum(bool(r['flags']) for r in rows),independently_verified=False))
        progress.update(done,'completed',detected_frames=detected)
    except Exception as e:
        progress.update(done,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
