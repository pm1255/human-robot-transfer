"""Track one garment with SAM3.1 instance prompts, resetting at reviewed anchors."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from .common import Progress,atomic_json,sha256
from .sam3_video import as_numpy,mask_box
from .sam3_compat import install_session_adapter


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--guides',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 root=Path(a.output);data=json.loads((root/'objects.json').read_text());meta=data['meta'];rows=data['frames'];h,w=meta['height'],meta['width']
 guides=json.loads(Path(a.guides).read_text());anchors={x['frame_index']:x for x in guides['anchors']}
 starts=sorted({r['chunk_start'] for r in rows});starts=[x for x in starts if x in [0,720,1080,3360,4800]] if a.pilot else starts
 total=sum(sum(r['chunk_start']==s for r in rows) for s in starts);stage='sam3_instance_pilot' if a.pilot else 'sam3_instance'
 progress=Progress(root,stage,total);done=0;outdir=root/'instance_masks';outdir.mkdir(exist_ok=True);preview=root/'instance_previews';preview.mkdir(exist_ok=True)
 try:
  from sam3.model_builder import build_sam3_predictor
  progress.update(0,phase='loading_model')
  predictor=build_sam3_predictor(checkpoint_path=meta['checkpoint'],version='sam3.1',compile=False,warm_up=False,use_fa3=False,async_loading_frames=False)
  install_session_adapter(predictor)
  for start in starts:
   cr=[r.copy() for r in rows if r['chunk_start']==start];n=len(cr);path=outdir/f'{start:06d}.npz'
   if not path.exists():
    sid=predictor.handle_request(dict(type='start_session',resource_path=str(root/'chunks'/f'{start:06d}'),offload_video_to_cpu=True))['session_id']
    anchor=30 if start==0 else start;box=[188,156,488,388] if start==0 else anchors[start]['bbox'];local=anchor-start
    points=[[box[0]/w,box[1]/h],[box[2]/w,box[3]/h]]
    prompt=predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=local,points=points,point_labels=[2,3],obj_id=1))
    pending={}
    for result in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=sid,start_frame_index=local,propagation_direction='both',evict_cached_frame_outputs=True)):
     i=int(result['frame_index']);o=result['outputs'];masks=as_numpy(o['out_binary_masks']).reshape(-1,h,w).astype(bool);ids=as_numpy(o['out_obj_ids']).reshape(-1)
     selected=[m for m,oid in zip(masks,ids) if oid==1];mask=selected[0] if selected else np.zeros((h,w),bool)
     pending[i]=np.packbits(mask.reshape(-1))
     if i%30==0:progress.update(done+len(pending),phase='instance_tracking',chunk_start=start)
    predictor.handle_request(dict(type='close_session',session_id=sid))
    if set(pending)!=set(range(n)):raise ValueError(f'Incomplete instance chunk {start}: {sorted(pending)}')
    np.savez_compressed(path,masks=np.stack([pending[i] for i in range(n)]),shape=np.array([h,w]),frame_indices=np.arange(start,start+n))
    atomic_json(path.with_suffix('.prompt.json'),dict(anchor_frame=anchor,prompt_type='SAM_instance_box_corner_points',bbox=box,guide_source=guides['source'],guide_sha256=sha256(a.guides)))
   with np.load(path) as pack:masks=np.unpackbits(pack['masks'],axis=1,count=h*w).reshape(-1,h,w).astype(bool)
   for j,row in enumerate(cr):
    i=row['frame_index'];mask=masks[j];area=int(mask.sum());box=mask_box(mask)
    row.update(mask_area=area,bbox=box,instances=[dict(model_id=1,score=None,score_semantics='interactive_instance_not_calibrated',bbox=box,area=area)] if area else [],mask_source='SAM3.1_interactive_instance_tracking',status='predicted' if area else 'no_detection',flags=[] if area else ['no_detection'])
    row.pop('guide_bbox',None)
    if i%30==0 or i==len(rows)-1:
     frame=cv2.imread(str(root/'chunks'/f'{start:06d}'/f'{j:06d}.jpg'));view=frame.copy();view[mask]=(view[mask]*.55+np.array([60,210,60])*.45).astype(np.uint8)
     if box:cv2.rectangle(view,tuple(box[:2]),tuple(box[2:]),(0,255,255),2)
     cv2.putText(view,f'{i} | INSTANCE', (10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,0,255),2);cv2.imwrite(str(preview/f'{i:06d}.jpg'),view)
   atomic_json(path.with_suffix('.json'),cr);done+=n;progress.update(done,phase='chunk_complete')
  if not a.pilot:
   previous=None;newrows=[]
   backup=root/'masks_before_instance'
   if not backup.exists():shutil.copytree(root/'masks',backup)
   for start in starts:
    path=outdir/f'{start:06d}.npz';cr=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as pack:masks=np.unpackbits(pack['masks'],axis=1,count=h*w).reshape(-1,h,w).astype(bool)
    for row,mask in zip(cr,masks):
     iou=None
     if previous is not None:
      union=(previous|mask).sum();iou=float((previous&mask).sum()/union) if union else None
      if iou is not None and iou<.5:row['flags'].append('mask_change_review')
     row['adjacent_mask_iou']=iou;previous=mask
    shutil.copy2(path,root/'masks'/path.name);atomic_json(root/'masks'/path.with_suffix('.json').name,cr);newrows.extend(cr)
   if not (root/'objects_before_instance.json').exists():shutil.copy2(root/'objects.json',root/'objects_before_instance.json')
   meta.update(mask_policy='one garment SAM3.1 interactive instance, box-corner initialization at visually reviewed 120-frame anchors, bidirectional tracking, no mask or box interpolation',guided_prompt_sha256=sha256(a.guides),guided_prompt_source=guides['source'])
   meta.pop('missing_mask_refinement',None)
   atomic_json(root/'objects.json',dict(meta=meta,frames=newrows));atomic_json(root/'segmentation_manifest.json',meta)
   detected=sum(r['mask_area']>0 for r in newrows)
   atomic_json(root/'segmentation_quality.json',dict(frames=len(newrows),detected_frames=detected,missing_frames=len(newrows)-detected,review_frames=sum(bool(r['flags']) for r in newrows),independently_verified=False))
  progress.update(done,'completed')
 except Exception as exc:progress.update(done,'failed',error=str(exc));raise
 finally:progress.close()

if __name__=='__main__':main()
