"""Real model outputs for the eight unified pilot episodes. Source videos stay read-only.

SAM3.1 tracks visually prompted instances; model IDs are not an
assertion of verified physical identity. Missing detections remain missing. Optical flow
is explicitly camera-inclusive 2D flow, never a metric object displacement.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from human2robot.hand_mesh.video import probe,frames,Writer
from .common import Progress,atomic_json,sha256
from .sam3_video import mask_box
from .sam3_video import as_numpy
from .sam3_compat import install_session_adapter, install_interactive_output_adapter, propagate_prompted_video

PROMPTS={
 'abc130k_ep_101288':[('bread','bread'),('plate','plate'),('tongs','kitchen tongs'),('gripper','robot gripper')],
 'abc130k_ep_110624':[('screwdriver','screwdriver'),('container','blue plastic container'),('gripper','robot gripper')],
 'abc130k_ep_042125':[('whiteboard','whiteboard'),('eraser','whiteboard eraser'),('gripper','robot gripper')],
 'umi_hifi_pot':[('pot','cooking pot'),('induction_cooker','induction cooker'),('gripper','robot gripper')],
 'umi_hifi_cabinet':[('cabinet_door','cabinet door'),('cabinet_handle','cabinet handle'),('gripper','robot gripper')],
 'umi_fast_sandwich':[('lettuce','lettuce'),('bread','bread'),('gripper','robot gripper')],
 'egodex_basic_pick_place-part1_000000':[('mustard_bottle','yellow mustard bottle'),('container','container'),('hand','human hand')],
 'egodex_add_remove_lid_000000':[('cup','cup'),('cup_lid','cup lid'),('hand','human hand')],
}
COLORS=[(80,205,60),(230,150,40),(190,75,200),(55,195,230)]
# Visually inspected source-frame rectangles are prompts, never output labels.
# Boxes are normalized xyxy on the full source image including any letterbox.
GUIDES={
 'abc130k_ep_101288':{'bread':[[.405,.34,.505,.44],[.475,.37,.55,.445]],'plate':[[.445,.50,.565,.62]],'tongs':[[.67,.43,.735,.56]],'gripper':[[.65,.40,.76,.52],[.29,.35,.40,.46]]},
 'abc130k_ep_110624':{'screwdriver':[[.435,.55,.515,.635]],'container':[[.39,.34,.505,.51]],'gripper':[[.28,.44,.38,.54],[.59,.44,.70,.54]]},
 'abc130k_ep_042125':{'whiteboard':[[.33,.36,.525,.545]],'eraser':[[.30,.515,.36,.555]],'gripper':[[.49,.37,.64,.48],[.15,.36,.245,.46]]},
 'umi_hifi_pot':{'pot':[[.38,.375,.52,.575]],'induction_cooker':[[.465,.27,.60,.395]],'gripper':[[.30,.61,.48,.755],[.625,.56,.755,.73]]},
 'umi_hifi_cabinet':{'cabinet_door':[[.095,.075,.295,.495],[.27,.015,.46,.37],[.45,.35,.59,.64]],'cabinet_handle':[[.36,.46,.405,.56]],'gripper':[[.40,.82,.55,.985],[.555,.80,.735,.99]]},
 'umi_fast_sandwich':{'lettuce':[[.205,.655,.285,.755]],'bread':[[.335,.59,.46,.69]],'gripper':[[.22,.70,.45,.96],[.555,.72,.80,.985]]},
 'egodex_basic_pick_place-part1_000000':{'mustard_bottle':[[.635,.53,.705,.75]],'container':[[.425,.54,.56,.74]],'hand':[[.185,.54,.31,.79],[.80,.70,.99,.94]]},
 'egodex_add_remove_lid_000000':{'cup':[[.22,.62,.31,.80],[.395,.52,.47,.70],[.52,.515,.60,.69],[.23,.73,.375,.98]],'cup_lid':[[.43,.69,.51,.81],[.52,.70,.61,.83],[.69,.695,.80,.83],[.49,.83,.61,.995]],'hand':[[.67,.83,.88,.995]]},
}

def mark(image,lines):
 cv2.rectangle(image,(0,0),(image.shape[1],23*len(lines)+7),(20,30,35),-1)
 for i,line in enumerate(lines):cv2.putText(image,line,(7,20+23*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1,cv2.LINE_AA)
 return image

def validate_movie(path,meta):
 info=probe(path)
 count=sum(1 for _ in frames(path,info))
 assert count==meta['frames'] and abs(info['fps']-meta['fps'])<1e-4,(path,info,count)
 return dict(frames=count,fps=info['fps'],sha256=sha256(path),complete_decode=True)

def tracked_sam_episode(predictor,source,out,prompts,limit=0):
 """Use the official SAM3.1 video interface including NMS and tracking."""
 out.mkdir(parents=True,exist_ok=True);meta=probe(source);n=min(meta['frames'],limit)if limit else meta['frames'];h,w=meta['height'],meta['width']
 cache=out/'rgb';cache.mkdir(exist_ok=True);md=out/'masks';md.mkdir(exist_ok=True)
 for i,frame in enumerate(frames(source,meta,limit=n)):
  path=cache/f'{i:06d}.jpg'
  if not path.exists():assert cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,97])
 config=dict(schema='unified.sam3.instances.v4',model='SAM3.1 official interactive video predictor',source_sha256=sha256(source),**meta,processed_frames=n,prompts=prompts,spatial_guides=GUIDES[source.parent.name],guide_method='assistant inspected stated prompt frames; normalized rectangles are prompts only; absent objects remain unverified',output_adapter='merge actual tracker masks into empty detector cache',score_threshold=.5,mask_threshold=.5,instance_identity='model_track_local_to_episode_and_prompt_unverified',bbox_format='xyxy_pixel_exclusive',bbox_source='exact_predicted_mask_extent',temporal_model=True,training_eligible=False,independently_verified=False)
 atomic_json(out/'sam3_manifest.json',config);prog=Progress(out,'sam3',n*len(prompts));done=0;class_rows=[]
 try:
  for ci,(label,prompt) in enumerate(prompts):
   classdir=out/'classes'/label;classdir.mkdir(parents=True,exist_ok=True)
   existing=classdir/'rows.json'
   if existing.exists():
    rr=json.loads(existing.read_text());assert len(rr)==n;class_rows.append(rr);done+=n;prog.update(done,phase='resume_class',label=label);continue
   sid=predictor.handle_request(dict(type='start_session',resource_path=str(cache),offload_video_to_cpu=True))['session_id']
   guide=GUIDES[source.parent.name][label];start=int(guide.get('frame',0))if isinstance(guide,dict)else 0;boxes=guide['boxes']if isinstance(guide,dict)else guide
   assert 0<=start<n
   for oid,b in enumerate(boxes,1):
    seed_result=predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=start,points=[[b[0],b[1]],[b[2],b[3]]],point_labels=[2,3],obj_id=oid))
   pending={}
   for result in propagate_prompted_video(predictor,sid,start,n,seed_result):
    i=int(result['frame_index']);o=result['outputs'];mm=as_numpy(o['out_binary_masks']).astype(bool).reshape(-1,h,w);ids=as_numpy(o['out_obj_ids']).reshape(-1);scores=as_numpy(o['out_probs']).reshape(-1)if 'out_probs'in o else [None]*len(ids);objects=[];bits=[]
    assert len(mm)==len(ids)==len(scores)
    for mask,oid,score in zip(mm,ids,scores):
     if not mask.any():continue
     objects.append(dict(index=len(bits),class_id=label,class_index=ci,model_track_id=int(oid),instance_id=f'{label}:{int(oid)}',score=float(score)if score is not None else None,score_calibrated=False,bbox=mask_box(mask),area=int(mask.sum()),review_required=True));bits.append(np.packbits(mask))
    np.savez_compressed(classdir/f'{i:06d}.npz',masks=np.stack(bits)if bits else np.zeros((0,(h*w+7)//8),np.uint8),frame_index=i)
    pending[i]=objects
    if i in [0,n//2,n-1]:
     preview=cv2.imread(str(cache/f'{i:06d}.jpg'));union=np.any(mm,axis=0)if len(mm)else np.zeros((h,w),bool);preview[union]=(preview[union]*.55+np.array(COLORS[ci%len(COLORS)])*.45).astype(np.uint8);cv2.imwrite(str(classdir/f'preview_{i:06d}.jpg'),preview)
    if len(pending)%20==0:prog.update(done+len(pending),phase='official_video_tracking',label=label,detected_frames=sum(bool(x)for x in pending.values()))
   predictor.handle_request(dict(type='close_session',session_id=sid))
   assert set(pending)==set(range(n)),'Incomplete video model output'
   rr=[pending[i]for i in range(n)];atomic_json(existing,rr);class_rows.append(rr);done+=n;prog.update(done,phase='class_complete',label=label)
  rows=[]
  for i in range(n):
   allbits=[];objects=[];missing=[]
   for ci,(label,_)in enumerate(prompts):
    bits=np.load(out/'classes'/label/f'{i:06d}.npz')['masks'];offset=len(allbits);allbits.extend(bits)
    objects.extend([dict(o,index=offset+j)for j,o in enumerate(class_rows[ci][i])])
    if not len(bits):missing.append(label)
   np.savez_compressed(md/f'{i:06d}.npz',masks=np.stack(allbits)if allbits else np.zeros((0,(h*w+7)//8),np.uint8),shape=np.array([h,w]),frame_index=i)
   rows.append(dict(frame_index=i,timestamp_s=i/meta['fps'],instances=objects,missing_classes=missing,status='predicted_review_required'if objects else'no_detection'))
  atomic_json(out/'objects.json',dict(meta=config,frames=rows));counts={label:sum(bool(rr[i])for i in range(n))for (label,_),rr in zip(prompts,class_rows)}
  atomic_json(out/'sam3_quality.json',dict(frames=n,class_detected_frames=counts,accuracy_verified=False,instance_tracks_verified=False))
  prog.update(done,'completed');return counts
 except Exception as e:prog.update(done,'failed',error=str(e));raise
 finally:prog.close()

def render_episode(source,out,sem_path):
 meta=probe(source);n=meta['frames'];data=json.loads((out/'objects.json').read_text());rows=data['frames'];assert len(rows)==n
 prog=Progress(out,'render_multimodal',n);writers={name:Writer(out/f'{name}.mp4',meta['width'],meta['height'],meta['fps'])for name in ['segmentation','boundingbox','optical_flow','contact']};previous=None;flow_rows=[];contact_rows=[];segs=json.loads(sem_path.read_text())['segments'];si=0
 try:
  for i,frame in enumerate(frames(source,meta)):
   h,w=frame.shape[:2];row=rows[i];pack=np.load(out/'masks'/f'{i:06d}.npz');mm=np.unpackbits(pack['masks'],axis=1,count=h*w).reshape(-1,h,w).astype(bool);seg=frame.copy();box=frame.copy()
   for obj,mask in zip(row['instances'],mm):
    color=COLORS[obj['class_index']%len(COLORS)];seg[mask]=(seg[mask]*.55+np.array(color)*.45).astype(np.uint8);x1,y1,x2,y2=obj['bbox'];cv2.rectangle(box,(x1,y1),(x2-1,y2-1),color,2);cv2.putText(box,obj['class_id']+' '+str(obj.get('model_track_id','')),(x1,max(65,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
   labels=', '.join(sorted({x['class_id']for x in row['instances']})) or 'NO DETECTION'
   writers['segmentation'].write(mark(seg,[f'SAM3.1 masks | f{i} | {i/meta["fps"]:.2f}s',labels,'MODEL PREDICTION - review required']))
   writers['boundingbox'].write(mark(box,[f'Boxes from exact SAM3.1 masks | f{i}',labels,'Model tracks; identity not independently verified']))
   gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   if previous is None:flow=np.zeros((h,w,2),np.float32);flow_valid=False
   else:flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,21,3,5,1.2,0);flow_valid=True
   previous=gray;mag,ang=cv2.cartToPolar(flow[:,:,0],flow[:,:,1]);hsv=np.zeros_like(frame);hsv[:,:,0]=ang*90/np.pi;hsv[:,:,1]=255;hsv[:,:,2]=np.clip(mag/12,0,1)*255
   flowview=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);flow_rows.append(dict(frame_index=i,valid=flow_valid,pair=[i-1,i] if flow_valid else None,median_magnitude_px=float(np.median(mag))if flow_valid else None,p95_magnitude_px=float(np.percentile(mag,95))if flow_valid else None))
   (out/'flow').mkdir(exist_ok=True);np.savez_compressed(out/'flow'/f'{i:06d}.npz',flow=flow.astype(np.float16),valid=flow_valid,from_frame=i-1,to_frame=i)
   writers['optical_flow'].write(mark(flowview,[f'2D optical flow prev -> current | f{i}','Farneback; includes camera motion; max color 12px','NO PREVIOUS FRAME' if not flow_valid else 'Not metric object displacement']))
   while i>=segs[si]['end_frame_exclusive']:si+=1
   s=segs[si];contacts=[o['index']for o in row['instances']if o['class_id']in ['gripper','hand','tongs']];targets=[o['index']for o in row['instances']if o['class_id']=={'bread_slice':'bread'}.get(s['object'],s['object'])]
   eff=np.any(mm[contacts],axis=0)if contacts else np.zeros((h,w),bool);target=np.any(mm[targets],axis=0)if targets else np.zeros((h,w),bool)
   gap=None;point=None
   if eff.any() and target.any():
    dist=cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,5);vals=np.where(eff,dist,np.inf);y,x=np.unravel_index(np.argmin(vals),vals.shape);gap=float(vals[y,x]);point=[int(x),int(y)]
   contact_rows.append(dict(frame_index=i,segment_id=s['id'],semantic_visual_initial=s.get('contact_transition','unknown'),mask_gap_px=gap,nearest_effector_pixel=point,contact_valid=False,affordance_valid=False,reason='2D proximity alone cannot establish physical contact or an affordance point'))
   view=seg.copy()
   if point:cv2.circle(view,tuple(point),5,(255,255,255),2)
   writers['contact'].write(mark(view,[f'Contact evidence review | f{i} | {s["verb"]}',f'2D mask gap: {gap:.1f}px' if gap is not None else '2D mask gap: unknown (missing target/effector)',f'Visual draft: {s.get("contact_transition","unknown")}','Physical contact: UNVERIFIED; point is not affordance']))
   if i in [0,n//2,n-1]:cv2.imwrite(str(out/f'preview_{i:06d}.jpg'),np.concatenate([seg,box],axis=1))
   if i%30==0:prog.update(i,phase='render_masks_boxes_flow_contact')
  for writer in writers.values():writer.close()
  atomic_json(out/'flow_manifest.json',dict(method='OpenCV_Farneback',units='pixels_per_source_frame',camera_motion_removed=False,parameters=dict(pyr_scale=.5,levels=3,winsize=21,iterations=3,poly_n=5,poly_sigma=1.2),frames=flow_rows,training_eligible=False))
  atomic_json(out/'contact_evidence.json',dict(physical_contact_verified=False,affordance_points_verified=False,frames=contact_rows))
  checks={name:validate_movie(out/f'{name}.mp4',meta)for name in writers};atomic_json(out/'video_validation.json',checks);prog.update(n,'completed')
 except Exception as e:
  for writer in writers.values():
   if writer.proc.poll() is None:writer.proc.terminate()
  prog.update(locals().get('i',0),'failed',error=str(e));raise
 finally:prog.close()

def render_depth(source,out):
 meta=probe(source);droot=out/'depth_model';manifest=json.loads((droot/'depth_manifest.json').read_text());n=meta['frames'];assert manifest['processed_target']==n
 values=[]
 content=np.ones((meta['height'],meta['width']),bool)
 if source.parent.name.startswith('abc130k_'):
  content[:meta['height']//8]=False;content[7*meta['height']//8:]=False
 for i in range(n):
  d=np.load(droot/'depth'/f'{i:06d}.npz');v=d['depth'][d['valid']&content];values.extend(v[::max(1,len(v)//500)].tolist())
 lo,hi=np.percentile(values,[3,97]);writer=Writer(out/'depth.mp4',meta['width'],meta['height'],meta['fps']);audit=[];source_frames=iter(frames(source,meta));previous_median=None;previous_focal=None
 for i in range(n):
  d=np.load(droot/'depth'/f'{i:06d}.npz');depth=d['depth'];raw=next(source_frames);valid=d['valid']&content
  if source.parent.name.startswith('umi_fast_'):valid&=np.max(raw,axis=2)>8
  gray=(np.nan_to_num(np.clip((depth-lo)/max(hi-lo,1e-6),0,1))*255).astype(np.uint8);im=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO);im[~valid]=0
  writer.write(mark(im,[f'MoGe-3 monocular depth | f{i}','Fixed color range across video; scale unverified',f'Model depth range {lo:.2f} - {hi:.2f} (not calibrated metres)']))
  median=float(np.median(depth[valid]))if valid.any()else None;focal=float(d['K'][0,0]);median_change=abs(median/previous_median-1)if median and previous_median else None;focal_change=abs(focal/previous_focal-1)if previous_focal else None
  audit.append(dict(frame_index=i,valid_fraction=float(valid.mean()),positive_finite=bool(valid.any()and np.isfinite(depth[valid]).all()and(depth[valid]>0).all()),depth_median=median,focal_px=focal,adjacent_median_relative_change=median_change,adjacent_focal_relative_change=focal_change,temporal_review=bool((median_change or 0)>.1 or(focal_change or 0)>.1)))
  previous_median=median;previous_focal=focal
 writer.close();atomic_json(out/'depth_validation.json',dict(video=validate_movie(out/'depth.mp4',meta),frames=audit,metric_scale_verified=False,temporal_review_frames=sum(x['temporal_review']for x in audit),black_fisheye_pixels_excluded=source.parent.name.startswith('umi_fast_'),excluded_source_letterbox=source.parent.name.startswith('abc130k_'),content_validity_policy='ABC square source top and bottom 1/8 padding excluded; FastUMI source pixels max(RGB)<=8 excluded from visualization and QC, which may also exclude very dark scene pixels; raw predictions retained',color_range=[float(lo),float(hi)]))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['preflight','sam','depth','render'],required=True);p.add_argument('--sample',default='');p.add_argument('--config',type=Path)
 repo=Path(__file__).resolve().parents[2]
 p.add_argument('--sam-checkpoint',type=Path,default=Path(os.environ.get('SAM3_CHECKPOINT',str(repo/'models/sam3.1/sam3.1_multiplex.pt'))))
 p.add_argument('--depth-checkpoint',type=Path,default=Path(os.environ.get('MOGE_CHECKPOINT',str(repo/'models/moge-3-vitl/model.pt'))))
 a=p.parse_args()
 if a.config:
  config=json.loads(a.config.read_text());PROMPTS.update({k:v['prompts']for k,v in config.items()});GUIDES.update({k:v['guides']for k,v in config.items()})
 samples=[a.sample]if a.sample else list(config if a.config else PROMPTS);ckpt=str(a.sam_checkpoint.expanduser().resolve());processor=None
 if a.stage in ['preflight','sam']:
  from sam3.model_builder import build_sam3_predictor
  predictor=build_sam3_predictor(checkpoint_path=ckpt,version='sam3.1',compile=False,warm_up=False,use_fa3=False,use_rope_real=False,async_loading_frames=False)
  install_session_adapter(predictor)
  install_interactive_output_adapter(predictor)
  checkpoint=torch.load(ckpt,map_location='cpu',weights_only=True);checkpoint=checkpoint.get('model',checkpoint)
  checkpoint={k.replace('sam3_model.','detector.',1)if k.startswith('sam3_model.')else k.replace('sam2_predictor.','tracker.',1)if k.startswith('sam2_predictor.')else k:v for k,v in checkpoint.items()}
  missing=[k for k,_ in predictor.model.named_parameters()if k not in checkpoint];assert not missing,missing[:10]
  atomic_json(a.root/'model_provenance.json',dict(checkpoint=ckpt,sha256=sha256(ckpt),missing_learned_parameters=missing,api='official_video_predictor',use_rope_real=False));del checkpoint
 for sid in samples:
  sample=a.root/'samples'/sid;source=sample/'cam_0.mp4';out=sample/('multimodal_preflight'if a.stage=='preflight'else'multimodal');out.mkdir(exist_ok=True)
  if a.stage in ['preflight','sam']:
   counts=tracked_sam_episode(predictor,source,out,PROMPTS[sid],limit=30 if a.stage=='preflight'else 0)
   if a.stage=='preflight':assert counts.get('pot',0)>=27,'Known visible pot missing in more than 10% of preflight frames'
  elif a.stage=='render':render_episode(source,out,sample/'segments.json')
  else:
   subprocess.run([os.sys.executable,'-m','human2robot.object_annotation.depth_video','--video',str(source),'--checkpoint',str(a.depth_checkpoint.expanduser().resolve()),'--output',str(out/'depth_model'),'--resolution-level','9'],check=True)
   render_depth(source,out)
 if a.stage=='preflight':atomic_json(a.root/'preflight.json',dict(status='passed_actual_sam_forward',samples=samples))

if __name__=='__main__':main()
