"""Audit actual model outputs; these tests cannot establish segmentation accuracy."""
import json,argparse
from pathlib import Path
import numpy as np
from .sam3_video import mask_box
from .common import atomic_json

def audit(out):
 d=json.loads((out/'objects.json').read_text());m=d['meta'];rows=d['frames'];h,w=m['height'],m['width'];n=m['frames'];assert len(rows)==n
 issues=[];previous={};coverage={p[0]:0 for p in m['prompts']};checks=0
 for i,r in enumerate(rows):
  assert r['frame_index']==i and abs(r['timestamp_s']-i/m['fps'])<1e-6
  p=np.load(out/'masks'/f'{i:06d}.npz');assert p['shape'].tolist()==[h,w] and int(p['frame_index'])==i
  mm=np.unpackbits(p['masks'],axis=1,count=h*w).reshape(-1,h,w).astype(bool);assert len(mm)==len(r['instances'])
  current={}
  for j,(o,mask)in enumerate(zip(r['instances'],mm)):
   assert o['index']==j and mask.any() and o['bbox']==mask_box(mask) and o['area']==int(mask.sum());checks+=1
   key=o['instance_id'];box=o['bbox'];center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2]);a=int(mask.sum());flags=[]
   if a/(h*w)>.65:flags.append('large_mask_review')
   if a<25:flags.append('very_small_mask_review')
   if key in previous:
    prev=previous[key];distance=float(np.linalg.norm(center-prev['center']));ratio=a/max(prev['area'],1);iou=float((mask&prev['mask']).sum()/max(1,(mask|prev['mask']).sum()))
    if distance>np.hypot(h,w)*.08:flags.append('center_jump_review')
    if ratio>3 or ratio<1/3:flags.append('area_change_review')
    if iou<.25:flags.append('low_adjacent_iou_review')
   if flags:issues.append(dict(frame_index=i,instance=key,flags=flags))
   current[key]=dict(mask=mask,center=center,area=a)
  for label in coverage:coverage[label]+=int(any(o['class_id']==label for o in r['instances']))
  for key in previous.keys()-current.keys():issues.append(dict(frame_index=i,instance=key,flags=['track_missing_visibility_unknown']))
  previous=current
 summary=dict(schema='unified.multimodal_qc.v1',frames=n,bbox_mask_exact_instances=checks,structure_passed=True,class_detected_frames=coverage,class_missing_frames={k:n-v for k,v in coverage.items()},flagged_frames=len({r['frame_index']for r in issues}),flags=issues,accuracy_verified=False,temporal_identity_verified=False,training_eligible=False,interpretation='Flags are review candidates. Missing masks may be occlusion; coverage is not precision or recall.')
 atomic_json(out/'multimodal_qc.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();allrows={}
 for s in sorted((a.root/'samples').iterdir()):
  if(s/'multimodal/objects.json').exists():allrows[s.name]=audit(s/'multimodal')
 atomic_json(a.root/'multimodal_audit.json',allrows)
