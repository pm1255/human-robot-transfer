"""Merge checked sequential shards, then audit every frame and dense mesh."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from .video import probe,frames,Writer
from .render import render_hands,composite


def spans(indices):
    out=[]
    for x in indices:
        if out and out[-1][1]+1==x:out[-1][1]=x
        else:out.append([x,x])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--parts',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corrections',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    parts=sorted(a.parts,key=lambda x:json.loads((x/'manifest.json').read_text())['start_frame'])
    first=json.loads((parts[0]/'manifest.json').read_text());expected=0;records=[];disagreements=[];counts=Counter();jumps=[];prev={}
    corrections={};correction_count=0
    if a.corrections:
        cm=json.loads((a.corrections/'manifest.json').read_text())
        for key in ['checkpoint_sha256','source_sha256','frames','focal_length_px','rescale_factor','neural_precision']:
            if cm[key]!=first[key]:raise ValueError(f'Correction mismatch: {key}')
        if not cm.get('sparse_identity_corrections'):raise ValueError('Expected sparse corrections')
        link=a.out/'identity_corrections'
        if not link.exists():link.symlink_to(a.corrections.resolve(),target_is_directory=True)
        for line in (a.corrections/'frames.jsonl').read_text().splitlines():
            f=json.loads(line)
            for h in f['hands']:corrections[(f['frame_index'],h['detector_hand_index'])]=h
    writer=SummaryWriter(str(a.out/'events'));dense_count=0
    for pi,part in enumerate(parts):
        meta=json.loads((part/'manifest.json').read_text())
        for key in ['checkpoint_sha256','source_sha256','detector_sha256','width','height','fps','focal_length_px','rescale_factor','neural_precision']:
            if first[key]!=meta[key]:raise ValueError(f'Shard mismatch: {key}')
        if meta['start_frame']!=expected:raise ValueError('Non-contiguous shards')
        expected=meta['end_frame']
        link=a.out/f'part{pi}'
        if not link.exists():link.symlink_to(part.resolve(),target_is_directory=True)
        cache={}
        for line in (part/'frames.jsonl').read_text().splitlines():
            f=json.loads(line)
            for hi,hand in enumerate(f['hands']):
                key=(f['frame_index'],hand['detector_hand_index'])
                if key in corrections:
                    f['hands'][hi]=dict(corrections[key],identity_reestimated=True);correction_count+=1
            f['status']='REVIEW' if any(h['quality']['status']!='CONSISTENT_2D' for h in f['hands']) else ('MESH' if f['hands'] else 'NO DETECTION')
            sides=Counter(h['handedness'] for h in f['hands'])
            for side,n in sides.items():
                if n>1:counts['duplicate_side_frames']+=1
            for hand in f['hands']:
                path=hand['prediction_file'];idx=hand['prediction_index']
                origin=a.corrections if hand.get('identity_reestimated') else part
                cache_key=str(origin/path)
                if cache_key not in cache:
                    if len(cache)>6:cache.clear()
                    cache[cache_key]=dict(np.load(origin/path))
                chunk=cache[cache_key]
                if chunk['vertices'][idx].shape!=(778,3) or not np.isfinite(chunk['vertices'][idx]).all():raise ValueError('Invalid dense vertices')
                if int(chunk['frame_index'][idx])!=f['frame_index']:raise ValueError('Prediction index mismatch')
                dense_count+=1
                xyz=chunk['joints'][idx]+chunk['cam_t'][idx]
                uv=meta['focal_length_px']*xyz[:,:2]/xyz[:,2:]+[meta['width']/2,meta['height']/2]
                side=hand['handedness']
                if side in prev and prev[side][0]==f['frame_index']-1 and sides[side]==1:
                    speed=float(np.linalg.norm(uv[0]-prev[side][1]))
                    if speed>0.08*np.hypot(meta['width'],meta['height']):jumps.append(dict(frame=f['frame_index'],handedness=side,wrist_jump_px=speed))
                prev[side]=(f['frame_index'],uv[0])
                error=hand['quality']['median_keypoint_disagreement_px']
                if error is not None:disagreements.append(error)
                counts.update(hand['quality']['reason_codes'])
                hand['prediction_file']=('identity_corrections/' if hand.get('identity_reestimated') else f'part{pi}/')+path
            records.append(f)
        writer.add_scalar('progress/validation',expected/first['frames'],expected);writer.flush()
    if expected!=first['frames'] or [r['frame_index'] for r in records]!=list(range(expected)):raise ValueError('Missing/duplicate frame')
    if correction_count!=len(corrections):raise ValueError('Unapplied identity corrections')
    if a.corrections:
        faces=np.load(parts[0]/'faces.npy');width,height=first['width'],first['height'];cache={}
        overlay_writer=Writer(a.out/'hamer_surface.mp4',width,height,first['fps'])
        mesh_writer=Writer(a.out/'hamer_mesh_only.mp4',width,height,first['fps'])
        for i,img in enumerate(frames(first['source_video'],first)):
            hands=[]
            for h in records[i]['hands']:
                path=h['prediction_file'];j=h['prediction_index']
                if path not in cache:
                    if len(cache)>6:cache.clear()
                    cache[path]=dict(np.load(a.out/path))
                chunk=cache[path]
                if h['quality']['status']!='INVALID':hands.append(dict(vertices=chunk['vertices'][j],cam_t=chunk['cam_t'][j],handedness=h['handedness']))
            rgb,depth,_=render_hands(hands,faces,first['focal_length_px'],width,height)
            overlay=composite(img,rgb,depth);surface=composite(np.full_like(img,235),rgb,depth,1)
            cv2.putText(overlay,f'HaMeR | {i}/{expected} | {records[i]["status"]}',(8,height-10),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            overlay_writer.write(overlay);mesh_writer.write(surface)
            if i in [0,120,600,1200,1800,2400,3000,3600,4200,expected-1]:
                image=np.concatenate([img,overlay,surface],axis=1)
                cv2.imwrite(str(a.out/f'frame_{i:05d}.jpg'),image);writer.add_image('review/raw_surface_mesh',image[:,:,::-1],i,dataformats='HWC')
            if i%120==0:writer.add_scalar('progress/final_render',(i+1)/expected,i);writer.flush()
        overlay_writer.close();mesh_writer.close();writer.add_scalar('progress/final_render',1,expected)
    for name in ([] if a.corrections else first['videos']):
        listing=a.out/(name+'.concat.txt')
        # Only internally generated shard names enter the concat manifest.
        listing.write_text(''.join(f"file 'part{i}/{name}'\n" for i in range(len(parts))))
        subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(a.out/name)],check=True)
        if probe(a.out/name)['frames']!=expected:raise ValueError('Merged video frame count mismatch')
    for name in ['faces.npy','camera.npz']:shutil.copy2(parts[0]/name,a.out/name)
    with (a.out/'frames.jsonl').open('w') as f:
        for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
    absent=[r['frame_index'] for r in records if not r['hands']]
    review=[r['frame_index'] for r in records if r['status']=='REVIEW']
    qc=dict(total_frames=expected,mesh_frames=expected-len(absent),missing_frames=len(absent),missing_spans=spans(absent),
        hands=dense_count,identity_reestimated_hands=correction_count,review_frames=len(review),review_spans=spans(review),reason_counts=dict(counts),
        keypoint_disagreement_px=dict(median=float(np.median(disagreements)),p95=float(np.percentile(disagreements,95)),max=float(max(disagreements))),
        temporal_jump_candidates=jumps,geometry_finite=True,all_dense_predictions_checked=True,video_frames_verified=True,
        status='REVIEW_NO_3D_GROUND_TRUTH',reference='MediaPipe estimates, not independent ground truth',
        limitations=['monocular metric depth unverified','object occlusion not reconstructed','finger visibility not measured','hand identity not independently verified'])
    (a.out/'quality.json').write_text(json.dumps(qc,indent=2,allow_nan=False))
    first.update(processed_frames=expected,start_frame=0,end_frame=expected,preflight_subset=False,mesh_frames=qc['mesh_frames'],
        hand_instances=dense_count,review_frames=len(review),shards=[str(x) for x in parts])
    if a.corrections:first.update(identity_corrections=str(a.corrections),corrected_detector_sha256=cm['detector_sha256'])
    (a.out/'manifest.json').write_text(json.dumps(first,indent=2));writer.add_text('status','Full video and dense predictions validated. No 3D ground truth available.',expected);writer.close()
    print(json.dumps(qc,ensure_ascii=False))


if __name__=='__main__':main()
