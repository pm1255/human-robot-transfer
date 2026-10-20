"""Run real HaMeR, save dense predictions, and render anatomical surfaces.

python -m human2robot.hand_mesh.infer --help
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .video import probe, frames, Writer
from .render import render_hands, composite
from .quality import assess


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def hand_box(hand,width,height):
    points=np.asarray([[p['x']*width,p['y']*height] for p in hand['landmarks']],np.float32)
    if points.shape!=(21,2) or not np.isfinite(points).all():raise ValueError('Invalid hand detector points')
    # Exactly one rescale in ViTDetDataset. Do not pad, clip, then rescale twice.
    low=points.min(0);high=points.max(0)
    if min(high-low)<4:raise ValueError('Degenerate hand bounding box')
    return np.concatenate((low,high))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',type=Path,required=True);ap.add_argument('--hands',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--hamer-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--rescale-factor',type=float,default=2.0)
    ap.add_argument('--max-frames',type=int,default=0,help='Explicit preflight subset; zero processes entire video')
    ap.add_argument('--start-frame',type=int,default=0)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--cpu-bf16',action='store_true',help='AMX acceleration; benchmark against FP32 before use')
    ap.add_argument('--sdpa',action='store_true',help='Equivalent PyTorch attention implementation; preflight required')
    ap.add_argument('--only-identity-changes',action='store_true',help='Sparse correction pass; not a full annotation')
    ap.add_argument('--device',default='cuda');args=ap.parse_args()
    if (args.out/'manifest.json').exists():raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    for n,v in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
        if n not in np.__dict__:setattr(np,n,v)
    sys.path.insert(0,str(args.hamer_root))
    import cv2
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from hamer.models import load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full
    torch.set_num_threads(args.threads)
    meta=probe(args.video);width,height=meta['width'],meta['height']
    expected=min(args.max_frames or meta['frames'],meta['frames']-args.start_frame)
    if args.start_frame<0 or expected<=0:raise ValueError('Invalid frame range')
    hand_data=json.loads(args.hands.read_text());by_frame={x['frame_index']:x for x in hand_data['frames']}
    if len(by_frame)!=meta['frames'] or hand_data['width']!=width or hand_data['height']!=height:
        raise ValueError('Hand detections do not match source video')
    log=SummaryWriter(str(args.out/'events'))
    log.add_text('status','Loading checkpoint; inference and rendering waiting.',0);log.add_scalar('progress/inference',0,0);log.flush()
    model,cfg=load_hamer(str(args.checkpoint));model=model.to(args.device).eval()
    if args.sdpa:
        import types
        from hamer.models.backbones.vit import Attention
        def attention(self,x):
            batch,n,_=x.shape
            q,k,v=self.qkv(x).reshape(batch,n,3,self.num_heads,-1).permute(2,0,3,1,4).unbind(0)
            result=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale)
            return self.proj_drop(self.proj(result.transpose(1,2).reshape(batch,n,-1)))
        for module in model.backbone.modules():
            if isinstance(module,Attention):module.forward=types.MethodType(attention,module)
    if args.cpu_bf16:
        if args.device!='cpu':raise ValueError('--cpu-bf16 requires --device cpu')
        # Keep MANO geometry in FP32. Only neural feature/parameter inference is autocast.
        for module in (model.backbone,model.mano_head):
            original=module.forward
            def accelerated(*a,_forward=original,**kw):
                with torch.autocast('cpu',dtype=torch.bfloat16):return _forward(*a,**kw)
            module.forward=accelerated
    faces=model.mano.faces.astype(np.int32)
    focal=float(cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*max(width,height))
    np.save(args.out/'faces.npy',faces)
    np.savez(args.out/'camera.npz',K=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]]))
    writer=Writer(args.out/'hamer_surface.mp4',width,height,meta['fps'])
    mesh_writer=Writer(args.out/'hamer_mesh_only.mp4',width,height,meta['fps'])
    chunkdir=args.out/'predictions';chunkdir.mkdir(exist_ok=True)
    records=[];pending=[];chunk_index=0;count=0;mesh_frames=0;hand_count=0;reviews=0;start=time.time()
    diagnostic_frames={0,120,600,1200,1800,2400,3000,3600,4200,meta['frames']-1}

    def process(items):
        nonlocal chunk_index,count,mesh_frames,hand_count,reviews
        samples=[];bindings=[];invalid={}
        for idx,img,detected in items:
            for k,hand in enumerate(detected):
                try:
                    side=(hand.get('handedness') or {}).get('category')
                    if side not in ('Left','Right'):raise ValueError('Unknown handedness')
                    box=hand_box(hand,width,height)
                    ds=ViTDetDataset(cfg,img,np.array([box]),np.array([side=='Right']),rescale_factor=args.rescale_factor)
                    samples.append(ds[0]);bindings.append((idx,hand.get('_source_hand_index',k),hand))
                except ValueError as exc:invalid.setdefault(idx,[]).append(str(exc))
        rendered={i:[] for i,_,_ in items};details={i:[] for i,_,_ in items}
        all_v=[];all_j=[];all_cam=[];all_rot=[];all_pose=[];all_shape=[];all_frame=[];all_right=[]
        for offset in range(0,len(samples),args.batch_size):
            b=recursive_to(torch.utils.data.default_collate(samples[offset:offset+args.batch_size]),args.device)
            with torch.inference_mode():out=model(b)
            cam=out['pred_cam'].clone();cam[:,1]*=(2*b['right']-1)
            full=cam_crop_to_full(cam,b['box_center'].float(),b['box_size'].float(),b['img_size'].float(),focal).cpu().numpy()
            vertices=out['pred_vertices'].cpu().numpy();joints=out['pred_keypoints_3d'].cpu().numpy()
            params=out['pred_mano_params']
            for j,(idx,k,hand) in enumerate(bindings[offset:offset+args.batch_size]):
                right=int(b['right'][j].item());vertices[j,:,0]*=2*right-1;joints[j,:,0]*=2*right-1
                quality=assess(vertices[j],full[j],joints[j],hand['landmarks'],focal,width,height)
                item=dict(handedness='Right' if right else 'Left',vertices=vertices[j],cam_t=full[j])
                if quality['status']!='INVALID':rendered[idx].append(item)
                detail=dict(detector_hand_index=k,handedness=item['handedness'],quality=quality,
                    prediction_file=f'predictions/chunk_{chunk_index:05d}.npz',prediction_index=len(all_v))
                detail['identity_method']=hand.get('identity_method','detector')
                detail['detector_handedness']=hand.get('detector_handedness',hand.get('handedness'))
                details[idx].append(detail)
                all_v.append(vertices[j]);all_j.append(joints[j]);all_cam.append(full[j]);all_frame.append(idx);all_right.append(right)
                all_rot.append(params['global_orient'][j].float().cpu().numpy());all_pose.append(params['hand_pose'][j].float().cpu().numpy());all_shape.append(params['betas'][j].float().cpu().numpy())
        if all_v:
            np.savez_compressed(chunkdir/f'chunk_{chunk_index:05d}.npz',vertices=all_v,joints=all_j,cam_t=all_cam,
                global_orient=all_rot,hand_pose=all_pose,betas=all_shape,frame_index=all_frame,is_right=all_right)
        chunk_index+=1
        for idx,img,_ in items:
            rgb,depth,owner=render_hands(rendered[idx],faces,focal,width,height)
            overlay=composite(img,rgb,depth)
            blank=np.full_like(img,235);surface=composite(blank,rgb,depth,1.0)
            frame_review=any(d['quality']['status']!='CONSISTENT_2D' for d in details[idx])
            state='REVIEW' if frame_review else ('MESH' if rendered[idx] else 'NO DETECTION')
            cv2.putText(overlay,f'HaMeR surface | {idx}/{meta["frames"]} | {state}',(8,height-10),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            writer.write(overlay);mesh_writer.write(surface)
            if idx in diagnostic_frames:
                cv2.imwrite(str(args.out/f'frame_{idx:05d}.jpg'),np.concatenate([img,overlay,surface],axis=1))
                log.add_image('visual_check/raw_surface_mesh',np.concatenate([img,overlay,surface],axis=1)[:,:,::-1],idx,dataformats='HWC')
            records.append(dict(frame_index=idx,timestamp_seconds=idx/meta['fps'],hands=details[idx],
                status=state,errors=invalid.get(idx,[]),rendered_pixels=int(np.isfinite(depth).sum())))
            mesh_frames+=int(bool(rendered[idx]));hand_count+=len(details[idx]);reviews+=int(frame_review);count+=1
        for tag,value in [('progress/inference',count/expected),('progress/render',count/expected),
            ('progress/frames',count),('quality/mesh_frames',mesh_frames),('quality/review_frames',reviews),('throughput/fps',count/max(time.time()-start,1))]:log.add_scalar(tag,value,count)
        log.flush();(args.out/'progress.json').write_text(json.dumps(dict(frames=count,total=expected,mesh_frames=mesh_frames,review_frames=reviews)))
        print(json.dumps(dict(frames=count,total=expected,mesh_frames=mesh_frames,review_frames=reviews)),flush=True)

    for idx,img in enumerate(frames(args.video,meta,args.start_frame,args.max_frames),start=args.start_frame):
        detected=by_frame[idx].get('hands',[])
        if args.only_identity_changes:
            detected=[dict(h,_source_hand_index=k) for k,h in enumerate(detected)
                if h['handedness']['category']!=h.get('detector_handedness',h['handedness'])['category']]
        pending.append((idx,img,detected))
        if len(pending)>=32:process(pending);pending=[]
    if pending:process(pending)
    writer.close();mesh_writer.close()
    with (args.out/'frames.jsonl').open('w') as f:
        for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
    manifest=dict(schema='hamer.surface.v1',model='HaMeR',checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        source_video=str(args.video),source_sha256=sha(args.video),detector_json=str(args.hands),detector_sha256=sha(args.hands),
        **meta,processed_frames=count,start_frame=args.start_frame,end_frame=args.start_frame+count,preflight_subset=count!=meta['frames'],mesh_frames=mesh_frames,hand_instances=hand_count,review_frames=reviews,
        vertices_per_hand=778,faces_per_hand=len(faces),renderer='perspective_correct_triangle_z_buffer',
        focal_length_px=focal,intrinsics_source='HaMeR model convention, not calibrated camera',
        coordinate_system='OpenCV x right / y down / z forward; unmirrored vertices + cam_t',
        mano_parameters='canonical right-hand rotations; is_right controls reflection',
        rescale_factor=args.rescale_factor,sparse_identity_corrections=args.only_identity_changes,sdpa=args.sdpa,neural_precision='bf16' if args.cpu_bf16 else 'float32',geometry_precision='float32',training_valid=False,quality_status='REVIEW_NO_3D_GROUND_TRUTH',
        quality_reference='MediaPipe predictions, not independent ground truth',
        occlusion='hand self-occlusion and hand-hand depth only; no scene depth',
        interpolation=False,videos=['hamer_surface.mp4','hamer_mesh_only.mp4'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    log.add_scalar('progress/validation',1,count);log.add_text('status','Completed. 3D accuracy and metric depth remain unverified.',count);log.close()


if __name__=='__main__':main()
