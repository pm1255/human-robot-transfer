"""Experimental FoundationPose rigid proxy, with deforming observations retained.

This is a baseline using a visible depth surface, not a SAM3D full canonical mesh
or an EgoMatrix reproduction. Rejected poses remain candidates and never labels.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from human2robot.hand_mesh.video import probe, frames
from .common import Progress, atomic_json, sha256
from .cloth_geometry import backproject, reference_surface, pose_quality


class Masks:
    def __init__(self,root,shape):
        self.root,self.shape,self.start=Path(root),shape,None

    def get(self,row):
        if row['chunk_start']!=self.start:
            self.start=row['chunk_start']
            with np.load(self.root/'masks'/f'{self.start:06d}.npz') as pack:
                self.data=np.unpackbits(pack['masks'],axis=1,count=np.prod(self.shape)).reshape(-1,*self.shape).astype(bool)
        return self.data[row['frame_index']-self.start]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference-frame',type=int,required=True,
                        help='Visually reviewed reference frame for the fixed partial surface')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--wait-depth-seconds',type=int,default=0,help='Wait per frame while the independent depth job produces it')
    parser.add_argument('--reinitialize-interval',type=int,default=30,help='Retry failed rigid registration at most once per this many frames')
    parser.add_argument('--reference-review',required=True,help='JSON review bound to the saved reference mask and input video')
    args=parser.parse_args();root=Path(args.output)
    annotations=json.loads((root/'objects.json').read_text())
    depth_meta=json.loads((root/'depth_manifest.json').read_text())
    meta=annotations['meta'];rows=annotations['frames'];total=len(rows)
    if args.max_frames:total=min(total,args.max_frames)
    if meta['source_sha256']!=depth_meta['source_sha256']:
        raise ValueError('SAM and MoGe inputs differ')
    if total<=0 or not 0<=args.reference_frame<total:
        raise ValueError('Invalid frame range')
    if args.reinitialize_interval<1:raise ValueError('Invalid reinitialization interval')
    progress=Progress(root,'pose6d',total);done=0
    try:
        progress.update(0,phase='loading_foundationpose')
        import torch
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose
        from learning.training.predict_score import ScorePredictor
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from Utils import nvdiffrast_render, set_seed
        set_seed(0)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        masks=Masks(root,(meta['height'],meta['width']))
        def depth_path(index):
            path=root/'depth'/f'{index:06d}.npz'
            deadline=time.monotonic()+args.wait_depth_seconds
            while not path.exists() and time.monotonic()<deadline:
                state_path=root/'depth_status.json'
                if state_path.exists() and json.loads(state_path.read_text()).get('status')=='failed':
                    raise RuntimeError('Upstream depth task failed')
                progress.update(done,'waiting_depth',waiting_for_frame=index)
                time.sleep(5)
            if not path.exists():raise FileNotFoundError(path)
            return path
        ref=args.reference_frame;ref_row=rows[ref]
        review=json.loads(Path(args.reference_review).read_text())
        mask_digest=__import__('hashlib').sha256(np.packbits(masks.get(ref_row)).tobytes()).hexdigest()
        if not ref_row['mask_area'] or review.get('frame_index')!=ref or review.get('mask_sha256')!=mask_digest or review.get('source_sha256')!=meta['source_sha256'] or not review.get('approved_for_proxy_experiment'):
            raise ValueError('Reference requires an explicit visual review matching this exact saved mask and video')
        with np.load(depth_path(ref)) as data:
            vertices,faces,centre,used=reference_surface(data['depth'],data['K'],masks.get(ref_row))
        reference_rgb=None
        for index,bgr in enumerate(frames(meta['video'],probe(meta['video']),limit=ref+1)):
            if index==ref:reference_rgb=bgr[:,:,::-1].copy()
        if reference_rgb is None:raise ValueError('Reference RGB frame is missing')
        # The geometry helper samples every fourth pixel and returns its used indices.
        colours=reference_rgb[::4,::4].reshape(-1,3)[used]
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=colours,process=False)
        mesh.export(root/'cloth_reference_surface.ply')
        config=dict(schema='object_annotation.cloth_pose.v1',rigid_object=False,
                    pose_type='cloth_global_proxy',estimator='FoundationPose',
                    reference_frame=ref,reference_review=review,reference_geometry='partial_visible_MoGe_surface',
                    reference_appearance='observed_RGB_vertex_colours',
                    reference_mesh_sha256=sha256(root/'cloth_reference_surface.ply'),
                    reference_camera_centre=centre.tolist(),
                    mesh_coordinates='reference_camera_axes_recentered_at_visible_surface_median',
                    matrix='T_camera_object_column_vectors',world_pose_available=False,
                    depth_source=depth_meta['depth_source'],scale_source=depth_meta['scale_source'],
                    input_sha256=meta['source_sha256'],processed_target=total,
                    training_ground_truth=False,full_egomatrix_reproduction=False,
                    inference_precision='FP32; mixed precision disabled',
                    tf32_matmul=False,tf32_cudnn=False,
                    reinitialize_interval_frames=args.reinitialize_interval,
                    deformation_retained_in=['masks/*.npz','depth/*.npz'],
                    limitation='Rigid proxy cannot represent folding, hidden surfaces or cloth point correspondence')
        atomic_json(root/'pose6d_manifest.json',config)
        context=dr.RasterizeCudaContext()
        scorer=ScorePredictor();scorer.amp=False
        refiner=PoseRefinePredictor();refiner.amp=False
        estimator=FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,
                                 mesh=mesh,scorer=scorer,refiner=refiner,
                                 glctx=context,debug=0,debug_dir=str(root/'pose_debug'))
        # NumPy 2 scalars otherwise promote FoundationPose's crop offsets to FP64.
        estimator.diameter=float(estimator.diameter)
        result_rows=[];last_accepted=None;has_initialized=False;accepted_count=0;next_registration=ref
        for i,bgr in enumerate(frames(meta['video'],probe(meta['video']),limit=total)):
            row=rows[i];mask=masks.get(row)
            record=dict(frame_index=i,timestamp_s=i/meta['fps'],rigid_object=False,
                        pose_type='cloth_global_proxy',T_camera_object=None,
                        pose_source='unavailable',occlusion_status='not_independently_measured',
                        deformation_risk='intrinsic_to_folding_cloth')
            with np.load(depth_path(i)) as data:
                depth=data['depth'].copy();K=data['K'].copy()
            valid=mask&np.isfinite(depth)&(depth>0)
            if valid.any():
                observed=backproject(depth,K)[valid]
                record['visible_surface_centroid_camera']=np.median(observed,axis=0).tolist()
                record['visible_surface_extents_camera']=(np.percentile(observed,95,axis=0)-np.percentile(observed,5,axis=0)).tolist()
            if i<ref:
                record['status']='before_reference'
            elif valid.sum()<100 or 'multiple_instances_review' in row['flags']:
                record['status']='insufficient_mask_or_depth';estimator.pose_last=None
            elif estimator.pose_last is None and i<next_registration:
                record['status']='waiting_for_reinitialization'
                record['next_registration_frame']=next_registration
            else:
                depth=np.where(np.isfinite(depth)&(depth>0),depth,0).astype(np.float32)
                rgb=bgr[:,:,::-1].copy()
                source='tracking'
                if estimator.pose_last is None:
                    source='reinitialization' if has_initialized else 'initialization'
                    candidate=estimator.register(K=K,rgb=rgb,depth=depth,ob_mask=mask,iteration=5)
                    has_initialized=True
                else:
                    candidate=estimator.track_one(rgb=rgb,depth=depth,K=K,iteration=2)
                _,rendered,_=nvdiffrast_render(K=K,H=meta['height'],W=meta['width'],
                    ob_in_cams=torch.as_tensor(candidate,dtype=torch.float32,device='cuda')[None],
                    context='cuda',glctx=context,mesh=mesh)
                quality=pose_quality(candidate,rendered[0].detach().cpu().numpy(),depth,mask,last_accepted)
                record.update(candidate_T_camera_object=candidate.tolist(),candidate_source=source,quality=quality)
                if quality['accepted']:
                    record.update(T_camera_object=candidate.tolist(),pose_source=source,status='proxy_review_required')
                    last_accepted=candidate;accepted_count+=1
                else:
                    record['status']='rejected_proxy';estimator.pose_last=None;last_accepted=None
                    next_registration=i+args.reinitialize_interval
            result_rows.append(record);done=i+1
            if done%30==0 or done==total:
                atomic_json(root/'pose6d_partial.json',dict(meta=config,frames=result_rows))
                progress.update(done,accepted_proxy_frames=accepted_count,phase='inference_and_consistency_checks')
        atomic_json(root/'pose6d.json',dict(meta=config,frames=result_rows))
        atomic_json(root/'pose6d_quality.json',dict(frames=done,accepted_proxy_frames=accepted_count,
                    rejected_or_unavailable_frames=done-accepted_count,independently_verified=False,
                    usable_as_ground_truth=False))
        progress.update(done,'completed',accepted_proxy_frames=accepted_count)
    except Exception as exc:
        progress.update(done,'failed',error=str(exc));raise
    finally:
        progress.close()


if __name__=='__main__':main()
