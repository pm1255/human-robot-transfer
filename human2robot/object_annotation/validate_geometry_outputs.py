"""Audit complete depth/pose files without calling consistency checks accuracy."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from .common import Progress,atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    depth_meta=json.loads((root/'depth_manifest.json').read_text())
    pose=json.loads((root/'pose6d.json').read_text());rows=pose['frames'];total=depth_meta['frames']
    if len(rows)!=total or pose['meta']['input_sha256']!=depth_meta['source_sha256']:
        raise ValueError('Pose/depth inputs or coverage differ')
    progress=Progress(root,'geometry_validation',total);done=0;focal=[];medians=[];valid_ratios=[]
    try:
        for i,row in enumerate(rows):
            if row['frame_index']!=i or abs(row['timestamp_s']-i/depth_meta['fps'])>1e-6:
                raise ValueError(f'Incorrect frame/time index {i}')
            with np.load(root/'depth'/f'{i:06d}.npz') as data:
                d,v,K=data['depth'],data['valid'],data['K']
                if d.shape!=(depth_meta['height'],depth_meta['width']) or v.shape!=d.shape or K.shape!=(3,3):
                    raise ValueError(f'Invalid depth shape {i}')
                if int(data['frame_index'])!=i or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:
                    raise ValueError(f'Invalid depth index/intrinsics {i}')
                if not np.all(np.isfinite(d[v])&(d[v]>0)) or v.sum()==0:raise ValueError(f'Invalid depth values {i}')
                focal.append(float(K[0,0]));medians.append(float(np.median(d[v])));valid_ratios.append(float(v.mean()))
            T=row['T_camera_object']
            if T is not None:
                T=np.asarray(T)
                if row['status']!='proxy_review_required' or not row['quality']['accepted']:
                    raise ValueError('Rejected pose published as accepted')
                if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
                    raise ValueError(f'Invalid transform {i}')
                if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-3):
                    raise ValueError(f'Invalid rotation {i}')
            elif row['status']=='proxy_review_required':raise ValueError('Missing accepted pose')
            done=i+1
            if done%120==0 or done==total:progress.update(done)
        counts=collections.Counter(r['status'] for r in rows)
        rejected=collections.Counter(flag for r in rows if r.get('quality') and not r['quality']['accepted'] for flag in r['quality']['flags'])
        spans=[]
        for i,r in enumerate(rows):
            state=r['status']
            if spans and spans[-1]['status']==state:spans[-1]['end_frame']=i
            else:spans.append(dict(start_frame=i,end_frame=i,status=state))
        focal=np.array(focal);medians=np.array(medians)
        accepted=[r['quality'] for r in rows if r['status']=='proxy_review_required']
        result=dict(frames=total,status_counts=dict(counts),rejection_flags=dict(rejected),
            accepted_proxy_count=len(accepted),
            accepted_mask_iou_median=float(np.median([q['mask_iou'] for q in accepted])) if accepted else None,
            depth_valid_fraction_min=float(min(valid_ratios)),
            estimated_focal_pixels_range=[float(focal.min()),float(focal.max())],
            estimated_full_frame_median_depth_range=[float(medians.min()),float(medians.max())],
            adjacent_focal_changes_over_10pct=np.flatnonzero(np.abs(np.diff(focal))/focal[:-1]>.1).tolist(),
            adjacent_median_depth_changes_over_10pct=np.flatnonzero(np.abs(np.diff(medians))/medians[:-1]>.1).tolist(),
            change_indices_mean='index of earlier frame; heuristic flags, not confirmed anomalies',
            source_sha256=depth_meta['source_sha256'],all_files_and_accepted_transforms_verified=True,
            model_accuracy_verified=False,metric_scale_verified=False,usable_as_training_ground_truth=False,
            deformation_preserved=True,temporal_cloth_point_correspondence=False,status_spans=spans)
        atomic_json(root/'geometry_validation.json',result)
        progress.update(done,'completed')
    except Exception as exc:
        progress.update(done,'failed',error=str(exc));raise
    finally:progress.close()


if __name__=='__main__':main()
