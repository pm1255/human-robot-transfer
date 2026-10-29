"""Image consistency diagnostics are not 3D ground-truth accuracy."""
import numpy as np
from .render import project


def assess(vertices, cam_t, joints, landmarks, focal, width, height):
    reasons=[]
    try:
        uv, xyz=project(vertices,cam_t,focal,width,height)
    except ValueError as exc:
        return dict(status='INVALID',reason_codes=['invalid_geometry'],detail=str(exc))
    if np.asarray(vertices).shape!=(778,3):
        return dict(status='INVALID',reason_codes=['invalid_topology'])
    in_frame=np.mean((uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height))
    if in_frame<0.5: reasons.append('mostly_out_of_frame')
    error=None;normalized=None
    if joints is not None and landmarks is not None:
        reference=np.asarray([[p['x']*width,p['y']*height] for p in landmarks])
        pred,_=project(joints,cam_t,focal,width,height)
        error=float(np.median(np.linalg.norm(pred-reference,axis=1)))
        normalized=error/max(float(np.linalg.norm(np.ptp(reference,axis=0))),1.)
        if normalized>0.20: reasons.append('landmark_disagreement')
    return dict(status='REVIEW' if reasons else 'CONSISTENT_2D',reason_codes=reasons,
        median_keypoint_disagreement_px=error,normalized_keypoint_disagreement=normalized,
        vertices_in_frame_fraction=float(in_frame),min_depth=float(xyz[:,2].min()),
        metric_depth_verified=False,reference='MediaPipe prediction, not human ground truth')
