"""Geometry utilities for an explicitly non-rigid garment proxy experiment."""
import numpy as np


def backproject(depth, K):
    v,u = np.indices(depth.shape, dtype=np.float32)
    return np.stack(((u-K[0,2])*depth/K[0,0],
                     (v-K[1,2])*depth/K[1,1],depth),axis=-1)


def reference_surface(depth, K, mask, stride=4, max_relative_edge=.04):
    """Visible surface only; do not invent the hidden side or fill mask gaps."""
    points = backproject(depth,K)[::stride,::stride]
    valid = (mask & np.isfinite(depth) & (depth>0))[::stride,::stride]
    h,w=valid.shape
    indices=np.arange(h*w).reshape(h,w)
    a,b,c,d=(indices[:-1,:-1],indices[:-1,1:],indices[1:,:-1],indices[1:,1:])
    faces=np.concatenate((np.stack((a,c,b),axis=-1).reshape(-1,3),
                          np.stack((b,c,d),axis=-1).reshape(-1,3)))
    vertices=points.reshape(-1,3)
    faces=faces[valid.ravel()[faces].all(axis=1)]
    triangles=vertices[faces]
    if len(faces):
        edge=np.max(np.stack([np.linalg.norm(triangles[:,i]-triangles[:,(i+1)%3],axis=1)
                              for i in range(3)]),axis=0)
        faces=faces[edge < max_relative_edge*np.median(triangles[:,:,2],axis=1)]
    if len(faces)<50:
        raise ValueError('Too little connected visible garment surface for a reference mesh')
    used=np.unique(faces);remap=np.full(h*w,-1,dtype=int);remap[used]=np.arange(len(used))
    centre=np.median(vertices[used],axis=0)
    return vertices[used]-centre,remap[faces],centre,used


def pose_quality(pose, rendered_depth, observed_depth, mask, previous_pose=None,
                 min_iou=.45, max_relative_depth_error=.08):
    valid_pose=(np.shape(pose)==(4,4) and np.isfinite(pose).all()
                and np.allclose(pose[3],[0,0,0,1],atol=1e-4)
                and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-3)
                and np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-3))
    if not valid_pose:
        return dict(accepted=False,flags=['invalid_transform'])
    predicted=np.isfinite(rendered_depth)&(rendered_depth>0)
    union=predicted|mask;intersection=predicted&mask
    iou=float(intersection.sum()/max(1,union.sum()))
    overlap=intersection&np.isfinite(observed_depth)&(observed_depth>0)
    relative_error=None
    flags=['nonrigid_target','unverified_monocular_scale','partial_reference_surface']
    rejected=[]
    if iou<min_iou:rejected.append('silhouette_mismatch_deformation_or_tracking_error')
    if overlap.sum()<100:
        rejected.append('insufficient_depth_overlap')
    else:
        relative_error=float(np.median(np.abs(rendered_depth[overlap]-observed_depth[overlap]))/
                             np.median(observed_depth[overlap]))
        if relative_error>max_relative_depth_error:rejected.append('depth_mismatch')
    rotation_jump=None;translation_jump=None
    if previous_pose is not None:
        rotation_jump=float(np.degrees(np.arccos(np.clip((np.trace(previous_pose[:3,:3].T@pose[:3,:3])-1)/2,-1,1))))
        translation_jump=float(np.linalg.norm(pose[:3,3]-previous_pose[:3,3])/max(abs(previous_pose[2,3]),1e-6))
        if rotation_jump>30 or translation_jump>.1:rejected.append('temporal_jump')
    return dict(accepted=not rejected,mask_iou=iou,relative_depth_error=relative_error,
                rotation_jump_deg=rotation_jump,translation_jump_relative_depth=translation_jump,
                flags=flags+rejected,thresholds=dict(min_mask_iou=min_iou,
                    max_relative_depth_error=max_relative_depth_error),
                thresholds_validated_against_ground_truth=False)
