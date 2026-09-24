"""Offline, noncausal reference-label repair. Never silently use as policy input."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .geometry import interp_rotations


def runs(mask):
    start = None
    for i, flag in enumerate(np.r_[mask, False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            yield start, i
            start = None


def repair_poses(t, poses, widths, observed, confidence, track_id, phase,
                 max_gap_s=0.25, sigma_m=0.006, smooth_weight=0.08):
    t=np.asarray(t); n=len(t)
    raw=np.asarray(poses).copy(); out=raw.copy(); widths=np.asarray(widths).copy()
    valid=np.asarray(observed,bool).copy()
    valid &= np.isfinite(raw).all(axis=(1,2)) & np.isfinite(widths)
    interpolated=np.zeros(n,bool); repaired=np.zeros(n,bool)
    issues=[]
    # Identity/phase are barrier labels: no interpolation or smoothing across them.
    boundaries=np.r_[0, np.flatnonzero((track_id[1:] != track_id[:-1]) | (phase[1:] != phase[:-1]))+1,n]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        for a,b in list(runs(~valid[lo:hi])):
            a+=lo;b+=lo
            if a>lo and b<hi and t[b]-t[a-1]<=max_gap_s:
                for dim in range(3):out[a:b,dim,3]=np.interp(t[a:b],t[[a-1,b]],raw[[a-1,b],dim,3])
                out[a:b,:3,:3]=interp_rotations(t[[a-1,b]],raw[[a-1,b],:3,:3],t[a:b])
                out[a:b,3,:]=[0,0,0,1]
                widths[a:b]=np.interp(t[a:b],t[[a-1,b]],widths[[a-1,b]])
                valid[a:b]=True;interpolated[a:b]=True
            else:
                issues.append({"code":"GAP_UNREPAIRED","start":int(a),"end_exclusive":int(b)})
        for a,b in runs(valid[lo:hi]):
            a+=lo;b+=lo;m=b-a
            if m<4:continue
            p=out[a:b,:3,3].copy()
            dt=np.diff(t[a:b]); weights=np.clip(confidence[a:b],0.05,1).copy()
            weights[interpolated[a:b]]=0.1
            # A localized jump with consistent neighbors is suspicious; fast sustained
            # motion remains constrained by the observations, not blindly clipped.
            mid=(p[:-2]+p[2:])/2
            spike=(np.linalg.norm(p[1:-1]-mid,axis=1)>0.06)&(np.linalg.norm(p[2:]-p[:-2],axis=1)<0.035)
            weights[1:-1][spike]=0.0004
            initial=p.copy()
            initial[1:-1][spike]=mid[spike]
            def residual(x):
                x=x.reshape(m,3)
                data=(x-p)/sigma_m*np.sqrt(weights[:,None])
                velocity=np.diff(x,axis=0)/dt[:,None]
                accel=np.diff(velocity,axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
                return np.r_[data.ravel(),(smooth_weight*accel).ravel()]
            sparsity=lil_matrix((m*3+(m-2)*3,m*3),dtype=int)
            for i in range(m):sparsity[i*3:i*3+3,i*3:i*3+3]=1
            for i in range(m-2):sparsity[m*3+i*3:m*3+i*3+3,i*3:(i+3)*3]=1
            result=least_squares(residual,initial.ravel(),loss="soft_l1",f_scale=1.0,jac_sparsity=sparsity,max_nfev=100)
            out[a:b,:3,3]=result.x.reshape(m,3)
            # Rotation is intentionally not flattened into Euler coordinates. Large
            # orientation discontinuities remain flagged; only short gaps use SLERP.
    repaired=valid & (interpolated | (np.linalg.norm(np.nan_to_num(out[:,:3,3]-raw[:,:3,3]),axis=1)>1e-5))
    return {"poses":out,"widths":widths,"valid":valid,"repair_mask":repaired,
            "interpolated_mask":interpolated,"issues":issues,"noncausal":True}
