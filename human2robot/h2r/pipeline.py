from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import pinch_frame, transform, invert, apply, transfer_object_goal, check_transform
from .repair import repair_poses
from .schema import split_for_group


def process(episode,robot,*,T_robot_world=None,illustrative_alignment=False,obstacles=(),seed=None):
    episode.validate();n=len(episode.timestamp)
    poses=np.full((n,4,4),np.nan);widths=np.full(n,np.nan);observed=episode.observed.copy()
    issues=[]
    for i in range(n):
        if not observed[i]:continue
        try:
            local,widths[i]=pinch_frame(episode.joints_camera[i])
            poses[i]=episode.camera_world[i]@local
        except ValueError:
            observed[i]=False;issues.append({"code":"DEGENERATE_HAND_FRAME","frame":i})
    repair=repair_poses(episode.timestamp,poses,widths,observed,episode.confidence,episode.track_id,episode.phase)
    valid=repair["valid"].copy();targets=repair["poses"].copy()
    alignment=np.eye(4) if T_robot_world is None else np.asarray(T_robot_world,float)
    check_transform(alignment)
    if illustrative_alignment and valid.any():
        # Explicit illustration only. Its alignment must never license robot actions.
        first=int(np.flatnonzero(valid)[0]); q_anchor=np.array([0,-0.7,1.1,0,-0.4,0]) if robot.n==6 else np.zeros(robot.n)
        alignment=robot.fk(q_anchor)@invert(targets[first])
    for i in np.flatnonzero(valid):targets[i]=alignment@targets[i]
    q=np.full((n,robot.n),np.nan);actual=np.full_like(targets,np.nan)
    links=np.full((n,len(robot.chain)+1,3),np.nan);ik_valid=np.zeros(n,bool)
    poserr=np.full(n,np.nan);roterr=np.full(n,np.nan);collision=np.full(n,np.nan)
    previous=None;last_time=None;last_track=None;last_phase=None
    seed=np.array([0,-0.7,1.1,0,-0.4,0]) if seed is None and robot.n==6 else (np.zeros(robot.n) if seed is None else np.asarray(seed,float))
    for i in range(n):
        if not valid[i]:previous=None;continue
        if last_track is not None and episode.track_id[i]!=last_track:previous=None
        dt=None if previous is None else float(episode.timestamp[i]-last_time)
        result=robot.ik(targets[i],seed=seed if previous is None else previous,
                        previous=previous,dt=dt,obstacles=obstacles,ground_z=-0.03)
        q[i]=result["q"];actual[i]=result["actual"];links[i]=result["link_points"]
        poserr[i]=result["position_error_m"];roterr[i]=result["rotation_error_rad"];collision[i]=result["collision_proxy_m"]
        ik_valid[i]=result["valid"]
        if ik_valid[i]:previous=q[i];last_time=episode.timestamp[i];last_track=episode.track_id[i]
        else:previous=None
    # Width is an estimated pinch aperture, not a measured gripper command/contact.
    width_valid=np.isfinite(repair["widths"]) & (repair["widths"]>=0.005) & (repair["widths"]<=0.085)
    geometry_valid=valid & ik_valid & width_valid
    m=episode.metadata
    reasons=[]
    if m["scale_status"] not in {"metric_calibrated","synthetic_metric"}:reasons.append("SCALE_NOT_VERIFIED")
    if m["camera_status"] not in {"calibrated","synthetic_truth"}:reasons.append("CAMERA_NOT_VERIFIED")
    if illustrative_alignment:reasons.append("ILLUSTRATIVE_ROBOT_ALIGNMENT")
    if m.get("grasp_status") not in {"verified_geometry","synthetic_fixture"}:reasons.append("GRASP_CANDIDATE_ONLY")
    if m.get("mirror_status","unknown")=="unknown":reasons.append("MIRROR_CONVENTION_UNKNOWN")
    if m.get("license_status") not in {"authorized_internal","self_generated","verified"}:reasons.append("LICENSE_UNVERIFIED")
    if m.get("source_kind")=="synthetic":reasons.append("SYNTHETIC_TEST_FIXTURE")
    robot_training_valid=geometry_valid.copy() if not reasons else np.zeros(n,bool)
    # Human reference supervision can still be used with masks and its own semantics.
    human_valid=valid & (episode.confidence>=0.2)
    if repair["interpolated_mask"].any():human_valid[repair["interpolated_mask"]]=True
    result={"episode_id":episode.episode_id,"metadata":m,"timestamp":episode.timestamp,
            "raw_pose_world":poses,"repaired_pose_world":repair["poses"],"target_pose_robot":targets,
            "robot_actual_pose":actual,"robot_joint_reference":q,"robot_link_points":links,
            "aperture_m":repair["widths"],"observed":observed,"human_valid":human_valid,
            "geometry_valid":geometry_valid,"robot_training_valid":robot_training_valid,
            "repair_mask":repair["repair_mask"],"interpolated_mask":repair["interpolated_mask"],
            "phase":episode.phase,"track_id":episode.track_id,
            "ik_position_error_m":poserr,"ik_rotation_error_rad":roterr,
            "collision_proxy_m":collision,"training_exclusion_reasons":reasons,
            "robot_joint_names":robot.names,"robot_model":robot.name,
            "split":split_for_group(m["split_group"]),"issues":issues+repair["issues"],
            "scope":{"repair":"offline robust TCP translation + short-gap SO(3) interpolation",
                     "collision":"link capsule versus configured sphere/plane; no complete self/mesh collision",
                     "dynamics":"not_verified","grasp_force":"not_estimated","action_type":"robot_joint_reference",
                     "observation_state":"not_robot_measured_state"}}
    result["summary"]={"frames":n,"observed_frames":int(observed.sum()),"repaired_frames":int(repair["repair_mask"].sum()),
                       "geometry_valid_frames":int(geometry_valid.sum()),"robot_training_frames":int(robot_training_valid.sum()),
                       "ik_median_mm":float(np.nanmedian(poserr)*1000) if np.isfinite(poserr).any() else None,
                       "source_kind":m["source_kind"]}
    return result


def reference_windows(result,horizon=4,allow_synthetic=False):
    """Future joint *reference*, explicitly not true robot commands or state.

    Reject every window crossing invalid frames, identity/phase boundaries or tail.
    """
    if horizon<1:raise ValueError("horizon must be >=1")
    mask=np.asarray(result["robot_training_valid"],bool)
    if allow_synthetic and result["metadata"]["source_kind"]=="synthetic":mask=np.asarray(result["geometry_valid"],bool)
    q=np.asarray(result["robot_joint_reference"]);width=np.asarray(result["aperture_m"])
    out=[]
    for i in range(len(mask)-horizon):
        sl=slice(i,i+horizon+1)
        if not mask[sl].all():continue
        if len(set(result["track_id"][sl]))!=1 or len(set(result["phase"][sl]))!=1:continue
        out.append({"frame_index":i,"timestamp":float(result["timestamp"][i]),
                    "reference_state":np.r_[q[i],width[i]],
                    "future_reference":np.column_stack([q[i+1:i+horizon+1],width[i+1:i+horizon+1]]),
                    "target_semantics":"future_robot_joint_reference_not_executed_command"})
    return out
