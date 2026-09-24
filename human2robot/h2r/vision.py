"""CPU RGB-only bootstrap. No device-recorded human pose is consumed.

MediaPipe local 3D + PnP with declared camera intrinsics is an estimated-scale
camera-frame reconstruction. Identity camera poses are ASSUMPTIONS, not SLAM.
Use imported calibrated camera trajectories for genuine world reconstruction.
"""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import numpy as np
from .schema import Episode, file_hash, write_json


def extract_video(video,model_path,out_dir,*,sample_fps=10,max_seconds=8,intrinsics=None,
                  source_uri=None,source_group=None,camera_trajectory=None):
    import cv2
    import mediapipe as mp
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    decode_proxy=False
    if not cap.isOpened():
        cap.release()
        # Some cluster OpenCV builds omit H264. Use an explicit, bounded proxy,
        # recording the resampling instead of silently fabricating source PTS.
        proxy=out/'decoded_proxy.avi'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-t',str(max_seconds),
            '-vf',f'fps={sample_fps}','-an','-c:v','mjpeg','-q:v','3',str(proxy)],check=True)
        cap=cv2.VideoCapture(str(proxy));decode_proxy=True
        if not cap.isOpened():raise ValueError(f"Cannot decode {video} or ffmpeg proxy")
    fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0 or sample_fps<=0:raise ValueError("Invalid FPS")
    K=np.array([[max(w,h),0,w/2],[0,max(w,h),h/2],[0,0,1]],float) if intrinsics is None else np.asarray(intrinsics,float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Camera intrinsics must be a finite 3x3 matrix with positive focal lengths')
    options=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=2,
        min_hand_detection_confidence=0.35,min_hand_presence_confidence=0.35,min_tracking_confidence=0.35)
    detector=mp.tasks.vision.HandLandmarker.create_from_options(options)
    records=[];rgb_frames=[];poses=[];xy=[];weights=[];observed=[];observed2d=[];times=[];tracks=[]
    previous_wrist=None;track=0;index=0;next_time=0;last_ms=-1
    timestamp_source="ffmpeg_resampled_proxy_grid" if decode_proxy else "video_pts"
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            pts=index/fps if decode_proxy else cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if (index>0 and pts<=0) or not np.isfinite(pts):pts=index/fps;timestamp_source="nominal_fps_fallback"
            index+=1
            if pts>max_seconds:break
            if pts+1e-6<next_time:continue
            next_time=pts+1/sample_fps
            ms=max(last_ms+1,round(pts*1000));last_ms=ms
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ms)
            joints=np.full((21,3),np.nan);landmarks=np.full((21,2),np.nan);weight=0.;valid=False;valid2d=False
            label="unknown";err=None
            if result.hand_landmarks:
                wrists=np.asarray([[r[0].x,r[0].y] for r in result.hand_landmarks])
                k=int(np.argmin(np.linalg.norm(wrists-previous_wrist,axis=1))) if previous_wrist is not None else 0
                if previous_wrist is not None and np.linalg.norm(wrists[k]-previous_wrist)>.22:track+=1
                previous_wrist=wrists[k]
                landmarks=np.array([[p.x,p.y] for p in result.hand_landmarks[k]],float)
                valid2d=True
                local=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[k]],float)
                label=result.handedness[k][0].category_name.lower()
                image_points=landmarks*np.array([w,h])
                try:
                    success,rvec,tvec=cv2.solvePnP(local,image_points,K,None,flags=cv2.SOLVEPNP_SQPNP)
                    if success:
                        R,_=cv2.Rodrigues(rvec);candidate=local@R.T+tvec.reshape(3)
                        projection,_=cv2.projectPoints(local,rvec,tvec,K,None)
                        err=float(np.linalg.norm(projection.reshape(-1,2)-image_points,axis=1).mean())
                        valid=bool((candidate[:,2]>0.03).all() and err<35 and np.isfinite(candidate).all())
                        if valid:joints=candidate;weight=float(np.exp(-err/15))
                except cv2.error:pass
            else:
                # A reappearance may be a different hand; do not interpolate across it.
                if previous_wrist is not None:track+=1
                previous_wrist=None
            times.append(pts);poses.append(joints);xy.append(landmarks);weights.append(weight);observed.append(valid);observed2d.append(valid2d);tracks.append(track)
            rgb_frames.append(cv2.resize(rgb,(96,96),interpolation=cv2.INTER_AREA))
            records.append({"source_frame_index":index-1,"timestamp":pts,"handedness_model":label,
                            "handedness_verified":False,"reprojection_error_px":err,"landmarks_2d":landmarks,
                            "track_id":track,"observed_2d":valid2d,"observed_3d":valid})
    finally:
        cap.release();detector.close()
    if len(times)<2:raise ValueError("Video has fewer than two usable timestamps")
    timestamps=np.asarray(times);n=len(times)
    cameras=np.repeat(np.eye(4)[None],n,axis=0);camera_status="assumed_stationary"
    if camera_trajectory is not None:
        data=json.loads(Path(camera_trajectory).read_text())
        if not np.allclose(data["timestamp"],timestamps,atol=1e-3):raise ValueError("Camera timestamps must match selected video frames")
        cameras=np.asarray(data["T_world_camera"]);camera_status=data["status"]
    metadata={"units":"m","scale_status":"metric_estimated","scale_source":"MediaPipe learned hand dimensions + PnP",
              "source_kind":"human_video","source_uri":source_uri or str(video),"source_sha256":file_hash(video),
              "split_group":source_group or file_hash(video),"camera_status":camera_status,
              "intrinsics_status":"supplied" if intrinsics is not None else "assumed_focal_length",
              "intrinsics":K.tolist(),"image_size":[w,h],"mirror_status":"unknown",
              "license_status":"authorized_internal","grasp_status":"pinch_heuristic_unverified",
              "estimator":"MediaPipe HandLandmarker + OpenCV SQPnP","model_sha256":file_hash(model_path),
              "timestamp_source":timestamp_source,"task":"unannotated human manipulation",
              "decoder_proxy":decode_proxy,"mediapipe_version":mp.__version__,"opencv_version":cv2.__version__,
              "task_source":"not_provided","confidence_semantics":"exp(-PnP reprojection_px/15), not calibrated probability"}
    ep=Episode(Path(video).stem,timestamps,np.asarray(poses),np.asarray(observed),np.asarray(weights),cameras,
               np.asarray(tracks),np.full(n,"unknown"),metadata)
    ep.save(out/'annotation.json');write_json(out/'keypoints.json',records)
    np.savez_compressed(out/'rgb_supervision.npz',images=np.asarray(rgb_frames,dtype=np.uint8),
                        landmarks_2d=np.asarray(xy,dtype=np.float32),observed_2d=np.asarray(observed2d),
                        timestamp=timestamps,track_id=np.asarray(tracks))
    return ep,records
