import copy
import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation
from h2r.geometry import transform,invert,apply,pinch_frame,transfer_object_goal,check_transform,interp_rotations
from h2r.kinematics import SerialRobot
from h2r.fixtures import make_case,hand_landmarks
from h2r.repair import repair_poses
from h2r.pipeline import process,reference_windows
from h2r.schema import Episode,split_for_group
from h2r.export import export_references

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def robot():return SerialRobot(ROOT/'configs/demo_arm.urdf')

def test_coordinate_roundtrip():
    T=transform(Rotation.from_euler('xyz',[.2,-.5,1.2]).as_matrix(),[.4,.2,1])
    p=np.random.default_rng(1).normal(size=(21,3))
    np.testing.assert_allclose(apply(invert(T),apply(T,p)),p,atol=1e-12)

def test_reflection_is_not_rotation():
    T=np.eye(4);T[0,0]=-1
    with pytest.raises(ValueError):check_transform(T)

def test_object_transfer_tracks_rotated_object():
    source=transform(translation=[1,.5,.8]);grasp=transform(translation=[.95,.5,.82])
    target=transform(Rotation.from_euler('z',90,degrees=True).as_matrix(),[.6,.2,.75])
    out=transfer_object_goal(source,grasp,target)
    np.testing.assert_allclose(out[:3,3],[.6,.15,.77],atol=1e-12)

def test_tool_offset_rotates_with_grasp():
    G=transform(Rotation.from_euler('y',90,degrees=True).as_matrix())
    out=transfer_object_goal(np.eye(4),G,np.eye(4),transform(translation=[0,0,.1]))
    np.testing.assert_allclose(out[:3,3],[.1,0,0],atol=1e-12)

def test_pinch_frame_recovers_constructed_pose():
    T=transform(Rotation.from_euler('xyz',[.1,.4,.7]).as_matrix(),[.2,.3,.4])
    predicted,width=pinch_frame(hand_landmarks(T,.04))
    np.testing.assert_allclose(predicted,T,atol=1e-12);assert width==pytest.approx(.04)

def test_degenerate_hand_rejected():
    with pytest.raises(ValueError):pinch_frame(np.zeros((21,3)))

def test_quaternion_sign_does_not_create_rotation():
    r=Rotation.from_quat([[0,0,0,1],[0,0,0,-1]]).as_matrix()
    np.testing.assert_allclose(interp_rotations([0,1],r,[.5])[0],np.eye(3))

def test_timestamp_rejection(robot):
    ep,_=make_case('clean_pinch',robot);ep.timestamp[2]=ep.timestamp[1]
    with pytest.raises(ValueError):ep.validate()

def test_units_not_guessed(robot):
    ep,_=make_case('clean_pinch',robot);ep.metadata['units']='mm'
    with pytest.raises(ValueError):ep.validate()

def test_nonfinite_observation_rejected(robot):
    ep,_=make_case('clean_pinch',robot);ep.joints_camera[0,0,0]=np.nan
    with pytest.raises(ValueError):ep.validate()

def test_episode_roundtrip(robot,tmp_path):
    ep,_=make_case('short_gap',robot);ep.save(tmp_path/'episode.json');loaded=Episode.load(tmp_path/'episode.json')
    np.testing.assert_allclose(loaded.joints_camera,ep.joints_camera,equal_nan=True)

def test_camera_compensation(robot):
    ep,truth=make_case('camera_motion',robot)
    for i in range(len(ep.timestamp)):
        p,_=pinch_frame(ep.joints_camera[i]);np.testing.assert_allclose(ep.camera_world[i]@p,truth[i],atol=1e-10)

def test_fk_ik_known_pose(robot):
    q=np.array([.2,-.6,1.2,.1,-.3,.2]);target=robot.fk(q)
    r=robot.ik(target,seed=q+.04)
    assert r['valid'];assert r['position_error_m']<1e-4

def test_ik_unreachable_is_not_valid(robot):
    target=transform(translation=[2,0,.3]);r=robot.ik(target)
    assert not r['valid']

def test_ik_velocity_bound(robot):
    q=np.array([0,-.7,1.1,0,-.4,0]);target=robot.fk(q+np.array([1,0,0,0,0,0]))
    r=robot.ik(target,seed=q,previous=q,dt=.01)
    assert np.all(np.abs(r['q']-q)<=robot.velocity*.01+1e-8)
    assert not r['valid']

def test_obstacle_is_checked(robot):
    q=np.array([0,-.7,1.1,0,-.4,0]);target=robot.fk(q)
    obstacle={'center':target[:3,3].tolist(),'radius':.05}
    r=robot.ik(target,seed=q,obstacles=[obstacle]);assert not r['valid']

def repair_fixture(case,robot):
    ep,truth=make_case(case,robot)
    poses=np.full_like(truth,np.nan);widths=np.full(len(poses),np.nan)
    for i in np.flatnonzero(ep.observed):poses[i],widths[i]=pinch_frame(ep.joints_camera[i])
    r=repair_poses(ep.timestamp,poses,widths,ep.observed,ep.confidence,ep.track_id,ep.phase)
    return ep,truth,poses,r

def test_short_gap_is_flagged(robot):
    ep,truth,raw,r=repair_fixture('short_gap',robot)
    assert r['valid'][30] and r['interpolated_mask'][30] and r['repair_mask'][30]

def test_long_gap_stays_invalid(robot):
    _,_,_,r=repair_fixture('long_gap',robot)
    assert not r['valid'][30:40].any()

def test_no_repair_across_event_boundary(robot):
    _,_,_,r=repair_fixture('event_boundary',robot);assert not r['valid'][30]

def test_spike_repair_improves_saved_truth(robot):
    _,truth,raw,r=repair_fixture('spike',robot)
    before=np.linalg.norm(raw[30,:3,3]-truth[30,:3,3]);after=np.linalg.norm(r['poses'][30,:3,3]-truth[30,:3,3])
    assert after<before*.5

def test_window_tail_mask_and_barriers(robot,tmp_path):
    ep,_=make_case('clean_pinch',robot,n=12);r=process(ep,robot)
    assert not r['robot_training_valid'].any() # synthetic never silently becomes robot data
    windows=reference_windows(r,4,allow_synthetic=True)
    assert windows and max(x['frame_index'] for x in windows)<=7
    r['track_id'][5:]=1
    assert all(not (x['frame_index']<=4<x['frame_index']+4) for x in reference_windows(r,4,True))
    manifest=export_references(r,tmp_path,4,False);assert manifest['windows']==0

def test_estimated_scale_rejects_robot_training(robot):
    ep,_=make_case('clean_pinch',robot,n=8);ep.metadata.update(source_kind='human_video',scale_status='metric_estimated')
    r=process(ep,robot);assert 'SCALE_NOT_VERIFIED' in r['training_exclusion_reasons'];assert not r['robot_training_valid'].any()

def test_split_stable_for_derived_clips():
    assert split_for_group('same-source-video')==split_for_group('same-source-video')

def test_nan_windows_never_export(robot):
    ep,_=make_case('long_gap',robot,n=45);r=process(ep,robot)
    for win in reference_windows(r,4,True):assert np.isfinite(win['future_reference']).all()

def test_robot_alignment_rejects_reflection(robot):
    ep,_=make_case('clean_pinch',robot,n=8);T=np.eye(4);T[1,1]=-1
    with pytest.raises(ValueError):process(ep,robot,T_robot_world=T)

def test_bad_robot_axis_is_rejected(tmp_path):
    source=(ROOT/'configs/demo_arm.urdf').read_text()
    source=source.replace('axis xyz="0 0 1"','axis xyz="0 0 0"')
    path=tmp_path/'bad.urdf';path.write_text(source)
    with pytest.raises(ValueError):SerialRobot(path)

def test_proxy_timestamp_correction_is_explicit_and_idempotent(robot,tmp_path):
    from scripts.correct_proxy_timestamps import correct
    import json
    folder=tmp_path/'clip';folder.mkdir();ep,_=make_case('clean_pinch',robot,n=8)
    ep.timestamp+=.1;ep.metadata['timestamp_source']='ffmpeg_resampled_proxy_grid';ep.save(folder/'annotation.json')
    (folder/'keypoints.json').write_text(json.dumps([{'timestamp':float(t)} for t in ep.timestamp]))
    images=np.zeros((8,2,2,3),np.uint8);np.savez_compressed(folder/'rgb_supervision.npz',images=images,timestamp=ep.timestamp)
    correct(tmp_path);first=(folder/'annotation.json').read_bytes();correct(tmp_path)
    assert (folder/'annotation.json').read_bytes()==first
    a=json.loads(first);assert a['timestamp'][0]==0;assert 'timestamp_correction' in a['metadata']
    d=np.load(folder/'rgb_supervision.npz');np.testing.assert_array_equal(d['images'],images)
    np.testing.assert_allclose(d['timestamp'],np.arange(8)/10)
