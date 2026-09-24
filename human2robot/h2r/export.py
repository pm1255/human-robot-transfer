from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .pipeline import reference_windows
from .schema import write_json


def export_references(result,destination,horizon=4,allow_synthetic=False):
    """Native reference archive, explicitly NOT advertised as a LeRobot dataset."""
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    windows=reference_windows(result,horizon,allow_synthetic)
    width=len(result['robot_joint_names'])+1
    states=np.stack([x['reference_state'] for x in windows]) if windows else np.empty((0,width))
    targets=np.stack([x['future_reference'] for x in windows]) if windows else np.empty((0,horizon,width))
    np.savez_compressed(out/'reference_windows.npz',reference_state=states,future_reference=targets,
                        frame_index=np.array([x['frame_index'] for x in windows],dtype=np.int64))
    manifest={'format':'h2r_reference_archive_v1','episode_id':result['episode_id'],'windows':len(windows),
              'horizon':horizon,'state_semantics':'retargeted_reference_not_measured_robot_state',
              'target_semantics':'future_robot_joint_reference_not_executed_command',
              'joint_order':result['robot_joint_names']+['gripper_aperture_m'],
              'source_kind':result['metadata']['source_kind'],'split':result['split'],
              'exclusion_reasons':result['training_exclusion_reasons'],
              'allow_synthetic_test_only':allow_synthetic,'schema_version':'human-reference/0.1'}
    write_json(out/'manifest.json',manifest)
    return manifest


def export_human_lerobot(annotation_dir,destination,repo_id='local/human-rgb-reference'):
    """Use official writer. Custom human-supervision view, not stock robot BC.

    No fabricated observation.state or robot action. All missing landmarks are
    finite-filled AND accompanied by a mask. This export does not upload to Hub.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=Path(annotation_dir);data=np.load(root/'rgb_supervision.npz')
    annotation=json.loads((root/'annotation.json').read_text());m=annotation['metadata']
    t=data['timestamp'];delta=np.diff(t)
    if len(t)<2 or not np.allclose(delta,np.median(delta),atol=0.012):
        raise ValueError('Resample VFR explicitly before fixed-FPS LeRobot export')
    fps=int(round(1/np.median(delta)))
    if abs(fps*np.median(delta)-1)>.03:raise ValueError('Unsupported fractional FPS without explicit resampling')
    h,w=data['images'].shape[1:3]
    features={'observation.images.head':{'dtype':'image','shape':(h,w,3),'names':['height','width','channels']},
              'annotation.hand_keypoints_2d':{'dtype':'float32','shape':(42,),'names':None},
              'annotation.hand_valid':{'dtype':'float32','shape':(1,),'names':None},
              'annotation.source_timestamp':{'dtype':'float32','shape':(1,),'names':None}}
    dataset=LeRobotDataset.create(repo_id=repo_id,fps=fps,root=Path(destination),
                                   robot_type='human_rgb_annotation',features=features,use_videos=False)
    try:
        for i in range(len(t)):
            valid=bool(data['observed_2d'][i])
            dataset.add_frame({'observation.images.head':data['images'][i],
                'annotation.hand_keypoints_2d':np.nan_to_num(data['landmarks_2d'][i]).reshape(-1).astype(np.float32),
                'annotation.hand_valid':np.array([valid],np.float32),
                'annotation.source_timestamp':np.array([t[i]],np.float32),
                'task':m.get('task') or 'unannotated human manipulation'})
        dataset.save_episode()
    finally:
        dataset.finalize()
    write_json(Path(destination)/'human_annotation_contract.json',{
        'purpose':'custom human auxiliary supervision; no robot action or robot state',
        'mask_required':'annotation.hand_valid','provenance':m})
    # Read-after-write through the same real loader, not a hand-built layout check.
    loaded=LeRobotDataset(repo_id,root=Path(destination),download_videos=False)
    if len(loaded)!=len(t):raise RuntimeError('LeRobot round-trip frame mismatch')
    _=loaded[0]
    return {'frames':len(loaded),'fps':fps,'native_loader_roundtrip':True}
