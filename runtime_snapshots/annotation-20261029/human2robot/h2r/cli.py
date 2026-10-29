from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .schema import Episode,write_json,file_hash
from .kinematics import SerialRobot
from .fixtures import CASES,make_case
from .pipeline import process
from .export import export_references,export_human_lerobot

ROOT=Path(__file__).resolve().parents[1]


def gallery(root=ROOT,real=True):
    root=Path(root);web=root/'web';out=root/'outputs';(web/'data').mkdir(parents=True,exist_ok=True);(web/'media').mkdir(exist_ok=True)
    robot=SerialRobot(root/'configs/demo_arm.urdf');items=[]
    for case,title,description in CASES:
        ep,truth=make_case(case,robot);ep.save(out/'synthetic'/case/'annotation.json')
        result=process(ep,robot)
        raw=result['raw_pose_world'][:,:3,3];fixed=result['repaired_pose_world'][:,:3,3]
        compare=np.isfinite(raw).all(axis=1)&result['human_valid']
        if compare.any():
            result['summary']['synthetic_raw_rmse_mm']=float(np.sqrt(np.mean(np.sum((raw[compare]-truth[compare,:3,3])**2,axis=1)))*1000)
            result['summary']['synthetic_repaired_rmse_mm']=float(np.sqrt(np.mean(np.sum((fixed[compare]-truth[compare,:3,3])**2,axis=1)))*1000)
        result['truth_pose_world']=truth
        write_json(web/'data'/f'{case}.json',result)
        export_references(result,out/'synthetic'/case/'export',allow_synthetic=True)
        item={'id':case,'title':title,'description':description,'kind':'synthetic','data':f'data/{case}.json','summary':result['summary']}
        items.append(item);print('fixture',case,result['summary'],flush=True)
    manifest_path=root/'data/raw/bootstrap/manifest.json'
    if real and manifest_path.exists():
        from .vision import extract_video
        manifest=json.loads(manifest_path.read_text())
        for source in manifest:
            case=source['id'];video=root/'data/raw/bootstrap'/source['local_filename'];ann=out/'real'/case
            if (ann/'annotation.json').exists() and (ann/'keypoints.json').exists() and (ann/'rgb_supervision.npz').exists():
                ep=Episode.load(ann/'annotation.json')
                if ep.metadata['source_sha256']!=file_hash(video):raise ValueError('Stale annotation source hash')
                keypoints=json.loads((ann/'keypoints.json').read_text())
            else:
                ep,keypoints=extract_video(video,root/'data/models/hand_landmarker.task',ann,
                                source_uri=source['source_uri'],source_group=source['split_group'])
            ep.metadata['task']=source['task'];ep.metadata['task_source']=source['task_source']
            ep.metadata['source_cluster']=source['source_cluster'];ep.metadata['source_episode_index']=source['source_episode_index']
            ep.save(ann/'annotation.json')
            result=process(ep,robot,illustrative_alignment=True)
            result['keypoints']=keypoints
            write_json(web/'data'/f'{case}.json',result)
            export_references(result,ann/'export')
            shutil.copy2(video,web/'media'/f'{case}.mp4')
            items.append({'id':case,'title':f"EPIC · {source['source_episode_index']:04d}",
                'description':source['task'],'description_source':'上游 VLM 描述，非人工确认的指令',
                'kind':'human_video','data':f'data/{case}.json','video':f'media/{case}.mp4',
                'summary':result['summary']})
            print('real',case,result['summary'],flush=True)
    for name in ['rovid_bottle_manipulation','rovid_kitchen_manipulation']:
        path=root/'data/raw'/f'{name}.mp4'
        if not path.exists():continue
        shutil.copy2(path,web/'media'/path.name)
        items.append({'id':name,'title':'来源拒绝 · '+('瓶子操作' if 'bottle' in name else '厨房操作'),
            'description':'抽帧人工核查为机器人视频；文件名含 manipulation 不能作为人类来源证据。',
            'kind':'rejected_robot_video','video':f'media/{path.name}',
            'summary':{'robot_training_frames':0,'source_kind':'robot_video','reason':'NOT_HUMAN_VIDEO'}})
    report={'created_at':datetime.now(timezone.utc).isoformat(),'schema_version':'human-reference/0.1',
            'items':items,'summary':{'synthetic_cases':sum(x['kind']=='synthetic' for x in items),
            'real_human_clips':sum(x['kind']=='human_video' for x in items),
            'rejected_sources':sum(x['kind']=='rejected_robot_video' for x in items)},
            'claim_boundaries':['真实视频没有独立三维真值','真实样本尚无经过验证的世界尺度、相机轨迹、物体接触',
                                '示例机械臂不是 Aloha/Piper 本体','IK 和代理碰撞检查不等于物理抓取成功',
                                'RoboTwin 微调和真机替代率尚未由本网页证明']}
    write_json(web/'data/index.json',report);write_json(out/'gallery_report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description='Human RGB to auditable reference trajectories')
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('gallery');g.add_argument('--synthetic-only',action='store_true')
    c=sub.add_parser('convert');c.add_argument('annotation');c.add_argument('--urdf',default=str(ROOT/'configs/demo_arm.urdf'));c.add_argument('--out',required=True);c.add_argument('--illustrative-alignment',action='store_true')
    c.add_argument('--base',default='base');c.add_argument('--tip',default='tcp');c.add_argument('--alignment',help='JSON 4x4 T_robot_world; calibrated values only')
    v=sub.add_parser('annotate');v.add_argument('video');v.add_argument('--out',required=True);v.add_argument('--model',default=str(ROOT/'data/models/hand_landmarker.task'))
    v.add_argument('--intrinsics',help='JSON 3x3 camera K');v.add_argument('--camera-trajectory',help='JSON timestamp, T_world_camera, status')
    v.add_argument('--source-uri');v.add_argument('--source-group',help='Original session identity shared by derived clips')
    l=sub.add_parser('export-lerobot');l.add_argument('annotation_dir');l.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='gallery':gallery(real=not a.synthetic_only)
    elif a.command=='annotate':
        from .vision import extract_video
        extract_video(a.video,a.model,a.out,intrinsics=json.loads(Path(a.intrinsics).read_text()) if a.intrinsics else None,
                      camera_trajectory=a.camera_trajectory,source_uri=a.source_uri,source_group=a.source_group)
    elif a.command=='convert':
        r=process(Episode.load(a.annotation),SerialRobot(a.urdf,a.base,a.tip),illustrative_alignment=a.illustrative_alignment,
                  T_robot_world=json.loads(Path(a.alignment).read_text()) if a.alignment else None)
        write_json(Path(a.out)/'result.json',r);export_references(r,a.out)
    elif a.command=='export-lerobot':print(export_human_lerobot(a.annotation_dir,a.out))

if __name__=='__main__':main()
