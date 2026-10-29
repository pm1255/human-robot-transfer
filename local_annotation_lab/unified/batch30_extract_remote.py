"""Extract selected full episodes and run the snapshotted EgoSuite quality checks."""
import json, sys, subprocess, hashlib, traceback, argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/user/panmiao/human-robot-transfer/outputs/unified_30_20261021')
sys.path.insert(0,str(ROOT/'vendor'))
def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)): return x.item()
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    return x
def dump(p,o): p.write_text(json.dumps(clean(o),ensure_ascii=False,allow_nan=False))
def video(src,start,n,fps,dst,ordinal=False):
    if dst.exists(): return
    tmp=dst.with_name(dst.stem+'.tmp.mp4')
    vf='scale=640:-2'+(f',setpts=N/({fps}*TB)' if ordinal else '')
    subprocess.run(['ffmpeg','-v','error','-threads','2','-ss',str(start),'-i',str(src),'-frames:v',str(n),'-vf',vf,'-r',str(fps),'-an','-c:v','libx264','-preset','fast','-crf','21','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(tmp)],check=True)
    tmp.replace(dst)

def ego_extract(r):
    from ego_visualization.dataset import LeRobotV3View
    from ego_visualization.hand_projection import pose7_to_transform,project,transform_points
    root=Path(r['root']);row=r['episode'];info=r['info'];sid=r['sample_id'];out=ROOT/'samples'/sid;out.mkdir(parents=True,exist_ok=True)
    data=root/info['data_path'].format(chunk_index=row['data/chunk_index'],file_index=row['data/file_index'])
    whole=pq.read_table(data);table=pq.read_table(data,filters=[('episode_index','=',row['episode_index'])]);assert table.num_rows==row['length']
    ids=set(whole['episode_index'].to_pylist())
    # Upstream checker assumes one episode per file. Enforce it before using upstream.
    assert ids=={row['episode_index']},('packed parquet requires filtered adapter',ids)
    assert info['features']['observation.state']['shape']==[148]
    dump(out/'native.json',table.to_pydict());pq.write_table(table,out/'native.parquet')
    dump(out/'source.json',{**r,'frames':row['length'],'fps':info['fps'],'task':row['tasks'][0],'dataset':'EgoSuite','videoLabels':['左目','右目'],'parquet_episode_ids':list(ids),'source_code_commit':'d06d30a0729bef6df0c8404364d8ce0332b654e7'})
    for j,key in enumerate(['observation.images.head_left','observation.images.head_right']):
        src=root/info['video_path'].format(video_key=key,chunk_index=row[f'videos/{key}/chunk_index'],file_index=row[f'videos/{key}/file_index'])
        assert row[f'videos/{key}/from_timestamp']==0, 'B2 decoder requires zero offset'
        video(src,0,row['length'],info['fps'],out/f'cam_{j}.mp4',ordinal=True)
    class SelectedView(LeRobotV3View):
        @property
        def episodes(self):return [row]
    episode=SelectedView(root).canonical_episode(row['episode_index'])
    points=[]
    for i in range(row['length']):
        transform=np.linalg.inv(pose7_to_transform(episode.camera_pose_world['head_left'][i]))
        pp=[]
        for hand in [episode.left_hand_keypoints_world[i],episode.right_hand_keypoints_world[i]]:
            uvz=project(transform_points(transform,hand),episode.cameras['head_left'].intrinsics)
            uvz[:,:2]/=np.array([1920,1456]);pp.append(uvz)
        points.append(pp)
    dump(out/'hand_projection.json',dict(schema='egosuite.native_projection.v1',uv_normalized_depth_m=points,valid=np.stack([episode.left_hand_valid,episode.right_hand_valid],1),camera='head_left',calibration_verified=False,source='native_world_3d_to_per_frame_camera'))
    print('EXTRACTED',sid,row['length'],flush=True)

def ego_qc(r):
    from ego_quality_experiments.state_self_check.scan import analyze_episode
    from ego_visualization.dataset import LeRobotV3View
    from ego_quality_experiments.visual_consistency.scan import camera_metrics,b2_camera_metrics,sample_indices
    root=Path(r['root']);row=r['episode'];out=ROOT/'samples'/r['sample_id']
    class SelectedView(LeRobotV3View):
        @property
        def episodes(self):return [row]
    result=dict(schema='unified.egosuite_cleaner_qc.v1',source_commit='d06d30a0729bef6df0c8404364d8ce0332b654e7',score_type='uncalibrated_rule_score_not_probability',source_read_only=True,state=analyze_episode(str(root),row))
    episode=SelectedView(root).canonical_episode(row['episode_index']);indices=sample_indices(episode.frame_count,24)
    result['b0']={name:camera_metrics(episode,name,indices) for name in episode.cameras}
    result['b2']={};result['sampled_frames']=indices.tolist();result['b1']=dict(status='not_implemented_upstream',score=None)
    model=Path('/user/yangqinglin/workspace/uni_cleaner_plus/models/hand_landmarker.task')
    for name in episode.cameras:
        try:result['b2'][name]=b2_camera_metrics(episode,name,indices,model)
        except Exception as e:result['b2'][name]={'status':'error','error':str(e),'score':None}
    result['limitations']=['规则分不是正确概率','B2 仅抽检 24 帧/相机，MediaPipe 不是人工真值','动作语义、SAM3 轮廓与物体 6D 未由此检查验证','运动时延 B1 上游尚未实现']
    dump(out/'cleaner_qc.json',result);print('QC',r['sample_id'],result['state']['decision'],result['state']['score'],flush=True)

def abc_extract(row):
    out=ROOT/'samples'/row['sample_id'];out.mkdir(parents=True,exist_ok=True)
    z=np.load(row['numeric_path']);native={k:z[k].tolist() for k in z.files}
    schema=pq.ParquetFile(row['source_data_files'][0]).schema_arrow.names
    columns=[c for c in schema if not c.startswith('observation.images') and ('joint' in c or 'grip' in c or c in ['episode_index','frame_index'])]
    native['source_auxiliary']=pq.read_table(row['source_data_files'][0],columns=columns,filters=[('episode_index','=',row['episode_index'])]).to_pydict()
    dump(out/'native.json',native);dump(out/'source.json',row)
    for cam,c in row['cameras'].items():video(c['path'],c['from_timestamp'],row['num_frames'],row['fps'],out/f'cam_{cam}.mp4')
    print('EXTRACTED',row['sample_id'],row['num_frames'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['ego','abc','qc']);a=p.parse_args()
    for row in json.loads((ROOT/('abc_selection.json' if a.stage=='abc' else 'ego_selection.json')).read_text()):
        (abc_extract if a.stage=='abc' else ego_qc if a.stage=='qc' else ego_extract)(row)
