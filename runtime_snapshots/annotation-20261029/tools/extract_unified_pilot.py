"""Read-only pilot extraction from source datasets; outputs go to a separate review directory."""
import json, subprocess, time, hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
ROOT=Path('/user/panmiao/human-robot-transfer/outputs/unified_three_domain_20261020');ROOT.mkdir(exist_ok=True)
writer=EventFileWriter('/user/panmiao/human-robot-transfer/outputs/object_annotation_20261020/events/unified_three_domain')
def log(stage,n,total,step):
 writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=f'{stage}/completed',simple_value=n),Summary.Value(tag=f'{stage}/total',simple_value=total),Summary.Value(tag=f'{stage}/fraction',simple_value=n/total)])));writer.flush()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False))
def video(src,start,n,fps,dst):
 subprocess.run(['ffmpeg','-v','error','-threads','2','-ss',str(start),'-i',str(src),'-frames:v',str(n),'-vf','scale=640:-2','-r',str(fps),'-an','-c:v','libx264','-preset','fast','-crf','21','-threads','2','-movflags','+faststart','-y',str(dst)],check=True)
log('selection',0,9,0);log('annotation',0,9,0);log('qc',0,9,0);log('independent_review',0,9,0)
rows=json.loads((ROOT/'abc_selection.json').read_text())
for i,row in enumerate(rows):
 out=ROOT/row['sample_id'];out.mkdir(exist_ok=True)
 z=np.load(row['numeric_path']);native={k:z[k].tolist() for k in z.files}
 schema=pq.ParquetFile(row['source_data_files'][0]).schema_arrow.names
 columns=[c for c in schema if not c.startswith('observation.images') and ('joint' in c or 'grip' in c or c in ['episode_index','frame_index'])]
 native['source_auxiliary'] = pq.read_table(row['source_data_files'][0],columns=columns,filters=[('episode_index','=',row['episode_index'])]).to_pydict()
 native['source_columns']=schema
 dump(out/'native.json',native);dump(out/'source.json',row)
 for cam,c in row['cameras'].items():video(c['path'],c['from_timestamp'],row['num_frames'],row['fps'],out/f'cam_{cam}.mp4')
 log('selection',i+1,9,i+1);print('extracted',row['sample_id'],flush=True)
base=Path('/user/zhangxueqian/dataset/lerobot_v3.0_v0/ego/EgoDex-LeRobot-v3.0/train')
for j,name in enumerate(['basic_pick_place-part1','add_remove_lid']):
 p=base/name;m=json.loads((p/'meta/info.json').read_text());epf=p/'meta/episodes/chunk-000/file-000.parquet';cols=[c for c in pq.ParquetFile(epf).schema_arrow.names if not c.startswith('stats/')];r=pq.read_table(epf,columns=cols).slice(0,1).to_pylist()[0]
 sid='egodex_'+name+'_000000';out=ROOT/sid;out.mkdir(exist_ok=True)
 data=p/m['data_path'].format(chunk_index=r['data/chunk_index'],file_index=r['data/file_index'])
 native=pq.read_table(data,filters=[('episode_index','=',r['episode_index'])]).to_pydict()
 dump(out/'native.json',native);dump(out/'source.json',{'root':str(p),'episode':r,'info':m})
 vkey='observation.images.camera';src=p/m['video_path'].format(video_key=vkey,chunk_index=r[f'videos/{vkey}/chunk_index'],file_index=r[f'videos/{vkey}/file_index'])
 video(src,r[f'videos/{vkey}/from_timestamp'],r['length'],m['fps'],out/'cam_0.mp4')
 log('selection',j+4,9,j+4);print('extracted',sid,r['length'],r['tasks'],flush=True)
writer.close()
