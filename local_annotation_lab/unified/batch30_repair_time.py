"""Preserve all ordinal video frames; use each episode's recorded state time grid."""
import json,pathlib,subprocess,hashlib
import numpy as np
R=pathlib.Path('/user/panmiao/human-robot-transfer/outputs/unified_30_20261021')
audit=[]
for item in json.loads((R/'ego_selection.json').read_text()):
 p=R/'samples'/item['sample_id'];s=json.loads((p/'source.json').read_text());native=json.loads((p/'native.json').read_text());t=np.asarray(native['timestamp']);fps=round(1/float(np.median(np.diff(t))));row=s['episode'];info=s['info'];n=len(t)
 entry=dict(id=p.name,frames=n,dataset_nominal_fps=info['fps'],episode_timestamp_fps=fps,state_time_grid_max_error_s=float(abs(t-t[0]-np.arange(n)/fps).max()),export_policy='one output frame per decoded source frame; setpts from ordinal and episode state timestamps; no frame interpolation',cameras=[])
 for j,key in enumerate(['observation.images.head_left','observation.images.head_right']):
  src=pathlib.Path(s['root'])/info['video_path'].format(video_key=key,chunk_index=row[f'videos/{key}/chunk_index'],file_index=row[f'videos/{key}/file_index'])
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration','-of','json',str(src)]))['streams'][0];assert int(probe['nb_read_frames'])==n
  tmp=p/f'cam_{j}.repaired.mp4';dst=p/f'cam_{j}.mp4'
  subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(src),'-vf',f'scale=640:-2,setpts=N/({fps}*TB)','-r',str(fps),'-frames:v',str(n),'-an','-c:v','libx264','-preset','fast','-crf','21','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(tmp)],check=True)
  check=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(tmp)]))['streams'][0];assert int(check['nb_read_frames'])==n
  tmp.replace(dst);entry['cameras'].append(dict(camera=key,source_probe=probe,preview_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),decoded_frames=n))
 s['fps']=fps;s['source_nominal_fps']=info['fps'];s['time_normalization']=entry;(p/'source.json').write_text(json.dumps(s,ensure_ascii=False));(p/'time_audit.json').write_text(json.dumps(entry,indent=2));audit.append(entry);print('TIME_FIXED',p.name,fps,n,flush=True)
(R/'ego_time_audit.json').write_text(json.dumps(audit,indent=2))
