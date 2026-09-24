"""Attach unchanged robot labels; keep synthetic-human alignment explicitly unverified."""
import argparse,json,pathlib,subprocess,hashlib,numpy as np
ROOT=pathlib.Path('/user/panmiao/workspace/robot2human-vace-20260924')
def run(model):
 records=[]
 for meta in sorted((ROOT/'results'/model).glob('*/seed*/result.json')):
  folder=meta.parent;data=json.loads(meta.read_text())
  if folder.name!='seed'+str(data['seed']):continue
  cached=folder/'paired_manifest.json'
  if cached.exists() and json.loads(cached.read_text()).get('schema')=='robot2human.paired-example.v1' and cached.stat().st_mtime>=meta.stat().st_mtime and (folder/'human_native_fps.mp4').exists():
   previous=json.loads(cached.read_text());records.append({k:v for k,v in previous.items() if k!='samples'});continue
  source=ROOT/'inputs'/data['id'];native=np.load(source/'source_frames.npz');fps=float(native['fps']);n=len(native['frames']);size=data['size']
  decoded=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(folder/'human.mp4'),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']);frames=np.frombuffer(decoded,np.uint8).reshape(-1,size,size,3)
  generated_times=np.asarray(data['source_timestamps']);mapping=[]
  for i in range(n):
   candidates=np.flatnonzero(np.asarray(data['frame_indices'])==i)
   if not len(candidates):continue
   j=int(candidates[np.argmin(abs(generated_times[candidates]-i/fps))]);mapping.append({'native_frame':i,'generated_frame':j,'native_time':i/fps,'generated_time':float(generated_times[j]),'time_error_seconds':float(generated_times[j]-i/fps)})
  assert [x['native_frame'] for x in mapping]==list(range(n)), 'Some native observations lack a generated counterpart'
  paired=frames[[x['generated_frame'] for x in mapping]]
  subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size}x{size}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'human_native_fps.mp4')],input=paired.tobytes(),check=True)
  labels=json.loads((source/'robot_labels.json').read_text());(folder/'robot_labels.json').write_text(json.dumps(labels,indent=2));(folder/'frame_alignment.json').write_text(json.dumps(mapping,indent=2))
  samples=[]
  for row in labels['samples']:
   identity='|'.join([row['uid'],data['model'],str(data['seed']),data['source_sha256'],hashlib.sha256((source/'masks.npz').read_bytes()).hexdigest()])
   samples.append({'pair_id':hashlib.sha256(identity.encode()).hexdigest()[:24],'robot_sample':row,'human_video':'human_native_fps.mp4','human_frame_index':row['t'],'human_timestamp':row['t']/fps,'future_horizon_within_clip':bool(row['t']+4*row['stride']<n),'human_alignment_verified':False,'training_eligible':False})
  record={'schema':'robot2human.paired-example.v1','can_load_directly_as_v4_train_manifest':False,'id':data['id'],'model':data['model'],'seed':data['seed'],'source_fps':fps,'human_preview':'human.mp4','human_native_fps':'human_native_fps.mp4','label_semantics':data['source_label_semantics'],'native_frame_count':n,'generated_frames':data['frames'],'max_resampling_time_error_seconds':max(abs(x['time_error_seconds']) for x in mapping),'human_alignment_verified':False,'training_eligible':False,'samples':samples}
  (folder/'paired_manifest.json').write_text(json.dumps(record,indent=2));records.append({k:v for k,v in record.items() if k!='samples'});print('PAIRED',data['id'],len(samples),flush=True)
 (ROOT/('PAIRED_'+model+'.json')).write_text(json.dumps(records,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);run(p.parse_args().model)
