"""Extract source clips referenced by the existing v4 manifests, without modifying v4."""
import argparse,pathlib,json,subprocess,numpy as np
DEFAULT_ROOT='/user/panmiao/workspace/robot2human-vace-20260924'
DEFAULT_V4='/user/panmiao/workspace/human2robot-wan-v4-20260924'
def run(a):
 root=pathlib.Path(a.root);v4=pathlib.Path(a.v4);rows=json.loads((v4/'manifests/bridge.json').read_text())['train'];jobs=json.loads((v4/'episode_jobs.json').read_text());records=[]
 for eid in a.episodes:
  row=next(x for x in rows if x['episode']==eid);job=jobs[eid];folder=root/'inputs'/eid;folder.mkdir(parents=True,exist_ok=True);duration=min(a.seconds,job['n']/job['fps'])
  subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',str(job['start']),'-i',row['source'],'-t',str(duration),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(folder/'source.mp4')],check=True)
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height','-of','json',str(folder/'source.mp4')]))['streams'][0];w,h=info['width'],info['height']
  b=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(folder/'source.mp4'),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']);frames=np.frombuffer(b,np.uint8).reshape(-1,h,w,3);np.savez_compressed(folder/'source_frames.npz',frames=frames,fps=job['fps'])
  records.append({'id':eid,'source':row['source'],'start':job['start'],'frames':job['n'],'fps':job['fps'],'text':row['text'],'duration':duration})
 (root/'inputs/manifest.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=DEFAULT_ROOT);p.add_argument('--v4',default=DEFAULT_V4);p.add_argument('--seconds',type=float,default=6);p.add_argument('episodes',nargs='+');run(p.parse_args())
