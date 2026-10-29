"""Portable entry points around the archived pipelines. No training is launched."""
from pathlib import Path
import argparse, importlib.util, json, shutil, subprocess, sys, urllib.request, urllib.error
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'human2robot'))
MODELS = {
 'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
 'pose_landmarker_full.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
 'efficientdet_lite0.tflite': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite',
}
def require(path):
 p=Path(path).expanduser().resolve()
 if not p.exists(): raise ValueError(f'Missing input: {p}')
 return p

def models(path):
 p=Path(path).resolve();p.mkdir(parents=True,exist_ok=True)
 for name,url in MODELS.items():
  target=p/name
  if target.exists() and target.stat().st_size>100_000: continue
  print('Downloading',name,flush=True);temporary=target.with_suffix('.part')
  with urllib.request.urlopen(url,timeout=90) as source, temporary.open('wb') as dest:shutil.copyfileobj(source,dest)
  if temporary.stat().st_size<100_000:raise ValueError(f'Invalid model download: {name}')
  temporary.replace(target)
 return p

def prepare_video(a,out):
 if not 0<a.seconds<=12:raise ValueError('--seconds must be between 0 and 12 (bounded preview).')
 video=require(a.video)
 if out.exists() and any(out.iterdir()):raise ValueError('Output directory is not empty; choose a fresh --out to avoid stale cached annotations.')
 out.mkdir(parents=True,exist_ok=True)
 target=out/'input.mp4'
 subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(video),'-t',str(a.seconds),'-vf','scale=640:-2,fps=10','-an','-c:v','libx264','-pix_fmt','yuv420p',str(target)],check=True)
 return target

def annotate(a,out,video):
 from v2.annotate import run
 from v2.review_gate import gate
 source=out/'input';source.mkdir();shutil.copy2(video,source/'clip.mp4')
 (source/'manifest.json').write_text(json.dumps([{'id':'clip','file':'clip.mp4','view':a.view,'task':a.task,'source_kind':'human_video','source_uri':str(require(a.video)),'training_eligible':False}]))
 run(source,models(a.models),out/'annotations')
 f=out/'annotations/clip/annotation.json';data=gate(json.loads(f.read_text()));f.write_text(json.dumps(data))
 (f.parent/'summary.json').write_text(json.dumps(data['meta'],indent=2));(out/'annotations/manifest.json').write_text(json.dumps([data['meta']],indent=2))
 from preview_annotation import render
 render(video,data,out/'annotated.mp4')
 return data

def human(a):
 out=Path(a.out).expanduser().resolve()
 # Check required third-party assets before running detectors.
 if a.command=='arm':require(Path(a.aloha_assets)/'model.urdf')
 if a.command=='humanoid':require(a.g1_xml)
 video=prepare_video(a,out)
 if a.command in ['annotate','humanoid']:
  data=annotate(a,out,video)
  if a.command=='annotate':print(out/'annotated.mp4');return
 from v2 import render_conversion as renderer
 renderer.ROOT=out;renderer.OUT=out/'conversion';renderer.OUT.mkdir()
 if a.command=='humanoid':
  renderer.XML=require(a.g1_xml)
  renderer.write_case('body','clip',video,data,renderer.G1(),max_frames=len(data['frames']))
 else:
  from h2r.vision import extract_video
  from h2r.pipeline import process
  from h2r.kinematics import SerialRobot
  from h2r.schema import write_json
  from v2 import build_aloha as builder
  ep,kp=extract_video(video,models(a.models)/'hand_landmarker.task',out/'annotation',max_seconds=a.seconds)
  result=process(ep,SerialRobot(REPO/'human2robot/configs/demo_arm.urdf'),illustrative_alignment=True);result['keypoints']=kp
  write_json(out/'web/data/clip.json',result)
  write_json(out/'web/data/index.json',{'items':[{'id':'clip','data':'data/clip.json'}]})
  asset=out/'data/aloha';asset.mkdir(parents=True)
  source=require(a.aloha_assets)
  # Work on a private copy: original third-party URDF and meshes stay unchanged.
  for f in source.rglob('*'):
   if f.is_file() and f.suffix.lower() in ['.urdf','.dae','.stl','.obj','.png','.jpg']:
    dest=asset/f.name
    if dest.exists():raise ValueError(f'Ambiguous asset basename: {f.name}; supply the flattened Aloha asset directory.')
    shutil.copy2(f,dest)
  builder.ROOT=out;builder.ASSET=asset;builder.OUT=out/'web/v2data/aloha';builder.main()
  result=json.loads((out/'web/data/clip.json').read_text());result['_aloha']=json.loads((builder.OUT/'clip.json').read_text())
  renderer.write_case('hand','clip',video,result,max_frames=len(result['timestamp']))
 print(out/'conversion/clip.mp4')

def robot(a):
 import numpy as np
 from PIL import Image
 for path in [a.video,a.mask,a.reference,a.wan,a.weights]:require(path)
 if not 0<a.seconds<=6:raise ValueError('For VACE preview use --seconds in (0, 6].')
 if not 1<=a.fps<=30:raise ValueError('--fps must be between 1 and 30.')
 if a.steps<=0:raise ValueError('--steps must be positive.')
 if not a.prompt.strip():raise ValueError('--prompt must describe the target action.')
 out=Path(a.out).expanduser().resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh --out; cached results must not be reused with new inputs.')
 import torch
 if not torch.cuda.is_available():raise ValueError('robot2human requires NVIDIA CUDA; CPU-only preview is not supported.')
 folder=out/'inputs/custom';folder.mkdir(parents=True);ref=out/'human_references';ref.mkdir()
 # Both videos must share timestamps/duration; do not repeat or truncate a short mask.
 def duration(path):
  return float(json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(path)]))['format']['duration'])
 length=min(a.seconds,duration(a.video))
 if duration(a.mask)+.5/a.fps<length:raise ValueError('Mask video is shorter than the source clip.')
 for input_path,target,mask in [(a.video,folder/'source.mp4',False),(a.mask,folder/'mask.mp4',True)]:
  vf=f'fps={a.fps},scale=256:256:flags='+('neighbor' if mask else 'bicubic')
  subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(require(input_path)),'-t',str(length),'-vf',vf,'-an','-c:v','libx264','-crf','10','-pix_fmt','yuv420p',str(target)],check=True)
 def decode(path,fmt,channels):
  b=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt',fmt,'pipe:1'])
  return np.frombuffer(b,np.uint8).reshape(-1,256,256,channels)
 frames=decode(folder/'source.mp4','rgb24',3);mask=decode(folder/'mask.mp4','gray',1)[...,0]
 if len(frames)!=len(mask):raise ValueError('Source/mask frame counts differ; align the videos before editing.')
 if len(frames)<2:raise ValueError('Need at least two source frames.')
 np.savez_compressed(folder/'source_frames.npz',frames=frames,fps=a.fps)
 np.savez_compressed(folder/'masks.npz',edit_mask=(mask>127).astype(np.uint8)*255,fps=a.fps)
 Image.open(a.reference).convert('RGB').save(ref/'custom_human.png')
 (out/'inputs/manifest.json').write_text(json.dumps([{'id':'custom','generation_prompt':a.prompt,'source_dataset':'user_supplied','text':a.prompt}]))
 model_dir=out/'models';model_dir.mkdir();(model_dir/'Wan2.1-VACE-1.3B').symlink_to(require(a.weights),target_is_directory=True)
 sys.path.insert(0,str(require(a.wan)))
 spec=importlib.util.spec_from_file_location('r2h_generate',REPO/'robot2human/generate_expanded.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=out
 module.run(argparse.Namespace(model='1.3B',size=512,steps=a.steps,seed=a.seed,seeds=[a.seed],only='custom',variant=a.variant))
 print(out/'results'/('1.3B_'+a.variant)/'custom'/f'seed{a.seed}'/'human.mp4')

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 m=sub.add_parser('models');m.add_argument('--models',default=str(REPO/'models/mediapipe'))
 for name in ['annotate','arm','humanoid']:
  q=sub.add_parser(name);q.add_argument('--video',required=True);q.add_argument('--out',required=True);q.add_argument('--models',default=str(REPO/'models/mediapipe'));q.add_argument('--seconds',type=float,default=6);q.add_argument('--task',default='');q.add_argument('--view',choices=['egocentric','exocentric'],default='exocentric' if name=='humanoid' else 'egocentric')
  if name=='arm':q.add_argument('--aloha-assets',required=True)
  if name=='humanoid':q.add_argument('--g1-xml',required=True)
 q=sub.add_parser('robot2human')
 for name in ['video','mask','reference','prompt','weights','wan','out']:q.add_argument('--'+name,required=True)
 q.add_argument('--seconds',type=float,default=6);q.add_argument('--fps',type=float,default=5);q.add_argument('--steps',type=int,default=40);q.add_argument('--seed',type=int,default=2026);q.add_argument('--variant',choices=['baseline','handroom'],default='baseline')
 a=p.parse_args()
 if not all(shutil.which(x) for x in ['ffmpeg','ffprobe']):p.error('Install ffmpeg/ffprobe first (see README).')
 try:
  if a.command=='models':models(a.models)
  elif a.command=='robot2human':robot(a)
  else:human(a)
 except (ValueError,FileNotFoundError,urllib.error.URLError) as error:p.error(str(error))
if __name__=='__main__':main()
