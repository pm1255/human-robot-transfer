"""SAM2 video masks from explicit first-frame robot prompts; no action labels inferred."""
import argparse,json,pathlib,subprocess,time
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2_video_predictor
ROOT=pathlib.Path('/user/panmiao/workspace/robot2human-vace-20260924')
PROMPTS={
 'bridge_bridge_042977':{'box':[95,0,182,172],'points':[[136,30],[135,95],[145,145],[70,113],[183,188]],'labels':[1,1,1,0,0]},
 'bridge_bridge_045160':{'box':[0,0,110,173],'points':[[31,31],[49,105],[81,143],[119,67],[189,99]],'labels':[1,1,1,0,0]},
 'bridge_bridge_007089':{'box':[105,0,207,177],'points':[[159,30],[171,95],[159,130],[219,176],[97,180]],'labels':[1,1,1,0,0]}}
def run():
 torch.set_num_threads(8)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'sam2_verified.pt'),device=device,apply_postprocessing=False)
 for row in json.loads((ROOT/'inputs/manifest.json').read_text()):
  folder=ROOT/'inputs'/row['id'];out=folder/'masks.npz'
  if out.exists():continue
  source=np.load(folder/'source_frames.npz');frames=source['frames'];fps=float(source['fps'])
  jpegs=folder/'frames';jpegs.mkdir(exist_ok=True)
  for i,im in enumerate(frames):cv2.imwrite(str(jpegs/f'{i:05d}.jpg'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
  prm=PROMPTS[row['id']];masks=np.zeros((len(frames),*frames[0].shape[:2]),np.uint8)
  with torch.inference_mode():
   state=predictor.init_state(str(jpegs),offload_video_to_cpu=True,offload_state_to_cpu=True)
   predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,points=np.array(prm['points'],np.float32),labels=np.array(prm['labels'],np.int32),box=np.array(prm['box'],np.float32))
   for index,ids,logits in predictor.propagate_in_video(state):masks[index]=(logits[0,0].cpu().numpy()>0).astype(np.uint8)*255
   predictor.reset_state(state)
  expanded=np.stack([cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))) for m in masks])
  np.savez_compressed(out,mask=masks,edit_mask=expanded,fps=fps)
  # Review overlays at native timing; white pixels in mask denote editable region.
  overlays=[]
  for im,m in zip(frames,expanded):
   overlay=im.copy();sel=m>0;overlay[sel]=(im[sel]*.45+np.array([52,229,202])*.55).astype(np.uint8);overlays.append(overlay)
  for name,arr in [('mask.mp4',np.repeat(expanded[...,None],3,-1)),('mask_overlay.mp4',np.array(overlays))]:
   h,w=arr.shape[1:3];subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/name)],input=arr.tobytes(),check=True)
  cv2.imwrite(str(folder/'mask_contact_sheet.jpg'),cv2.cvtColor(np.concatenate([overlays[i] for i in np.linspace(0,len(overlays)-1,6).astype(int)],axis=1),cv2.COLOR_RGB2BGR))
  (folder/'mask_provenance.json').write_text(json.dumps({'model':'SAM2.1 Hiera tiny','prompt':prm,'native_fps':fps,'frames':len(frames),'dilation_pixels':7,'coverage_mean':float((expanded>0).mean()),'manually_verified':False},indent=2))
  print('MASK_DONE',row['id'],len(frames),flush=True)
 (ROOT/'MASKS_READY.json').write_text(json.dumps({'completed':True}))
if __name__=='__main__':run()
