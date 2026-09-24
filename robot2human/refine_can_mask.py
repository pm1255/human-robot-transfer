import pathlib,json,shutil,subprocess
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2_video_predictor
R=pathlib.Path('/user/panmiao/workspace/robot2human-vace-20260924');p=R/'inputs/bridge_bridge_045160'
def run():
 torch.set_num_threads(8);original=p/'masks_initial.npz'
 if not original.exists():shutil.copy2(p/'masks.npz',original)
 old=np.load(original);frames=np.load(p/'source_frames.npz')['frames'];masks=old['mask'].copy();fps=float(old['fps'])
 predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(R/'sam2_verified.pt'),device='cuda',apply_postprocessing=False)
 prompts={'frame':29,'box':[25,0,195,165],'points':[[70,12],[112,31],[153,78],[146,120],[62,63],[133,184],[74,94],[214,114]],'labels':[1,1,1,1,1,0,0,0]}
 with torch.inference_mode():
  state=predictor.init_state(str(p/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
  predictor.add_new_points_or_box(state,frame_idx=29,obj_id=1,points=np.asarray(prompts['points'],np.float32),labels=np.asarray(prompts['labels'],np.int32),box=np.asarray(prompts['box'],np.float32))
  for i,ids,logits in predictor.propagate_in_video(state,start_frame_idx=29,reverse=True):masks[i]=np.maximum(masks[i],(logits[0,0].cpu().numpy()>0).astype(np.uint8)*255)
 expanded=np.stack([cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))) for m in masks]);np.savez_compressed(p/'masks.npz',mask=masks,edit_mask=expanded,fps=fps)
 overlays=frames.copy();sel=expanded>0;overlays[sel]=(frames[sel]*.45+np.array([52,229,202])*.55).astype(np.uint8)
 for name,arr in [('mask.mp4',np.repeat(expanded[...,None],3,-1)),('mask_overlay.mp4',overlays)]:
  subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','256x256','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(p/name)],input=arr.tobytes(),check=True)
 cv2.imwrite(str(p/'mask_contact_sheet_refined.jpg'),cv2.cvtColor(np.concatenate([overlays[i] for i in [0,6,12,18,24,29]],axis=1),cv2.COLOR_RGB2BGR))
 (p/'mask_refinement.json').write_text(json.dumps({'prompts':prompts,'method':'union initial forward mask with corrected last-frame backward propagation','manual_quality_verified':False},indent=2));print('REFINEMENT_DONE',flush=True)
if __name__=='__main__':run()
