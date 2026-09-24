"""Reference + masked source video -> human edit. Never uses v4 policy weights."""
import argparse,json,pathlib,time,subprocess,hashlib,gc
import numpy as np,cv2,torch
from PIL import Image
from wan.vace import WanVace
from wan.configs import WAN_CONFIGS
from wan.modules.vace_model import VaceWanModel
# Native Wan requires the optional flash-attn extension; preserve variable lengths
# with per-batch slicing when using PyTorch's built-in fused SDPA instead.
import wan.modules.model as wan_model
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
 def sdpa_attention(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,q_scale=None,causal=False,window_size=(-1,-1),deterministic=False,dtype=torch.bfloat16,version=None):
  assert window_size==(-1,-1)
  output=torch.zeros_like(q)
  for b in range(q.shape[0]):
   nq=int(q_lens[b]) if q_lens is not None else q.shape[1]
   nk=int(k_lens[b]) if k_lens is not None else k.shape[1]
   qq=q[b:b+1,:nq].to(dtype).transpose(1,2)
   if q_scale is not None:qq=qq*q_scale
   result=torch.nn.functional.scaled_dot_product_attention(qq,k[b:b+1,:nk].to(dtype).transpose(1,2),v[b:b+1,:nk].to(dtype).transpose(1,2),dropout_p=dropout_p,is_causal=causal,scale=softmax_scale)
   output[b:b+1,:nq]=result.transpose(1,2).to(q.dtype)
  return output
 wan_model.flash_attention=sdpa_attention
ROOT=pathlib.Path('/user/panmiao/workspace/robot2human-vace-20260924')
PROMPTS={
'bridge_bridge_042977':'A realistic adult human bare hand and forearm reaches down from the upper edge of the image, grasps the rim of the small stainless steel pot on the wooden countertop, lifts and moves the pot to the left of the yellow bottle, then releases it. The five human fingers curl naturally around the metal rim. Same fixed camera and kitchen scene. The yellow cloth and bottle remain unchanged.',
'bridge_bridge_045160':'A realistic adult human bare hand and forearm enters from the upper edge of the image, reaches toward the gray cylindrical can on the wooden countertop, wraps the fingers around the can, and moves it toward the lower right side of the orange cloth. Same fixed camera, tabletop and surrounding objects. Natural five-finger anatomy and a continuous grasp.',
'bridge_bridge_007089':'A realistic adult human bare hand and forearm enters from the top of the image and reaches down into the metal dish rack, closing its fingers around the clear glass cup and lifting it slightly. The human fingers stay in contact with the glass while it moves. Same fixed camera and dish rack with white mugs, transparent glasses and a dark cup. Preserve the seated person in the background.'}
NEGATIVE='robot, robotic arm, mechanical gripper, metal fingers, machinery, extra hand, extra arm, extra fingers, six fingers, missing fingers, fused fingers, deformed anatomy, floating object, changing object, warped cup, disappearing object, camera movement, flickering, blurry hand, subtitles, text, watermark'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_video(path,frames,fps=16):
 h,w=frames.shape[1:3]
 subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],input=frames.tobytes(),check=True)
PROMPTS.update({
 'bridge_bridge_041207':'One adult bare human hand reaches down, grasps the red cylindrical can naturally with the thumb opposed to curved fingers, and moves it above the toy fish and blue object on the cooktop.',
 'bridge_bridge_015993':'One adult bare human hand reaches down, holds the rim of the small blue bowl, and lifts the bowl from the sink ledge with a natural stable wrist.',
 'bridge_bridge_040489':'One adult bare human hand reaches toward the metal bowl, grasps its rim, lifts and places it onto the purple towel on the wooden countertop.'
})
def run(a):
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
 model_name='Wan2.1-VACE-'+a.model;checkpoint=ROOT/'models'/model_name
 # Official native checkpoints contain fp32 weights. Load the large DIT in bf16,
 # but preserve the documented fp32 timestep embedding/projection path.
 original=VaceWanModel.from_pretrained
 VaceWanModel.from_pretrained=classmethod(lambda cls,*args,**kw:original(*args,**dict(kw,torch_dtype=torch.bfloat16)))
 pipe=WanVace(WAN_CONFIGS['vace-'+a.model],str(checkpoint),device_id=0,t5_cpu=False)
 pipe.model.time_embedding.float();pipe.model.time_projection.float()
 rows=json.loads((ROOT/'inputs/manifest.json').read_text())
 import itertools
 for row,seed in itertools.product(rows,a.seeds):
  a.seed=seed
  if a.only and row['id']!=a.only:continue
  folder=ROOT/'inputs'/row['id'];output=ROOT/'results'/('1.3B_'+a.variant)/row['id']/f'seed{a.seed}';output.mkdir(parents=True,exist_ok=True)
  if (output/'result.json').exists():continue
  source_data=np.load(folder/'source_frames.npz');fps=float(source_data['fps']);raw=source_data['frames'];mask=np.load(folder/'masks.npz')['edit_mask']
  if a.variant=='handroom':
   # Extra 10 native pixels plus +/-1-frame union, allowing palm/wrist room.
   grown=np.stack([cv2.dilate(x,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))) for x in mask])
   mask=np.stack([np.max(grown[max(0,i-1):min(len(grown),i+2)],axis=0) for i in range(len(grown))])
  # Preserve absolute time; repeat nearest source frames, no synthetic optical flow.
  count=min(97,4*int((len(raw)/fps*16-1)//4)+1);ts=np.arange(count)/16
  indices=np.minimum(np.floor(ts*fps+.5).astype(int),len(raw)-1)
  source=np.stack([cv2.resize(raw[i],(a.size,a.size),interpolation=cv2.INTER_CUBIC) for i in indices])
  m=np.stack([cv2.resize(mask[i],(a.size,a.size),interpolation=cv2.INTER_NEAREST) for i in indices])
  refpath=ROOT/'human_references'/(row['id']+'_human.png')
  ref=np.array(Image.open(refpath).convert('RGB').resize((a.size,a.size),Image.Resampling.LANCZOS))
  # Official VACE inpainting preprocessing replaces the editable pixels by gray.
  masked_source=source.copy();masked_source[m>127]=128
  src=torch.from_numpy(masked_source).permute(3,0,1,2).float().cuda()/127.5-1
  masks=torch.from_numpy(m.copy()).unsqueeze(0).float().cuda()/255
  reference=torch.from_numpy(ref.copy()).permute(2,0,1).unsqueeze(1).float().cuda()/127.5-1
  prompt=PROMPTS[row['id']]+' Replace the robotic manipulator with the human hand shown in the reference image. Follow the source action timing and object motion precisely. Photorealistic skin, consistent hand identity, continuous movement.'
  start=time.time();print('GENERATE',row['id'],count,a.size,a.seed,flush=True)
  with torch.inference_mode():
   video=pipe.generate(prompt,[src],[masks],[[reference]],size=(a.size,a.size),frame_num=count,shift=16,sampling_steps=a.steps,guide_scale=5.,n_prompt=NEGATIVE,seed=a.seed,offload_model=True)
  generated=((video.float().clamp(-1,1)+1)*127.5).byte().permute(1,2,3,0).cpu().numpy()
  assert generated.shape==source.shape,(generated.shape,source.shape)
  # Preserve the unmodified original outside the expanded edit mask; keep raw output too.
  alpha=np.stack([cv2.GaussianBlur(x.astype(np.float32)/255,(9,9),2) for x in m])[...,None]
  composited=np.clip(generated*alpha+source*(1-alpha),0,255).astype(np.uint8)
  write_video(output/'raw_generated.mp4',generated);write_video(output/'human.mp4',composited);write_video(output/'source_aligned.mp4',source)
  overlay=source.copy();sel=m>0;overlay[sel]=(source[sel]*.45+np.array([52,229,202])*.55).astype(np.uint8);write_video(output/'mask_overlay.mp4',overlay)
  pick=np.linspace(0,count-1,8).astype(int)
  contact=np.concatenate([np.concatenate([source[i],composited[i]],axis=0) for i in pick],axis=1)
  Image.fromarray(contact).save(output/'contact_sheet.jpg',quality=90)
  data={'id':row['id'],'variant':a.variant,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'mask_coverage':float((mask>0).mean()),'model':model_name,'seed':a.seed,'steps':a.steps,'size':a.size,'frames':count,'fps':16,'source_fps':fps,'frame_indices':indices.tolist(),'source_timestamps':ts.tolist(),'source_sha256':sha(folder/'source.mp4'),'reference_sha256':sha(refpath),'prompt':prompt,'negative_prompt':NEGATIVE,'elapsed_seconds':time.time()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,'background_composited':True,'masked_input_fill':128,'paired_alignment_verified':False,'training_eligible':False,'source_dataset':'Bridge','source_label_semantics':'realized TCP states, not native robot command actions'}
  (output/'result.json').write_text(json.dumps(data,indent=2));print('RESULT_DONE',str(output),time.time()-start,flush=True)
  del src,masks,reference,video;gc.collect();torch.cuda.empty_cache()
 (ROOT/('GENERATION_'+a.model+'_DONE.json')).write_text(json.dumps({'variant':a.variant,'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'mask_coverage':float((mask>0).mean()),'model':model_name,'seed':a.seed}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['1.3B','14B'],default='1.3B');p.add_argument('--size',type=int,default=512);p.add_argument('--steps',type=int,default=40);p.add_argument('--seed',type=int,default=2026);p.add_argument('--only');p.add_argument('--variant',choices=['baseline','handroom'],default='handroom');p.add_argument('--seeds',type=int,nargs='+',default=[2026,2027]);run(p.parse_args())
