import os, sys, json, cv2, numpy as np, torch
# HaMeR's MANO dependency still imports NumPy 1.x aliases.  The 4090 image
# uses NumPy 2.x, so provide the aliases before smplx/chumpy are imported.
for _name, _value in {
    'bool': bool, 'int': int, 'float': float, 'complex': complex,
    'object': object, 'unicode': str, 'str': str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
HAMER_ROOT=os.environ.get('HAMER_ROOT', '/user/panmiao/tmp/humanego_upload/src/hamer-main')
sys.path.insert(0,HAMER_ROOT)
from hamer.models import load_hamer
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

INPUT=os.environ.get('HUMANEGO_INPUT','/user/panmiao/tmp/humanego_upload/humanego_serve_bread_000_6s__raw.mp4')
HAND_JSON=os.environ.get('HUMANEGO_HAND_JSON','/user/panmiao/tmp/humanego_upload/hand/humanego_serve_bread_000_6s__raw_hand.json')
OUT_DIR=os.environ.get('HAMER_OUT_DIR','/user/panmiao/tmp/humanego_upload/hamer')
os.makedirs(OUT_DIR,exist_ok=True)
stem=os.path.splitext(os.path.basename(INPUT))[0]
OUT_VIDEO=os.path.join(OUT_DIR,stem+'_hamer_mesh.mp4')
OUT_JSON=os.path.join(OUT_DIR,stem+'_hamer_mesh.json')
CK=os.environ.get('HAMER_CHECKPOINT', '/root/.cache/hamer/hamer_ckpts/checkpoints/hamer.ckpt')

def bbox_from_hand(hnd,w,h):
    pts=np.array([[p['x']*w,p['y']*h] for p in hnd['landmarks']],dtype=np.float32)
    x1,y1=pts.min(0); x2,y2=pts.max(0); bw=max(x2-x1,20); bh=max(y2-y1,20)
    pad=max(bw,bh)*0.25
    return [max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)]

print('[HaMeR] loading checkpoint',flush=True)
model,cfg=load_hamer(CK); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=model.to(device).eval(); renderer=Renderer(cfg,faces=model.mano.faces)
hand_data={x['frame_index']:x for x in json.load(open(HAND_JSON))['frames']}
cap=cv2.VideoCapture(INPUT); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
writer=cv2.VideoWriter(OUT_VIDEO,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); results=[]; idx=0; mesh_frames=0
while True:
  ok,img=cv2.VideoCapture if False else (None,None)
  ok,img=cap.read()
  if not ok: break
  clean=img.copy(); frame_res={'frame_index':idx,'hands':[]}; entry=hand_data.get(idx,{'hands':[]})
  hands=entry.get('hands',[])
  if hands:
    boxes=np.array([bbox_from_hand(x,w,h) for x in hands],dtype=np.float32)
    right=np.array([1 if (x.get('handedness') or {}).get('category')=='Right' else 0 for x in hands],dtype=np.float32)
    try:
      ds=ViTDetDataset(cfg,img,boxes,right,rescale_factor=2.0)
      loader=torch.utils.data.DataLoader(ds,batch_size=len(ds),shuffle=False,num_workers=0)
      batch=recursive_to(next(iter(loader)),device)
      with torch.no_grad(): out=model(batch)
      mult=(2*batch['right']-1); pred_cam=out['pred_cam']; pred_cam[:,1]=mult*pred_cam[:,1]
      center=batch['box_center'].float(); size=batch['box_size'].float(); img_size=batch['img_size'].float(); fl=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*img_size.max()
      cam_full=cam_crop_to_full(pred_cam,center,size,img_size,fl).detach().cpu().numpy()
      verts_all=[]; cams=[]; rights=[]
      for j,hnd in enumerate(hands):
        verts=out['pred_vertices'][j].detach().cpu().numpy(); is_right=float(right[j]); verts[:,0]=(2*is_right-1)*verts[:,0]
        cam_t=cam_full[j]; verts_all.append(verts); cams.append(cam_t); rights.append(is_right)
        # HaMeR's output is the mesh itself.  Do not propagate the legacy
        # open/pinch/fist categorical heuristic from the MediaPipe sidecar.
        frame_res['hands'].append({'handedness':(hnd.get('handedness') or {}).get('category'),'vertices':verts.tolist(),'cam_t':cam_t.tolist(),'mesh_vertices':778,'model':'HaMeR'})
      rgba=renderer.render_rgba_multiple(verts_all,cam_t=cams,render_res=img_size[0],is_right=rights,mesh_base_color=(0.18,0.72,0.95),scene_bg_color=(1,1,1),focal_length=fl)
      # renderer output is RGB(A), align to BGR frame
      if rgba.shape[:2] != (h,w):
        rgba=cv2.resize(rgba,(w,h),interpolation=cv2.INTER_LINEAR)
      rgb=clean[:,:,::-1].astype(np.float32)/255.0; alpha=rgba[:,:,3:4].astype(np.float32); overlay=rgb*(1-alpha)+rgba[:,:,:3].astype(np.float32)*alpha; clean=np.clip(overlay[:,:,::-1]*255,0,255).astype(np.uint8)
      mesh_frames+=1
    except Exception as e:
      frame_res['error']=str(e)[:300]
  cv2.putText(clean,f'frame={idx} HaMeR mesh={len(frame_res["hands"])}', (5,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
  writer.write(clean); results.append(frame_res); idx+=1
  if idx%30==0: print('frame',idx,'mesh_frames',mesh_frames,flush=True)
cap.release(); writer.release()
with open(OUT_JSON,'w') as f: json.dump({'input':INPUT,'model':'HaMeR','num_frames':len(results),'mesh_frames':mesh_frames,'vertices_per_hand':778,'frames':results},f)
print(json.dumps({'video':OUT_VIDEO,'json':OUT_JSON,'frames':len(results),'mesh_frames':mesh_frames}),flush=True)
