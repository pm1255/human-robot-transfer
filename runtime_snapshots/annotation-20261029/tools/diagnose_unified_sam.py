import json,cv2,numpy as np,torch
from pathlib import Path
from types import MethodType
from sam3.model_builder import build_sam3_predictor
from human2robot.object_annotation.sam3_compat import install_session_adapter
from human2robot.object_annotation.unified_multimodal import GUIDES,as_numpy
R=Path('/user/panmiao/human-robot-transfer/outputs/unified_multimodal_20261020');src=R/'samples/umi_hifi_pot/multimodal_preflight/rgb';dest=R/'diagnostic';dest.mkdir(exist_ok=True)
for rope in [True,False]:
 p=build_sam3_predictor(checkpoint_path='/mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt',version='sam3.1',compile=False,warm_up=False,use_fa3=False,use_rope_real=rope,async_loading_frames=False);install_session_adapter(p)
 orig=p.model._propogate_tracker_one_frame_local_gpu
 def wrapped(self,*args,**kwargs):
  ids,masks,scores=orig(*args,**kwargs);print('RAW',rope,kwargs['frame_idx'],ids,float(masks.min()),float(masks.max()),scores.flatten().tolist(),flush=True);return ids,masks,scores
 p.model._propogate_tracker_one_frame_local_gpu=MethodType(wrapped,p.model)
 sid=p.handle_request(dict(type='start_session',resource_path=str(src),offload_video_to_cpu=True))['session_id'];b=GUIDES['umi_hifi_pot']['pot'][0]
 result=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,points=[b[:2],b[2:]],point_labels=[2,3],obj_id=1));print('HISTORY',p._all_inference_states[sid]['state']['action_history'],flush=True)
 records=[]
 for r in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,start_frame_index=0,max_frame_num_to_track=10,propagation_direction='forward',evict_cached_frame_outputs=False)):
  i=r['frame_index'];m=as_numpy(r['outputs']['out_binary_masks']);records.append(dict(frame=i,area=int(m.sum()),ids=as_numpy(r['outputs']['out_obj_ids']).tolist()));im=cv2.imread(str(src/f'{i:06d}.jpg'));mask=np.any(m,axis=0)if len(m)else np.zeros(im.shape[:2],bool);im[mask]=im[mask]*.5+np.array([0,255,0])*.5;cv2.imwrite(str(dest/f'rope{rope}_{i}.jpg'),im);print('OUTPUT',records[-1],flush=True)
 (dest/f'rope{rope}.json').write_text(json.dumps(records));p.handle_request(dict(type='close_session',session_id=sid));del p;torch.cuda.empty_cache()
