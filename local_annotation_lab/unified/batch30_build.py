"""Build the selected 30-episode page, with explicit modality evidence and gaps."""
import json,sys,shutil,subprocess,hashlib,copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cv2
import build_pilot as old
from batch30_semantics import CONFIG,refine_objects
from batch30_progress import update
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper');ROOT=BASE/'outputs/unified_30_20261021';SITE=BASE/'outputs/hand_mesh_repair_20261020';CODE=BASE/'code/human-robot-transfer/local_annotation_lab'
def dump(p,d):old.dump(p,d)
def ego_state(p,n,fps):
 d=json.loads((p/'native.json').read_text());a=np.array(d['observation.state'],float);t=np.array(d['timestamp']);xyz=np.stack([a[:,14:17],a[:,77:80]],1);valid=np.stack([d['left_hand_valid'],d['right_hand_valid']],1)&np.array(d['frame_valid'])[:,None]&np.array(d['in_valid_range'])[:,None]&np.isfinite(xyz).all(2)
 delta=np.diff(t);pair=valid[1:]&valid[:-1]&(delta[:,None]>0);speed=np.zeros((n,2));speed[1:]=np.where(pair,np.linalg.norm(np.diff(xyz,axis=0),axis=2)/delta[:,None],0)
 result=dict(schema='unified.native_state.v1',frames=n,fps=fps,actors=['left','right'],timestamp_s=(t-t[0]).tolist(),position_m=np.where(np.isfinite(xyz),xyz,np.nan).tolist(),speed_m_s=speed.tolist(),speed_valid=np.vstack([np.zeros((1,2),bool),pair]).tolist(),gripper=None,gripper_unit='not_applicable',valid=valid.tolist(),source_kind='egosuite',frame_convention='native world coordinate frame, metres, xyzw wrist quaternion; source camera calibration used for projections',physical_calibration_verified=False,boundary_candidates=[],native_fields='native.json')
 def nulls(x):
  if isinstance(x,list):return [nulls(v)for v in x]
  if isinstance(x,float)and not np.isfinite(x):return None
  return x
 result['position_m']=nulls(result['position_m'])
 qc=dict(frames=n,time_monotonic=bool(np.all(delta>0)),max_time_grid_error_s=float(abs(t-t[0]-np.arange(n)/fps).max()),finite_position=bool(np.isfinite(xyz).all()),valid_actor_frames=int(valid.sum()),total_actor_frames=valid.size,invalid_actor_frames=int((~valid).sum()),max_speed_m_s=float(np.nanmax(speed)),native_gripper_unit='not_applicable')
 return result,qc
def hand_video(p,fps,n):
 target=p/'hands.mp4'
 if target.exists():return
 data=json.loads((p/'hand_projection.json').read_text());uv=np.asarray(data['uv_normalized_depth_m'],float);valid=np.asarray(data['valid'],bool);cap=cv2.VideoCapture(str(p/'cam_0.mp4'));w,h=int(cap.get(3)),int(cap.get(4));edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
 proc=subprocess.Popen([str(old.FF),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(target)],stdin=subprocess.PIPE)
 font=ImageFont.truetype(old.FONT,17)
 for i in range(n):
  ok,a=cap.read();assert ok,(p,i);im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
  for side in range(2):
   if not valid[i,side]:continue
   pts=uv[i,side];visible=np.isfinite(pts).all(1)&(pts[:,2]>.03)&(pts[:,0]>=0)&(pts[:,0]<1)&(pts[:,1]>=0)&(pts[:,1]<1);xy=pts[:,:2]*[w,h];color='#48e1ce'if side==0 else'#ffb26a'
   for l,r in edges:
    if visible[l]and visible[r]:d.line([tuple(xy[l]),tuple(xy[r])],fill=color,width=2)
   for j in np.flatnonzero(visible):x,y=xy[j];d.ellipse((x-2,y-2,x+2,y+2),fill=color)
  d.rectangle((0,0,w,29),fill='#12232d');d.text((8,4),f'原生 3D 关节投影 · 帧 {i} · 有效性保留 · 非手型网格真值',font=font,fill='white');proc.stdin.write(np.asarray(im,np.uint8).tobytes())
 cap.release();proc.stdin.close();assert proc.wait()==0
def cleaner_summary(p):
 d=json.loads((p/'cleaner_qc.json').read_text());score=100*d['state']['score'];b2=d['b2'];usable=[v for v in b2.values()if v.get('compared_points',0)>0];pck=min([x['pck_0.2']for x in usable],default=None);coverage=min([x['detection_recall']for x in usable],default=0)
 reason=[]
 if d['state']['decision']!='pass':reason.append('原生手部有效帧比例偏低')
 if pck is None:reason.append('图像对齐证据不足')
 elif pck<.8:reason.append('抽检图像与原生关节的一致性偏弱')
 if coverage<.6:reason.append('独立模型可比对手部数量不足')
 time=json.loads((p/'time_audit.json').read_text());changed=abs(time['episode_timestamp_fps']-time['dataset_nominal_fps'])>.1
 if changed:reason.append('已按本 episode 原生时间间隔修正目录帧率')
 return dict(state_score=round(score,1),pck20=None if pck is None else round(100*pck,1),match_coverage=round(100*coverage,1),notes=reason,decision=d['state']['decision'],time_fixed=changed)
def build():
 mp=SITE/'assets/multi_manifest.json';backup=ROOT/'manifest_before_batch30.json'
 if not backup.exists():shutil.copy2(mp,backup)
 before=json.loads(backup.read_text());oldclips={c['id']:c for c in before['clips']};selection=json.loads((ROOT/'selection_manifest.json').read_text())['samples'];clips=[];audits=[]
 for number,item in enumerate(selection,1):
  sid=item['id'];p=ROOT/'samples'/sid;s=json.loads((p/'source.json').read_text());fps=float(s['fps']);n=s.get('frames',s.get('num_frames'));pref=f'unified/{sid}/'
  if sid not in CONFIG:
   c=copy.deepcopy(oldclips[sid]);clips.append(c);audits.append(dict(id=sid,reused=True,frames=n));continue
  title,obj,target,uncertainty,rows=CONFIG[sid];cfg=dict(object=obj,target=target,actor='both',step=uncertainty,rows=[(end,phase,verb,label,'unknown','unknown')for end,phase,verb,label in rows]);seg=old.segments(sid,cfg,n,fps)
  for z in seg:
   z['contact_transition']='unknown';z['contact_evidence']='not_measured';z['state_before']='unknown';z['state_after']='unknown';z['evidence']['method']='assistant_visual_sampled_review_initial';z['evidence']['sampling_period_s']=uncertainty;z['evidence']['note']='边界由可见动作变化估计，接触与状态未获独立验证';z['annotation_unit']='observable_action_phase'
  refine_objects(sid,seg)
  sem=dict(schema='unified.temporal_semantics.v1',taxonomy_version='1.0',episode_id=sid,frames=n,fps=fps,task_instruction=s.get('task'),annotation_method='assistant_visual_initial_from_16_sampled_frames_and_source_subtasks',training_eligible=False,independent_human_verification=False,source_video_sha256=old.sha(p/'cam_0.mp4'),segments=seg,source_subtasks=s.get('episode',{}).get('sub_tasks',[]) if isinstance(s.get('episode'),dict)else[])
  if sid=='egosuite_p05_ep_033793':sem['source_conflict']='源描述称 180–312 帧擦拭床头柜下地面；抽帧中该区间抹布已回到托盘附近，按当前可见动作标注，原描述保留。'
  dump(p/'segments.json',sem)
  state,qc=ego_state(p,n,fps)if sid.startswith('egosuite')else old.native_state(p,s,n,fps);dump(p/'state.json',state)
  if not(p/'semantics.mp4').exists():
   old.render(p,seg,state,n,fps)
  qc.update(semantic_segments=len(seg),semantic_structure='contiguous_half_open_complete',independent_semantic_acceptance='unverified',training_eligible=False,sync_status='ordinal_frames_and_native_time_grid_checked; image_motion_lag_not_measured');dump(p/'qc.json',qc)
  cap=cv2.VideoCapture(str(p/'cam_0.mp4'));w,h=int(cap.get(3)),int(cap.get(4));cap.release();labels=s.get('videoLabels')or['外部','左腕','右腕'];layers=[old.layer('source','原视频 · '+labels[0],pref+'cam_0.mp4','同一 episode 全部帧，按原生时间轴播放。'),old.layer('semantics','片段语义',pref+'semantics.mp4','统一动作阶段词表，随动作变化分段；保留源描述及不确定边界。',[dict(label='分段 JSON',url=pref+'segments.json')]),old.layer('state','手腕 / 末端与夹爪时序',pref+'state.mp4','源坐标系与原生有效性保留；不等于物体 6D 位姿。',[dict(label='原生记录',url=pref+'native.json'),dict(label='统一时序',url=pref+'state.json')])]
  if sid.startswith('egosuite'):
   hand_video(p,fps,n);layers.append(old.layer('keypoints','原生手部 3D 关节投影',pref+'hands.mp4','使用原生 21 点手部关节与逐帧相机位姿；无效手部不绘制。',[dict(label='投影记录',url=pref+'hand_projection.json')]))
  for j,f in enumerate(sorted(p.glob('cam_*.mp4'))[1:],1):layers.append(old.layer('camera'+str(j),'原视频 · '+labels[j],pref+f.name,'同一 episode 的另一相机视角。'))
  for lid,name,reason in [('segmentation','SAM3 分割','模型计算中'),('bbox','Bounding Box','等待真实分割输出'),('depth','单目深度','模型计算中'),('flow','光流','等待渲染'),('contact','接触证据','等待分割；二维接近不等于接触'),('object_pose','物体 6D 位姿','缺少物体几何与有效标定'),('affordance','可操作区域','未有可靠功能部位标签'),('obstacles','障碍与碰撞','未有完整场景与碰撞证据')]:layers.append(dict(id=lid,title=name,short_title=name,video=None,badge='等待'if lid in ['segmentation','bbox','depth','flow','contact']else'证据不足',badge_tone='review',description=reason,empty_title=reason,missing_reason=reason,assessment='未检测不算通过，也不赋零分。',tags=[],resources=[],annotation_complete=False))
  c=dict(id=sid,dataset={'human':'EgoSuite 人类 Ego','robot':'ABC-130K 真机','umi':'UMI'}[item['domain']],domain=item['domain'],label=title+' · '+sid,frames=n,fps=fps,width=w,height=h,initial_frame=0,semantics=pref+'segments.json',state=pref+'state.json',qc=pref+'qc.json',provenance=pref+'source.json',layers=layers,loose_verified=False,strict_verified=False,semantic_protocol=f'依据抽帧可见动作与源片段描述初标，边界不确定度约 ±{uncertainty} 秒；统一阶段词表，源任务不代替片段标签。',sample_note='完整 episode · 自动初标 · 未校准，尚未获得训练准入',native_summary=f'{n} 帧 · {len(seg)} 个片段 · 原生末端有效 {qc["valid_actor_frames"]}/{qc["total_actor_frames"]}',task_instruction=s.get('task'))
  if sid.startswith('egosuite'):c['cleaner_qc']=pref+'cleaner_qc.json';c['cleaner_summary']=cleaner_summary(p);c['time_audit']=pref+'time_audit.json'
  dest=SITE/'unified'/sid;dest.mkdir(parents=True,exist_ok=True)
  for f in p.iterdir():
   if f.is_file()and f.suffix in ['.json','.mp4','.jpg']:shutil.copy2(f,dest/f.name)
  clips.append(c);audits.append(dict(id=sid,**qc));update('semantics',number);print('BUILT',sid,flush=True)
 manifest=copy.deepcopy(before);manifest.update(clips=clips,pilot_report='batch30/report_summary.md',batch=dict(episodes=30,domains=dict(human=10,robot=10,umi=10),monitor='http://127.0.0.1:16069/',source_selection='batch30/selection_manifest.json'))
 assert all(sum(c['domain']==d for c in clips)==10 for d in ['human','robot','umi'])
 dump(ROOT/'manifest_stage.json',manifest);dump(ROOT/'native_audit.json',audits)
 for f in ['index.html','app.js','styles.css']:shutil.copy2(CODE/f,SITE/f)
 archive=SITE/'archives';archive.mkdir(exist_ok=True);shutil.copy2(backup,archive/'manifest_before_egosuite.json')
 dump(mp,manifest);dump(CODE/'assets/multi_manifest.json',manifest);update('semantics',30,'complete');update('native_qc',30,'complete');update('website',30,'complete')
if __name__=='__main__':build()
