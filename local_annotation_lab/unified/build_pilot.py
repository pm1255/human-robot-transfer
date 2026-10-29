"""Build draft pilot semantics and native-state displays without changing source data."""
from pathlib import Path
import json, math, hashlib, shutil, subprocess, io, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper')
ROOT=BASE/'outputs/unified_three_domain_20261020'; SAMPLES=ROOT/'samples'
SITE=BASE/'outputs/hand_mesh_repair_20261020'; CODE=BASE/'code/human-robot-transfer/local_annotation_lab'
FF=BASE/'outputs/unified_annotation_web_20261020/render_dependencies/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
FONT='/System/Library/Fonts/STHeiti Light.ttc'
COLORS={'approach':'#5c8cac','contact':'#c79652','align':'#9a81af','transport':'#538e88','manipulate':'#8c79a9','release':'#ae7d75','retract':'#71949b','idle':'#a3aeb4','unknown':'#8c939c'}
LABELS={'approach':'接近','contact':'建立接触','align':'对齐','transport':'搬运','manipulate':'操作','release':'释放','retract':'撤离','idle':'等待','unknown':'证据不足'}
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Each row was authored after inspecting source RGB contact sheets and available wrist views.
# End times are approximate visual event boundaries, not equal-duration cuts or task-text labels.
CONFIG={
'egodex_basic_pick_place-part1_000000':dict(title='黄色调味瓶放入容器',domain='人类 Ego',object='mustard_bottle',target='container',actor='right',step=.25,rows=[
(.65,'approach','reach','右手接近黄色调味瓶','瓶子在桌面','手接近瓶子'),
(.95,'contact','grasp','右手抓住瓶子上部','未抓持','开始抓持'),
(1.70,'transport','move','提起瓶子并移到容器上方','瓶子在桌面','瓶子悬于容器上方'),
(2.15,'align','place','将瓶子向容器内放下','瓶子悬空','瓶子接触容器底部'),
(2.4,'release','release','松开瓶子','手握瓶子','瓶子留在容器内'),
(None,'retract','withdraw','右手撤离','手靠近瓶子','手离开瓶子')]),
'egodex_add_remove_lid_000000':dict(title='依次给纸杯盖盖子',domain='人类 Ego',object='cup_lid',target='cup',actor='both',step=.5,rows=[
(.5,'approach','reach','双手分别接近杯子和杯盖','杯盖分离','手接近目标'),
(1.2,'transport','move','左手提杯，右手拿盖并靠拢','杯盖在桌面','杯盖相互接近'),
(2.1,'manipulate','close','将第一个杯盖压到杯口','杯口敞开','杯盖覆盖杯口'),
(2.7,'release','place','放下第一个带盖杯','杯被持握','杯置于桌面'),
(3.4,'transport','move','拿起第二个杯子和杯盖','杯盖分离','双手持杯盖'),
(4.7,'manipulate','close','对齐并压合第二个杯盖','杯口敞开','杯盖覆盖杯口'),
(5.2,'release','place','放下第二个带盖杯','杯被持握','杯置于桌面'),
(5.9,'transport','move','拿起第三个杯子和杯盖','杯盖分离','双手持杯盖'),
(7.2,'manipulate','close','压合第三个杯盖','杯口敞开','杯盖覆盖杯口'),
(7.6,'release','place','放下第三个带盖杯','杯被持握','杯置于桌面'),
(8.3,'transport','move','拿起第四个杯子和杯盖','杯盖分离','双手持杯盖'),
(None,'manipulate','close','压合第四个杯盖，视频结束时仍在操作','杯口敞开','盖子接近闭合，最终结果未见')]),
'abc130k_ep_101288':dict(title='夹取面包并放到餐盘',domain='ABC-130K 真机',object='bread',target='plate',actor='right',step=.5,rows=[
(1.7,'align','adjust','夹具在桌前调整开合与姿态','夹具靠近桌前','夹具转向面包'),
(2.7,'approach','reach','持夹具接近面包','面包在后方托盘','夹具位于面包旁'),
(5.5,'manipulate','grasp','夹具对准并夹取面包','面包位于托盘','面包被夹具带起'),
(6.5,'transport','move','将面包移向前方餐盘','面包离开托盘','面包到达餐盘上方'),
(7.2,'release','place','把面包放入餐盘','面包被夹持','面包留在餐盘'),
(None,'retract','withdraw','夹具撤离并回到桌前','夹具位于餐盘旁','夹具回到桌前')]),
'abc130k_ep_110624':dict(title='将螺丝刀放入蓝色盒',domain='ABC-130K 真机',object='screwdriver',target='bin',actor='right',step=.5,rows=[
(1.2,'idle','wait','双臂处于起始位置','工具在桌面','工具在桌面'),
(2.7,'approach','reach','右夹爪接近螺丝刀','工具在桌面','夹爪接近工具'),
(5.7,'contact','grasp','夹爪闭合并调整抓持','夹爪与工具分离','工具随夹爪运动'),
(7.1,'transport','move','提起工具移向蓝色盒','工具在桌面','工具在蓝色盒上方'),
(9.4,'align','place','将工具下放并调整在盒中的位置','工具悬于盒上方','工具下降至盒内'),
(9.8,'release','release','张开夹爪释放工具','工具被夹持','工具留在盒内'),
(10.8,'retract','withdraw','右臂撤回','夹爪在盒上方','右臂回到起始区'),
(None,'idle','wait','动作结束后保持','工具在盒内','工具在盒内')]),
'abc130k_ep_042125':dict(title='双臂配合擦白板',domain='ABC-130K 真机',object='eraser',target='whiteboard',actor='both',step=.5,rows=[
(1.0,'align','adjust','双臂调整起始姿态','工具和白板在桌面','双臂转向目标'),
(2.4,'approach','reach','左夹爪接近板擦','板擦在桌面','夹爪接近板擦'),
(3.1,'contact','grasp','左夹爪抓起板擦','板擦在桌面','板擦被持握'),
(4.8,'align','align','左臂送板擦到白板，右臂扶持白板','板擦与白板分离','板擦接近白板表面'),
(13.5,'manipulate','wipe','左臂往复擦拭，右臂固定白板','白板有笔迹','白板笔迹减少，清洁程度待复核'),
(14.6,'transport','move','板擦离开白板并向桌面移动','板擦贴近白板','板擦回到桌面附近'),
(15.1,'release','release','释放板擦并放开白板','板擦与白板被持握','物体留在桌面'),
(None,'retract','withdraw','双臂撤离','夹爪靠近物体','夹爪离开物体')]),
'umi_hifi_pot':dict(title='搬运锅具并放上电磁炉',domain='UMI',object='pot',target='induction_cooker',actor='left',step=.5,rows=[
(1.5,'idle','hold','左夹爪已持锅，停留观察','锅已被抓持','锅保持被抓持'),
(4.7,'transport','move','持锅向电磁炉移动','锅离开炉面','锅接近炉面上方'),
(5.85,'align','place','对齐炉面并放下锅','锅悬在炉面上方','锅落在炉面上'),
(6.65,'release','release','左夹爪张开释放锅柄','夹爪夹持锅柄','锅柄离开夹爪'),
(7.9,'retract','withdraw','左夹爪撤离锅柄','夹爪在锅柄旁','夹爪离开锅柄'),
(None,'idle','wait','放置后保持','锅位于炉面','锅位于炉面')]),
'umi_hifi_cabinet':dict(title='柜门开合与调整（原指令不完整）',domain='UMI',object='cabinet_door',target='cabinet',actor='right',step=.5,rows=[
(4.2,'approach','reach','移动到柜门附近','柜门部分敞开','操作者接近柜门'),
(7.3,'align','align','右夹爪接近并对准柜门边缘','夹爪远离门边','夹爪接近门边'),
(8.3,'contact','grasp','夹爪在门边建立接触','夹爪与门分离','夹爪接触门边'),
(10.5,'manipulate','open','向外拉开柜门','柜门开度较小','柜门开度增大'),
(11.5,'manipulate','close','将柜门向内推回','柜门开度较大','柜门开度减小'),
(16.4,'manipulate','adjust','反复调整柜门开度，未见所有柜门关闭','柜门未完全关闭','柜门仍部分敞开'),
(17.6,'release','release','夹爪松开门边','夹爪接触柜门','夹爪离开门边'),
(None,'retract','withdraw','夹爪撤离，柜门仍有敞开','夹爪在门边','夹爪撤离')],flags=['instruction_does_not_describe_all_visible_actions','task_success_not_confirmed']),
'umi_fast_sandwich':dict(title='把生菜和面包叠成三明治',domain='UMI',object='lettuce',target='bread',actor='single',step=.5,rows=[
(2.3,'approach','reach','夹爪接近生菜','生菜在盘中','夹爪接近生菜'),
(2.7,'contact','grasp','闭合夹爪夹取生菜','生菜在盘中','生菜被夹起'),
(4.5,'transport','move','把生菜移到面包上方','生菜在夹爪中','生菜到达面包上方'),
(5.2,'release','place','张开夹爪把生菜放到面包上','生菜被夹持','生菜落在面包上'),
(6.1,'approach','reach','夹爪转向面包盒','夹爪在生菜上方','夹爪接近面包片'),
(8.3,'manipulate','grasp','夹取并调整面包片','面包片在盒中','面包片被夹持'),
(10.5,'transport','move','将面包片移向生菜','面包片离开面包盒','面包片接近生菜'),
(12.1,'release','place','把面包片放在生菜上','面包片被夹持','面包片覆盖生菜'),
(None,'retract','withdraw','夹爪离开叠放区域','夹爪靠近三明治','夹爪撤离')])}

def segments(sid,c,n,fps):
 rows=[];start=0
 for i,(end,phase,verb,label,before,after) in enumerate(c['rows']):
  stop=n if end is None else round(end*fps)
  assert start<stop<=n
  obj='bread_slice' if sid=='umi_fast_sandwich' and i>=4 else c['object']
  rows.append(dict(id=f'{sid}_s{i+1:03d}',start_frame=start,end_frame_exclusive=stop,start_s=start/fps,end_s=stop/fps,phase=phase,verb=verb,label=label,object=obj,target=c['target'],actor=c['actor'],roles={'left':'support' if c['actor']=='both' and sid.startswith('egodex') else ('active' if c['actor'] in ['left','both'] else 'not_labeled'), 'right':'active' if c['actor'] in ['right','both','single'] else 'not_labeled'},state_before=before,state_after=after,outcome='not_independently_verified',contact_transition={'contact':'establish_contact','release':'break_contact','transport':'maintain_contact','approach':'remain_separated','retract':'remain_separated'}.get(phase,'unknown'),contact_evidence='visual_inference_no_force_sensor',gripper_transition='unknown',color=COLORS[phase],boundary_uncertainty_s=c['step'],boundary_window_frames=[max(0,start-round(c['step']*fps)),min(n-1,start+round(c['step']*fps))],review_required=True,review_status='draft',evidence={'method':'assistant_visual_contact_sheet_review','sampling_period_s':c['step'],'views':['cam_0']+(['cam_1'] if sid=='umi_hifi_pot' else ['cam_2'] if sid in ['umi_hifi_cabinet','abc130k_ep_101288'] else []),'note':before+' → '+after,'kinematic_events':'see state.json boundary_candidates; proposals do not prove contact'}))
  start=stop
 return rows

def native_state(p,s,n,fps):
 m=json.loads((p/'native.json').read_text());kind='egodex' if p.name.startswith('ego') else 'abc' if p.name.startswith('abc') else 'umi'
 if kind=='egodex':
  mats=np.stack([np.asarray(m['observation.state.leftHand']),np.asarray(m['observation.state.rightHand'])],1);xyz=mats[:,:,:3,3];grip=None;valid=np.isfinite(mats).all((2,3));arms=['left','right'];times=np.array(m['timestamp']);quaterr=None
  det=np.linalg.det(mats[:,:,:3,:3]);geometry=dict(rotation_det_max_error=float(abs(det-1).max()),homogeneous_bottom_row_max_error=float(abs(mats[:,:,3,:]-[0,0,0,1]).max()))
 else:
  a=np.asarray(m['state'] if kind=='abc' else m['poses']);arms=['left','right'] if a.shape[1]==14 else ['single'];xyz=np.stack([a[:,j*7:j*7+3] for j in range(len(arms))],1);q=np.stack([a[:,j*7+3:j*7+7] for j in range(len(arms))],1);valid=np.isfinite(a).all(1)[:,None]*np.ones((n,len(arms)),bool);quaterr=float(abs(np.linalg.norm(q,axis=2)-1).max());geometry={'quaternion_norm_max_error':quaterr}
  if kind=='abc':grip=np.asarray(m['source_auxiliary']['observation.state.gripper']);times=np.asarray(m['timestamp'])
  else:
   grip=np.asarray(m['gripper']);times=np.asarray(m['time']);vv=np.asarray(m.get('validHands',m['valid']),dtype=bool);valid &= vv[:,None] if vv.ndim==1 else vv
 times=times-times[0];speeds=np.zeros((n,len(arms)));delta=np.diff(times);pair=valid[1:]&valid[:-1]&(delta[:,None]>0);speeds[1:]=np.where(pair,np.linalg.norm(np.diff(xyz,axis=0),axis=2)/np.maximum(delta[:,None],1e-9),np.nan)
 def finite(v):return np.where(np.isfinite(v),v,np.nan).tolist()
 candidates=[]
 for j,arm in enumerate(arms):
  values=grip[:,j] if grip is not None else speeds[:,j]
  dif=np.abs(np.diff(values));threshold=max(float(np.nanquantile(dif,.95)),.015 if grip is not None else .03);raw=np.where(dif>threshold)[0]+1
  groups=[]
  for t in raw:
   if not groups or t-groups[-1][-1]>.25*fps:groups.append([int(t)])
   else:groups[-1].append(int(t))
  for g in groups:candidates.append({'frame':g[int(np.argmax(dif[np.array(g)-1]))],'actor':arm,'signal':'gripper_change' if grip is not None else 'wrist_speed_change','interval':[g[0],g[-1]],'semantic_decision':'proposal_only'})
 unit=('source_normalized_value' if kind=='abc' else m.get('gripperUnit')) if grip is not None else 'not_applicable'
 result=dict(schema='unified.native_state.v1',frames=n,fps=fps,actors=arms,timestamp_s=times.tolist(),position_m=xyz.tolist(),speed_m_s=np.nan_to_num(speeds,nan=0).tolist(),speed_valid=np.vstack([np.zeros((1,len(arms)),bool),pair]).tolist(),gripper=None if grip is None else grip.tolist(),gripper_unit=unit,valid=valid.tolist(),frame_convention='source reference frame retained; cross-domain axes and camera calibration not unified',physical_calibration_verified=False,source_kind=kind,boundary_candidates=sorted(candidates,key=lambda x:x['frame']),native_fields=str(p/'native.json'))
 qc=dict(frames=n,time_monotonic=bool(np.all(delta>0)),max_time_grid_error_s=float(abs(times-np.arange(n)/fps).max()),finite_position=bool(np.isfinite(xyz).all()),valid_actor_frames=int(valid.sum()),total_actor_frames=int(valid.size),invalid_actor_frames=int((~valid).sum()),max_speed_m_s=float(np.nanmax(speeds)),native_gripper_unit=unit,**geometry)
 if 'invalidFrames' in m and m['invalidFrames']<0:qc['source_summary_error']={'field':'invalidFrames','value':m['invalidFrames'],'recomputed_invalid_actor_frames':int((~valid).sum()),'handling':'source summary not used; recomputed from per-frame flags'}
 return result,qc

def read_frames(path):
 p=subprocess.run([str(FF),'-v','error','-i',str(path),'-vf','scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True);return np.frombuffer(p.stdout,np.uint8).reshape(-1,360,640,3)
def encode(path,frames,fps):
 proc=subprocess.Popen([str(FF),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(path)],stdin=subprocess.PIPE)
 for im in frames:proc.stdin.write(np.asarray(im,dtype=np.uint8).tobytes())
 proc.stdin.close();assert proc.wait()==0

def render(p,seg,state,n,fps):
 raw=read_frames(p/'cam_0.mp4');assert len(raw)==n,(p.name,len(raw),n)
 font=ImageFont.truetype(FONT,21);small=ImageFont.truetype(FONT,16)
 def semantic_frames():
  k=0
  for f,a in enumerate(raw):
   while f>=seg[k]['end_frame_exclusive']:k+=1
   s=seg[k];im=Image.new('RGB',(640,480),'#12232d');im.paste(Image.fromarray(a),(0,0));d=ImageDraw.Draw(im)
   d.text((14,368),f'{k+1:02d}  {LABELS[s["phase"]]} · {s["label"]}',font=font,fill='white');d.text((14,401),f'{f/fps:.2f}s   {s["verb"]} · {s["object"]} → {s["target"]}',font=small,fill='#c3d4dc');d.text((14,428),f'语义初标 · 边界约 ±{s["boundary_uncertainty_s"]}s · 待独立复核',font=small,fill='#e8c086')
   for t in seg:d.rectangle((int(t['start_frame']/n*640),462,int(t['end_frame_exclusive']/n*640),479),fill=t['color'])
   x=round(f/max(n-1,1)*639);d.line((x,460,x,479),fill='white',width=3);yield im
 encode(p/'semantics.mp4',semantic_frames(),fps)
 xy=np.asarray(state['position_m'],dtype=float);v=np.asarray(state['speed_m_s'],dtype=float);g=None if state['gripper'] is None else np.asarray(state['gripper'],dtype=float);actors=state['actors'];colors=['#176c70','#b66b50']
 valid=np.asarray(state['valid'],dtype=bool);speed_valid=np.asarray(state['speed_valid'],dtype=bool)
 xy=np.where(valid[:,:,None],xy,np.nan);v=np.where(speed_valid,v,np.nan)
 if g is not None:g=np.where(valid,g,np.nan)
 background=Image.new('RGB',(640,480),'#f5f7f8');d=ImageDraw.Draw(background);d.text((20,12),'原生位姿与运动信号',font=font,fill='#20323c');d.text((20,44),'保留源坐标系 · 未统一物理轴 · 非图像投影',font=small,fill='#647680')
 series=[(v,'末端速度 (m/s)',95,210),(g if g is not None else xy[:,:,2],f'夹爪 ({state["gripper_unit"]})' if g is not None else '手腕 Z (源记录 m)',265,380)]
 for arr,label,top,bottom in series:
  d.text((20,top-24),label,font=small,fill='#20323c');finite=arr[np.isfinite(arr)];lo=float(finite.min())if len(finite)else 0;hi=float(finite.max())if len(finite)else 1;span=max(hi-lo,1e-6)
  d.line((45,top,45,bottom,620,bottom),fill='#c4d2d7',width=1)
  for j in range(len(actors)):
   pts=[]
   for i in range(n):
    if np.isfinite(arr[i,j]):pts.append((45+i/(n-1)*575,bottom-(float(arr[i,j])-lo)/span*(bottom-top)))
    else:
     if len(pts)>1:d.line(pts,fill=colors[j],width=2)
     pts=[]
   if len(pts)>1:d.line(pts,fill=colors[j],width=2)
 def state_frames():
  for f in range(n):
   im=background.copy();d=ImageDraw.Draw(im);x=45+f/(n-1)*575
   for _,_,t,b in series:d.line((x,t,x,b),fill='#d1483e',width=2)
   for j,arm in enumerate(actors):d.text((20+j*305,403),f'{arm}: {v[f,j]:.3f} m/s'if np.isfinite(v[f,j])else f'{arm}: 证据不足',font=small,fill=colors[j])
   d.text((20,434),f'帧 {f}/{n-1} · {f/fps:.2f}s · 夹爪变化仅提供边界线索',font=small,fill='#647680');yield im
 encode(p/'state.mp4',state_frames(),fps)
 raw_image=Image.fromarray(raw[min(n-1,round(n*.35))]);raw_image.save(p/'poster.jpg',quality=88)
 return {'cam_0_decoded_frames':len(raw)}

def layer(id,title,video,description,resources=None):
 return dict(id=id,title=title,short_title=title,video=video,badge='初标 · 待复核' if id in ['semantics','state'] else '源数据',badge_tone='review' if id in ['semantics','state'] else 'neutral',description=description,tags=['统一时间轴'],assessment='可回放，标注精度尚未独立验收。',resources=resources or [],annotation_complete=False)

def main():
 backup=ROOT/'manifest_before.json'
 if not backup.exists():shutil.copy2(SITE/'assets/multi_manifest.json',backup)
 manifest=json.loads(backup.read_text());old=manifest['clips'][0];old['dataset']='人类 Ego';old['domain']='human';old['label']='EgoVerse · 折叠衣物（原样本）'
 oldsem=json.loads((SITE/old['semantics']).read_text());oldsem['schema']='unified.temporal_semantics.v1';oldsem['taxonomy_version']='1.0';oldsem['source_episode']=old['id'];oldsem['training_eligible']=False
 oldsem['source_phase_definitions']=oldsem.get('source_phase_definitions',oldsem.get('phases',[]));oldsem['phases']=[{'id':p,'label':LABELS[p],'color':COLORS[p]} for p in COLORS]
 for s in oldsem['segments']:
  original=s.get('source_phase',s['phase']);s['source_phase']=original;s['phase']='unknown' if original=='reset' else 'align' if original=='align_edge' else 'manipulate';s['verb']={'smooth':'smooth','align_edge':'align','fold_first':'fold','fold_again':'fold','fold_side':'fold','press':'press','reset':'unknown'}[original];s.update(object='cloth',target='cloth',actor='both',roles={'left':'active','right':'active'},state_before='unknown',state_after='unknown',outcome='not_independently_verified',contact_transition='unknown',gripper_transition='not_applicable',review_status='draft',evidence={'method':'prior_assistant_visual_review_2hz','note':'沿用已观察的90段边界；本轮仅映射统一词表，不追加未观察状态'});s['color']=COLORS[s['phase']]
 dump(SITE/'unified/egoverse/segments.json',oldsem);old['semantics']='unified/egoverse/segments.json';old['semantic_protocol']='原样本沿用 90 段视觉初标；统一映射动作词表。边界约 ±0.5 秒，独立复核等待中。'
 for l in old['layers']:
  for resource in l.get('resources',[]):
   if resource.get('label')=='分段数据':resource['url']=old['semantics']
 clips=[old];audits=[]
 for sid,c in CONFIG.items():
  p=SAMPLES/sid;s=json.loads((p/'source.json').read_text());fps=float(s.get('fps',30));n=s.get('frames',s.get('num_frames',0))
  if sid.startswith('egodex'):n=s['episode']['length'];fps=s['info']['fps']
  seg=segments(sid,c,n,fps);sem=dict(schema='unified.temporal_semantics.v1',taxonomy_version='1.0',episode_id=sid,frames=n,fps=fps,task_instruction=s.get('task',s.get('episode',{}).get('tasks') if isinstance(s.get('episode'),dict) else None),annotation_method='assistant_visual_review_with_native_signal_proposals',independent_human_verification=False,training_eligible=False,segments=seg,flags=c.get('flags',[]),source_video_sha256=sha(p/'cam_0.mp4'))
  dump(p/'segments.json',sem);state,qc=native_state(p,s,n,fps);dump(p/'state.json',state);qc.update(render(p,seg,state,n,fps))
  qc['semantic_segments']=len(seg);qc['semantic_structure']='contiguous_complete_half_open_intervals';qc['independent_semantic_acceptance']='waiting';qc['geometry_accuracy']='not_independently_verified';qc['sync_status']='frame_count_and_source_timestamps_checked_visual_lag_not_measured';qc['flags']=c.get('flags',[]);dump(p/'qc.json',qc);audits.append(dict(id=sid,**qc))
  dest=SITE/'unified'/sid;dest.mkdir(parents=True,exist_ok=True)
  for f in p.iterdir():shutil.copy2(f,dest/f.name)
  pref=f'unified/{sid}/';cams=sorted(p.glob('cam_*.mp4'));labels=s.get('videoLabels') or (['外部','左腕','右腕'] if sid.startswith('abc') else ['头部'])
  layers=[layer('source',f'原视频 · {labels[0]}',pref+'cam_0.mp4','保留原始动作；预览按源帧率导出，帧号从 0 开始。'),layer('semantics','片段语义',pref+'semantics.mp4','依据可见动作与物体状态变化分段，原任务指令另行保留。',[{'label':'分段 JSON','url':pref+'segments.json'}]),layer('state','手腕 / 末端与夹爪时序',pref+'state.mp4','使用数据集原生记录；曲线不等于已校准到图像的轨迹。',[{'label':'原生记录','url':pref+'native.json'},{'label':'统一时序','url':pref+'state.json'}])]
  for j,f in enumerate(cams[1:],1):layers.append(layer('camera'+str(j),f'原视频 · {labels[j] if j<len(labels) else "视角 "+str(j)}',pref+f.name,'同一 episode 的同步相机，可用于复核接触和遮挡。'))
  layers.append(dict(id='objects',title='物体分割 / 框 / 6D',short_title='待补标注',badge='待标注',badge_tone='neutral',video=None,empty_title='尚未运行此样本的物体标注',missing_reason='本批先完成片段语义与原生动作接入；分割、框和物体位姿需要逐样本生成和质检。',description='明确保留缺失状态，不能将末端位姿当作物体位姿。',tags=['等待'],assessment='未检测，不计为检测失败。',annotation_complete=False))
  clips.append(dict(id=sid,dataset=c['domain'],label=c['title']+' · '+sid,domain='human' if c['domain']=='人类 Ego' else 'robot' if sid.startswith('abc') else 'umi',frames=n,fps=fps,width=640,height=360,initial_frame=0,semantics=pref+'segments.json',state=pref+'state.json',qc=pref+'qc.json',provenance=pref+'source.json',layers=layers,loose_verified=False,strict_verified=False,semantic_protocol=f'按可见接触与物体状态变化分段，抽帧间隔 {c["step"]} 秒；结合原生夹爪/手腕变化提供边界候选。独立复核等待中。',sample_note='语义初标与原生运动信号已接入；相机同步精度、接触和物理坐标尚待独立复核。'+(' 原指令未覆盖全部可见动作。' if c.get('flags') else ''),native_summary=f'{len(state["actors"])} 路末端记录 · 有效 {qc["valid_actor_frames"]}/{qc["total_actor_frames"]} 个末端帧 · {len(seg)} 个动作片段',task_instruction=sem['task_instruction']))
 manifest.update(schema='human_robot.annotation_lab.v2',clips=clips,taxonomy='unified/taxonomy.json',qc_plan='unified/quality_plan.html',pilot_report='unified/report_summary.md')
 dump(SITE/'assets/multi_manifest.json',manifest);dump(CODE/'assets/multi_manifest.json',manifest)
 dump(ROOT/'pilot_audit.json',{'episodes':len(clips),'new_episode_audits':audits,'existing_episode':{'id':old['id'],'frames':old['frames'],'segments':len(oldsem['segments']),'new_full_decode':False,'reason':'previous complete audit retained'},'total_segments':sum(a['semantic_segments']for a in audits)+len(oldsem['segments']),'independent_review':'waiting','training_eligible_episodes':0})
 taxonomy=dict(version='1.0',phases={p:{'label':LABELS[p],'color':COLORS[p],'vla_delta_compatible':p not in ['idle','unknown']}for p in COLORS},verbs=sorted({s['verb'] for p in SAMPLES.iterdir() for s in json.loads((p/'segments.json').read_text())['segments']}|{s['verb'] for s in oldsem['segments']}),interval_convention='start inclusive, end exclusive',unknown_policy='null / unknown with valid_mask=0; never zero-fill unavailable geometry',reference_commit='6b54a95580729bca1726d9c3fdfebf7727435729')
 dump(SITE/'unified/taxonomy.json',taxonomy);dump(ROOT/'taxonomy.json',taxonomy)
 print('BUILT',len(clips),'episodes',json.loads((ROOT/'pilot_audit.json').read_text())['total_segments'],'segments',flush=True)
if __name__=='__main__':main()
