"""Publish only structurally validated, source-hash-matched model videos."""
import json,shutil,hashlib
from pathlib import Path
from batch30_progress import update
B=Path('/Users/panmiao/Documents/ChatGPT/paper');R=B/'outputs/unified_30_20261021';S=B/'outputs/hand_mesh_repair_20261020';C=B/'code/human-robot-transfer/local_annotation_lab'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def publish():
 mp=S/'assets/multi_manifest.json';m=json.loads(mp.read_text());published=[]
 for c in m['clips']:
  sid=c['id'];o=R/'samples'/sid/'multimodal'
  if sid not in json.loads((R/'sam_config.json').read_text())or not(o/'multimodal_qc.json').exists():continue
  q=json.loads((o/'multimodal_qc.json').read_text());v=json.loads((o/'video_validation.json').read_text());depth=json.loads((o/'depth_validation.json').read_text());sam=json.loads((o/'sam3_manifest.json').read_text());assert q['structure_passed']and q['frames']==c['frames'];assert sam['source_sha256']==sha(S/'unified'/sid/'cam_0.mp4')
  for name,check in [*v.items(),('depth',depth['video'])]:
   assert check['complete_decode']and check['frames']==c['frames']and abs(check['fps']-c['fps'])<.001,(sid,name,check)
   assert sha(o/(name+'.mp4'))==check['sha256']
  target=S/'unified'/sid/'multimodal';target.mkdir(exist_ok=True)
  for f in o.iterdir():
   if f.is_file()and f.suffix in ['.json','.mp4','.jpg','.zip']:shutil.copy2(f,target/f.name)
  pref=f'unified/{sid}/multimodal/'
  specs={
   'segmentation':('SAM3 物体与执行器分割','segmentation.mp4','根据已观察画面中的位置提示，逐帧追踪物体与手 / 夹爪。缺失预测保持缺失。',[('实例与掩码框记录','objects.json'),('掩码数据','masks.zip'),('逐帧质检','multimodal_qc.json'),('提示与模型来源','sam3_manifest.json')]),
   'bbox':('物体与执行器 Bounding Box','boundingbox.mp4','框取自实际 SAM3 掩码边界，分割错误也会传递给框。',[('框与实例','objects.json')]),
   'depth':('单目深度','depth.mp4','MoGe-3 逐帧估计，整段采用固定色标；未经米制标定。',[('深度时序检查','depth_validation.json'),('模型记录','depth_manifest.json'),('原始数组位置','raw_data_locations.json')]),
   'flow':('二维运动 / 光流','optical_flow.mp4','相邻源视频帧的二维运动，包含相机运动；不是物体 3D 位移。',[('光流方法与统计','flow_manifest.json')]),
   'contact':('二维接触证据','contact.mp4','显示轮廓最近点与语义片段；二维接近不能证明真实接触。',[('接触证据','contact_evidence.json')]),
  }
  review=json.loads((R/'visual_review.json').read_text())['samples'].get(sid,{})if(R/'visual_review.json').exists()else{}
  for l in c['layers']:
   if l['id']not in specs:continue
   title,video,desc,files=specs[l['id']];l.update(title=title,short_title=title,video=pref+video,badge='模型初标 · 未校准',badge_tone='review',description=desc,assessment=f'{q["flagged_frames"]}/{c["frames"]} 帧触发轮廓变化或缺失提示；自动提示不等于已确认错误。'+(' '.join(review.get('observations',[]))if l['id']in['segmentation','bbox']else'精度未独立验收。'),tags=['同源逐帧','自动证据筛查'],annotation_complete=False,resources=[dict(label=a,url=pref+b)for a,b in files]);l.pop('missing_reason',None);l.pop('empty_title',None)
   if l['id']=='contact':l['badge']='证据展示 · 非接触真值'
   if l['id']=='flow':l['badge']='算法结果 · 未校准'
   if review.get('confirmed_errors')and l['id']in['segmentation','bbox']:l['badge']='含已知问题 · 暂不采用'
  c['multimodal']=dict(objects=pref+'objects.json',qc=pref+'multimodal_qc.json');c['multimodal_summary']=f'{c["frames"]} 帧完整接入：原视频、分段语义、原生状态及分割 / 框 / 深度 / 光流 / 接触证据，各自独立视频。物体 6D、功能部位和碰撞标签仍缺证据。';c['multimodal_quality']=f'{q["flagged_frames"]}/{c["frames"]} 帧触发自动提示；分割身份、深度尺度和真实接触未获验证。'+' '.join(review.get('observations',[]))
  score=R/'confidence/samples'/f'{sid}.json'
  if score.exists():
   dest=S/'confidence'/score.name;shutil.copy2(score,dest);c['confidence']='confidence/'+score.name
  published.append(sid)
 dump(mp,m);dump(C/'assets/multi_manifest.json',m)
 for f in ['index.html','app.js','styles.css']:shutil.copy2(C/f,S/f)
 dump(R/'published_models.json',dict(samples=published,complete=len(published)==24))
 update('sam3',6+len(published),'complete'if len(published)==24 else'running')
 depths=sum(any(l['id']=='depth'and l.get('video')for l in c['layers'])for c in m['clips'])
 update('depth',depths,'complete'if depths==30 else'running')
 print('PUBLISHED_MODELS',len(published),flush=True)
if __name__=='__main__':publish()
