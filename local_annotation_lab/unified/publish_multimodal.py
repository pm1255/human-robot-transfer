"""Publish only complete, decoded, structurally audited multimodal pilots."""
from pathlib import Path
import json,shutil,sys,hashlib
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper');ROOT=BASE/'outputs/unified_multimodal_20261020';SITE=BASE/'outputs/hand_mesh_repair_20261020';CODE=BASE/'code/human-robot-transfer/local_annotation_lab'
sys.path.insert(0,str(BASE/'outputs/unified_three_domain_20261020/render_dependencies'))
import markdown

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest_path=SITE/'assets/multi_manifest.json';m=json.loads(manifest_path.read_text());backup=ROOT/'manifest_before_multimodal.json'
 if not backup.exists():shutil.copy2(manifest_path,backup)
 release=[]
 for c in m['clips']:
  src=ROOT/'samples'/c['id']/'multimodal'
  if not src.exists():continue
  qc=json.loads((src/'multimodal_qc.json').read_text());v=json.loads((src/'video_validation.json').read_text());d=json.loads((src/'depth_validation.json').read_text());assert qc['structure_passed']
  for name,info in [*v.items(),('depth',d['video'])]:
   assert info['complete_decode'] and info['frames']==c['frames'] and abs(info['fps']-c['fps'])<1e-4
   assert digest(src/(name+'.mp4'))==info['sha256']
  sam=json.loads((src/'sam3_manifest.json').read_text());assert sam['source_sha256']==digest(SITE/'unified'/c['id']/'cam_0.mp4')
  review=json.loads((ROOT/'visual_review.json').read_text())['samples'][c['id']]
  target=SITE/'unified'/c['id']/'multimodal';target.mkdir(parents=True,exist_ok=True)
  for f in src.iterdir():
   if f.is_file():shutil.copy2(f,target/f.name)
  (target/'visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
  prefix=f"unified/{c['id']}/multimodal/";c['layers']=[l for l in c['layers']if l['id']not in ['objects','segmentation','bbox','depth','flow','contact','object_pose','affordance','obstacles']]
  def layer(id,title,video,description,assessment,files):
   return dict(id=id,title=title,short_title=title,video=prefix+video if video else None,badge='模型初标 · 待复核'if video else'条件待补齐',badge_tone='review',description=description,assessment=assessment,tags=['逐帧同源','未获训练准入']if video else['尚未生成'],annotation_complete=False,resources=[dict(label=label,url=prefix+name)for label,name in files])
  qnote=f"{qc['flagged_frames']} / {c['frames']} 帧有自动复核提示。提示不等于错误；遮挡时允许缺失。"+' '.join(review['observations'])
  new_start=len(c['layers'])
  c['layers'] += [
   layer('segmentation','SAM3 物体与执行器分割','segmentation.mp4','根据首帧位置提示，逐帧预测操作物体、目标物体和夹爪 / 人手轮廓。',qnote+'实例身份与分割精度尚未独立验收。',[('掩码与框记录','objects.json'),('掩码数据','masks.zip'),('逐帧质检','multimodal_qc.json'),('抽帧观察','visual_review.json')]),
   layer('bbox','物体与执行器 Bounding Box','boundingbox.mp4','框由实际分割轮廓的边界计算。模型预测实例有独立编号。','框与保存的掩码逐个一致；轮廓错误也会传递给框。',[('框与实例 JSON','objects.json')]),
   layer('depth','单目深度','depth.mp4','MoGe-3 对每帧估计深度，整段使用固定颜色范围。','相机参数与尺度由模型估计，尚未标定成真实距离；可能存在时序漂移。',[('深度检查','depth_validation.json'),('模型来源','depth_manifest.json'),('原始数组位置','raw_data_locations.json')]),
   layer('flow','二维运动 / 光流','optical_flow.mp4','颜色表示相邻原视频帧的运动方向和像素位移。第一帧没有前一帧，显示为空。','当前使用传统光流算法；包含相机运动，不能直接当作物体三维位移。',[('光流方法与统计','flow_manifest.json')]),
   layer('contact','接触证据与片段状态','contact.mp4','对照当前动作片段、物体轮廓和执行器距离，定位需要复核的接触时刻。','图中点仅为二维轮廓最近点。它不等于真实接触点或可抓取点；接触状态仍为待复核。',[('逐帧接触证据','contact_evidence.json')]),
  ]
  for item in c['layers'][new_start:]:
   if item['id'] in ['segmentation','bbox'] and review['confirmed_errors']:item['badge']='含已知问题 · 待修正'
   if item['id']=='flow':item['badge']='算法结果 · 待复核'
   if item['id']=='contact':item['badge']='证据展示 · 非接触真值'
  for id,title,reason in [('object_pose','物体 6D 位姿','需要目标几何模型、可靠深度和相机标定。关节物体与柔性物体还需独立形变表达。原生末端位姿不能替代物体位姿。'),('affordance','可操作点与可抓取区域','需要根据操作目的确认功能部位并标注；不能将轮廓中心或最近点当作可操作点。'),('obstacles','障碍物与碰撞风险','需要场景物体、运动路径和机器人几何。未标注不能解释为没有障碍或没有风险。')]:
   l=layer(id,title,None,reason,'未生成有效标签，训练有效掩码保持关闭。',[]);l.update(empty_title='此项仍需补齐证据',missing_reason=reason);c['layers'].append(l)
  c['multimodal_summary']=f"已新增 5 路初标视频：分割、框、深度、光流、接触证据。共 {c['frames']} 帧，与原视频同步。"+('分割含已知问题，详情见卡片和逐帧复核。' if review['confirmed_errors'] else '标注精度待独立验收。')+'物体 6D、可操作点与碰撞风险待补齐。'
  c['multimodal_quality']=qnote+'分割精度、实例身份、深度尺度与真实接触尚未独立验收；接触视频只展示证据，不代表有效接触真值。'
  c['multimodal']=dict(objects=prefix+'objects.json',qc=prefix+'multimodal_qc.json')
  c['sample_note']='逐片段语义、原生状态与五路多模态初标已接入。所有新增标注处于复核阶段。'
  release.append(dict(id=c['id'],frames=c['frames'],quality=qc,resources=[dict(name=f.name,bytes=f.stat().st_size,sha256=digest(f))for f in target.iterdir()if f.is_file()]))
 for f in ['index.html','app.js','styles.css']:shutil.copy2(CODE/f,SITE/f)
 reportdir=SITE/'multimodal';reportdir.mkdir(exist_ok=True)
 for name in ['multimodal_audit.json','depth_audit_summary.json','visual_review.json','browser_validation.json','transfer_manifest.json','model_provenance.json']:
  if (ROOT/name).exists():shutil.copy2(ROOT/name,reportdir/name)
 for stem in ['report_summary','vla_annotation_map']:
  path=ROOT/(stem+'.md');shutil.copy2(path,reportdir/path.name);body=markdown.markdown(path.read_text(),extensions=['tables','fenced_code'])
  css='body{font:16px/1.8 system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#21343c;background:#f8fafb}a,h1,h2,h3{color:#176c70}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd9dc;padding:10px;text-align:left}code,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap}'
  (reportdir/(stem+'.html')).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多模态标注说明</title><style>'+css+'</style><body><a href="../">← 返回工作台</a>'+body+'</body></html>')
 tmp=manifest_path.with_suffix('.tmp');tmp.write_text(json.dumps(m,ensure_ascii=False,indent=2));tmp.replace(manifest_path)
 (ROOT/'release_manifest.json').write_text(json.dumps(dict(schema='unified.multimodal_release.v1',samples=release,training_eligible=False),ensure_ascii=False,indent=2));shutil.copy2(ROOT/'release_manifest.json',reportdir/'release_manifest.json')
 print('Published',len(release),'samples,',len(release)*5,'new videos')
if __name__=='__main__':main()
