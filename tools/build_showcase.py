from pathlib import Path
import json,subprocess,shutil,cv2,numpy as np
src=Path('/Users/panmiao/Documents/Codex/2026-09-22/https-github-com-lightorigins-light-o1/human2robot/web')
dst=Path('outputs/robot-to-human/site/dist');assets=dst/'showcase';assets.mkdir(exist_ok=True)
manifest=json.loads((src/'v2data/manifest.json').read_text());items=[]
handedges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
bodyedges=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
chosen=[]
for view,count in [('egocentric',24),('exocentric',12)]:
 rows=[r for r in manifest if r['view']==view]
 if not rows and view=='exocentric':rows=[r for r in manifest if r['view']!='egocentric']
 rows.sort(key=lambda r:(r.get('any_hand_frames',0) if view=='egocentric' else r.get('full_body_frames',0))/max(1,r['frames']))
 chosen += [rows[i] for i in np.linspace(0,len(rows)-1,count).round().astype(int)]
for n,m in enumerate(chosen):
 id=m['id'];a=json.loads((src/'v2data/annotations'/f'{id}.json').read_text());cap=cv2.VideoCapture(str(src/'v2data/media'/f'{id}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
 for f in a['frames'][:50]:
  ok,im=cap.read()
  if not ok:break
  im=cv2.resize(im,(480,360));h,w=im.shape[:2]
  for hand in f.get('hands',[]):
   pts=np.asarray(hand['xy'])*[w,h];col=(90,220,130) if hand['side']=='Left' else (240,160,75)
   for u,v in handedges:cv2.line(im,tuple(pts[u].astype(int)),tuple(pts[v].astype(int)),col,2,cv2.LINE_AA)
   for pt in pts:cv2.circle(im,tuple(pt.astype(int)),2,(250,250,250),-1)
  b=np.asarray(f.get('body',[]))
  if b.shape==(33,4):
   for u,v in bodyedges:
    if min(b[u,2:4].min(),b[v,2:4].min())>.6:cv2.line(im,tuple((b[u,:2]*[w,h]).astype(int)),tuple((b[v,:2]*[w,h]).astype(int)),(160,220,90),2)
  for obj in f.get('objects',[]):
   if obj['label']=='person':continue
   x,y,bw,bh=obj['box'];cv2.rectangle(im,(int(x*w),int(y*h)),(int((x+bw)*w),int((y+bh)*h)),(60,210,255),1)
  cv2.rectangle(im,(0,0),(480,23),(23,28,34),-1);cv2.putText(im,f"MODEL ANNOTATION | {id}",(8,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,245,245),1,cv2.LINE_AA)
  frames.append(im)
 cap.release()
 assert frames,id
 poster=assets/f'{id}.jpg';cv2.imwrite(str(poster),frames[len(frames)//2]);video=assets/f'{id}.mp4'
 proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','480x360','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','28','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 proc.communicate(np.stack(frames).tobytes());assert proc.returncode==0
 cov=m['any_hand_frames']/max(1,m['frames']);bcov=m.get('full_body_frames',0)/max(1,m['frames'])
 items.append(dict(id='ann_'+id,sourceId=id,category='annotation',title=('双手与物体标注' if m['view']=='egocentric' else '全身标注')+' · '+id,video='showcase/'+video.name,poster='showcase/'+poster.name,status='模型标注 · 非人工真值',note=f"整段手部检出 {cov:.0%}，全身检出 {bcov:.0%}。检出率不是准确率。本卡展示开头 {len(frames)/fps:.1f} 秒；漏检帧保留，不补造关键点。",source='EPIC-KITCHENS' if id.startswith('epic') else 'HMDB51',kind='annotation'))
 print('annotation',n+1,id,flush=True)
for row in json.loads((src/'v2data/conversion/manifest.json').read_text()):
 id=row['id'];video=assets/f'convert_{id}.mp4';poster=assets/f'convert_{id}.jpg'
 subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(src/row['video']),'-vf','scale=960:-2','-c:v','libx264','-crf','28','-an','-movflags','+faststart',str(video)],check=True)
 shutil.copy2(src/row['poster'],poster)
 cat='arm' if row['kind']=='hand' else 'humanoid'
 items.append(dict(id='conv_'+id,sourceId=id,category=cat,title=('人手 → Aloha 机械臂' if cat=='arm' else '人体 → G1 机器人')+' · '+id,video='showcase/'+video.name,poster='showcase/'+poster.name,status='几何合成预览 · 待改进',note=f"原图 / 删除掩码 / 机器人渲染三栏对照。{row['robot_frames']}/{row['frames']} 帧获得机器人拟合，不等于动作成功率。逐帧 Telea 补背景，未校准尺度、遮挡和接触。",source='EPIC-KITCHENS' if id.startswith('epic') else 'HMDB51',kind='render'))
r2h=json.loads((dst/'examples.json').read_text())['examples']
for row in r2h:
 for key,title in [('baseline','原版 1.3B'),('larger','14B 对照'),('optimized','扩大掩码 A'),('alternate','扩大掩码 B')]:
  if row.get(key):items.append(dict(id=row['id']+'_'+key,sourceId=row['id'],category='robot2human',title=row['title']+' · '+title,video=row[key],poster=row.get('poster',''),status='AI 视频编辑 · 待验证',note=row['review'],source='BridgeData V2',detail='compare.html?scene='+row['id'],kind='generation'))
(dst/'showcase.json').write_text(json.dumps({'items':items,'counts':{'results':len(items),'sources':len(set(x['sourceId'] for x in items))}},ensure_ascii=False,indent=2))
print('TOTAL',len(items),'UNIQUE',len(set(x['sourceId'] for x in items)))
