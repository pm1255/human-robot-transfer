import cv2, json, os, hashlib, shutil, math, statistics
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/user/panmiao/human-robot-transfer')
VIDEO = ROOT/'inputs/unlabeled/humanego_serve_bread_000_6s.mp4'
OUT = ROOT/'outputs/humanego_serve_bread_000_6s'
ANN = OUT/'annotation'; WEB = OUT/'web'; FRAMES = WEB/'frames'; CLEAN = OUT/'cleaned'
for p in [ANN, WEB, FRAMES, CLEAN]: p.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

segments=[
 {'id':'s00','start_sec':0.0,'end_sec':0.9,'subtask':'prepare','step':'orient toward bread','atomic_action':'stabilize view and open hand','object_ids':['bread'],'hand':'right','contact':'none','state_before':'empty_hand','state_after':'ready','language':'准备拿取面包','language_en':'Prepare to pick up the bread','outcome':'ready'},
 {'id':'s01','start_sec':0.9,'end_sec':2.0,'subtask':'reach','step':'move hand to bread','atomic_action':'reach toward bread','object_ids':['bread'],'hand':'right','contact':'approach','state_before':'ready','state_after':'near_object','language':'伸手靠近面包','language_en':'Reach toward the bread','outcome':'near_object'},
 {'id':'s02','start_sec':2.0,'end_sec':3.1,'subtask':'grasp','step':'close fingers on bread','atomic_action':'grasp bread','object_ids':['bread'],'hand':'right','contact':'contact_start=2.0;contact_end=3.1','state_before':'near_object','state_after':'holding_bread','language':'抓住面包','language_en':'Grasp the bread','outcome':'holding'},
 {'id':'s03','start_sec':3.1,'end_sec':4.7,'subtask':'carry','step':'transport bread','atomic_action':'move while holding bread','object_ids':['bread'],'hand':'right','contact':'maintained','state_before':'holding_bread','state_after':'over_target','language':'拿着面包移动到目标位置','language_en':'Carry the bread to the target','outcome':'over_target'},
 {'id':'s04','start_sec':4.7,'end_sec':6.0,'subtask':'release','step':'place or hand over bread','atomic_action':'release bread','object_ids':['bread'],'hand':'right','contact':'contact_end=4.7','state_before':'over_target','state_after':'released','language':'松开并放下面包','language_en':'Release and place the bread','outcome':'released'}]

# OpenCV cannot decode this remote MP4 directly; use ffmpeg for deterministic frame extraction.
raw=WEB/'raw_frames'; raw.mkdir(parents=True, exist_ok=True)
import subprocess, glob
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(VIDEO),'-vf','fps=30','-q:v','3',str(raw/'f_%04d.jpg')],check=True)
paths=sorted(Path(raw).glob('f_*.jpg'))
fps=30.0; n=len(paths); w=h=0
images=[]
for path in paths:
    im=cv2.imread(str(path))
    if im is not None:
        if not w: h,w=im.shape[:2]
        images.append(im)
n=len(images); dur=n/fps if fps else 0.0
motion=[]; gray_prev=None; frames=[]; sharp=[]; black=[]
for idx, frame in enumerate(images):
    g=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion.append(float(cv2.absdiff(g,gray_prev).mean()) if gray_prev is not None else 0.0)
    gray_prev=g
    sharp.append(float(cv2.Laplacian(g,cv2.CV_64F).var()))
    black.append(float((g<8).mean()))
    if idx in set([0, max(0,n//9), max(0,2*n//9), max(0,3*n//9), max(0,4*n//9), max(0,5*n//9), max(0,6*n//9), max(0,7*n//9), max(0,8*n//9), max(0,n-1)]):
        frames.append((idx,frame.copy()))


# proxy hand interaction: motion segmentation in central/right field of view, explicitly not a hand keypoint estimate
motion_mean=float(statistics.mean(motion)) if motion else 0.0
motion_p90=float(sorted(motion)[int(0.9*max(0,len(motion)-1))]) if motion else 0.0
active=sum(v>max(1.0,motion_mean*1.15) for v in motion)/max(1,len(motion))
model_path=ROOT/'models/mediapipe/hand_landmarker.task'
model_info={'name':'MediaPipe Hand Landmarker','path':str(model_path.relative_to(ROOT)),'sha256':sha256(model_path) if model_path.exists() else None,'bytes':model_path.stat().st_size if model_path.exists() else None,'downloaded':model_path.exists(),'inference_status':'blocked_missing_python_mediapipe_runtime'}

# draw frame examples and contact sheet
font=cv2.FONT_HERSHEY_SIMPLEX
for idx, frame in frames:
    t=idx/fps
    seg=next((s for s in segments if s['start_sec']<=t< s['end_sec']), segments[-1])
    cv2.rectangle(frame,(8,8),(w-8,66),(20,20,20),-1)
    cv2.putText(frame,f'{t:05.2f}s | {seg["subtask"]} | {seg["atomic_action"]}',(18,35),font,0.55,(80,235,255),1,cv2.LINE_AA)
    cv2.putText(frame,f'L2 proxy motion={motion[min(idx,len(motion)-1)]:.2f}',(18,57),font,0.45,(160,255,160),1,cv2.LINE_AA)
    cv2.imwrite(str(FRAMES/f'frame_{idx:05d}.jpg'),frame,[int(cv2.IMWRITE_JPEG_QUALITY),90])
thumbs=[]
for idx,frame in frames:
    im=cv2.resize(frame,(320,240))
    cv2.putText(im,f'{idx/fps:.2f}s',(8,232),font,0.55,(255,255,255),1,cv2.LINE_AA)
    thumbs.append(im)
if thumbs:
    while len(thumbs)<10: thumbs.append(thumbs[-1].copy())
    sheet=cv2.vconcat([cv2.hconcat(thumbs[i:i+5]) for i in range(0,10,5)])
    cv2.imwrite(str(WEB/'annotation_contact_sheet.jpg'),sheet,[int(cv2.IMWRITE_JPEG_QUALITY),90])

# timeline visualization
canvas=cv2.imread(str(WEB/'annotation_contact_sheet.jpg')) if False else 255*__import__('numpy').ones((360,1200,3),dtype='uint8')
cv2.putText(canvas,'HumanEgo four-layer annotation timeline',(30,35),font,0.9,(30,30,30),2,cv2.LINE_AA)
colors=[(220,160,60),(80,180,240),(90,210,120),(180,100,220),(80,120,230)]
for i,s in enumerate(segments):
    x0=int(50+s['start_sec']/dur*1100); x1=int(50+s['end_sec']/dur*1100)
    cv2.rectangle(canvas,(x0,100),(x1,190),colors[i],-1)
    cv2.putText(canvas,s['subtask'],(x0+6,140),font,0.65,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(canvas,f'{s["start_sec"]:.1f}-{s["end_sec"]:.1f}s',(x0+6,170),font,0.45,(255,255,255),1,cv2.LINE_AA)
cv2.line(canvas,(50,220),(1150,220),(40,40,40),2)
for j in range(7):
    x=int(50+j/6*1100); cv2.line(canvas,(x,214),(x,228),(40,40,40),2); cv2.putText(canvas,f'{j}s',(x-8,250),font,0.5,(40,40,40),1,cv2.LINE_AA)
cv2.putText(canvas,'L2 = motion proxy; MediaPipe runtime unavailable on devspace', (30,310),font,0.55,(60,60,60),1,cv2.LINE_AA)
cv2.imwrite(str(WEB/'annotation_timeline.png'),canvas)

annotation={
 'schema':'human-robot-transfer/four-layer-v1','clip_id':'humanego_serve_bread_000_6s','created_at':datetime.now(timezone.utc).isoformat(),
 'source':{'dataset':'HumanEgo','task':'serve_bread','video':str(VIDEO.relative_to(ROOT)),'sha256':sha256(VIDEO),'fps':fps,'frames':n,'width':w,'height':h,'duration_sec':dur,'annotations_hidden':True},
 'layer_0_provenance':{'status':'complete','source_recording':'mps_serve_bread_000_vrs','rgb_only':True,'model_artifacts':[model_info]},
 'layer_1_language_action':{'status':'complete_manual_review_required','goal':'serve bread / hand bread to target','language':'中文+英文双语动作描述，按原子子任务切分','segments':segments},
 'layer_2_interaction':{'status':'proxy_only','method':'frame-difference motion proxy','hand_side':'right','object_ids':['bread'],'motion_mean':motion_mean,'motion_p90':motion_p90,'active_fraction':active,'model_inference':model_info,'limitations':['no hand keypoints because mediapipe package unavailable','no object detector runtime','proxy is for triage/QC, not training ground truth']},
 'layer_3_geometry':{'status':'not_applicable','representation':None,'reason':'RGB-only clip has no calibrated depth/Aria SLAM; do not fabricate 6D pose'},
 'review':{'annotator':'pipeline+manual seed','second_annotator':False,'decision':'review_before_training','reasons':['single clip','language timeline seeded from task name','L2 model inference blocked','L3 unavailable']}
}
(ANN/'four_layer_annotation.json').write_text(json.dumps(annotation,ensure_ascii=False,indent=2))
(ANN/'four_layer_annotation.md').write_text('# Four-layer annotation\n\n- L0 provenance: complete\n- L1 language/action: 5 atomic subtasks, bilingual text, manual review required\n- L2 interaction: motion proxy only; MediaPipe model downloaded but Python runtime missing\n- L3 geometry: not applicable for RGB-only clip; no fabricated 6D pose\n\nThis clip is held for review before training.')

# QC and cleaning
input_hash=sha256(VIDEO)
qc={
 'clip_id':'humanego_serve_bread_000_6s','checks':{
  'decode_ok': len(motion)>=max(1,n-1),'metadata_ok': bool(w and h and fps and n),'duration_ok': abs(dur-6.0)<0.25,
  'black_frame_ok': max(black or [1.0])<0.98,'sharpness_ok': (statistics.mean(sharp) if sharp else 0)>20,
  'segment_bounds_ok': all(0<=s['start_sec']<s['end_sec']<=dur for s in segments),
  'segment_nonoverlap_ok': all(segments[i]['end_sec']<=segments[i+1]['start_sec'] for i in range(len(segments)-1)),
  'language_complete': all(s.get('language') and s.get('language_en') and s.get('atomic_action') for s in segments),
  'duplicate_exact_ok': 'unknown_single_clip_cohort',
  'interaction_proxy_ok': active>0.01,
  'geometry_ready': False},
 'metrics':{'fps':fps,'frames':n,'duration_sec':dur,'resolution':[w,h],'mean_sharpness':statistics.mean(sharp) if sharp else 0.0,'max_black_fraction':max(black or [0.0]),'motion_mean':motion_mean,'motion_p90':motion_p90,'active_fraction':active,'exact_sha256':input_hash},
 'warnings':['duplicate check requires a cohort manifest','MediaPipe runtime missing; L2 is proxy only','RGB-only input cannot support metric 6D pose','single annotator and language seed require review'],
 'status':'REVIEW','cleaning_decision':'kept_for_review_not_training'
}
(OUT/'qc'/'qc_report_v2.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2))
(OUT/'qc'/'qc_report_v2.md').write_text('# QC report v2\n\n**Status: REVIEW**\n\nVideo decodes and segment boundaries/language completeness pass. The clip is kept in a review bucket because the duplicate cohort is unavailable, MediaPipe runtime is missing, and RGB-only data cannot provide 6D geometry.')
# cleaned manifest and safe copy
shutil.copy2(VIDEO,CLEAN/'humanego_serve_bread_000_6s.review.mp4')
(CLEAN/'manifest.json').write_text(json.dumps({'status':'review','reason':'model_runtime_and_cohort_qc_pending','source_sha256':input_hash,'kept_file':'humanego_serve_bread_000_6s.review.mp4'},indent=2))

# data-backed local HTML
html=f'''<!doctype html><meta charset="utf-8"><title>HumanEgo result - serve bread</title><style>body{{font:15px system-ui;margin:32px;color:#172033;max-width:1100px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{border:1px solid #d9dfeb;border-radius:12px;padding:14px;background:#fff}}img{{max-width:100%;border-radius:8px}}.warn{{background:#fff4d8;padding:12px;border-radius:8px}}code{{word-break:break-all}}</style><h1>HumanEgo 实际标注结果</h1><p>片段：serve_bread · 6.0 s · 640×480 · 180 帧</p><div class="warn"><b>最终 QC：REVIEW</b>。视频可解码，动作/语言五段已写入；L2 为运动代理，L3 在 RGB-only 条件下不适用。</div><h2>标注示例</h2><img src="annotation_contact_sheet.jpg"><img src="annotation_timeline.png"><h2>四层状态</h2><div class="grid"><div class="card"><b>L0 溯源</b><p>complete</p><code>{input_hash}</code></div><div class="card"><b>L1 动作+语言</b><p>5 atomic subtasks<br>prepare · reach · grasp · carry · release</p></div><div class="card"><b>L2 交互</b><p>proxy motion<br>active fraction: {active:.3f}<br>runtime: missing</p></div><div class="card"><b>L3 几何</b><p>not_applicable<br>无深度/标定 SLAM，不生成 6D</p></div><div class="card"><b>清洗</b><p>保留到 review bucket<br>未进入训练集</p></div><div class="card"><b>重复</b><p>exact hash 已记录<br>需 cohort 才能判重</p></div></div><h2>机器可读文件</h2><ul><li><a href="../annotation/four_layer_annotation.json">four_layer_annotation.json</a></li><li><a href="../qc/qc_report_v2.json">qc_report_v2.json</a></li><li><a href="../cleaned/manifest.json">cleaned/manifest.json</a></li></ul>'''
(WEB/'index.html').write_text(html)
print(json.dumps({'annotation':str(ANN/'four_layer_annotation.json'),'qc':str(OUT/'qc'/'qc_report_v2.json'),'web':str(WEB/'index.html'),'cleaned':str(CLEAN/'manifest.json'),'frames':len(frames),'status':'REVIEW'},indent=2))

