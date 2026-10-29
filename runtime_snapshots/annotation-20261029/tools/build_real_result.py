#!/usr/bin/env python3
import argparse, hashlib, json, math, shutil, subprocess, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np

HAND_EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
BODY_EDGES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

def sha256(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def run_source_qc(repo, video, qc_dir, segment):
    qc_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(repo/"tools/qc_unlabeled.py"),"--video",str(video),"--out",str(qc_dir),"--segment-manifest",str(segment)],check=True)
    return json.loads((qc_dir/"qc_report.json").read_text())

def frame_phash(path):
    try:
        raw=subprocess.check_output(["ffmpeg","-v","error","-ss",str(path[1]),"-i",str(path[0]),"-frames:v","1","-f","image2pipe","-vcodec","png","pipe:1"],timeout=20)
        img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        small=cv2.resize(img,(8,8),interpolation=cv2.INTER_AREA); m=float(small.mean())
        return ''.join('1' if float(x)>=m else '0' for x in small.reshape(-1))
    except Exception:
        return None

def hamming(a,b): return sum(x!=y for x,y in zip(a,b))

def dedup_report(cohort_dir):
    files=sorted(Path(cohort_dir).glob('*.mp4'))
    exact={}; reps={}
    for p in files:
        exact.setdefault(sha256(p),[]).append(p.name)
        reps[p.name]=[frame_phash((p,t)) for t in (0.0,0.5,1.0)]
    exact_pairs=[v for v in exact.values() if len(v)>1]
    near_pairs=[]
    names=list(reps)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            aa=[x for x in reps[a] if x]; bb=[x for x in reps[b] if x]
            if aa and bb and min(hamming(x,y) for x in aa for y in bb)<=5: near_pairs.append([a,b])
    return {"cohort_files":[p.name for p in files],"file_count":len(files),"exact_sha_groups":list(exact.values()),"exact_duplicate_pairs":exact_pairs,"near_duplicate_pairs":near_pairs,"status":"unknown_single_clip_cohort" if len(files)<2 else ("FAIL" if exact_pairs or near_pairs else "PASS")}

def safe_float(v):
    try: return float(v)
    except Exception: return None

def annotation_qc(model, semantic):
    meta=model["meta"]; frames=model["frames"]
    hands=[h for f in frames for h in f.get("hands",[])]
    objects=[o for f in frames for o in f.get("objects",[])]
    skeleton_valid=sum(len(h.get("xy",[]))==21 and len(h.get("local_xyz",[]))==21 for h in hands)
    skeleton_bad=sum(not(len(h.get("xy",[]))==21 and len(h.get("local_xyz",[]))==21) for h in hands)
    shape_counts=Counter(h.get("shape","unknown") for h in hands)
    track_ids=[o.get("track_id") for o in objects]
    pose2d=sum(bool(o.get("pose_2d")) for o in objects)
    body_frames=sum(bool(f.get("body_raw_prediction") or f.get("body")) for f in frames)
    target=sum(bool(o.get("target_candidate")) for o in objects)
    language=semantic["layer_1_language_action"]["segments"]
    language_complete=all(s.get("language") and s.get("language_en") and s.get("atomic_action") for s in language)
    return {"frames":len(frames),"hands_detected":len(hands),"hand_skeleton_valid":skeleton_valid,"hand_skeleton_invalid":skeleton_bad,"hand_shape_counts":dict(shape_counts),"body_pose_frames":body_frames,"object_detections":len(objects),"object_track_count":len(set(x for x in track_ids if x is not None)),"object_pose_2d_valid":pose2d,"target_candidate_detections":target,"language_segments":len(language),"language_complete":language_complete,"no_nan":all(np.isfinite(np.asarray(h.get("xy",[]),dtype=float)).all() for h in hands if h.get("xy"))}

def draw_frame(img, rec, idx, t):
    h,w=img.shape[:2]
    for o in rec.get("objects",[]):
        x,y,bw,bh=o["box"]; x1,y1=int(x*w),int(y*h); x2,y2=int((x+bw)*w),int((y+bh)*h)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,180,40),2)
        label=f'{o.get("label","?")}#{o.get("track_id","?")} {o.get("score",0):.2f}'
        cv2.putText(img,label,(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,210,100),1,cv2.LINE_AA)
    for hand in rec.get("hands",[]):
        pts=np.asarray(hand.get("xy",[]),dtype=float)
        if len(pts)!=21: continue
        q=(pts*np.array([w,h])).astype(int)
        for a,b in HAND_EDGES: cv2.line(img,tuple(q[a]),tuple(q[b]),(70,240,180),1,cv2.LINE_AA)
        for x,y in q: cv2.circle(img,(int(x),int(y)),2,(80,255,220),-1)
        cv2.putText(img,f'{hand.get("side","?")} {hand.get("shape","?")} H#{hand.get("track_id","?")}',tuple(q[0]),cv2.FONT_HERSHEY_SIMPLEX,.45,(80,255,220),1,cv2.LINE_AA)
    body=rec.get("body_raw_prediction") or rec.get("body") or []
    if len(body)>=29:
        q=np.asarray([[p[0],p[1]] for p in body],dtype=float)*np.array([w,h])
        for a,b in BODY_EDGES:
            if 0<=a<len(q) and 0<=b<len(q): cv2.line(img,tuple(q[a].astype(int)),tuple(q[b].astype(int)),(180,100,255),1,cv2.LINE_AA)
        for x,y in q: cv2.circle(img,(int(x),int(y)),1,(220,140,255),-1)
    cv2.rectangle(img,(0,0),(w,28),(8,20,30),-1)
    cv2.putText(img,f'frame {idx:03d}  t={t:.1f}s | hand skeleton + hand shape + object tracks', (8,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,247,250),1,cv2.LINE_AA)

def render_examples(video, model, web):
    frames_dir=web/"frames"; frames_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run(["ffmpeg","-v","error","-y","-i",str(video),"-vf","fps=10,scale=640:-2","-q:v","3",str(frames_dir/"raw_%04d.jpg")],check=True)
    for p in sorted(frames_dir.glob("raw_*.jpg")):
        p.rename(p.with_name(p.name.replace("raw_","frame_")))
    for i,rec in enumerate(model["frames"]):
        src=frames_dir/f"frame_{i+1:04d}.jpg"
        if src.exists():
            img=cv2.imread(str(src)); draw_frame(img,rec,i,i/10); cv2.imwrite(str(src),img)
    selected=[frames_dir/f"frame_{i+1:04d}.jpg" for i in np.linspace(0,len(model["frames"])-1,10,dtype=int)]
    imgs=[cv2.imread(str(p)) for p in selected if p.exists()]
    if imgs:
        thumb_w=320; thumb_h=int(imgs[0].shape[0]*thumb_w/imgs[0].shape[1]); thumbs=[cv2.resize(x,(thumb_w,thumb_h)) for x in imgs]
        sheet=np.zeros((thumb_h*2,thumb_w*5,3),np.uint8)
        for j,x in enumerate(thumbs): sheet[(j//5)*thumb_h:(j//5+1)*thumb_h,(j%5)*thumb_w:(j%5+1)*thumb_w]=x
        cv2.imwrite(str(web/"annotation_contact_sheet.jpg"),sheet)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",required=True); ap.add_argument("--video",required=True); ap.add_argument("--model-annotation",required=True); ap.add_argument("--semantic-annotation",required=True); ap.add_argument("--segment-manifest",required=True); ap.add_argument("--cohort-dir",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    repo=Path(args.repo); video=Path(args.video); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    model=json.loads(Path(args.model_annotation).read_text()); semantic=json.loads(Path(args.semantic_annotation).read_text())
    source_qc=run_source_qc(repo,video,out/"qc",Path(args.segment_manifest))
    ann_qc=annotation_qc(model,semantic); dedup=dedup_report(args.cohort_dir)
    reasons=[]
    if not all(source_qc["checks"].values()): reasons.append("source_quality_gate")
    if ann_qc["hand_skeleton_valid"]==0: reasons.append("no_hand_skeleton")
    if ann_qc["object_track_count"]==0: reasons.append("no_object_tracks")
    if not ann_qc["language_complete"]: reasons.append("language_incomplete")
    if dedup["status"]!="PASS": reasons.append(dedup["status"])
    if ann_qc["target_candidate_detections"]==0: reasons.append("object_detector_target_mismatch")
    reasons += ["single_annotator_language_review","rgb_only_no_metric_6d","egocentric_full_body_gate"]
    decision="PASS" if not reasons else "REVIEW"
    combined={"schema":"human-robot-transfer/four-layer-real-v2","clip_id":semantic["clip_id"],"source":semantic["source"],"layer_0_provenance":semantic["layer_0_provenance"],"layer_1_language_action":semantic["layer_1_language_action"],"layer_2_interaction":{**semantic["layer_2_interaction"],"model_annotation_path":str(Path(args.model_annotation).resolve()),"model_qc":ann_qc},"layer_3_geometry":{**semantic["layer_3_geometry"],"object_pose_status":model["meta"].get("object_pose_status"),"metric_6d":False},"model_summary":model["meta"],"review":{"decision":decision,"reasons":reasons}}
    (out/"annotation").mkdir(exist_ok=True); (out/"annotation/four_layer_real_v2.json").write_text(json.dumps(combined,ensure_ascii=False,indent=2))
    report={"status":decision,"source_qc":source_qc,"annotation_qc":ann_qc,"dedup":dedup,"filter":{"decision":decision,"training_eligible":False,"reasons":reasons},"model_hashes":model["meta"].get("model_hashes",{})}
    (out/"qc/qc_report_real_v3.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)); (out/"qc/qc_report_real_v3.md").write_text('# Real annotation QC v3\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    cleaned=out/"cleaned"; cleaned.mkdir(exist_ok=True); shutil.copy2(video,cleaned/video.name); (cleaned/"manifest.json").write_text(json.dumps({"clip_id":semantic["clip_id"],"file":video.name,"sha256":sha256(video),"decision":decision,"training_eligible":False,"reasons":reasons},ensure_ascii=False,indent=2))
    web=out/"web"; web.mkdir(exist_ok=True); render_examples(video,model,web)
    (web/"result_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"status":decision,"annotation":str(out/"annotation/four_layer_real_v2.json"),"qc":str(out/"qc/qc_report_real_v3.json"),"cleaned":str(cleaned/"manifest.json"),"web":str(web),"annotation_qc":ann_qc,"dedup":dedup},ensure_ascii=False))
if __name__=='__main__': main()
