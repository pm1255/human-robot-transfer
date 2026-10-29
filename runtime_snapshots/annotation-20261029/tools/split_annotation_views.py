import argparse,json,shutil,cv2,math
from pathlib import Path
import numpy as np

def targets(s):
    seg=s.get('layer_1_language_action',[])
    if isinstance(seg,dict): seg=seg.get('segments',[])
    return sorted({str(x).lower() for q in seg if isinstance(q,dict) for x in q.get('object_ids',[])})

def exact(label,target):
    syn={'bread':{'bread','loaf','bun','roll'},'tray':{'tray','plate','platter'},'cup':{'cup','mug'},'bottle':{'bottle'},'bowl':{'bowl'}}
    return str(label).lower() in syn.get(target,{target})

def hand_center(f):
    q=[]
    for h in f.get('hands',[]):
        for p in h.get('xy',[]):
            if len(p)>=2:q.append((float(p[0]),float(p[1])))
    return (sum(x for x,y in q)/len(q),sum(y for x,y in q)/len(q)) if q else None

def aff(o,hc,strict):
    b=o.get('box');
    if not b:return None
    x,y,w,h=map(float,b[:4]); pts={'center':(x+w/2,y+h/2),'left':(x+w*.28,y+h/2),'right':(x+w*.72,y+h/2),'top':(x+w/2,y+h*.28),'bottom':(x+w/2,y+h*.72)}
    k='center'
    if hc:k=min(pts,key=lambda n:(pts[n][0]-hc[0])**2+(pts[n][1]-hc[1])**2)
    return {'type':'grasp','preferred_point':list(pts[k]),'grasp_points':[list(v) for v in pts.values()],'preferred_region':k,'confidence':round(.55 if strict else .18,3),'source':'hand_proximity+box_geometry' if hc else 'box_geometry_fallback','status':'candidate_strict_target' if strict else 'candidate_only_unresolved_target','requires_mask_refinement':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-video',required=True);ap.add_argument('--model-json',required=True);ap.add_argument('--semantic-json',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True); src=Path(a.input_video)
    m=json.load(open(a.model_json)); s=json.load(open(a.semantic_json)); fs=m['frames']; ts=targets(s)
    by={}
    for i,f in enumerate(fs):
        for o in f.get('objects',[]):
            tid=o.get('track_id'); lab=o.get('label','')
            for t in ts:
                if tid is not None and exact(lab,t):by.setdefault((t,int(tid)),[]).append((i,float(o.get('score',0))))
    good={};bad=[]
    for t in ts:
        cand=[]
        for (tt,tid),rows in by.items():
            if tt==t and len(rows)>=3 and np.median([r[1] for r in rows])>=.35 and len({r[0] for r in rows})/len(fs)>=.03:cand.append((np.median([r[1] for r in rows]),tid,len(rows)))
        if cand:
            cand.sort(reverse=True); good[cand[0][1]]=t
        else: bad.append({'target':t,'status':'unresolved_no_exact_track'})
    ann={'meta':{'schema':'human2robot.annotation_views.v1','target_terms':ts,'strict_target_association':{'rule':'exact label/curated synonym + >=3 detections + median score >=0.35 + coverage >=3%','matched':[{'target':t,'track_id':tid,'status':'accepted_strict'} for tid,t in good.items()],'unresolved':bad,'status':'accepted' if good and not bad else ('partial' if good else 'unresolved')},'segmentation':{'backend':'sam3','status':'not_executed_no_visible_cuda_or_checkpoint','mask_rle':None},'affordance':{'status':'heuristic_candidates_only','training_gate':'strict_target_and_mask_required'}},'frames':[]}
    for f in fs:
        g=dict(f);g['objects']=[];hc=hand_center(f)
        for oo in f.get('objects',[]):
            o=dict(oo); tid=o.get('track_id'); strict=tid is not None and int(tid) in good;o['target_association']={'status':'accepted_strict','target':good[int(tid)]} if strict else {'status':'unresolved','reason':'no exact stable target track'};o['target_candidate']=strict;o['affordance']=aff(o,hc,strict);o['segmentation']={'backend':'sam3','status':'not_executed_no_visible_cuda_or_checkpoint','mask_rle':None};g['objects'].append(o)
        ann['frames'].append(g)
    json.dump(ann,open(out/'affordance_segmentation_annotation.json','w'),ensure_ascii=False,indent=2)
    cap=cv2.VideoCapture(str(src));fps=cap.get(cv2.CAP_PROP_FPS) or 30;w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    def pts(hand):return [(int(float(p[0])*w),int(float(p[1])*h)) for p in hand.get('xy',[]) if len(p)>=2]
    modes={'raw':None,'hand_skeleton':'hand','object_track':'object','affordance_grasp':'affordance','segmentation_sam3':'segmentation','language_action':'language'}
    for name,mode in modes.items():
        dst=out/(src.stem+'__'+name+'.mp4')
        if mode is None:shutil.copy2(src,dst);continue
        cap=cv2.VideoCapture(str(src));wr=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        for i in range(n):
            ok,img=cap.read()
            if not ok:break
            t=i/fps;j=min(range(len(fs)),key=lambda z:abs(float(fs[z].get('t',z/fps))-t));f=fs[j]
            if mode=='hand':
                for hh in f.get('hands',[]):
                    pp=pts(hh)
                    for x,y in pp:cv2.circle(img,(x,y),3,(80,220,255),-1)
                    for u,v in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                        if u<len(pp) and v<len(pp):cv2.line(img,pp[u],pp[v],(80,220,255),2)
            elif mode=='language':cv2.putText(img,'language/action layer · review required',(12,28),0,.65,(190,240,240),2)
            else:
                for o in f.get('objects',[]):
                    b=o.get('box');
                    if not b:continue
                    x,y,bw,bh=[float(z) for z in b];x1,y1=int(x*w),int(y*h);x2,y2=int((x+bw)*w),int((y+bh)*h);tid=o.get('track_id');strict=tid in good
                    if mode=='object':cv2.rectangle(img,(x1,y1),(x2,y2),(90,230,160) if strict else (255,170,70),2)
                    elif mode=='affordance':
                        p=(o.get('affordance') or {}).get('preferred_point');
                        if p:cv2.drawMarker(img,(int(p[0]*w),int(p[1]*h)),(80,230,255) if strict else (170,170,170),cv2.MARKER_CROSS,18,3)
                    else:
                        cv2.rectangle(img,(x1,y1),(x2,y2),(100,100,255),2);cv2.putText(img,'SAM3 mask pending',(x1,max(14,y1-5)),0,.38,(100,100,255),1)
            wr.write(img)
        cap.release();wr.release()
    json.dump({'source_video':str(src),'copies':{k:src.stem+'__'+k+'.mp4' for k in modes},'strict_association':ann['meta']['strict_target_association'],'segmentation_status':ann['meta']['segmentation'],'affordance_status':ann['meta']['affordance'],'training_eligible':ann['meta']['strict_target_association']['status']=='accepted'},open(out/'view_manifest.json','w'),ensure_ascii=False,indent=2)
    print(json.dumps({'status':ann['meta']['strict_target_association']['status'],'matched':good,'unresolved':bad,'videos':len(modes)},ensure_ascii=False))
if __name__=='__main__':main()
