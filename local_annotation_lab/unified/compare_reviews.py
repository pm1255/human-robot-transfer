"""Compare two independently authored temporal annotations; never treats self-comparison as acceptance.
Usage: python compare_reviews.py reviewer_a.json reviewer_b.json --output agreement.json
"""
import argparse,json
from pathlib import Path
import numpy as np

def validate(d):
 rows=d['segments'];n=d['frames'];fps=d['fps'];assert rows and rows[0]['start_frame']==0 and rows[-1]['end_frame_exclusive']==n
 assert all(0<=s['start_frame']<s['end_frame_exclusive']<=n and (i==0 or rows[i-1]['end_frame_exclusive']==s['start_frame']) for i,s in enumerate(rows))
 return rows,n,fps

def match_boundaries(a,b,tolerance):
 # Ordered DP maximizes unique matches before minimizing absolute error.
 dp=[[(0,0.,[]) for _ in range(len(b)+1)] for _ in range(len(a)+1)]
 key=lambda x:(x[0],-x[1])
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1):
   candidates=[dp[i-1][j],dp[i][j-1]];dist=abs(a[i-1]-b[j-1])
   if dist<=tolerance:
    count,error,pairs=dp[i-1][j-1];candidates.append((count+1,error+dist,pairs+[(a[i-1],b[j-1])]))
   dp[i][j]=max(candidates,key=key)
 count,_,pairs=dp[-1][-1];p=count/len(a) if a else float(not b);r=count/len(b) if b else float(not a)
 return dict(matches=count,predicted=len(a),reference=len(b),precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0,matched_errors_s=[abs(x-y)for x,y in pairs])

def compare(a,b):
 ra,n,fps=validate(a);rb,nb,fb=validate(b);assert n==nb and fps==fb,'different video timebase'
 assert a.get('episode_id',a.get('source_episode'))==b.get('episode_id',b.get('source_episode')),'different episode'
 metrics={};bounda=[s['start_frame']/fps for s in ra[1:]];boundb=[s['start_frame']/fps for s in rb[1:]]
 for tol in [.1,.25,.5]:metrics[f'boundary_at_{tol}s']=match_boundaries(bounda,boundb,tol)
 for field in ['phase','verb','object','actor']:
  aa=np.empty(n,dtype=object);bb=np.empty(n,dtype=object)
  for s in ra:aa[s['start_frame']:s['end_frame_exclusive']]=s.get(field,'unknown')
  for s in rb:bb[s['start_frame']:s['end_frame_exclusive']]=s.get(field,'unknown')
  valid=(aa!='unknown')&(bb!='unknown')&(aa!=None)&(bb!=None);labels=sorted(set(aa[valid])|set(bb[valid]));cm={x:{y:int(((aa==x)&(bb==y)&valid).sum())for y in labels}for x in labels};f1=[]
  for label in labels:
   tp=((aa==label)&(bb==label)&valid).sum();den=((aa==label)&valid).sum()+((bb==label)&valid).sum();f1.append(float(2*tp/den)if den else 0)
  agreement=float((aa[valid]==bb[valid]).mean()) if valid.any() else None
  metrics[field]={'jointly_known_frames':int(valid.sum()),'total_frames':n,'accuracy_known_only':agreement,'macro_f1':float(np.mean(f1))if f1 else None,'confusion':cm}
 # Segment matching at IoU .5 with phase+verb labels, one-to-one.
 candidates=[]
 for i,x in enumerate(ra):
  for j,y in enumerate(rb):
   inter=max(0,min(x['end_frame_exclusive'],y['end_frame_exclusive'])-max(x['start_frame'],y['start_frame']));union=max(x['end_frame_exclusive'],y['end_frame_exclusive'])-min(x['start_frame'],y['start_frame']);iou=inter/union
   if iou>=.5 and (x['phase'],x['verb'])==(y['phase'],y['verb']):candidates.append((iou,i,j))
 useda=set();usedb=set()
 for iou,i,j in sorted(candidates,reverse=True):
  if i not in useda and j not in usedb:useda.add(i);usedb.add(j)
 metrics['segment_f1_iou_0_5']=2*len(useda)/(len(ra)+len(rb))
 independent=bool(a.get('reviewer_id') and b.get('reviewer_id') and a['reviewer_id']!=b['reviewer_id'] and a.get('independent_review') is True and b.get('independent_review') is True)
 return {'status':'agreement_measured_requires_adjudication'if independent else 'comparison_only_independence_not_established','independent_review_evidence_present':independent,'metrics':metrics,'training_acceptance':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('review_a');p.add_argument('review_b');p.add_argument('--output',required=True);args=p.parse_args();result=compare(json.loads(Path(args.review_a).read_text()),json.loads(Path(args.review_b).read_text()));Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['status'])
