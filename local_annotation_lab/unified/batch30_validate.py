"""Full source/generated decode, distinctness, labels, hashes, references and range serving."""
import json,hashlib,urllib.request
from pathlib import Path
import cv2
B=Path('/Users/panmiao/Documents/ChatGPT/paper');R=B/'outputs/unified_30_20261021';S=B/'outputs/hand_mesh_repair_20261020'
m=json.loads((S/'assets/multi_manifest.json').read_text());taxonomy=json.loads((S/m['taxonomy']).read_text());results=[];source_hashes={};refs=set()
assert len(m['clips'])==30
assert {d:sum(c['domain']==d for c in m['clips'])for d in ['human','robot','umi']}==dict(human=10,robot=10,umi=10)
for c in m['clips']:
 src=S/'unified'/c['id']/'cam_0.mp4';sha=hashlib.sha256(src.read_bytes()).hexdigest();assert sha not in source_hashes,('duplicate source',c['id'],source_hashes.get(sha));source_hashes[sha]=c['id']
 sem=json.loads((S/c['semantics']).read_text());segs=sem['segments'];assert segs[0]['start_frame']==0 and segs[-1]['end_frame_exclusive']==c['frames'];assert all(x['start_frame']<x['end_frame_exclusive']and x['phase']in taxonomy['phases']and x['verb']in taxonomy['verbs']for x in segs);assert all(x['end_frame_exclusive']==y['start_frame']for x,y in zip(segs,segs[1:]));assert sem['source_video_sha256']==sha
 refs.update(c[k]for k in ['semantics','state','qc','provenance','confidence','cleaner_qc','time_audit']if c.get(k))
 for l in c['layers']:
  refs.update(x['url']for x in l.get('resources',[])if not x['url'].startswith('http'))
  if not l.get('video'):continue
  f=S/l['video'];cap=cv2.VideoCapture(str(f));n=0;fps=cap.get(5)
  while cap.read()[0]:n+=1
  cap.release();assert n==c['frames']and abs(fps-c['fps'])<.001,(c['id'],l['id'],n,fps)
  results.append(dict(sample=c['id'],layer=l['id'],frames=n,fps=fps,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),path=l['video'],passed=True))
 for key in ['cleaner_qc','confidence']:
  if c.get(key):json.loads((S/c[key]).read_text())
 print('VALIDATED',c['id'],flush=True)
missing=[p for p in sorted(refs)if not(S/p).is_file()];assert not missing,missing
for c in [m['clips'][0],next(x for x in m['clips']if x['domain']=='robot'),next(x for x in m['clips']if x['domain']=='umi')]:
 req=urllib.request.Request('http://127.0.0.1:53419/'+c['layers'][0]['video'],headers={'Range':'bytes=0-1023'})
 with urllib.request.urlopen(req)as response:assert response.status==206 and len(response.read())==1024
report=dict(episodes=30,domain_counts=dict(human=10,robot=10,umi=10),distinct_video_hashes=30,total_frames=sum(c['frames']for c in m['clips']),segments=sum(len(json.loads((S/c['semantics']).read_text())['segments'])for c in m['clips']),videos=results,resource_paths_checked=len(refs),missing_references=missing,range_checks=3,accuracy_verified=False)
(R/'release_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('ALL_OK',len(results),'videos',flush=True)
