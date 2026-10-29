"""Full video decoding, time contracts, semantic coverage, native-record and source checks."""
from pathlib import Path
import json,subprocess,re,hashlib,collections
import numpy as np
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper');ROOT=BASE/'outputs/unified_three_domain_20261020';SITE=BASE/'outputs/hand_mesh_repair_20261020'
FF=BASE/'outputs/unified_annotation_web_20261020/render_dependencies/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
m=json.loads((SITE/'assets/multi_manifest.json').read_text());tax=json.loads((SITE/'unified/taxonomy.json').read_text());report={'status':'running','episodes':[],'errors':[],'videos':[],'checks_not_performed':['independent_human_semantic_agreement','frame_exact_boundary_accuracy','physical_camera_state_lag','cross_domain_metric_calibration','new_object_masks_bbox_6d']};hashes=set()
for c in m['clips']:
 s=json.loads((SITE/c['semantics']).read_text());rows=s['segments'];checks={'start_at_zero':rows[0]['start_frame']==0,'end_at_total_frames':rows[-1]['end_frame_exclusive']==c['frames'],'positive_contiguous':all(x['end_frame_exclusive']>x['start_frame'] and (i==0 or rows[i-1]['end_frame_exclusive']==x['start_frame']) for i,x in enumerate(rows)),'shared_phase_vocabulary':all(x['phase'] in tax['phases'] for x in rows),'shared_verb_vocabulary':all(x['verb'] in tax['verbs'] for x in rows),'time_frame_agreement':all(abs(x['start_s']-x['start_frame']/c['fps'])<1e-5 and abs(x['end_s']-x['end_frame_exclusive']/c['fps'])<1e-5 for x in rows),'all_require_review':all(x['review_required'] for x in rows)}
 report['episodes'].append({'id':c['id'],'domain':c['dataset'],'frames':c['frames'],'segments':len(rows),'checks':checks,'training_eligible':False})
 if c.get('state'):
  st=json.loads((SITE/c['state']).read_text());a=np.array(st['position_m']);checks['state_length']=len(a)==c['frames'];checks['state_finite']=bool(np.isfinite(a).all());checks['timestamp_length']=len(st['timestamp_s'])==c['frames'];checks['timestamp_finite']=bool(np.isfinite(st['timestamp_s']).all());checks['timestamp_monotonic']=bool(np.all(np.diff(st['timestamp_s'])>0))
  for layer in c['layers']:
   if not layer.get('video'):continue
   path=SITE/layer['video'];h=hashlib.sha256(path.read_bytes()).hexdigest()
   proc=subprocess.run([str(FF),'-hide_banner','-i',str(path),'-map','0:v:0','-an','-vf','scale=1:1','-pix_fmt','gray','-f','rawvideo','-'],capture_output=True)
   stderr=proc.stderr.decode(errors='replace');stream=next((l for l in stderr.splitlines() if 'Video:' in l and 'Stream' in l),'');dim=re.search(r'\b(\d{2,5})x(\d{2,5})\b',stream);fps=re.search(r'([0-9.]+) fps',stream);n=len(proc.stdout)
   video={'episode':c['id'],'layer':layer['id'],'path':layer['video'],'sha256':h,'decoded_frames':n,'expected_frames':c['frames'],'decode_exit':proc.returncode,'source_fps':float(fps[1]) if fps else None,'dimensions':[int(dim[1]),int(dim[2])] if dim else None,'status':'passed' if proc.returncode==0 and n==c['frames'] and fps and abs(float(fps[1])-c['fps'])<.001 else 'failed'};report['videos'].append(video)
   if video['status']!='passed':report['errors'].append(video)
   if layer['id']=='source':
    if h in hashes:report['errors'].append([c['id'],'duplicate_source_hash'])
    hashes.add(h);c['width'],c['height']=video['dimensions']
  qc=json.loads((SITE/c['qc']).read_text());qc['native_validity_basis']='source_boolean_flags_and_finite' if st['source_kind']=='umi' else 'finite_numeric_only_no_source_confidence_flags';qc['video_decodes']=[x for x in report['videos'] if x['episode']==c['id']];qc['structural_checks']=checks
  (SITE/c['qc']).write_text(json.dumps(qc,ensure_ascii=False,indent=2));(ROOT/'samples'/c['id']/'qc.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2))
 for k,v in checks.items():
  if not v:report['errors'].append([c['id'],k])
report['status']='passed_structural_checks_review_required' if not report['errors'] else 'failed';report['new_videos_fully_decoded']=len(report['videos']);report['total_frames_physical_episodes']=sum(c['frames']for c in m['clips']);report['domain_counts']=dict(collections.Counter(c['dataset']for c in m['clips']))
(ROOT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(SITE/'unified/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for target in [SITE/'assets/multi_manifest.json',BASE/'code/human-robot-transfer/local_annotation_lab/assets/multi_manifest.json']:target.write_text(json.dumps(m,ensure_ascii=False,indent=2))
print({k:v for k,v in report.items() if k not in ['episodes','videos']});assert not report['errors']
