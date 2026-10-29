"""Read actual stage files; generation and independent acceptance stay separate."""
import json,time
from pathlib import Path
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
ROOT=Path('/user/panmiao/human-robot-transfer/outputs/unified_multimodal_20261020')
LOG=Path('/user/panmiao/human-robot-transfer/outputs/object_annotation_20261020/events/unified_multimodal_v4')
writer=EventFileWriter(str(LOG));step=0
try:
 while True:
  values=[];totals={k:[0,0]for k in ['sam3','render_multimodal','depth']};summary={}
  def scalar(tag,v):values.append(Summary.Value(tag=tag,simple_value=v))
  for s in sorted((ROOT/'samples').iterdir()):
   if not s.is_dir():continue
   n=json.loads((s/'segments.json').read_text())['frames'];current={}
   for stage in totals:
    p=s/'multimodal'/('depth_model' if stage=='depth' else '')/f'{stage}_status.json'
    expected=n*(4 if s.name=='abc130k_ep_101288'else 3)if stage=='sam3'else n
    d=json.loads(p.read_text())if p.exists()else dict(completed=0,total=expected,status='waiting')
    totals[stage][0]+=d['completed'];totals[stage][1]+=d['total'];current[stage]=d
    for tag,v in [('completed',d['completed']),('total',d['total']),('waiting',int(d['status'].startswith('waiting'))),('failed',int(d['status']=='failed'))]:scalar(f'{s.name}/{stage}/{tag}',v)
   summary[s.name]=current
  for stage,(done,total)in totals.items():
   for tag,v in [('completed',done),('total',total),('fraction',done/max(total,1))]:scalar(f'overall/{stage}/{tag}',v)
  pre=ROOT/'samples/umi_hifi_pot/multimodal_preflight/sam3_status.json';pre=json.loads(pre.read_text())if pre.exists()else dict(status='waiting',completed=0,total=90)
  prepass=(ROOT/'preflight.json').exists()
  for tag,v in [('completed',pre['completed']),('total',pre['total']),('passed',int(prepass)),('waiting',int(pre['status']=='waiting')),('failed',int(pre['status']=='failed'))]:scalar('preflight/'+tag,v)
  for tag,v in [('sam3_gpu0_fraction',totals['sam3'][0]/max(1,totals['sam3'][1])),('depth_gpu0_fraction',totals['depth'][0]/max(1,totals['depth'][1])),('rejected_previous_attempts',3)]:scalar('pipeline/'+tag,v)
  scalar('independent_review/completed',0);scalar('independent_review/waiting',1)
  writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=values)));writer.flush()
  tmp=ROOT/'progress.partial.json';tmp.write_text(json.dumps(dict(updated=time.time(),totals=totals,samples=summary,preflight=pre,preflight_passed=prepass),ensure_ascii=False,indent=2));tmp.replace(ROOT/'progress.json');step+=1
  # Keep monitoring other independent stages even when one stage failed.
  time.sleep(15)
finally:writer.close()
