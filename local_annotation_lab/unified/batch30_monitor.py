"""Read progress without interrupting remote inference; record actual counters."""
import json,time,subprocess
from pathlib import Path
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
ROOT=Path('/Users/panmiao/Documents/ChatGPT/paper/outputs/unified_30_20261021')
REMOTE='/user/panmiao/human-robot-transfer/outputs/unified_30_20261021'
writer=EventFileWriter(str(ROOT/'events/model_generation_v2'))
script="import pathlib,json; r=pathlib.Path('"+REMOTE+"'); d={str(p.relative_to(r)):json.loads(p.read_text())for base in [r/'samples',r/'repair_run/samples'] for p in base.glob('*/multimodal/**/*_status.json')}; a=r/'repair_execution_assignment.json'; d['__assignment__']=json.loads(a.read_text())if a.exists()else {}; print(json.dumps(d))"
step=int(time.time())
try:
 while True:
  try:
   data=json.loads(subprocess.check_output(['ssh','devspace-panmiao-pm-1055716','python -c '+__import__('shlex').quote(script)],timeout=25));assignments=data.pop('__assignment__',{});values=[];totals={}
   for sid,config in json.loads((ROOT/'sam_config.json').read_text()).items():
    source=json.loads((ROOT/'samples'/sid/'source.json').read_text());n=source.get('frames',source.get('num_frames'))
    for stage,tail,total in [('depth','depth_model/depth_status.json',n),('sam3','sam3_status.json',n*len(config['prompts'])),('render_multimodal','render_multimodal_status.json',n)]:
     data.setdefault(f'samples/{sid}/multimodal/{tail}',dict(stage=stage,status='waiting',completed=0,total=total))
   gpu_totals={}
   for shard in ROOT.glob('repair_shard_*.json'):
    for sid,config in json.loads(shard.read_text()).items():
     source=json.loads((ROOT/'samples'/sid/'source.json').read_text());n=source.get('frames',source.get('num_frames'))
     for stage,total in [('sam3',n*len(config['prompts'])),('render_multimodal',n)]:
      data.setdefault(f'repair_run/samples/{sid}/multimodal/{stage}_status.json',dict(stage=stage,status='waiting',completed=0,total=total))
   for path,d in data.items():
    repair=path.startswith('repair_run/');stage=('repair_'if repair else'')+d.get('stage',path.split('/')[-1]);sid=path.split('/')[2 if repair else 1];done=d.get('completed',0);total=d.get('total',0)
    for key,v in [('completed',done),('fraction',done/max(total,1)),('waiting',float(d.get('status')=='waiting')),('failed',float(d.get('status')=='failed'))]:values.append(Summary.Value(tag=f'{sid}/{stage}/{key}',simple_value=v))
    totals.setdefault(stage,[0,0]);totals[stage][0]+=done;totals[stage][1]+=total
    for gpu in range(4):
     shard=ROOT/f'{"repair"if repair else"depth"if stage=="depth"else"sam"}_shard_{gpu}.json'
     matches=(assignments.get(sid)==gpu)if repair and sid in assignments else(shard.exists()and sid in json.loads(shard.read_text()))
     if matches:
      gpu_totals.setdefault((stage,gpu),[0,0]);gpu_totals[stage,gpu][0]+=done;gpu_totals[stage,gpu][1]+=total
   for stage,(done,total)in totals.items():
    values.extend([Summary.Value(tag=f'overall/{stage}/processed_frames',simple_value=done),Summary.Value(tag=f'overall/{stage}/fraction_of_started',simple_value=done/max(total,1))])
   for (stage,gpu),(done,total)in gpu_totals.items():
    values.extend([Summary.Value(tag=f'gpu{gpu}/{stage}/processed_frames',simple_value=done),Summary.Value(tag=f'gpu{gpu}/{stage}/fraction_of_started',simple_value=done/max(total,1))])
   writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=values)));writer.flush();(ROOT/'model_progress.json').write_text(json.dumps(data,indent=2));step+=1
  except Exception as e:print(type(e).__name__,str(e)[:150],flush=True)
  time.sleep(20)
finally:writer.close()
