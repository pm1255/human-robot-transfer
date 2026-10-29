"""Write real completed counts and explicit stage states to TensorBoard."""
import json,time,argparse
from pathlib import Path
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
ROOT=Path('/Users/panmiao/Documents/ChatGPT/paper/outputs/unified_30_20261021')
STAGES={'selection':30,'extraction':30,'native_qc':30,'semantics':30,'sam3':30,'depth':30,'website':30}
def update(stage,completed,status='running'):
    p=ROOT/'progress.json';data=json.loads(p.read_text()) if p.exists() else {k:dict(completed=0,total=v,status='waiting') for k,v in STAGES.items()}
    if stage:data[stage].update(completed=completed,status=status)
    p.write_text(json.dumps(data,indent=2))
    writer=EventFileWriter(str(ROOT/'events/batch30'))
    for k,v in data.items():
        values=[Summary.Value(tag=k+'/completed',simple_value=v['completed']),Summary.Value(tag=k+'/total',simple_value=v['total']),Summary.Value(tag=k+'/fraction',simple_value=v['completed']/v['total']),Summary.Value(tag=k+'/waiting',simple_value=float(v['status']=='waiting'))]
        writer.add_event(Event(wall_time=time.time(),step=int(time.time()),summary=Summary(value=values)))
    writer.flush();writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',default='');p.add_argument('completed',nargs='?',type=int,default=0);p.add_argument('--status',default='running');a=p.parse_args();update(a.stage,a.completed,a.status)
