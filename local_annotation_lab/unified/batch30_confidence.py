"""Compute automated evidence for new model products; reuse no human gold labels."""
import json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
B=Path('/Users/panmiao/Documents/ChatGPT/paper');R=B/'outputs/unified_30_20261021';S=B/'outputs/hand_mesh_repair_20261020'
sys.path.insert(0,str(B/'code/human-robot-transfer'))
from human2robot.object_annotation.confidence import score_episode
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
def emit(writer,step,values):
 writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=k,simple_value=v)for k,v in values.items()])));writer.flush()
def run(c):
 import cv2
 cv2.setNumThreads(2)
 review=json.loads((R/'visual_review.json').read_text());destination=R/'confidence';target=destination/'samples'/f'{c["id"]}.json'
 if target.exists():return c['id'],'existing'
 writer=EventFileWriter(str(R/'events'/('confidence_'+c['id'])))
 try:
  summary=score_episode(c,S,R,destination,review,lambda n:emit(writer,n,dict(processed_frames=n,fraction=n/c['frames'])))
  return c['id'],summary['summary']['statuses']
 finally:writer.close()
if __name__=='__main__':
 clips=json.loads((S/'assets/multi_manifest.json').read_text())['clips'];config=json.loads((R/'sam_config.json').read_text());clips=[c for c in clips if c['id']in config and(R/'samples'/c['id']/'multimodal/masks.zip').exists()]
 writer=EventFileWriter(str(R/'events/confidence_evaluation'));emit(writer,int(time.time()),dict(completed=6,total=30,waiting=0))
 try:
  with ProcessPoolExecutor(max_workers=4)as pool:
   for count,(sid,summary)in enumerate(pool.map(run,clips),7):
    print('SCORED',sid,summary,flush=True);emit(writer,int(time.time()),dict(completed=count,total=30,fraction=count/30))
 finally:writer.close()
