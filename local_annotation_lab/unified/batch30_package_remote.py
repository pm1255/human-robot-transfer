"""Audit finished per-episode products and package a resumable transfer."""
import json,sys,zipfile,tarfile,hashlib,shutil
from pathlib import Path
from human2robot.object_annotation.audit_multimodal import audit
R=Path('/user/panmiao/human-robot-transfer/outputs/unified_30_20261021')
incremental='--incremental'in sys.argv
resources=json.loads((R/'transfer_manifest.json').read_text())if incremental else[]
summary=json.loads((R/'multimodal_audit.json').read_text())if incremental else{}
paths={}
for sid in json.loads((R/'sam_config.json').read_text()):
 if incremental and sid in summary:continue
 p=R/'samples'/sid;o=(R/'repair_run/samples'/sid/'multimodal')if sid in json.loads((R/'repair_classes.json').read_text())else p/'multimodal'
 if not(o/'video_validation.json').exists()or not(o/'depth_validation.json').exists():continue
 q=audit(o);summary[sid]=q
 with zipfile.ZipFile(o/'masks.zip','w',zipfile.ZIP_STORED)as z:
  for f in sorted((o/'masks').glob('*.npz')):z.write(f,'masks/'+f.name)
 d=json.loads((o/'depth_model/depth_manifest.json').read_text());d['rigid_constraint_applied']=False;(o/'depth_manifest.json').write_text(json.dumps(d,indent=2))
 (o/'raw_data_locations.json').write_text(json.dumps(dict(source=str(p/'cam_0.mp4'),depth=str((o/'depth_model/depth').resolve()),flow=str(o/'flow'),masks=str(o/'masks'),training_eligible=False),indent=2))
 for f in o.iterdir():
  if f.is_file()and f.suffix in ['.json','.mp4','.jpg','.zip']:
   name=f'samples/{sid}/multimodal/{f.name}';paths[name]=f;resources.append(dict(path=name,remote_origin=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
(R/'multimodal_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(R/'transfer_manifest.json').write_text(json.dumps(resources,indent=2))
with tarfile.open(R/'deliverables.tar','w',dereference=True)as t:
 for name,path in paths.items():t.add(path,arcname=name)
 for name in ['multimodal_audit.json','transfer_manifest.json']:t.add(R/name,arcname=name)
print('PACKAGED',len(summary),'samples',len(resources),'files',flush=True)
