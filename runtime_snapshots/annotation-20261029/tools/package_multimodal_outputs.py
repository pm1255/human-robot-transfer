"""Package verified generated resources; keep large raw depth/flow arrays on server."""
import json,tarfile,zipfile,shutil,hashlib
from pathlib import Path
from human2robot.object_annotation.audit_multimodal import audit
from human2robot.object_annotation.unified_multimodal import render_depth
R=Path('/user/panmiao/human-robot-transfer/outputs/unified_multimodal_20261020')
allqc={};resources=[]
for s in sorted((R/'samples').iterdir()):
 out=s/'multimodal';assert (out/'video_validation.json').exists() and (out/'depth_validation.json').exists(),s.name
 # Exclude actual source letterbox, keeping original raw depth intact.
 render_depth(s/'cam_0.mp4',out)
 q=audit(out);allqc[s.name]=q
 depth_manifest=json.loads((out/'depth_model/depth_manifest.json').read_text());depth_manifest.pop('rigid_object',None);depth_manifest['rigid_constraint_applied']=False;depth_manifest['metric_scale_verified']=False;(out/'depth_manifest.json').write_text(json.dumps(depth_manifest,indent=2))
 with zipfile.ZipFile(out/'masks.zip','w',zipfile.ZIP_STORED)as z:
  for p in sorted((out/'masks').glob('*.npz')):z.write(p,'masks/'+p.name)
  z.write(out/'objects.json','objects.json');z.write(out/'sam3_manifest.json','sam3_manifest.json')
 files=[p for p in out.iterdir()if p.is_file() and p.suffix in ['.json','.mp4','.jpg','.zip']]
 for p in files:resources.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 # Raw data are not deleted or approximated for transfer.
 manifest=dict(raw_depth_directory=str(out/'depth_model/depth'),raw_flow_directory=str(out/'flow'),raw_mask_directory=str(out/'masks'),source_video=str(s/'cam_0.mp4'),raw_depth_frames=len(list((out/'depth_model/depth').glob('*.npz'))),raw_flow_frames=len(list((out/'flow').glob('*.npz'))),raw_mask_frames=len(list((out/'masks').glob('*.npz'))))
 (out/'raw_data_locations.json').write_text(json.dumps(manifest,indent=2));resources.append(dict(path=str((out/'raw_data_locations.json').relative_to(R))))
(R/'multimodal_audit.json').write_text(json.dumps(allqc,ensure_ascii=False,indent=2));(R/'transfer_manifest.json').write_text(json.dumps(resources,indent=2))
with tarfile.open(R/'deliverables.tar.gz','w:gz')as t:
 for r in resources:t.add(R/r['path'],arcname=r['path'])
 for name in ['model_provenance.json','multimodal_audit.json','transfer_manifest.json','preflight.json','progress.json']:t.add(R/name,arcname=name)
print('Packaged',len(resources),'files',flush=True)
