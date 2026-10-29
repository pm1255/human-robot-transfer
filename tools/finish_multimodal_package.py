"""Wait for current writers then validate/package without modifying active outputs."""
import json,time,subprocess,os
from pathlib import Path
R=Path('/user/panmiao/human-robot-transfer/outputs/unified_multimodal_20261020')
while True:
 statuses=[]
 for s in (R/'samples').iterdir():
  p=s/'multimodal/render_multimodal_status.json'
  statuses.append(json.loads(p.read_text())if p.exists()else {'status':'waiting'})
 if any(d['status']=='failed'for d in statuses):raise RuntimeError('Rendering failed; do not package')
 if len(statuses)==8 and all(d['status']=='completed'for d in statuses):break
 time.sleep(15)
subprocess.run([os.sys.executable,'tools/package_multimodal_outputs.py'],check=True)
