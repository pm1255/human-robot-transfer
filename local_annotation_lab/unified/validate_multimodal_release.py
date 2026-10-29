from pathlib import Path
import json,hashlib,urllib.request
B=Path('/Users/panmiao/Documents/ChatGPT/paper');S=B/'outputs/hand_mesh_repair_20261020';R=B/'outputs/unified_multimodal_20261020'
m=json.loads((S/'assets/multi_manifest.json').read_text());old=json.loads((R/'manifest_before_multimodal.json').read_text());rows=[];new_count=0
assert m['clips'][0]['layers']==old['clips'][0]['layers'],'Original folding layers changed'
for c in m['clips']:
 for l in c['layers']:
  if not l.get('video'):continue
  p=S/l['video'];assert p.exists(),p
  req=urllib.request.Request('http://127.0.0.1:53419/'+l['video'],headers={'Range':'bytes=0-1023'})
  with urllib.request.urlopen(req)as resp:
   assert resp.status==206 and len(resp.read())==1024
  rows.append(dict(clip=c['id'],layer=l['id'],range_request=True))
  if '/multimodal/'in l['video']:new_count+=1
 for l in c['layers']:
  for link in l.get('resources',[]):
   if not link['url'].startswith(('#','http')):assert (S/link['url']).is_file(),link
assert new_count==40
result=dict(new_videos=new_count,total_videos=len(rows),original_folding_layers_unchanged=True,http_range_resources=rows,independent_accuracy_verified=False)
(R/'release_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print({k:v for k,v in result.items()if k!='http_range_resources'})
