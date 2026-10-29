"""Render pilot reports and freeze a resource manifest after validation."""
from pathlib import Path
import hashlib,json,shutil,sys
from datetime import datetime,timezone
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper')
ROOT=BASE/'outputs/unified_three_domain_20261020'
SITE=BASE/'outputs/hand_mesh_repair_20261020'
CODE=BASE/'code/human-robot-transfer/local_annotation_lab'
sys.path.insert(0,str(ROOT/'render_dependencies'))
import markdown

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return {'path':str(p.relative_to(BASE)),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def main():
 for filename in ['index.html','app.js','styles.css']:
  shutil.copy2(CODE/filename,SITE/filename)
 for filename in ['report_summary.md','quality_plan.md','validation.json','pilot_audit.json','taxonomy.json','review_queue.json','browser_validation.json']:
  shutil.copy2(ROOT/filename,SITE/'unified'/filename)
 css='body{font:16px/1.8 system-ui;margin:32px auto;padding:0 24px;max-width:1100px;color:#21343c;background:#f8fafb}h1,h2,h3,a{color:#176c70}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}th,td{border:1px solid #cbd9dc;padding:10px;text-align:left}code,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap}'
 for stem in ['report_summary','quality_plan']:
  content=markdown.markdown((ROOT/(stem+'.md')).read_text(),extensions=['tables','fenced_code'])
  html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>统一标注 · '+stem+'</title><style>'+css+'</style></head><body><a href="../">← 返回统一工作台</a>'+content+'</body></html>'
  (SITE/'unified'/(stem+'.html')).write_text(html)
 paths={SITE/'index.html',SITE/'app.js',SITE/'styles.css',SITE/'assets/multi_manifest.json'}
 paths.update(p for p in (SITE/'unified').rglob('*') if p.is_file() and p.name!='release_manifest.json')
 m=json.loads((SITE/'assets/multi_manifest.json').read_text())
 for c in m['clips']:
  paths.update(SITE/l['video'] for l in c['layers'] if l.get('video'))
 paths.update(p for p in (CODE/'unified').iterdir() if p.is_file())
 release={'schema':'unified.release_manifest.v1','created_utc':datetime.now(timezone.utc).isoformat(),'scope':'served frontend, unified annotations/reports, all 43 video resources and pilot tools; original auxiliary geometry resources remain in their previous release','reference_commit':'6b54a95580729bca1726d9c3fdfebf7727435729','episodes':len(m['clips']),'training_eligible':False,'resources':[digest(p) for p in sorted(paths)]}
 (ROOT/'release_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2))
 shutil.copy2(ROOT/'release_manifest.json',SITE/'unified/release_manifest.json')
 print('Packaged',len(paths),'resources')

if __name__=='__main__':main()
