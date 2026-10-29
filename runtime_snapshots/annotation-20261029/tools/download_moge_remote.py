from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time,hashlib
root=Path('/user/panmiao/human-robot-transfer/models/moge-3-vitl')
url='https://hf-mirror.com/Ruicheng/moge-3-vitl/resolve/184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be/model.pt'
size=1481333394
path=root/'model.pt'
if path.exists() and path.stat().st_size!=size:path.rename(root/'model.pt.partial_initial')
parts=root/'parts';parts.mkdir(exist_ok=True)
chunk=8*1024*1024

def get(i):
 a=i*chunk;b=min(size,a+chunk)-1;p=parts/f'{i:04d}'
 if p.exists() and p.stat().st_size==b-a+1:return i
 for attempt in range(8):
  try:
   req=Request(url+f'?download=true&part={i}',headers={'Range':f'bytes={a}-{b}'})
   with urlopen(req,timeout=45) as r:
    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{size}':raise RuntimeError(f'bad range {r.status} {r.headers.get("Content-Range")}')
    data=r.read()
   if len(data)!=b-a+1:raise RuntimeError(f'truncated {len(data)}')
   tmp=p.with_suffix(".tmp");tmp.write_bytes(data);tmp.replace(p);return i
  except Exception as e:
   if attempt==7:raise
 print('failed',i,flush=True)

with ThreadPoolExecutor(8) as pool:
 for j,f in enumerate(as_completed([pool.submit(get,i) for i in range((size+chunk-1)//chunk)])):
  print('downloaded_part',f.result(),'count',j+1,flush=True)
with path.open('wb') as out:
 for i in range((size+chunk-1)//chunk):out.write((parts/f'{i:04d}').read_bytes())
assert path.stat().st_size==size
assert hashlib.sha256(path.read_bytes()).hexdigest() == '9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925'
print('complete_verified_sha256',size,flush=True)
