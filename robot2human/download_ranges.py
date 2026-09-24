"""Resumable, checksum-verified public model download with bounded parallel ranges."""
import os,json,pathlib,urllib.request,urllib.parse,time,concurrent.futures,hashlib,threading
ROOT=pathlib.Path('/user/panmiao/workspace/robot2human-vace-20260924');SHARED=pathlib.Path('/user/liuhanyu/model_ckpt/Wan2.1-T2V-1.3B')
CHUNK=64<<20;state={};lock=threading.Lock()
def model(name,listing):
 dest=ROOT/'models'/name;dest.mkdir(parents=True,exist_ok=True)
 files=json.loads((ROOT/listing).read_text())['Data']['Files'];tasks=[];handles=[]
 for f in files:
  p=f['Path'];size=f.get('Size',0)
  if not size or p.startswith('assets/') or p in ['README.md','.gitattributes']:continue
  out=dest/p;out.parent.mkdir(parents=True,exist_ok=True)
  if (p.startswith('google/') or p in ['models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth']) and (SHARED/p).is_file() and (SHARED/p).stat().st_size==size:
   if not out.exists():out.symlink_to(SHARED/p)
   continue
  if out.is_file() and out.stat().st_size==size:continue
  part=out.with_suffix(out.suffix+'.ranges.part');marks=out.with_suffix(out.suffix+'.chunks');marks.mkdir(exist_ok=True)
  old=out.with_suffix(out.suffix+'.part')
  if old.exists() and not part.exists():
   previous=old.stat().st_size;old.rename(part)
   for j in range(previous//CHUNK):(marks/str(j)).touch()
  fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,size)
  handles.append((fd,part,out,marks,f))
  for j,start in enumerate(range(0,size,CHUNK)):
   if (marks/str(j)).exists():continue
   tasks.append((fd,f,start,min(size,start+CHUNK)-1,marks/str(j)))
 def chunk(task):
  fd,f,start,end,mark=task;p=f['Path'];url='https://www.modelscope.cn/api/v1/models/Wan-AI/'+name+'/repo?Revision=master&FilePath='+urllib.parse.quote(p,safe='')
  for attempt in range(5):
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=90) as r:
     if r.status!=206 and (start!=0 or end+1!=f['Size']):raise RuntimeError('Range unsupported '+str(r.status))
     n=0
     while n<=end-start:
      b=r.read(min(4<<20,end-start+1-n))
      if not b:break
      os.pwrite(fd,b,start+n);n+=len(b)
     if n!=end-start+1:raise RuntimeError('Incomplete range')
    mark.touch();return n
   except Exception as e:
    print('RETRY',p,start,attempt,repr(e),flush=True);time.sleep(2+attempt*3)
  raise RuntimeError('Range failed '+p+':'+str(start))
 started=time.time();done=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
  for fut in concurrent.futures.as_completed([pool.submit(chunk,t) for t in tasks]):
   done+=fut.result()
   progress={'model':name,'downloaded_this_run':done,'seconds':time.time()-started,'mb_per_s':done/1e6/max(1,time.time()-started),'ranges_remaining':sum(not t[4].exists() for t in tasks)}
   (ROOT/'range_progress.json').write_text(json.dumps(progress));print('PROGRESS',json.dumps(progress),flush=True)
 for fd,part,out,marks,f in handles:
  os.fsync(fd);os.close(fd);expected=f.get('Sha256') or f.get('SHA256')
  if expected:
   h=hashlib.sha256()
   with part.open('rb') as src:
    for b in iter(lambda:src.read(8<<20),b''):h.update(b)
   if h.hexdigest()!=expected:raise RuntimeError('SHA256 mismatch '+str(part))
  part.replace(out)
 (ROOT/(name+'_READY.json')).write_text(json.dumps({'model':str(dest)}));print('MODEL_READY',name,flush=True)
model('Wan2.1-VACE-1.3B','Wan2.1-VACE-1.3B_files.json')
model('Wan2.1-VACE-14B','model_files.json')
