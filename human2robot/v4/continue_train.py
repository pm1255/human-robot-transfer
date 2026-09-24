"""Wait on concrete preparation/smoke gates, then submit the authorized training once."""
import os,time,subprocess,json,traceback,fcntl
from v4.common import *

env=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
env['CCTL_SERVER']='https://cybertron.modelbest.co'
try:
    lock=(ROOT/'continue_train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(ROOT/'CONTINUATION_STARTED.json',{'pid':os.getpid(),'time':time.time()})
    for attempt in range(2880):
        if (ROOT/'SMOKE_JOB.json').exists():
            job=json.loads((ROOT/'SMOKE_JOB.json').read_text())
            status=json.loads(subprocess.check_output(['cctl','--no-input','job','get',f'tasks/{job["id"]}','-F','id,status'],env=env,text=True))['status']
            if status in ['Failed','Stopped','Cancelled']:raise RuntimeError(f'Smoke job {job["id"]} {status}; no formal training submitted')
            if status=='Succeeded' and (ROOT/'WAN_SMOKE_PASSED.json').exists() and (ROOT/'RGB_READY.json').exists():break
        time.sleep(30)
    else:raise RuntimeError('Preparation or smoke did not finish within 24 hours')
    subprocess.run(['/user/panmiao/workspace/envs/windtunnel/bin/python','-m','v4.submit','--phase','train'],env=env,cwd=ROOT/'code',check=True)
except Exception as e:
    dump(ROOT/'CONTINUATION_BLOCKED.json',{'error':repr(e),'trace':traceback.format_exc()});raise
