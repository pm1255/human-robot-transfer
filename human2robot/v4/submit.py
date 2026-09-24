"""Explicit user authorization: re-run the experiment on Wan in embody-train2."""
import os,subprocess,json,time,argparse
from v4.common import *

p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','train'],default='train');phase=p.parse_args().phase

env=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
env['CCTL_SERVER']='https://cybertron.modelbest.co'
cmd=['cctl','--no-input','pytorchjob','create','--project','pm','--cluster','yingbo_train',
     '--resource-pool','embody-train2','--billing-account-id','N00003','--priority','NORMAL',
     '--image','embody/embody-train:py312pt210cu128-230058','--gpu-model','h100',
     '--gpu','8','--cpu','80','--memory','512','--nodes','1',
     '--description',f'Human2Robot Wan2.1-1.3B {phase}: UMT5 language/video/action LoRA, matched 10k, 4 groups x 4 seeds',
     '--entry',f'bash {ROOT}/code/v4/'+('smoke.sh' if phase=='smoke' else 'run.sh')]
prefix=phase.upper()
if (ROOT/f'{prefix}_SUBMIT_ATTEMPT.json').exists():raise RuntimeError('Existing submission attempt; inspect before retrying')
dry=subprocess.check_output(cmd+['--dry-run'],env=env,text=True);(ROOT/f'{prefix}_DRY_RUN.json').write_text(dry)
dump(ROOT/f'{prefix}_SUBMIT_ATTEMPT.json',{'time':time.time(),'reason':'User explicitly requested Wan re-experiment and resource submission'})
r=subprocess.run(cmd,env=env,text=True,capture_output=True)
(ROOT/f'{prefix}_submit.stdout').write_text(r.stdout);(ROOT/f'{prefix}_submit.stderr').write_text(r.stderr);r.check_returncode()
job=json.loads(r.stdout);dump(ROOT/f'{prefix}_JOB.json',job);print(json.dumps(job),flush=True)
