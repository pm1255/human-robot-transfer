"""Authorized continuation: submit the human cohort only after data gates and real cohort completion."""
import json,os,subprocess,time,traceback,fcntl
from pathlib import Path
from v3.common import ROOT,dump

ENV=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:ENV.pop(k,None)
ENV['CCTL_SERVER']='https://cybertron.modelbest.co'
PY='/user/panmiao/workspace/envs/windtunnel/bin/python'
base=['cctl','--no-input','pytorchjob','create','--project','pm','--cluster','yingbo_train','--resource-pool','embody-train2','--billing-account-id','N00003','--priority','NORMAL','--image','embody/embody-train:py312pt210cu128-230058','--gpu-model','h100','--gpu','8','--cpu','80','--memory','256','--nodes','1','--description','human2robot v3: human RGB gripper 10k vs shuffled, 4 paired seeds, 20k pretrain + 10k RoboTwin cup finetune','--entry',f'bash {ROOT}/code/v3/run_human.sh']
try:
    lock=(ROOT/'human_continuation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(ROOT/'HUMAN_CONTINUATION_STARTED.json',{'pid':os.getpid(),'started':time.time()})
    # Never issue a second request if one has already been submitted or its outcome is uncertain.
    assert not (ROOT/'human_submit_attempt.json').exists()
    for _ in range(960):
        if (ROOT/'human_stable_manifest.json').exists() and (ROOT/'REAL_JOB.json').exists():break
        time.sleep(30)
    assert (ROOT/'human_stable_manifest.json').exists(),'Human preparation did not complete within 8 hours'
    subprocess.run([PY,'-m','v3.freeze_data'],cwd=ROOT/'code',check=True)
    subprocess.run([PY,'-m','v3.pack_data','--domains','human'],cwd=ROOT/'code',check=True)
    job=json.loads((ROOT/'REAL_JOB.json').read_text());jid=job['id']
    for _ in range(960):
        raw=subprocess.check_output(['cctl','--no-input','job','get',f'tasks/{jid}','-o','json'],env=ENV,text=True)
        status=json.loads(raw)['status']
        if status=='Succeeded':break
        if status in ['Failed','Cancelled','Stopped']:raise RuntimeError(f'Real cohort {jid} {status}; inspect before starting human cohort')
        time.sleep(30)
    assert status=='Succeeded','Real cohort has not finished within 8 hours'
    for file in ['common.py','policy.py','train.py']:
        assert (ROOT/'training/code_snapshot_real/v3'/file).read_bytes()==(ROOT/'code/v3'/file).read_bytes(),f'Training code changed: {file}'
    dry=subprocess.check_output(base+['--dry-run'],env=ENV,text=True);(ROOT/'human_dry_run.json').write_text(dry)
    dump(ROOT/'human_submit_attempt.json',{'authorized_by':'user asked to apply embody-train2 resources and start matched 10k experiment','timestamp':time.time(),'state':'requesting; do not automatically retry uncertain request'})
    result=subprocess.run(base,env=ENV,text=True,capture_output=True)
    (ROOT/'human_submit_stdout.txt').write_text(result.stdout);(ROOT/'human_submit_stderr.txt').write_text(result.stderr)
    result.check_returncode();job=json.loads(result.stdout);dump(ROOT/'HUMAN_JOB.json',job);print('HUMAN_JOB_SUBMITTED',job['id'],flush=True)
except Exception as e:
    dump(ROOT/'HUMAN_CONTINUATION_BLOCKED.json',{'error':repr(e),'trace':traceback.format_exc()});raise
