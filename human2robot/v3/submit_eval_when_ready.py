"""Authorized post-training transfer and eval submission, with fail-closed resource gates."""
import os,json,time,subprocess,tarfile,fcntl,traceback
from v3.common import ROOT,dump

env=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
env['CCTL_SERVER']='https://cybertron.modelbest.co'

def call(args):
    return json.loads(subprocess.check_output(['cctl','--no-input']+args,env=env,text=True))

def wait_job(jid):
    for _ in range(720):
        state=call(['job','get',f'tasks/{jid}','-o','json'])['status']
        if state=='Succeeded':return
        if state in ['Failed','Stopped','Cancelled']:raise RuntimeError(f'Upstream job {jid} {state}')
        time.sleep(60)
    raise RuntimeError(f'Upstream job {jid} still unfinished after 12 hours')

try:
    lock=(ROOT/'eval_continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'eval_submit_attempt.json').exists()
    dump(ROOT/'EVAL_CONTINUATION_STARTED.json',{'pid':os.getpid(),'started':time.time()})
    for _ in range(720):
        if (ROOT/'HUMAN_JOB.json').exists():break
        if (ROOT/'HUMAN_CONTINUATION_BLOCKED.json').exists():raise RuntimeError('Human continuation blocked; inspect its recorded reason')
        time.sleep(60)
    for name in ['REAL_JOB.json','HUMAN_JOB.json']:wait_job(json.loads((ROOT/name).read_text())['id'])
    archive=ROOT/'eval-release-v3.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in (ROOT/'code/v3').glob('*.py'):tar.add(p,arcname='code/v3/'+p.name)
        for seed in [17,29,43,71]:
            for group in ['human','bridge','human_shuffled','robotwin_only']:
                d=ROOT/f'training/seed{seed}'/group
                assert (d/'DONE.json').exists(),f'Incomplete training {d}'
                for name in ['finetune_best.pt','finetune_offline.json','protocol.json']:tar.add(d/name,arcname=f'training/seed{seed}/{group}/{name}')
    source=f'yingbo_train:/yingbo_train{archive}';target=f'paratera_ningxia:/paratera_ningxia{archive}'
    args=['file-task','sync',source,target,'--description','human2robot v3 sixteen checkpoints and native evaluator']
    dump(ROOT/'eval_transfer_dryrun.json',call(args+['--dry-run']))
    transfer=call(args);dump(ROOT/'EVAL_TRANSFER.json',transfer)
    tid=transfer.get('id') or transfer.get('name','').rsplit('/',1)[-1]
    assert tid,transfer
    for _ in range(360):
        result=call(['file-task','get',str(tid)]);state=result.get('status',result.get('state'))
        if state in ['Completed','Succeeded','completed','succeeded']:break
        if state in ['Failed','Cancelled','failed','cancelled']:raise RuntimeError(result)
        if state is None:raise RuntimeError(f'Unrecognized transfer response: {result}')
        time.sleep(30)
    assert state in ['Completed','Succeeded','completed','succeeded']
    entry=f'tar -xzf {archive} -C {ROOT} && cd {ROOT}/code && timeout 36000 /user/chenzilong/envs/robotwin/bin/python -u -m v3.run_evaluation'
    args=['job','create','--project','pm','--cluster','paratera_ningxia','--resource-pool','embody-eval','--billing-account-id','N00003','--priority','NORMAL','--image','embody/embody-vla:docker-evaluation-base-4090-202606141624','--gpu-model','4090','--gpu','8','--cpu','60','--memory','256','--description','human2robot v3 16 gripper policies; paired 100 cup scenes; chain smoke and expert feasibility gates','--entry',entry]
    dump(ROOT/'eval_job_dryrun.json',call(args+['--dry-run']))
    dump(ROOT/'eval_submit_attempt.json',{'timestamp':time.time(),'state':'requesting; uncertain attempts require inspection, never auto-retry'})
    dump(ROOT/'EVAL_JOB.json',call(args))
except Exception as e:
    dump(ROOT/'EVAL_CONTINUATION_BLOCKED.json',{'error':repr(e),'trace':traceback.format_exc()});raise
