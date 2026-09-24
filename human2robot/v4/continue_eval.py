"""Transfer actual trained Wan adapters + base, then submit native RoboTwin evaluation once."""
import os,json,time,subprocess,tarfile,traceback,fcntl
from v4.common import *

ENV=os.environ.copy()
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:ENV.pop(k,None)
ENV['CCTL_SERVER']='https://cybertron.modelbest.co'
def cctl(args):return json.loads(subprocess.check_output(['cctl','--no-input']+args,env=ENV,text=True))

try:
    lock=(ROOT/'continue_eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(ROOT/'EVAL_CONTINUATION_STARTED.json',{'pid':os.getpid(),'time':time.time()})
    for attempt in range(2016):
        if (ROOT/'CONTINUATION_BLOCKED.json').exists():raise RuntimeError('Training preparation blocked')
        if (ROOT/'TRAIN_JOB.json').exists():
            job=json.loads((ROOT/'TRAIN_JOB.json').read_text());state=cctl(['job','get',f'tasks/{job["id"]}','-F','id,status'])['status']
            if state in ['Failed','Stopped','Cancelled']:raise RuntimeError(f'Training {job["id"]} {state}; no eval submitted')
            if state=='Succeeded':break
        time.sleep(300)
    else:raise RuntimeError('Training did not complete within seven days')
    assert (ROOT/'TRAIN_DONE.json').exists()
    assert not (ROOT/'EVAL_SUBMIT_ATTEMPT.json').exists(),'Existing evaluation submission attempt'
    bundle=ROOT/'eval-release.tar'
    with tarfile.open(bundle,'w') as tar:
        tar.add(ROOT/'code',arcname='code',filter=lambda x:None if '__pycache__' in x.name else x)
        tar.add(ROOT/'deps',arcname='deps',filter=lambda x:None if '__pycache__' in x.name else x)
        tar.add(ROOT/'text'/f'{key(EVAL_TEXT)}.pt',arcname=f'text/{key(EVAL_TEXT)}.pt')
        tar.add(ROOT/'test_seeds.json',arcname='test_seeds.json')
        tar.add(ROOT/'DATA_AUDIT.json',arcname='DATA_AUDIT.json')
        for name in ['config.json','diffusion_pytorch_model.safetensors','Wan2.1_VAE.pth']:
            tar.add(WEIGHTS/name,arcname='models/'+name)
        for seed in SEEDS:
            for group in GROUPS:
                p=ROOT/f'training/seed{seed}'/group/'finetune_best.pt';assert p.exists()
                tar.add(p,arcname=str(p.relative_to(ROOT)))
    transfer=['file-task','sync',f'yingbo_train:/yingbo_train{bundle}',f'paratera_ningxia:/paratera_ningxia{bundle}',
        '--description','Human2Robot Wan base, VAE, UMT5 prompt embedding and 16 trained adapters to RoboTwin eval']
    dump(ROOT/'EVAL_TRANSFER_DRY_RUN.json',cctl(transfer+['--dry-run']))
    assert not (ROOT/'EVAL_TRANSFER_ATTEMPT.json').exists()
    dump(ROOT/'EVAL_TRANSFER_ATTEMPT.json',{'time':time.time(),'bundle_sha':sha(bundle)})
    tr=cctl(transfer);dump(ROOT/'EVAL_TRANSFER.json',tr);tid=tr['name'].split('/')[-1]
    for attempt in range(2880):
        status=cctl(['file-task','get',tid])['status']
        if status=='Completed':break
        if status in ['Failed','Cancelled','Stopped']:raise RuntimeError('Transfer '+status)
        time.sleep(30)
    else:raise RuntimeError('Cross-filesystem transfer timed out')
    entry=f'tar -xf {bundle} -C {ROOT} && cd {ROOT}/code && export WAN_WEIGHTS={ROOT}/models PYTHONPATH={ROOT}/deps:{ROOT}/code && timeout 259200 /user/chenzilong/envs/robotwin/bin/python -u -m v4.run_eval'
    cmd=['job','create','--cluster','paratera_ningxia','--billing-account-id','N00003','--project','pm',
         '--resource-pool','embody-eval','--priority','NORMAL','--gpu-model','4090','--gpu','8','--cpu','60','--memory','256',
         '--image','embody/embody-vla:docker-evaluation-base-4090-202606141624',
         '--description','Wan Human2Robot 16 models x 100 paired cup scenes; real checkpoint chain smoke; resumable episodes','--entry',entry]
    dump(ROOT/'EVAL_DRY_RUN.json',cctl(cmd+['--dry-run']));dump(ROOT/'EVAL_SUBMIT_ATTEMPT.json',{'time':time.time()})
    job=cctl(cmd);dump(ROOT/'EVAL_JOB.json',job);print('EVAL_SUBMITTED',job['id'],flush=True)
except Exception as e:
    dump(ROOT/'EVAL_CONTINUATION_BLOCKED.json',{'error':repr(e),'trace':traceback.format_exc()});raise
