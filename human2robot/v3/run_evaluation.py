"""One formal eval job: chain smoke -> expert seed feasibility -> 16 paired policy evaluations."""
import concurrent.futures,os,subprocess,json
from pathlib import Path
from v3.common import ROOT,dump

PY='/user/chenzilong/envs/robotwin/bin/python'

def run(args,gpu,log,timeout=14400):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with open(log,'w') as f:subprocess.run([PY,'-u','-m','v3.evaluate']+args,env=env,cwd=ROOT/'code',stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)

out=ROOT/'evaluation';out.mkdir(exist_ok=True)
# Mandatory actual policy/server/simulator chain smoke; task failure is permitted, infrastructure error is not.
smoke=out/'smoke';ckpt=ROOT/'training/seed17/bridge/finetune_best.pt'
run(['--checkpoint',str(ckpt),'--out',str(smoke),'--episodes','1','--max-steps','30'],0,out/'smoke.log',1200)
r=json.loads((smoke/'summary.json').read_text())['episodes'][0]
assert 'error' not in r and r.get('steps',0)>0,'End-to-end policy smoke failed'
dump(out/'POLICY_CHAIN_SMOKE_PASSED.json',r)

def expert(worker):
    seeds=list(range(220000+worker*25,220000+(worker+1)*25));file=out/f'expert_seeds_{worker}.json';dump(file,seeds)
    dest=out/f'expert_{worker}'
    run(['--seeds',str(file),'--out',str(dest)],worker,out/f'expert_{worker}.log',2400)
    return json.loads((dest/'summary.json').read_text())['episodes']
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:expert_rows=sum(list(pool.map(expert,range(8))),[])
valid=sorted(r['seed'] for r in expert_rows if r['valid_scene'] and r['success'] and 'error' not in r)
assert len(valid)>=100, f'Only {len(valid)} feasible expert scenes; refusing to shrink test set silently'
seeds=valid[:100];dump(out/'test_seeds.json',seeds);dump(out/'expert_audit.json',expert_rows)

def policies(worker):
    groups=['human','bridge','human_shuffled','robotwin_only'];group=groups[worker%4]
    training_seeds=[17,43] if worker<4 else [29,71]
    for seed in training_seeds:
        ckpt=ROOT/f'training/seed{seed}'/group/'finetune_best.pt';dest=out/f'seed{seed}'/group
        run(['--checkpoint',str(ckpt),'--seeds',str(out/'test_seeds.json'),'--out',str(dest)],worker,out/f'seed{seed}_{group}.log')
        rows=json.loads((dest/'summary.json').read_text())['episodes'];assert len(rows)==100
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(policies,range(8)))
subprocess.run([PY,'-m','v3.summarize_eval','--root',str(out),'--out',str(out/'FINAL_REPORT.json')],cwd=ROOT/'code',check=True)
