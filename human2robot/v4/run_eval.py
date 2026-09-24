"""Paired native RoboTwin evaluation. Resume saved complete episodes, never shrink denominator."""
import concurrent.futures,json,os,subprocess
from v4.common import *

PY='/user/chenzilong/envs/robotwin/bin/python'
out=ROOT/'evaluation';out.mkdir(exist_ok=True)

def evaluate(args,gpu,log,timeout=86400):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with open(log,'a') as f:
        subprocess.run([PY,'-u','-m','v4.evaluate']+args,cwd=ROOT/'code',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=True)

ckpt=ROOT/'training/seed17/bridge_joint/finetune_best.pt'
evaluate(['--checkpoint',str(ckpt),'--out',str(out/'smoke'),'--max-steps','10'],0,out/'smoke.log',1800)
r=json.loads((out/'smoke/summary.json').read_text())['episodes'][0]
assert 'error' not in r and r.get('steps',0)>0
dump(out/'POLICY_CHAIN_SMOKE_PASSED.json',r)
# Use the previously frozen expert-feasible scene set; never select using policy outcomes.
seeds=json.loads((ROOT/'test_seeds.json').read_text());assert len(seeds)==100 and len(set(seeds))==100

def worker(gpu):
    group=GROUPS[gpu%4]
    for seed in ([17,43] if gpu<4 else [29,71]):
        path=ROOT/f'training/seed{seed}'/group/'finetune_best.pt';dest=out/f'seed{seed}'/group
        evaluate(['--checkpoint',str(path),'--out',str(dest),'--seeds',str(ROOT/'test_seeds.json')],gpu,out/f'{group}_{seed}.log')
        rows=json.loads((dest/'summary.json').read_text())['episodes'];assert len(rows)==100 and not any('error' in x for x in rows)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
report={}
for group in GROUPS:
    rows=[json.loads((out/f'seed{s}'/group/'summary.json').read_text()) for s in SEEDS]
    report[group]={'seeds':SEEDS,'successes':[r['successes'] for r in rows],'attempted':[r['attempted'] for r in rows],
        'mean_success_rate':sum(r['successes'] for r in rows)/400}
dump(out/'FINAL_REPORT.json',report)
