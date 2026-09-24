import json,collections
import numpy as np
from v3.common import *


def pack(domain,split):
    folder='human_stable' if domain=='human' else domain
    rows=json.loads((ROOT/f'{domain}_index.json').read_text())[split]
    images=np.empty((len(rows),SIZE,SIZE,3),np.uint8);state=np.empty((len(rows),7),np.float32)
    target=np.empty((len(rows),HORIZON,7),np.float32); side=np.empty(len(rows),np.int64);mask=np.ones((len(rows),HORIZON),np.float32)
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):groups[row[0]].append((i,row))
    for file,items in groups.items():
        with np.load(ROOT/folder/file) as d:
            lookup={tuple(w):i for i,w in enumerate(d['windows'])}
            im=d['images'];pp=d['poses'];actions=d['actions'] if 'actions' in d else None
            for i,(_,t,s,ep) in items:
                p=pp[t,s];future=pp[np.minimum(np.arange(t+1,t+HORIZON+1),len(pp)-1),s] if actions is None else actions[t:t+HORIZON,s]
                if domain!='sim':mask[i]=[1,0,0,0]
                images[i]=im[t];state[i]=state7(p);target[i]=local_targets(p,future);side[i]=s
    assert np.isfinite(state).all() and np.isfinite(target).all()
    dest=ROOT/'packed'/domain/split;dest.mkdir(parents=True,exist_ok=True)
    for name,a in dict(images=images,state=state,target=target,side=side,mask=mask).items():np.save(dest/f'{name}.npy',a)
    dump(dest/'manifest.json',{'domain':domain,'split':split,'n':len(rows),'index_sha256':sha(ROOT/f'{domain}_index.json'),'shapes':{'state':list(state.shape),'target':list(target.shape)}})
    print('PACKED',domain,split,len(rows),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--domains',nargs='+',default=['human','bridge','sim']);a=p.parse_args()
    for d in a.domains:
        for s in ['train','validation','test']:pack(d,s)
    print('PACKED: pretraining exactly one 0.2-second transition per sample; finetuning 4 native 15Hz actions',flush=True)
