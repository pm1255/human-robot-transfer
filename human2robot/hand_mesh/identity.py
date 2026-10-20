"""Globally assign anatomical left/right labels with injectivity and continuity.

This is an estimated identity repair, never an independent handedness truth.
Existing detector output is preserved in the sidecar. Empty frames reset tracks.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import numpy as np


def repair(data):
    result=copy.deepcopy(data);changes=[];duplicates=[];frames=result['frames']
    start=0
    while start<len(frames):
        if not frames[start]['hands']:start+=1;continue
        end=start
        while end<len(frames) and frames[end]['hands']:end+=1
        all_states=[];costs=[];backs=[]
        previous_hands=[]
        for frame in frames[start:end]:
            hands=frame['hands']
            if len(hands)>2:raise ValueError('Expected at most two hands per frame')
            labels=[h['handedness']['category'] for h in hands]
            if len(set(labels))<len(labels):duplicates.append(frame['frame_index'])
            states=list(itertools.permutations(('Left','Right'),len(hands)))
            emissions=[]
            for state in states:
                score=0.
                for hand,side in zip(hands,state):
                    confidence=np.clip(hand['handedness'].get('score',.5),.51,.98)
                    prob=confidence if hand['handedness']['category']==side else 1-confidence
                    score-=np.log(prob)
                emissions.append(score)
            if not costs:
                total=np.asarray(emissions);back=[-1]*len(states)
            else:
                transition=np.zeros((len(all_states[-1]),len(states)))
                for i,old in enumerate(all_states[-1]):
                    for j,new in enumerate(states):
                        for side in set(old)&set(new):
                            p=previous_hands[old.index(side)]['landmarks'];q=hands[new.index(side)]['landmarks']
                            x=np.array([[p[k]['x'],p[k]['y']] for k in [0,5,9,13,17]])
                            y=np.array([[q[k]['x'],q[k]['y']] for k in [0,5,9,13,17]])
                            delta=np.median(np.linalg.norm(x-y,axis=1))
                            transition[i,j]+=min(float((delta/.035)**2),40.)
                score=np.asarray(costs[-1])[:,None]+transition
                back=np.argmin(score,axis=0);total=np.asarray(emissions)+score[back,np.arange(len(states))]
            all_states.append(states);costs.append(total);backs.append(back);previous_hands=hands
        best=int(np.argmin(costs[-1]));path=[]
        for k in range(len(all_states)-1,-1,-1):
            path.append(all_states[k][best]);best=int(backs[k][best])
        for frame,assignment in zip(frames[start:end],reversed(path)):
            for i,(hand,side) in enumerate(zip(frame['hands'],assignment)):
                hand['detector_handedness']=copy.deepcopy(hand['handedness'])
                hand['identity_method']='viterbi_injective_wrist_palm_continuity'
                hand['identity_verified']=False
                if side!=hand['handedness']['category']:
                    changes.append(dict(frame_index=frame['frame_index'],hand_index=i,before=hand['handedness']['category'],after=side))
                    hand['handedness']={'category':side,'score':None,'source':'temporal_identity_estimate'}
        start=end
    result['identity_repair']=dict(changed_hands=len(changes),changed_frames=len({c['frame_index'] for c in changes}),
        original_duplicate_side_frames=len(duplicates),changes=changes,ground_truth_verified=False)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=repair(json.loads(a.input.read_text()));a.output.write_text(json.dumps(d,allow_nan=False))
    print(json.dumps({k:v for k,v in d['identity_repair'].items() if k!='changes'}))


if __name__=='__main__':main()
