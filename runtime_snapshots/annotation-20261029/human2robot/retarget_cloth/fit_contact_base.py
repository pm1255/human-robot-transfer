"""Select fixed arm bases from sequence-wide pose reachability, never per frame."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from human2robot.retarget_cloth.run_preview import Robot,transform
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);data=json.loads(a.input.read_text());rows=data['frames'];robot=Robot(Path('/user/panmiao/workspace/robodojo-data-windtunnel/runtime/RoboTwin/assets/embodiments/aloha-agilex'),a.root,face_budget=0)
    sample=np.array([120,300,600,960,1200,1680,2160,2640,3120,3600,4080,4560]);results={};progress=Progress('outputs/object_annotation_20261020','robot_contact_base_search',18)
    for si,side in enumerate(['Left','Right']):
        candidates=[]
        for x in [.22,.30,.38]:
            for y in [-.22,-.32,-.42]:
                base=transform(Rotation.from_euler('z',np.pi/2).as_matrix(),[-x if side=='Left' else x,y,.045]);inv=np.linalg.inv(base);solutions=[];costs=[]
                for i in sample:
                    goal=inv@np.asarray(rows[i]['hands'][side]['target']);seed=np.array(rows[i]['hands'][side]['q'])
                    def residual(q):
                        T,pts=robot.robot.fk(q,True);rot1=Rotation.from_matrix(goal[:3,:3]@T[:3,:3].T).as_rotvec();rot2=Rotation.from_matrix(goal[:3,:3]@np.diag([1,-1,-1])@T[:3,:3].T).as_rotvec();rot=rot1 if np.linalg.norm(rot1)<np.linalg.norm(rot2) else rot2
                        return np.r_[(T[:3,3]-goal[:3,3])/.012,rot/.20,np.maximum(0,.018-(pts[2:,2]+.045))/.002]
                    fits=[least_squares(residual,s,bounds=(-3*np.pi,3*np.pi),max_nfev=80) for s in [seed,np.array([0.,-1.5,1.8,0,.5,0.]),np.array([0.,1.5,-1.8,0,-.5,0.])]]
                    best=min(fits,key=lambda f:np.linalg.norm(residual(f.x)));costs.append(float(np.linalg.norm(residual(best.x))));solutions.append(best.x.tolist())
                score=float(np.mean(costs)+.3*np.percentile(costs,90));candidates.append(dict(base=base.tolist(),score=score,costs=costs,seed=solutions[0]));progress.update(si*9+len(candidates),phase=f'fixed_base_{side}')
        best=min(candidates,key=lambda c:c['score']);results[side]=dict(selected=best,candidates=candidates)
        for row in rows:row['robot_bases'][side]=best['base']
        rows[120]['hands'][side]['q']=best['seed']
    atomic_json(a.root/'robot_trajectory.json',data);atomic_json(a.root/'base_search.json',results);progress.update(18,'completed');progress.close();print(json.dumps({s:r['selected'] for s,r in results.items()}))

if __name__=='__main__':main()
