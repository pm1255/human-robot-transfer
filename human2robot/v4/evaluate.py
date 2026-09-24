"""RoboTwin native EE controller. Outcome comes only from env.check_success()."""
import argparse,importlib,json,os,sys,time,traceback,subprocess
from pathlib import Path
import cv2,numpy as np,torch,yaml
from v3.common import state7,apply_target,dump,sha
from v4.common import SIZE,EVAL_TEXT,WEIGHTS
from v4.policy import WanEvalPolicy


def configuration(root,out):
    args=yaml.safe_load((root/'task_config/demo_clean.yml').read_text())
    emb=yaml.safe_load((root/'task_config/_embodiment_config.yml').read_text())['aloha-agilex']['file_path']
    robot=yaml.safe_load((root/emb/'config.yml').read_text())
    camera=yaml.safe_load((root/'task_config/_camera_config.yml').read_text())[args['camera']['head_camera_type']]
    args.update(task_name='place_empty_cup',task_config='demo_clean',ckpt_setting='human2robot_wan_v4',policy_name='h2r_gripper',left_robot_file=emb,right_robot_file=emb,dual_arm_embodied=True,left_embodiment_config=robot,right_embodiment_config=robot,head_camera_h=camera['h'],head_camera_w=camera['w'],eval_mode=True,render_freq=0,save_data=False,collect_data=False,eval_video_log=False,save_path=str(out/'scratch'))
    args['data_type']['endpose']=True
    return args


def poses(obs):
    rows=[]
    for side in ['left','right']:
        p=np.asarray(obs['endpose'][f'{side}_endpose'],np.float32)
        assert p.shape==(7,)
        rows.append(np.r_[p[:3],p[4:7],p[3],obs['endpose'][f'{side}_gripper']])
    return np.array(rows,np.float32)


def run(a):
    root=Path(a.robotwin).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    os.chdir(root);sys.path.insert(0,str(root));sys.path.insert(0,str(root/'description/utils'))
    from envs.place_empty_cup import place_empty_cup
    args=configuration(root,out);torch.set_num_threads(4)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False) if a.checkpoint else None
    checkpoint_hash=sha(a.checkpoint) if ckpt else None
    if ckpt:
        assert sha(WEIGHTS/'diffusion_pytorch_model.safetensors')==ckpt['base_weight_sha'], 'Wan base weight hash mismatch'
        model=WanEvalPolicy().cuda().eval();model.load_adapter(ckpt['adapter'])
        st={k:v.cuda() for k,v in ckpt['sim_stats'].items()}
        print('STRICT_LOAD_OK',checkpoint_hash,flush=True)
    seeds=json.loads(Path(a.seeds).read_text()) if a.seeds else list(range(210000,210000+a.episodes))
    records=[]
    for seed in seeds:
        existing=out/f'episode_{seed}.json'
        if existing.exists():
            old=json.loads(existing.read_text())
            if 'error' not in old and old.get('checkpoint')==checkpoint_hash and old.get('step_limit')==a.max_steps:
                records.append(old);continue
        env=place_empty_cup();env.test_num=0;env.suc=0
        rec={'seed':seed,'mode':'policy' if ckpt else 'expert','valid_scene':False,'success':False,'checkpoint':checkpoint_hash,'step_limit':a.max_steps,'video_fps':15,'video_time_is_physics_time':False};ff=None;states=[];raw=[];commands=[];after=[];started=time.time()
        try:
            env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
            if not ckpt:
                env.play_once();rec.update(valid_scene=bool(env.plan_success),success=bool(env.check_success()))
            else:
                rec['valid_scene']=True;env.set_instruction(instruction=EVAL_TEXT)
                env.step_lim=a.max_steps
                while env.take_action_cnt<env.step_lim:
                    obs=env.get_obs();rgb=obs['observation']['head_camera']['rgb'];pp=poses(obs)
                    if ff is None:
                        h,w=rgb.shape[:2];ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','15','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',str(out/f'rollout_{seed}.mp4')],stdin=subprocess.PIPE)
                    ff.stdin.write(np.ascontiguousarray(rgb).tobytes())
                    image=cv2.resize(rgb,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
                    with torch.no_grad():
                        pred=model(torch.as_tensor(np.stack([image,image]),device='cuda'),(torch.as_tensor(state7(pp),device='cuda')-st['sm'])/st['ss'],torch.tensor([0,1],device='cuda'))
                        pred=(pred*st['ts']+st['tm']).cpu().numpy()[:,0]
                    if not np.isfinite(pred).all():raise ValueError('non-finite policy command')
                    # Same local displacement cap for all policies, with raw and clipped actions saved.
                    action=pred.copy();norm=np.linalg.norm(action[:,:3],axis=1);action[:,:3]*=np.minimum(1,.06/np.maximum(norm,1e-8))[:,None]
                    angle=np.linalg.norm(action[:,3:6],axis=1);action[:,3:6]*=np.minimum(1,.35/np.maximum(angle,1e-8))[:,None]
                    action[:,6]=np.clip(action[:,6],0,1)
                    target=np.stack([apply_target(p,x) for p,x in zip(pp,action)])
                    # Native simulator expects wxyz quaternion and normalized opening per arm.
                    native=target[:,[0,1,2,6,3,4,5,7]].reshape(16)
                    env.take_action(native,action_type='ee')
                    states.append(pp);raw.append(pred);commands.append(target);after.append(poses(env.get_obs()))
                    if env.check_success() or env.eval_success:rec['success']=True;break
                rec.update(steps=len(raw),checkpoint=checkpoint_hash,gripper_threshold=None)
        except Exception as e:rec.update(error=repr(e),trace=traceback.format_exc()[-2500:])
        finally:
            if ff is not None:ff.stdin.close();ff.wait(timeout=60)
            try:env.close_env()
            except Exception:pass
        rec['seconds']=time.time()-started;records.append(rec)
        np.savez_compressed(out/f'actions_{seed}.npz',state=np.array(states),raw_action=np.array(raw),command=np.array(commands),executed=np.array(after))
        dump(out/f'episode_{seed}.json',rec);dump(out/'summary.json',{'episodes':records,'successes':sum(r['success'] for r in records),'attempted':len(records),'errors':sum('error' in r for r in records),'success_rate_including_execution_failures':sum(r['success'] for r in records)/len(records)})
        print('ROLLOUT',json.dumps(rec),flush=True)

    dump(out/'summary.json',{'episodes':records,'successes':sum(r['success'] for r in records),'attempted':len(records),'errors':sum('error' in r for r in records),'success_rate_including_execution_failures':sum(r['success'] for r in records)/len(records)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robotwin',default='/user/panmiao/workspace/robodojo-data-windtunnel/runtime/RoboTwin');p.add_argument('--checkpoint');p.add_argument('--out',required=True);p.add_argument('--seeds');p.add_argument('--episodes',type=int,default=1);p.add_argument('--max-steps',type=int,default=400);run(p.parse_args())
