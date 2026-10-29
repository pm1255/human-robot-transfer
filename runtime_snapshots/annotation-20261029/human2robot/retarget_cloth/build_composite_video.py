"""Render independent video ranges, concatenate and validate the complete clip."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from human2robot.object_annotation.common import Progress,atomic_json
from human2robot.hand_mesh.video import probe

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/robot_composite_20261020/full'));p.add_argument('--removal',default='outputs/robot_composite_20261020/flow_full');p.add_argument('--trajectory',default='outputs/semantic_robot_20261020/full_v3/robot_trajectory.json');p.add_argument('--prefix',default='source_composite');p.add_argument('--blend-removal',action='store_true');p.add_argument('--temporal-removal',action='store_true');a=p.parse_args()
    root=a.root;root.mkdir(exist_ok=True,parents=True)
    objects=Path('outputs/object_annotation_20261020');progress=Progress(objects,f'{a.prefix}_total',4834);started=time.time()
    bounds=[0,600,1200,1800,2400,3000,3600,4200,4834];jobs=[]
    try:
        for start,stop in zip(bounds[:-1],bounds[1:]):
            part=root/f'part_{start:06d}';part.mkdir(exist_ok=True)
            args=[sys.executable,'-m','human2robot.retarget_cloth.composite_source','--root',str(part),'--objects',str(objects),'--removal',a.removal,'--trajectory',a.trajectory,'--assets','/user/panmiao/workspace/robodojo-data-windtunnel/runtime/RoboTwin/assets/embodiments/aloha-agilex','--start',str(start),'--stop',str(stop)]
            args+=['--prefix',a.prefix]
            if a.blend_removal:args+=['--blend-removal']
            if a.temporal_removal:args+=['--temporal-removal','--warmup','60']
            log=(part/'render.log').open('w');proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((start,stop,part,proc,log))
        while True:
            done=0
            for start,stop,part,proc,log in jobs:
                status=objects/f'{a.prefix}_{start:06d}_{stop:06d}_status.json'
                if status.exists():
                    state=json.loads(status.read_text())
                    if state['updated_at_unix']>=started:done+=state['completed']
                if proc.poll() not in (None,0):raise RuntimeError(f'Render failed: {part}')
            progress.update(done,phase='rendering_original_view',active_workers=sum(p.poll() is None for _,_,_,p,_ in jobs))
            if all(p.poll() is not None for _,_,_,p,_ in jobs):break
            time.sleep(10)
        rows=[]
        for start,stop,part,proc,log in jobs:
            log.close();rows+=json.loads((part/'composite_validation.json').read_text())['frames']
        assert [r['frame_index'] for r in rows]==list(range(4834))
        listing=root/'parts.txt';listing.write_text(''.join(f"file '{(part/'robot_source_composite.mp4').resolve()}'\n" for _,_,part,_,_ in jobs))
        video=root/'robot_source_composite.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(video)],check=True)
        meta=probe(video);assert meta['frames']==4834 and meta['fps']==30
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
        masks=json.loads((Path(a.removal)/'human_mask_validation.json').read_text())['summary']
        summary=dict(total=4834,video=meta,complete_decode=True,unchanged_outside_edit=all(r['original_pixels_outside_edit_unchanged'] for r in rows),frames_without_robot=sum(r['robot_pixels']==0 for r in rows),frames_without_each_arm={s:sum(r.get('arm_pixels',{}).get(s,0)==0 for r in rows) for s in ['Left','Right']},human_masks=masks,background_donor_pixels=sum(r['background_donor_pixels'] for r in rows),background_estimated_pixels=sum(r['background_inpaint_pixels'] for r in rows),cloth_estimated_pixels=sum(r['cloth_completion_pixels'] for r in rows),cloth_physically_simulated=False,hidden_content_verified=False,training_eligible=False)
        if a.temporal_removal:
            records=[r['temporal_removal'] for r in rows]
            scored=[r for r in records if 'background_rgb_mae_before' in r]
            n=sum(r['background_temporal_pixels'] for r in scored)
            summary['temporal_removal']=dict(history_reset_frames=sum(r['history_reset'] for r in records),observed_cloth_unchanged=all(r['observed_cloth_unchanged'] for r in records),scored_transitions=len(scored),background_temporal_pixels=n,background_rgb_mae_before=sum(r['background_rgb_mae_before']*r['background_temporal_pixels'] for r in scored)/max(n,1),background_rgb_mae_after=sum(r['background_rgb_mae_after']*r['background_temporal_pixels'] for r in scored)/max(n,1),metric_scope='motion-compensated estimated background before robot rendering, not truth accuracy or cloth contact',chunk_warmup_frames=60)
        atomic_json(root/'composite_validation.json',dict(summary=summary,frames=rows))
        progress.update(4834,'completed',phase='encoded_and_decoded')
    except Exception as e:
        for _,_,_,proc,_ in jobs:
            if proc.poll() is None:proc.terminate()
        progress.update(0,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
