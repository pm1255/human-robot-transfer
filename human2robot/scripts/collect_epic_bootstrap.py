"""Run on the train2 filesystem. RGB only; writes only a new user-owned folder."""
from pathlib import Path
import hashlib,json,subprocess
import cv2
import pyarrow.parquet as pq

SOURCE=Path('/user/zhangxueqian/dataset/lerobot_v3.0_v0/ego/EpicKitchens/epic_kitchens_action_clips_lda_full_v3')
DEST=Path('/user/panmiao/workspace/human2robot-rgb-pilot-20260923/data/bootstrap')
SELECTED={4,7,15,40,100,250,500,1000}

def main():
    DEST.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    rows=[]
    for p in sorted((SOURCE/'meta/episodes').rglob('*.parquet')):
        for r in pq.read_table(p).to_pylist():
            if int(r['episode_index']) in SELECTED:rows.append(r)
    manifest=[]
    for r in sorted(rows,key=lambda r:r['episode_index']):
        source=SOURCE/r['video_path'];dest=DEST/f"epic_{r['episode_index']:06d}.mp4"
        if dest.exists():raise FileExistsError(f'Refusing overwrite: {dest}')
        start=float(r.get('videos/observation.images.head/from_timestamp',0))
        duration=min(7,float(r['videos/observation.images.head/to_timestamp'])-start)
        command=['ffmpeg','-nostdin','-v','error','-threads','1','-ss',str(start),'-i',str(source),
                 '-t',str(duration),'-an','-vf','scale=640:-2,fps=10','-c:v','libx264','-threads','1',
                 '-preset','veryfast','-crf','23','-movflags','+faststart',str(dest)]
        subprocess.run(command,check=True)
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                              '-show_entries','stream=nb_read_frames','-of','json',str(dest)],check=True,capture_output=True,text=True)
        count=int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
        source_times=[start+i/10 for i in range(count)]
        record={'id':dest.stem,'source_cluster':'yingbo_train','source_pool_context':'embody-train2',
                'source_uri':str(source),'local_filename':dest.name,'source_episode_index':r['episode_index'],
                'source_time_grid_seconds':source_times,'resampling':'ffmpeg fps=10; grid is not an exact source-frame PTS list',
                'task':r.get('vlm_video_instruction',''),
                'task_source':'upstream_vlm_description_not_verified_instruction',
                'source_kind':'human_video','pose_labels_consumed':False,
                'camera_pan_upstream':r.get('camera_pan'),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                'split_group':'epic_bootstrap_parent_session_unresolved',
                'license_status':'internal_user_authorized_research_review_before_redistribution'}
        manifest.append(record);print(json.dumps({'id':dest.stem,'frames':len(source_times),'bytes':dest.stat().st_size}),flush=True)
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
