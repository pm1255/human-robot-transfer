"""Wait for dense segmentation, merge removal evidence, render and validate."""
import json,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    root=Path('outputs/robot_repair_20261020');objects=Path('outputs/object_annotation_20261020');out=root/'masks_final';(out/'human_masks').mkdir(parents=True,exist_ok=True)
    progress=Progress(objects,'robot_repair_removal_final',4834)
    try:
        progress.update(0,'waiting_for_dense_segmentation')
        while not (root/'dense_flow/human_mask_validation.json').exists():time.sleep(5)
        dense=json.loads((root/'dense_flow/human_mask_validation.json').read_text());records=[]
        for i,r in enumerate(dense['frames']):
            def read(folder):return np.unpackbits(np.load(root/folder/'human_masks'/f'{i:06d}.npz')['mask'],count=640*480).reshape(480,640).astype(bool)
            a=read('masks_pilot');b=read('dense_flow');m=a|cv2.dilate(b.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
            np.savez_compressed(out/'human_masks'/f'{i:06d}.npz',mask=np.packbits(m),frame_index=i)
            rec=dict(frame_index=i,area=int(m.sum()),method='dense_SAM3.1_3Hz_flow_union_MANO_every_frame',dense_mask_pixels=int(b.sum()),dense_mask_coverage=float((m&b).sum()/max(1,b.sum())),review_required=r.get('review_required',False),accuracy_verified=False);records.append(rec);atomic_json(out/'human_masks'/f'{i:06d}.json',rec)
            if i%120==0:progress.update(i,phase='combining_dense_segmentation_and_hand_mesh')
        atomic_json(out/'human_mask_validation.json',dict(frames=records,summary=dict(total=4834,model_inference_frames=dense['summary']['model_inference_frames'],flow_estimated_frames=dense['summary']['flow_estimated_frames'],review_frames=sum(r['review_required'] for r in records),all_dense_masks_covered=True,accuracy_verified=False)))
        progress.update(4834,'completed')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable,'-u','-m','human2robot.retarget_cloth.build_composite_video','--root',str(root/'full'),'--removal',str(out),'--trajectory',str(root/'trajectory/robot_trajectory.json'),'--prefix','robot_repair_composite'],env=env,check=True)
    except Exception as e:progress.update(0,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
