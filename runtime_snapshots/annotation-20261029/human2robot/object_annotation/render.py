"""Render separate segmentation and bounding-box movies from saved predictions."""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from .common import Progress, atomic_json
from human2robot.hand_mesh.video import probe, frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);data=json.loads((root/'objects.json').read_text());meta=data['meta'];rows=data['frames']
    h,w,fps=meta['height'],meta['width'],meta['fps'];total=len(rows)
    if fps <= 0 or total <= 0 or total != meta['processed_target']:
        raise ValueError('Empty or incomplete annotations cannot be rendered')
    video_meta=probe(meta['video'])
    prog=Progress(root,'render',total);writers=[]
    for name in ['segmentation','boundingbox']:
        args=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
              '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20',
              '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(root/f'{name}.partial.mp4')]
        writers.append(subprocess.Popen(args,stdin=subprocess.PIPE))
    chunk_start=None; masks=None
    try:
        for i,frame in enumerate(frames(meta['video'],video_meta,limit=total)):
            row=rows[i]
            if row['frame_index']!=i:raise ValueError('Non-consecutive annotation indices')
            if row['chunk_start']!=chunk_start:
                chunk_start=row['chunk_start'];pack=np.load(root/'masks'/f'{chunk_start:06d}.npz')
                masks=np.unpackbits(pack['masks'],axis=1,count=h*w).reshape(-1,h,w).astype(bool)
            mask=masks[i-chunk_start];seg=frame.copy();box=frame.copy()
            seg[mask]=(seg[mask]*.55+np.array([80,190,50])*.45).astype(np.uint8)
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg,contours,-1,(80,220,90),1)
            if row['bbox']:
                x1,y1,x2,y2=row['bbox'];cv2.rectangle(box,(x1,y1),(x2-1,y2-1),(40,190,255),2)
                cv2.putText(box,'t-shirt | garment_1',(x1,max(15,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,190,255),1,cv2.LINE_AA)
            for name,img in [('SAM 3.1 mask',seg),('BBox from SAM 3.1 mask',box)]:
                cv2.putText(img,f'{name} | frame {i} | {i/fps:.2f}s',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
                if not row['mask_area']:cv2.putText(img,'NO DETECTION',(8,42),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,80,255),1)
            writers[0].stdin.write(seg.tobytes());writers[1].stdin.write(box.tobytes())
            if i==600 or (total<=600 and i==0):
                cv2.imwrite(str(root/'segmentation_poster.jpg'),seg);cv2.imwrite(str(root/'boundingbox_poster.jpg'),box)
            if i%100==0:prog.update(i,phase='encoding')
        for proc in writers:
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
        checks={}
        for name in ['segmentation','boundingbox']:
            path=root/f'{name}.partial.mp4'
            n=sum(1 for _ in frames(path,probe(path)))
            checks[name]=n
            if n!=total:raise RuntimeError(f'{name} decoded {n}/{total}')
            path.replace(root/f'{name}.mp4')
        atomic_json(root/'video_validation.json',dict(decoded_frames=checks,width=w,height=h,fps=fps))
        prog.update(total,'completed')
    except Exception as exc:
        prog.update(locals().get('i',0),'failed',error=str(exc));raise
    finally:
        for proc in writers:
            if proc.poll() is None:proc.terminate()
        prog.close()

if __name__=='__main__':main()
