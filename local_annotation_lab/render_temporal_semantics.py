"""Render per-segment semantic labels, keeping provenance and boundary limits."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--segments',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.segments.read_text());rows=d['segments']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    font=lambda s:ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc',s)
    banners={}
    for j,s in enumerate(rows):
        im=Image.new('RGB',(640,110),'#182b34');draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,6,110),fill=s['color'])
        draw.text((14,7),f'片段 {j+1:02d}/{len(rows)} · 第 {s["cycle"]} 次演示 · {s["start_s"]:.1f}–{s["end_s"]:.1f} 秒',font=font(15),fill='#c5d6d9')
        draw.text((14,32),s['label'],font=font(23),fill='white')
        draw.text((14,67),f'{s["verb"]} · {s["noun"]}',font=font(15),fill='#c5d6d9')
        draw.text((14,88),'按画面重新分段 · 边界约 ±0.5 秒，待复核',font=font(13),fill='#ffd690')
        banners[s['id']]=np.asarray(im)
    reader=imageio_ffmpeg.read_frames(str(a.video),pix_fmt='rgb24');meta=next(reader)
    writer=imageio_ffmpeg.write_frames(str(a.output),(640,480),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','18','-preset','fast','-movflags','+faststart','-threads','4']);writer.send(None)
    j=0;count=0
    for i,b in enumerate(reader):
        while i>=rows[j]['end_frame_exclusive']:j+=1
        frame=np.frombuffer(b,np.uint8).reshape(480,640,3).copy();frame[:110]=banners[rows[j]['id']]
        writer.send(frame);count+=1
        if i in [30,120,180,240,300,360,390,600,2000,4800]:Image.fromarray(frame).save(a.output.parent/f'semantic_{i:06d}.jpg')
    writer.close();assert count==d['frames']
    print(json.dumps(dict(frames=count,segments=len(rows),video=str(a.output))))


if __name__=='__main__':main()
