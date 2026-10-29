import json,sys,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper');ROOT=BASE/'outputs/unified_30_20261021'
FF=BASE/'outputs/unified_annotation_web_20261020/render_dependencies/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
for p in sorted((ROOT/'samples').iterdir()):
    dest=ROOT/'sheets'/f'{p.name}.jpg';dest.parent.mkdir(exist_ok=True)
    if dest.exists() or not (p/'cam_0.mp4').exists():continue
    src=json.loads((p/'source.json').read_text());n=src.get('frames',src.get('num_frames'));fps=src['fps'];indices=np.unique(np.linspace(0,n-1,16,dtype=int));selection='+'.join(f'eq(n,{i})'for i in indices)
    proc=subprocess.run([str(FF),'-v','error','-i',str(p/'cam_0.mp4'),'-vf',f'select={selection.replace(",",chr(92)+",")},scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True)
    frames=np.frombuffer(proc.stdout,np.uint8).reshape(-1,240,320,3);im=Image.new('RGB',(1280,4*263),'#12232d');d=ImageDraw.Draw(im)
    for j,(a,f)in enumerate(zip(frames,indices)):
        x=(j%4)*320;y=(j//4)*263;im.paste(Image.fromarray(a),(x,y));d.text((x+4,y+242),f'{p.name} f{f} {f/fps:.2f}s',fill='white')
    im.save(dest,quality=85);print(p.name,flush=True)
