import json, subprocess, io, math
from pathlib import Path
from PIL import Image, ImageDraw
BASE=Path('/Users/panmiao/Documents/ChatGPT/paper')
FF=BASE/'outputs/unified_annotation_web_20261020/render_dependencies/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
ROOT=BASE/'outputs/unified_three_domain_20261020'
for p in sorted((ROOT/'samples').iterdir()):
 if not (p/'cam_0.mp4').exists(): continue
 s=json.loads((p/'source.json').read_text());fps=s.get('fps',30);n=s.get('frames',s.get('num_frames',s.get('episode',{}).get('length',0))) if isinstance(s.get('episode',{}),dict) else s.get('frames',s.get('num_frames',0))
 step=.25 if n/fps<5 else .5
 result=subprocess.run([str(FF),'-v','error','-i',str(p/'cam_0.mp4'),'-vf',f'fps={1/step},scale=240:160:force_original_aspect_ratio=decrease,pad=240:160:(ow-iw)/2:(oh-ih)/2','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True)
 count=len(result.stdout)//(240*160*3);cols=5;rows=math.ceil(count/cols);sheet=Image.new('RGB',(cols*240,rows*184),'#edf1f3');d=ImageDraw.Draw(sheet)
 for i in range(count):
  im=Image.frombytes('RGB',(240,160),result.stdout[i*240*160*3:(i+1)*240*160*3]);x=i%cols*240;y=i//cols*184;sheet.paste(im,(x,y));d.text((x+5,y+162),f'{i*step:.2f}s  ~f{round(i*step*fps)}',fill='black')
 sheet.save(ROOT/f'{p.name}_sheet.jpg',quality=90);print(p.name,n,fps,count)
