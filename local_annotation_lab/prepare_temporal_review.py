"""Prepare matched frame-count comparisons and visual sheets from final video."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--before',type=Path,required=True);p.add_argument('--ffmpeg',required=True);a=p.parse_args()
    root=a.root
    for start,name in [(3450,'temporal_comparison.mp4'),(510,'temporal_comparison_cut.mp4')]:
        filters=f'[0:v]trim=start_frame={start}:end_frame={start+300},setpts=PTS-STARTPTS,fps=30[a];[1:v]trim=start_frame={start}:end_frame={start+300},setpts=PTS-STARTPTS,fps=30[b];[a][b]hstack=inputs=2[out]'
        subprocess.run([a.ffmpeg,'-v','error','-y','-i',str(a.before),'-i',str(root/'full/robot_source_composite.mp4'),'-filter_complex',filters,'-map','[out]','-frames:v','300','-an','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(root/name)],check=True)
        subprocess.run([a.ffmpeg,'-v','error','-i',str(root/name),'-f','null','-'],check=True)
    files=sorted((root/'full').glob('part_*/composite_*.jpg'))
    sheet=Image.new('RGB',(1280,((len(files)+3)//4)*263),'white');draw=ImageDraw.Draw(sheet)
    for k,path in enumerate(files):
        x=k%4*320;y=k//4*263;sheet.paste(Image.open(path).resize((320,240)),(x,y+23));draw.text((x+5,y+4),path.stem,fill='black')
    sheet.save(root/'review_sheet.jpg')
    print(json.dumps(dict(sample_count=len(files),comparison_frames_each=300)))


if __name__=='__main__':main()
