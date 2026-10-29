"""Add per-frame Chinese review status to the saved robot visualization."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--font', default='/System/Library/Fonts/STHeiti Light.ttc')
    a = p.parse_args()
    data = json.loads((a.root/'robot_trajectory.json').read_text())
    output = a.root/'robot_preview_review.mp4'
    font = ImageFont.truetype(a.font, 16)
    small = ImageFont.truetype(a.font, 13)
    source = imageio_ffmpeg.read_frames(str(a.root/'robot_preview.mp4'), pix_fmt='rgb24')
    next(source)
    writer = imageio_ffmpeg.write_frames(str(output), (640,480), fps=30, codec='libx264', pix_fmt_out='yuv420p', output_params=['-crf','18','-preset','fast','-movflags','+faststart','-threads','4'])
    writer.send(None)
    saved = {0,90,600,696,781,2400,3600,3948,4800,4833}
    count = 0
    states = Counter()
    for i, raw in enumerate(source):
        row = data['frames'][i]
        hands = list(row['hands'].values())
        below = any(h['link_point_table_clearance_m'] < 0 for h in hands)
        bad = any(h['status'] == 'tracking_error_review' for h in hands)
        status = '数据不足，无机械臂输出' if not hands else '连杆点低于估计桌面，需复核' if below else '跟踪偏差超阈值，需复核' if bad else '通过拟合筛查，精度待复核'
        states[status] += 1
        im = Image.frombytes('RGB',(640,480),raw)
        draw = ImageDraw.Draw(im)
        draw.rectangle((0,0,639,57),fill='#202c34')
        draw.text((10,4),f'人手 → 双 ARX5 机械臂 · 运动学试验 · {i/30:.2f} 秒',font=font,fill='white')
        draw.text((10,24),status+(' · 重初始化' if any(h['reinitialized'] for h in hands) else ''),font=small,fill='#ffd18a' if bad or below or not hands else '#c4e2d4')
        draw.text((10,41),'10 Hz 动作 / 30 FPS 展示 · 未验证布料接触与物理执行',font=small,fill='#d4dce0')
        writer.send(np.asarray(im)); count += 1
        if i in saved:
            im.save(a.root/f'review_{i:06d}.jpg')
    writer.close()
    assert count == len(data['frames']) == 4834
    decoded = imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24')
    video_meta = next(decoded)
    n = sum(1 for _ in decoded)
    assert n == count and video_meta['fps'] == 30
    (a.root/'published_video_validation.json').write_text(json.dumps(dict(decoded_frames=n,fps=video_meta['fps'],states=dict(states),source='robot_preview.mp4',trajectory='robot_trajectory.json',output=output.name,scope='Chinese review overlay only; geometry and motion unchanged'),ensure_ascii=False,indent=2)+'\n')
    compact = dict(meta=data['meta'],frames=[{k: v for k,v in row.items() if k in ['frame_index','timestamp_s','sample_source_frame','resampled_hold','status']} | {'hands':{side:{k:v for k,v in hand.items() if k in ['error_m','rotation_error_rad','status','reinitialized','link_point_table_clearance_m']} for side,hand in row['hands'].items()}} for row in data['frames']])
    (a.root/'robot_review.json').write_text(json.dumps(compact,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(frames=n,states=dict(states)),ensure_ascii=False))


if __name__ == '__main__':
    main()
