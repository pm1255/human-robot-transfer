"""Package existing annotations as separate videos; no new model inference.

Usage: python render_individual_videos.py --root <served-output> --ffmpeg <binary>
The semantic overlay preserves only the recorded clip-level label. It does not
invent per-frame actions or the missing seven segment boundaries.
"""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    args = parser.parse_args()
    root = args.root
    assets = root / 'annotation_assets'
    font_path = '/System/Library/Fonts/STHeiti Light.ttc'
    font = lambda size: ImageFont.truetype(font_path, size)

    def banner(name, height, lines):
        image = Image.new('RGBA', (640, height), (24, 35, 42, 245))
        draw = ImageDraw.Draw(image)
        for text, x, y, size, color in lines:
            draw.text((x, y), text, font=font(size), fill=color)
        target = assets / name
        image.save(target)
        return target

    depth_banner = banner('depth_banner.png', 50, [
        ('MoGe-3 · 可见衣物深度', 12, 5, 17, '#ffffff'),
        ('逐帧估计 · 颜色逐帧缩放 · 尺度未经标定', 12, 29, 13, '#ffd58a'),
    ])
    semantic_banner = banner('semantic_banner.png', 112, [
        ('语义 · 历史整体标签（非逐帧动作）', 12, 6, 17, '#ffd58a'),
        ('fold the bottom of the t-shirt', 12, 32, 20, '#ffffff'),
        ('动词：fold（折叠）   名词：t-shirt（T 恤）', 12, 60, 15, '#e4efee'),
        ('原语义视频未找回；7 个子任务时间段缺失', 12, 87, 14, '#ffd58a'),
    ])
    common = [args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y']
    encoding = ['-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '4']
    # Both geometry views are derived from the already reviewed dual view.
    subprocess.run(common + ['-i', str(root / 'geometry/cloth_pose_review.mp4'),
        '-i', str(depth_banner), '-filter_complex',
        '[0:v]split=2[left][right];[left]crop=640:480:0:0[pose];'
        '[right]crop=640:480:640:0[depth];[depth][1:v]overlay=0:0[depth_labeled]',
        '-map', '[pose]'] + encoding + [str(root / 'geometry/pose.mp4'),
        '-map', '[depth_labeled]'] + encoding + [str(root / 'geometry/depth.mp4')], check=True)
    print('Separate pose and depth videos rendered', flush=True)
    subprocess.run(common + ['-i', str(root / 'source.mp4'), '-i', str(semantic_banner),
        '-filter_complex', '[0:v][1:v]overlay=0:0[v]', '-map', '[v]'] + encoding +
        [str(assets / 'semantic_summary.mp4')], check=True)
    print('Historical clip-level semantic view rendered', flush=True)
    for video, poster in [('geometry/pose.mp4', 'geometry/pose_poster.jpg'),
                          ('geometry/depth.mp4', 'geometry/depth_poster.jpg'),
                          ('annotation_assets/semantic_summary.mp4', 'annotation_assets/semantic_poster.jpg')]:
        subprocess.run(common + ['-ss', '20', '-i', str(root / video), '-frames:v', '1',
                                 '-update', '1', str(root / poster)], check=True)
    (assets / 'individual_video_provenance.json').write_text(json.dumps({
        'operation': 'presentation_only_no_new_model_inference',
        'geometry_input': 'geometry/cloth_pose_review.mp4',
        'pose': 'left view, preserved axes and contour',
        'depth': 'right view, pose-status header replaced with depth-specific label',
        'semantic_input': 'source.mp4 + manifest historical clip-level fields',
        'semantic_temporal_annotation_available': False,
        'semantic_original_video_recovered': False,
        'semantic_segment_boundaries': None,
        'fps': 30, 'expected_frames': 4834,
        'model_accuracy_verified': False,
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
