"""Visually reviewed, approximate temporal labels for the current folding video.

Boundaries are estimated from 2 Hz source frames, not original dataset labels.
Transitions with an incomplete view are explicitly retained as transitions.
"""
import argparse
import json
import hashlib
from pathlib import Path

PHASES = [
    ('smooth', '双手铺平、整理衣物', 'smooth / arrange', '衣身', '#618b83'),
    ('align_edge', '双手提起下侧衣边并对齐', 'lift / align', '下侧衣边', '#668ba1'),
    ('fold_first', '将下侧衣边向内折叠', 'fold inward', '衣物下侧', '#8874a5'),
    ('fold_again', '继续折叠，收成较窄长条', 'fold again', '已折叠衣物', '#b08a5b'),
    ('fold_side', '将右侧衣物向左侧折合', 'fold right to left', '长条衣物', '#ae737f'),
    ('press', '按压、整理折叠后的衣物', 'press / finish', '折叠衣物', '#709787'),
    ('reset', '演示切换：折叠态变为铺开态', 'transition / reset', '衣物', '#a1a8ad'),
]

# Each row is independently selected from the source contact sheets. Not a
# fixed-period replication; reset intervals retain uncertain/unobserved motion.
CYCLES = [
    [0, 3, 5.5, 7.5, 9.5, 11.5, 12.5, 13.5],
    [13.5, 15.5, 18.5, 20.5, 22.5, 24.5, 25.5, 26.5],
    [26.5, 29, 31, 33.5, 35.5, 37, 37.5, 38.5],
    [38.5, 41, 43, 46, 47.5, 49, 50, 51],
    [51, 53, 55, 57.5, 59.5, 61, 62, 63],
    [63, 65, 67.5, 70.5, 72.5, 74, 74.5, 75.5],
    [75.5, 78, 80.5, 83.5, 85, 86.5, 88, 89],
    [89, 91, 93, 95.5, 97.5, 99.5, 100.5, 101.5],
    [101.5, 104, 106.5, 109, 110.5, 112, 113, 114],
    [114, 117, 119, 121, 123, 124.5, 125, 126],
    [126, 128.5, 130, 132, 133.5, 135, 136.5, 137.5],
    [137.5, 139.5, 141.5, 143.5, 145.5, 147, 148, 149],
    [149, 152, 154, 156.5, 158, 159.5, 4834 / 30],
]


def build(video, target):
    rows = []
    for cycle, bounds in enumerate(CYCLES, 1):
        for phase, (start, end) in enumerate(zip(bounds, bounds[1:])):
            key, label, verb, noun, color = PHASES[phase]
            rows.append(dict(id=f'segment_{len(rows)+1:03d}', cycle=cycle,
                start_frame=round(start*30), end_frame_exclusive=round(end*30),
                start_s=start, end_s=end, phase=key, label=label, verb=verb,
                noun=noun, color=color, source='assistant_visual_review_2hz',
                boundary_uncertainty_s=0.5, review_required=True,
                observation='transition_incomplete_view_no_unfold_action_inferred' if key == 'reset' else 'visible_action_approximate_boundary'))
    assert rows[0]['start_frame'] == 0 and rows[-1]['end_frame_exclusive'] == 4834
    assert all(a['end_frame_exclusive'] == b['start_frame'] for a,b in zip(rows,rows[1:]))
    assert all(r['end_frame_exclusive'] > r['start_frame'] for r in rows)
    data = dict(schema='human_robot.temporal_semantics.v1', fps=30, frames=4834,
        source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        original_annotations_recovered=False, original_seven_segments_recovered=False,
        annotation_method='assistant visual review of 323 source images sampled at 2 Hz',
        independent_human_verification=False, boundary_precision='approximate ±0.5 s; not frame-exact ground truth',
        phases=[dict(id=p[0],label=p[1],color=p[4]) for p in PHASES], segments=rows)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return data


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=build(a.video,a.output);print(len(d['segments']), 'segments')
