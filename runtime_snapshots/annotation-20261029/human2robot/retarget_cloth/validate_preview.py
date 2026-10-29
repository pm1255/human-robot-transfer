"""Independently audit saved IK trajectories against their URDF and source detections."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from human2robot.hand_mesh.video import frames, probe


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1-math.cos(angle)) * (skew @ skew)


def origin(joint):
    e = joint.find('origin')
    result = np.eye(4)
    if e is not None:
        r, p, y = [float(v) for v in e.get('rpy', '0 0 0').split()]
        result[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
        result[:3, 3] = [float(v) for v in e.get('xyz', '0 0 0').split()]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hands', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.root / 'robot_trajectory.json').read_text())
    rows, meta = data['frames'], data['meta']
    source = [json.loads(line) for line in (args.hands / 'frames.jsonl').read_text().splitlines()]
    tree = ET.parse(args.root / 'preview_arm.urdf').getroot()
    by_child = {j.find('child').get('link'): j for j in tree.findall('joint')}
    chain, link = [], 'tcp'
    while link != 'fl_base_link':
        joint = by_child[link]
        chain.insert(0, joint)
        link = joint.find('parent').get('link')
    failures = []
    def check(condition, name, frame):
        if not condition:
            failures.append(dict(check=name, frame=frame))
    counters = Counter()
    residuals, angles, clearances, velocities, smooth, worst = [], [], [], [], [], []
    previous = {}
    fk_delta = 0.0
    for index, row in enumerate(rows):
        frame = row['frame_index']
        check(frame == index + meta['start_frame'], 'frame_order', frame)
        check(abs(row['timestamp_s'] - frame / 30) < 1e-8, 'timestamp', frame)
        check(row['training_eligible'] is False and row['physics_executed'] is False, 'scope_flags', frame)
        observed = {h['handedness'] for h in source[frame]['hands']}
        check(set(row['hands']).issubset(observed), 'source_detection_present', frame)
        counters['display_' + row['status']] += 1
        if row['resampled_hold']:
            sample = row['sample_source_frame'] - meta['start_frame']
            check(0 <= index - sample < 3 and not rows[sample]['resampled_hold'], 'hold_source', frame)
            check(row['hands'] == rows[sample]['hands'] or not row['hands'], 'hold_data', frame)
            continue
        counters['motion_samples'] += 1
        for side, hand in row['hands'].items():
            counters['solved_hands'] += 1
            counters[hand['status']] += 1
            counters['reinitialized_hands'] += int(hand['reinitialized'])
            q = np.asarray(hand['q'], float)
            check(q.shape == (6,) and np.isfinite(q).all() and max(abs(q)) <= np.pi + 1e-7, 'numeric_joint_range', frame)
            check(0 <= hand['gripper_joint_m'] <= .04765, 'gripper_range', frame)
            T = np.eye(4)
            points, active = [T[:3, 3].copy()], 0
            for joint in chain:
                T = T @ origin(joint)
                if joint.get('type') != 'fixed':
                    motion = np.eye(4)
                    axis = [float(v) for v in joint.find('axis').get('xyz').split()]
                    motion[:3, :3] = rotation(axis, q[active])
                    T = T @ motion
                    active += 1
                points.append(T[:3, 3].copy())
            base = np.eye(4)
            base[:3, :3] = rotation([0, 0, 1], np.pi/2)
            base[:3, 3] = [-.38 if side == 'Left' else .38, -.42, .015]
            actual = base @ T
            target = np.asarray(hand['target'])
            delta = float(np.max(abs(actual - hand['actual'])))
            fk_delta = max(delta, fk_delta)
            check(delta < 1e-8, 'independent_urdf_fk', frame)
            error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(target[:3, :3] @ actual[:3, :3].T)-1)/2, -1, 1)))
            clearance = float(np.min(np.asarray(points)[2:, 2] + .015))
            check(abs(error - hand['error_m']) < 1e-8, 'position_residual', frame)
            check(abs(angle - hand['rotation_error_rad']) < 1e-6, 'rotation_residual', frame)
            check(abs(clearance - hand['link_point_table_clearance_m']) < 1e-8, 'link_point_clearance', frame)
            check((hand['status'] == 'kinematic_match_review') == (error < .035 and angle < .65), 'fit_status', frame)
            if side in previous and not hand['reinitialized']:
                old_frame, old_q = previous[side]
                check(frame - old_frame == 3, 'contiguous_sample', frame)
                speed = float(max(abs(q-old_q)) / ((frame-old_frame)/30))
                velocities.append(speed)
                check(speed <= 3 + 1e-6, 'continuous_velocity_bound', frame)
            previous[side] = (frame, q)
            residuals.append(error)
            angles.append(angle)
            clearances.append(clearance)
            smooth.append(hand['target_smoothing_displacement_m'])
            worst.append(dict(frame=frame, side=side, error_m=error, rotation_error_rad=angle))
    video_path = args.root / 'robot_preview.mp4'
    video_meta = probe(video_path)
    video_meta['container_frames'] = video_meta['frames']
    decoded = 0
    for image in frames(video_path, video_meta):
        check(image.shape == (480, 640, 3), 'decoded_shape', decoded)
        decoded += 1
    video_meta['decoded_frames'] = decoded
    check(decoded == len(rows) == meta['frames'] == 4834, 'full_frame_count', -1)
    check(abs(video_meta['fps'] - 30) < .001, 'video_fps', -1)
    result = dict(
        result='integrity_pass' if not failures else 'integrity_fail',
        scope='Data integrity and numerical fit only; not calibrated accuracy, full mesh collision, contact, or execution validation.',
        video=video_meta, counts=dict(counters), failures=failures,
        independent_fk_max_absolute_delta=fk_delta,
        fit_thresholds=dict(position_m=.035, angle_rad=.65),
        solved_sample_position_residual=dict(median_m=float(np.median(residuals)), p95_m=float(np.percentile(residuals, 95)), max_m=max(residuals)),
        solved_sample_angle_residual=dict(median_rad=float(np.median(angles)), p95_rad=float(np.percentile(angles, 95))),
        minimum_link_point_table_clearance_m=min(clearances),
        hand_samples_with_link_points_below_table=sum(v < 0 for v in clearances),
        hand_samples_below_12mm_clearance_target=sum(v < .012 for v in clearances),
        continuous_sample_max_joint_velocity_rad_s=max(velocities),
        smoothing_displacement_p95_m=float(np.percentile(smooth, 95)),
        worst_position_samples=sorted(worst, key=lambda x:x['error_m'], reverse=True)[:10],
        metric_scale_verified=False, full_collision_verified=False, physics_executed=False,
    )
    (args.root / 'robot_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
