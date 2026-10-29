"""Compare encoded videos on shared removed-table pixels, excluding robots.

Temporal consistency is not reconstruction accuracy. This audit does not
measure hidden cloth truth, physical contact, or complete human removal.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from human2robot.hand_mesh.video import frames, probe
from human2robot.retarget_cloth.composite_source import Inputs
from human2robot.retarget_cloth.temporal_removal import TemporalRemoval
from human2robot.object_annotation.common import atomic_json, Progress


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--after-start', type=int, default=0)
    p.add_argument('--count', type=int, default=0)
    p.add_argument('--objects', type=Path, default=Path('outputs/object_annotation_20261020'))
    p.add_argument('--removal', type=Path, default=Path('outputs/robot_repair_20261020/masks_final'))
    a = p.parse_args(); a.root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    inputs = Inputs(a.objects, a.removal)
    meta = probe(a.after); n = min(a.count or meta['frames'], meta['frames']-a.after_start); source = inputs.data['meta']['video']
    iterators = [frames(source, probe(source), a.start, n), frames(a.before, probe(a.before), a.start, n), frames(a.after, meta, a.after_start, n)]
    previous = None; records = []; progress = Progress(a.objects, f'robot_temporal_encoded_audit_{a.start:06d}_{n:06d}', n)
    for k, (raw, before, after) in enumerate(zip(*iterators)):
        index = a.start + k; cloth, human = inputs.mask(index)
        exclude = cv2.dilate((cloth | human).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        # Remove rendered gray robot pixels, cloth, and uncertain edit edges
        # identically in both videos. Restrict this metric to brown tabletop.
        region = cv2.erode((human & ~cv2.dilate(cloth.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        for img in [before, after]:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            table = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 30)
            region &= cv2.erode(table.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        record = dict(frame_index=index, scored=False)
        if previous is not None:
            H, registration = TemporalRemoval.registration(previous, raw, exclude)
            record.update(registration)
            if H is not None:
                shared = region & (cv2.warpPerspective(previous['region'].astype(np.uint8), H, (640, 480), flags=cv2.INTER_NEAREST) > 0)
                if shared.sum() > 100:
                    errors = {}
                    for label, current in [('before', before), ('after', after)]:
                        prior = cv2.warpPerspective(previous[label], H, (640, 480))
                        residual = np.abs(current.astype(float)-prior)[shared]
                        errors[label] = dict(mean_rgb_difference=float(residual.mean()), p95_pixel_difference=float(np.percentile(residual.mean(1), 95)))
                    record.update(scored=True, pixels=int(shared.sum()), errors=errors)
        records.append(record)
        previous = dict(raw=raw, exclude=exclude, before=before, after=after, region=region)
        if k % 60 == 0: progress.update(k, phase='encoded_shared_region_temporal_error')
    scored = [r for r in records if r['scored']]; pixels = sum(r['pixels'] for r in scored)
    values = {side: sum(r['pixels']*r['errors'][side]['mean_rgb_difference'] for r in scored)/max(pixels, 1) for side in ['before', 'after']}
    summary = dict(frames=n, start=a.start, scored_transitions=len(scored), pixels=pixels, mean_rgb_difference=values, reduction_fraction=1-values['after']/max(values['before'], 1e-6), scope='shared estimated brown tabletop pixels in encoded videos after camera compensation; robot and cloth excluded', full_hidden_surface_accuracy_verified=False)
    atomic_json(a.root/'encoded_temporal_audit.json', dict(summary=summary, frames=records))
    progress.update(n, 'completed', reduction_fraction=summary['reduction_fraction']); progress.close()
    print(json.dumps(summary))


if __name__ == '__main__': main()
