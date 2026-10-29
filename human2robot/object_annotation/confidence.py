"""Uncalibrated, abstaining temporal evidence scores; never correctness probabilities.

SAM output probabilities are deliberately excluded. RGB flow is separate evidence
from the masks, but shared failure modes and stable wrong masks remain possible.
"""
import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

VERSION = 'temporal-evidence-v1'
PARAMETERS = dict(flow_width=320, min_mask_pixels=16, min_textured_pixels=16,
                  min_textured_fraction=.15, sobel_texture_threshold=8,
                  fb_error_px=1., fb_relative_error=.05,
                  evidence_high=80, evidence_medium=50, depth_change_scale=.1)
KNOWN_CLASSES = {
    'effector_oversegmentation': ['gripper'],
    'partial_cabinet_door_mask': ['cabinet_door'],
    'visible_right_hand_missed': ['hand'],
    'visible_cups_lids_missed': ['cup', 'cup_lid'],
    'visible_hands_missed': ['hand'],
}
UNKNOWN_REASONS = {
    'semantics': '尚无独立视觉模型或原生信号验证片段内容；结构合法不代表语义正确。',
    'contact': '二维接近不是物理接触；且依赖未验收的分割和语义。',
    'object_pose': '未有有效标定、对象几何与参考位姿。',
    'affordance': '尚无独立功能部位证据。',
    'obstacles': '尚无完整场景与碰撞证据。',
    'hand_mesh': '本轮未计算手型证据分。',
    'retarget': '本轮未计算背景修复与机器人接触证据分。',
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def pair_context(previous, current):
    args = (.5, 3, 21, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(previous, current, None, *args)
    backward = cv2.calcOpticalFlowFarneback(current, previous, None, *args)
    h, w = current.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    px, py = x + backward[..., 0], y + backward[..., 1]
    inbound = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    sampled = cv2.remap(forward, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    error = np.linalg.norm(backward + sampled, axis=-1)
    tolerance = PARAMETERS['fb_error_px'] + PARAMETERS['fb_relative_error'] * np.linalg.norm(backward, axis=-1)
    texture = np.hypot(cv2.Sobel(current, cv2.CV_32F, 1, 0), cv2.Sobel(current, cv2.CV_32F, 0, 1)) > PARAMETERS['sobel_texture_threshold']
    return dict(px=px, py=py, inbound=inbound, reliable=(error <= tolerance) & inbound,
                texture=texture, forward=forward, backward=backward)


def pair_evidence(previous_mask, current_mask, context):
    count = int(current_mask.sum())
    textured = current_mask & context['texture'] & context['inbound']
    support = int(textured.sum())
    if min(count, int(previous_mask.sum())) < PARAMETERS['min_mask_pixels']:
        return dict(score=None, reason='目标在评分分辨率下过小', texture_support_pixels=support)
    if support < PARAMETERS['min_textured_pixels'] or support / count < PARAMETERS['min_textured_fraction']:
        return dict(score=None, reason='纹理或可见对应不足，不能用零运动证明稳定', texture_support_pixels=support)
    warped = cv2.remap(previous_mask.astype(np.uint8), context['px'], context['py'], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
    union = warped | current_mask
    iou = float((warped & current_mask).sum() / max(1, union.sum()))
    fraction = float(context['reliable'][textured].mean())
    score = round(100 * min(iou, fraction), 1)
    return dict(score=score, warped_mask_iou=round(iou, 5), flow_roundtrip_fraction=round(fraction, 5),
                texture_support_pixels=support, texture_fraction=round(support/count, 5),
                reason='光流补偿轮廓一致性与前后向光流对应的较小值')


def final_instance(obj, evidences, conflicts):
    usable = [e['score'] for e in evidences if e['score'] is not None]
    score = min(usable) if usable else None
    reasons = []
    if conflicts: reasons.append('既有助手抽检发现该类别问题；保守阻断整类，不表示每帧均已证实错误')
    if not usable: reasons.append('相邻帧、纹理或目标面积不足，未给分')
    elif score < 50: reasons.append('图像运动与轮廓变化的一致性较弱')
    if len(usable) == 1: reasons.append('仅一个相邻帧对提供有效证据')
    status = 'known_issue' if conflicts else 'insufficient' if score is None else 'strong_consistency' if score >= 80 else 'medium_consistency' if score >= 50 else 'weak_consistency'
    return dict(instance_id=obj['instance_id'], class_id=obj['class_id'], temporal_score=score,
                status=status, evidence_pairs=len(usable), evidence=evidences, reasons=reasons,
                known_issue_codes=conflicts, probability=None, pseudo_label_candidate=False)


def summary(values):
    a = [v for v in values if v is not None]
    return dict(scored=len(a), total=len(values), median=round(float(np.median(a)), 1) if a else None,
                p10=round(float(np.percentile(a, 10)), 1) if a else None,
                high=sum(v >= 80 for v in a), medium=sum(50 <= v < 80 for v in a), low=sum(v < 50 for v in a))


def depth_score(row):
    if not row['positive_finite']:
        return dict(stability_score=None, status='invalid', reason='保存的有效区域没有有限正深度')
    changes = [row.get('adjacent_median_relative_change'), row.get('adjacent_focal_relative_change')]
    if any(v is None or not math.isfinite(v) for v in changes):
        return dict(stability_score=None, status='insufficient', reason='没有完整相邻帧变化证据')
    score = round(100 * math.exp(-max(changes) / PARAMETERS['depth_change_scale']), 1)
    return dict(stability_score=score, status='relative_stability_only',
                reason='深度中位数与焦距变化的稳定性；真实相机运动也会降分', metric_confidence=None)


def score_episode(clip, site, model_root, destination, review, progress):
    sid = clip['id']; src = site / 'unified' / sid / 'cam_0.mp4'
    root = model_root / 'samples' / sid / 'multimodal'
    objects = json.loads((root / 'objects.json').read_text()); meta = objects['meta']; rows = objects['frames']
    assert len(rows) == clip['frames'] and digest(src) == meta['source_sha256']
    width = PARAMETERS['flow_width']; height = round(meta['height'] * width / meta['width'])
    known = review['samples'][sid]['confirmed_errors']
    by_class = defaultdict(list)
    for error in known:
        for label in KNOWN_CLASSES.get(error, []): by_class[label].append(error)
    for label, errors in review['samples'][sid].get('known_issue_by_class', {}).items():
        by_class[label].extend(e for e in errors if e not in by_class[label])
    evidence = [defaultdict(list) for _ in rows]
    flow_scores = [None] * len(rows)
    cap = cv2.VideoCapture(str(src)); previous = previous_masks = None
    with zipfile.ZipFile(root / 'masks.zip') as archive:
        for i, row in enumerate(rows):
            ok, bgr = cap.read(); assert ok, (sid, i, 'decode failed')
            gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (width, height), interpolation=cv2.INTER_AREA)
            with np.load(io.BytesIO(archive.read(f'masks/{i:06d}.npz'))) as pack:
                assert pack['shape'].tolist() == [meta['height'], meta['width']]
                masks = np.unpackbits(pack['masks'], axis=1, count=meta['height'] * meta['width']).reshape(-1, meta['height'], meta['width'])
            assert len(masks) == len(row['instances'])
            current_masks = {o['instance_id']: cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool) for o, mask in zip(row['instances'], masks)}
            if previous is not None:
                context = pair_context(previous, gray)
                support = context['texture'] & context['inbound']
                flow_scores[i] = round(100 * float(context['reliable'][support].mean()), 1) if support.sum() >= 16 else None
                for key in current_masks.keys() & previous_masks.keys():
                    e = pair_evidence(previous_masks[key], current_masks[key], context)
                    evidence[i][key].append(dict(e, pair=[i-1, i]))
                    evidence[i-1][key].append(dict(e, pair=[i-1, i]))
            previous, previous_masks = gray, current_masks
            if i % 20 == 0 or i == len(rows)-1: progress(i+1)
        assert not cap.read()[0], (sid, 'extra source frames')
    cap.release()
    depths = json.loads((root / 'depth_validation.json').read_text())['frames']
    output = []; class_scores = defaultdict(list); statuses = Counter()
    for i, row in enumerate(rows):
        instances = [final_instance(o, evidence[i][o['instance_id']], by_class[o['class_id']]) for o in row['instances']]
        for o in instances: class_scores[o['class_id']].append(o['temporal_score']); statuses[o['status']] += 1
        output.append(dict(frame_index=i, timestamp_s=row['timestamp_s'], instances=instances,
                           missing_classes=[dict(class_id=c, score=None, status='visibility_unknown') for c in row['missing_classes']],
                           depth=depth_score(depths[i]), flow_roundtrip_score=flow_scores[i]))
    labels=set(class_scores)|set(by_class)|{c for r in rows for c in r['missing_classes']}
    classes = {k: dict(summary(class_scores[k]), detected_frames=sum(any(o['class_id']==k for o in r['instances']) for r in rows),
                       known_issue_codes=by_class[k]) for k in sorted(labels)}
    raw = [o['score'] for r in rows for o in r['instances'] if o.get('score') is not None]
    report = dict(schema='annotation.evidence_confidence.v1', version=VERSION, sample=sid, fps=clip['fps'],
                  calibrated_probability=False, accuracy_verified=False, training_eligible=False,
                  score_meaning='0–100 时序一致性分；不是正确概率，不证明物体身份或真实接触',
                  model_score_audit=dict(count=len(raw), distinct=len(set(raw)), min=min(raw) if raw else None,
                                         max=max(raw) if raw else None, used_in_score=False),
                  source_sha256=meta['source_sha256'], input_hashes={name:digest(root/name) for name in ['objects.json','masks.zip','depth_validation.json']},
                  conflict_record_sha256=digest(model_root/'visual_review.json'),
                  scoring_code_sha256=digest(Path(__file__)), runtime=dict(opencv=cv2.__version__,numpy=np.__version__),
                  parameters=PARAMETERS, conflict_source='previous_assistant_visual_review_not_human_gold',
                  unavailable=UNKNOWN_REASONS,
                  summary=dict(frames=len(rows), classes=classes, statuses=dict(statuses),
                               depth_stability=summary([r['depth']['stability_score'] for r in output]),
                               flow_roundtrip=summary(flow_scores), global_correctness_score=None), frames=output)
    write_json(destination / 'samples' / f'{sid}.json', report)
    return {k:v for k,v in report.items() if k != 'frames'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(2)
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    clips=[c for c in json.loads((a.site/'assets/multi_manifest.json').read_text())['clips'] if c.get('multimodal')]
    review=json.loads((a.model_root/'visual_review.json').read_text());total=sum(c['frames'] for c in clips);done=0;results={}
    writer=EventFileWriter(str(a.output/'events'/'automatic_evidence_v1'))
    def log(extra=None):
        values={'progress/completed_frames':done,'progress/total_frames':total,'progress/fraction':done/total,'evidence/accuracy_calibrated':0,'execution/gpu_used':0,'evidence/semantic_scoring_available':0}
        values.update(extra or {})
        writer.add_event(Event(wall_time=time.time(),step=done,summary=Summary(value=[Summary.Value(tag=k,simple_value=v) for k,v in values.items()])));writer.flush()
    log()
    try:
        for c in clips:
            base=done
            def progress(count):
                nonlocal done
                done=base+count;log({f'samples/{c["id"]}/fraction':count/c['frames']})
                write_json(a.output/'progress.json',dict(completed=done,total=total,current_sample=c['id'],status='running'))
            results[c['id']]=score_episode(c,a.site,a.model_root,a.output,review,progress)
            print(c['id'],results[c['id']]['summary']['statuses'],flush=True)
        write_json(a.output/'summary.json',dict(version=VERSION,frames=total,samples=results,calibrated_probability=False))
        write_json(a.output/'progress.json',dict(completed=done,total=total,status='completed'))
        log({'progress/completed':1})
    except Exception:
        log({'progress/failed':1});raise
    finally:writer.close()


if __name__=='__main__':main()
