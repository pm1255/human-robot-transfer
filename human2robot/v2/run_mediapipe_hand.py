import cv2
import json
import os
import math
import mediapipe as mp

INPUT = os.environ.get('HUMANEGO_INPUT', '/user/panmiao/tmp/humanego_upload/humanego_short.mp4')
MODEL = os.environ.get('HAND_LANDMARKER_MODEL', '/user/panmiao/tmp/humanego_upload/models/hand_landmarker.task')
OUT_DIR = os.environ.get('HUMANEGO_HAND_OUT', '/user/panmiao/tmp/humanego_upload/hand')
os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.splitext(os.path.basename(INPUT))[0]
OUT_SKELETON = os.path.join(OUT_DIR, stem + '_hand_skeleton.mp4')
OUT_SHAPE = os.path.join(OUT_DIR, stem + '_hand_shape.mp4')
# Backward-compatible alias: the legacy hand video now means skeleton only.
OUT_VIDEO = OUT_SKELETON
OUT_JSON = os.path.join(OUT_DIR, stem + '_hand.json')

EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
         (13,17),(17,18),(18,19),(19,20), (0,17)]
FINGERS = [(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]

def dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def angle(a, b, c):
    """Angle ABC in degrees, used for a less brittle discrete hand state."""
    u = (a.x-b.x, a.y-b.y, a.z-b.z)
    v = (c.x-b.x, c.y-b.y, c.z-b.z)
    nu = math.sqrt(sum(x*x for x in u)); nv = math.sqrt(sum(x*x for x in v))
    if nu*nv < 1e-8:
        return 0.0
    dot = sum(x*y for x,y in zip(u,v)) / (nu*nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))

def classify_shape(lm):
    # Geometry-only state, but with joint angles and temporal filtering below.
    # This is deliberately a conservative discrete label; uncertain frames are
    # kept as ``unknown`` instead of forcing an open-palm/fist decision.
    palm = max(dist(lm[0], lm[9]), 1e-6)
    extended = 0
    finger_scores = {}
    for name, (mcp, pip, dip, tip) in zip(['index','middle','ring','pinky'], FINGERS[1:]):
        a = angle(lm[mcp], lm[pip], lm[dip])
        b = angle(lm[pip], lm[dip], lm[tip])
        reach = dist(lm[tip], lm[0]) / palm
        is_ext = bool(a > 145.0 and b > 145.0 and reach > 1.45)
        finger_scores[name] = {'pip_angle': a, 'dip_angle': b, 'reach': reach, 'extended': is_ext}
        extended += int(is_ext)
    thumb_open = bool(angle(lm[1], lm[2], lm[4]) > 130.0 and dist(lm[4], lm[0]) / palm > 1.0)
    pinch_ratio = dist(lm[4], lm[8]) / palm
    pinch = pinch_ratio < 0.35
    if pinch:
        label = 'pinch'
    elif extended >= 4 and thumb_open:
        label = 'open_palm'
    elif extended == 0 and max(dist(lm[i], lm[0]) for i in [4,8,12,16,20]) / palm < 1.55:
        label = 'fist'
    elif extended == 1 and finger_scores['index']['extended']:
        label = 'point'
    elif extended <= 2:
        label = 'power_grasp'
    else:
        label = 'mixed_open'
    # Low margin means the label should be reviewed, even if a class was chosen.
    margin = min(abs(finger_scores[n]['reach'] - 1.45) for n in finger_scores)
    shape_conf = max(0.0, min(1.0, 0.45 + 0.22 * margin + (0.18 if pinch else 0.0)))
    return label, {'extended_non_thumb': extended, 'thumb_open': bool(thumb_open),
                   'pinch_ratio': float(pinch_ratio), 'finger_scores': finger_scores,
                   'shape_confidence': float(shape_conf)}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    # Keep the detector permissive; the temporal gate below is the anti-
    # hallucination filter. Raising these thresholds loses the real hand in
    # egocentric wide views before the continuity check can see it.
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
)

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 320)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 240)
frames = []
raw_frames = []
with HandLandmarker.create_from_options(options) as detector:
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        base_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(image, int(idx * 1000.0 / fps))
        hands = []
        for hidx, lms in enumerate(result.hand_landmarks):
            pts = [{'x': float(p.x), 'y': float(p.y), 'z': float(p.z), 'visibility': float(getattr(p, 'visibility', 1.0) or 1.0)} for p in lms]
            label, stats = classify_shape(lms)
            handed = result.handedness[hidx][0] if hidx < len(result.handedness) and result.handedness[hidx] else None
            handedness = {'category': handed.category_name, 'score': float(handed.score)} if handed else None
            hands.append({'hand_index': hidx, 'handedness': handedness,
                          # retain the normalized MediaPipe coordinates used
                          # by the retargeting solver
                          'landmarks': pts,
                          'shape_label': label, 'shape_method': 'landmark_geometry_heuristic', 'shape_stats': stats})
            pix = [(int(p.x * width), int(p.y * height)) for p in lms]
            for a, b in EDGES:
                cv2.line(frame, pix[a], pix[b], (0, 220, 255), 2, cv2.LINE_AA)
            for x, y in pix:
                cv2.circle(frame, (x, y), 2, (30, 255, 30), -1, cv2.LINE_AA)
            anchor = pix[0] if pix else (5, 18)
            cv2.putText(frame, f'{label}', (anchor[0] + 4, anchor[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'frame={idx} hands={len(hands)}', (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        raw_frames.append(base_frame)
        frames.append({'frame_index': idx, 'timestamp_ms': int(idx * 1000.0 / fps), 'hands': hands})
        idx += 1
cap.release()

# Remove one-off texture matches and short hallucinated runs.  A hand state
# becomes usable only after it is present for eight consecutive frames.  This
# removes the flower-pot false positive at the beginning of the debug clip.
for handed in ('Left', 'Right'):
    present = [any((h.get('handedness') or {}).get('category') == handed for h in f['hands']) for f in frames]
    keep = set(); i = 0
    while i < len(present):
        if not present[i]: i += 1; continue
        j = i
        while j < len(present) and present[j]: j += 1
        if j - i >= 8:
            keep.update(range(i, j))
        i = j
    for i, f in enumerate(frames):
        f['hands'] = [h for h in f['hands'] if (h.get('handedness') or {}).get('category') != handed or i in keep]
        for h in f['hands']:
            h['temporal_gate'] = 'pass' if i in keep else 'short_run_removed'

# Re-render only temporally accepted detections.  The old implementation drew
# the raw detector output before the gate, which made a rejected flower-pot
# match visible even though it had been removed from JSON.
writer_skeleton = cv2.VideoWriter(OUT_SKELETON, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
writer_shape = cv2.VideoWriter(OUT_SHAPE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for idx, (frame, data) in enumerate(zip(raw_frames, frames)):
    skeleton = frame.copy()
    shape = frame.copy()
    shape_labels = []
    for h in data['hands']:
        pix = [(int(p['x'] * width), int(p['y'] * height)) for p in h['landmarks']]
        for a, b in EDGES:
            cv2.line(skeleton, pix[a], pix[b], (0, 220, 255), 2, cv2.LINE_AA)
        for x, y in pix:
            cv2.circle(skeleton, (x, y), 2, (30, 255, 30), -1, cv2.LINE_AA)
        anchor = pix[0] if pix else (5, 18)
        label = (h.get('handedness') or {}).get('category', f"hand{len(shape_labels)}")
        shape_labels.append(f"{label}: {h['shape_label']} ({h['shape_stats']['shape_confidence']:.2f})")
    cv2.putText(skeleton, f"HAND SKELETON | frame={idx} hands={len(data['hands'])} temporal-gated", (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(shape, (6, 6), (min(width - 6, 390), 46 + 21 * max(1, len(shape_labels))), (8, 18, 25), -1)
    cv2.putText(shape, "DISCRETE HAND SHAPE", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, .48, (235,255,255), 1, cv2.LINE_AA)
    for k, label in enumerate(shape_labels or ['no hand detected']):
        cv2.putText(shape, label, (14, 49 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,230,255) if shape_labels else (180,190,200), 1, cv2.LINE_AA)
    cv2.putText(shape, f"frame={idx} | no skeleton overlay", (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    writer_skeleton.write(skeleton)
    writer_shape.write(shape)
writer_skeleton.release()
writer_shape.release()
with open(OUT_JSON, 'w') as f:
    json.dump({'schema': 'humanego.hand_annotation.v2', 'input': INPUT, 'fps': fps, 'width': width, 'height': height,
               'num_frames': len(frames), 'model': MODEL,
               'shape_note': 'Discrete shape is a geometry heuristic over MediaPipe landmarks; it is not a trained hand-shape classifier.',
               'video': OUT_SKELETON, 'skeleton_video': OUT_SKELETON, 'shape_video': OUT_SHAPE,
               'frames': frames}, f)
print(json.dumps({'video': OUT_SKELETON, 'skeleton_video': OUT_SKELETON, 'shape_video': OUT_SHAPE, 'json': OUT_JSON, 'frames': len(frames),
                  'frames_with_hands': sum(bool(x['hands']) for x in frames)}))
