# Blind human-video annotation and QC plan

## Scope

This experiment uses one short RGB-only clip from HumanEgo. Public annotations are treated as unavailable. The dataset archive is kept in _humanego_stage/reference_holdout and is outside every input, training, and QC path.

The current experiment does not perform human-to-robot retargeting or robot supervision.

## Four annotation layers

1. Video and provenance layer: source id, clip id, frame range, timestamp, fps, resolution, codec, camera/view, file hash, and decode status.
2. Temporal and language layer: clip boundaries, task/subtask/step hierarchy, action start/end, language text, normalized action, language reviewer, confidence, and approval state. A dataset folder name is provenance metadata, never a training label.
3. Hand and object interaction layer: left/right hand visibility, 2D keypoints and confidence, optional 3D keypoints/relative hand-object pose, object instance id, mask or box, visibility/occlusion, contact/grasp/release events, and track confidence.
4. High-value geometry layer: only for rigid, orientation-sensitive objects and a small gold subset, record object 6D pose, hand-object relative pose, uncertainty, annotator/evidence source, and review status. Do not require object 6D for every clip.

## QC gates

### File and video QC

- Verify file hash, frame count, fps, width/height, duration, and codec.
- Decode all frames with FFmpeg; record failures and missing ranges.
- Measure black-frame and clipping ratios, median Laplacian sharpness, brightness range, and audio presence.
- Compute byte-level duplicate frames and a perceptual near-duplicate signal. A near-duplicate warning is not an automatic rejection for slow motions; inspect the flagged intervals.
- Reject or quarantine corrupt files, missing frames, severe exposure failure, and clips outside the duration policy.

### Segment and language QC

- Bounds must satisfy 0 <= start < end <= video duration. Segments must not overlap.
- Keep a pending state until a reviewer checks the action boundary and language.
- Use two independent reviewers for a gold subset. Store boundary disagreement, normalized-language agreement, and a reason code for every correction.
- Keep source task names separate from human language labels and do not use them as supervision.

### Tracking and interaction QC

- Check left/right hand and object id continuity; flag implausible jumps, long missing runs, and identity swaps.
- Check visibility and occlusion reasons rather than forcing a coordinate through an occlusion.
- For 3D points, store confidence and camera/reference frame. Use reprojection as a soft diagnostic and calibrate thresholds per device.
- Contact/grasp/release must agree with hand-object distance, relative motion, and the temporal language segment. Send disagreements to review.

### Duplicate and split QC

- Deduplicate exact file hashes and adjacent frame hashes.
- Use perceptual or embedding retrieval to find visually repeated clips; review high-similarity pairs.
- Split by participant and recording session before training. Never let neighboring clips from one long recording cross train/eval.
- Freeze the blind manifest and annotations before comparing with any hidden reference labels.

## Current smoke test

Input: inputs/unlabeled/humanego_serve_bread_000_6s.mp4
Manifest: manifests/humanego_serve_bread_000_6s.json
Segment manifest: manifests/humanego_serve_bread_000_6s.segment.json
QC: outputs/humanego_serve_bread_000_6s/qc/qc_report.json
Blocked run record: outputs/humanego_serve_bread_000_6s/RUN_BLOCKED.md

The clip has 180/180 frames decoded, 640x480, 30 fps, and 6 seconds. The language field is intentionally unverified. The current annotation command stops before inference because the devspace cannot download hand_landmarker.task and mediapipe is not installed.

## Reproducible commands

python3 tools/qc_unlabeled.py --video inputs/unlabeled/humanego_serve_bread_000_6s.mp4 --out outputs/humanego_serve_bread_000_6s/qc --segment-manifest manifests/humanego_serve_bread_000_6s.segment.json
python3 tools/run.py annotate --video inputs/unlabeled/humanego_serve_bread_000_6s.mp4 --view egocentric --out outputs/humanego_serve_bread_000_6s

Mount the three model files through an internal mirror before rerunning annotation. Do not copy overlay videos or official JSON labels into inputs.
