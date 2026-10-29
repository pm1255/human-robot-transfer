"""SAM 3.1 text-prompt video masks; boxes derive from the exact saved masks.

One known manipulated T-shirt is the target. Every model instance is retained;
its union forms the garment mask. Instance IDs are local to each tracking window.
No empty or failed mask is filled by optical flow or an invented rectangle.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from .common import Progress, atomic_json, sha256
from .sam3_compat import install_session_adapter
from human2robot.hand_mesh.video import probe, frames


def mask_box(mask):
    y, x = np.nonzero(mask)
    return None if not len(x) else [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompt', default='white t-shirt')
    parser.add_argument('--chunk-size', type=int, default=300)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    video_meta = probe(args.video)
    count, fps = video_meta['frames'], video_meta['fps']
    width, height = video_meta['width'], video_meta['height']
    if min(count, fps, width, height) <= 0 or args.chunk_size <= 0 or args.max_frames < 0:
        raise ValueError('Video metadata and chunk size must be positive')
    total = min(count, args.max_frames) if args.max_frames else count
    progress = Progress(root, 'sam3', total)
    progress.update(0, 'running', phase='loading_model')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('SAM 3.1 official predictor requires a CUDA GPU')
        from sam3.model_builder import build_sam3_predictor
        predictor = build_sam3_predictor(checkpoint_path=args.checkpoint, version='sam3.1',
                                        compile=False, warm_up=False, use_fa3=False,
                                        async_loading_frames=False)
        adapted = install_session_adapter(predictor)
        provenance = dict(schema='object_annotation.sam3.v1', video=args.video,
                          source_sha256=sha256(args.video), checkpoint=args.checkpoint,
                          checkpoint_sha256=sha256(args.checkpoint), model='SAM 3.1 multiplex',
                          prompt=args.prompt, fps=fps, width=width, height=height,
                          source_frames=count, processed_target=total, chunk_size=args.chunk_size,
                          threshold=0.5, bbox_format='xyxy_pixels_exclusive_max',
                          object_id='garment_1', instance_ids='local_to_chunk',
                          mask_policy='union_of_all_tshirt_instances; no interpolation',
                          session_compatibility_adapter=adapted,
                          rigid_object=False, training_ground_truth=False)
        if (root/'segmentation_manifest.json').exists() and list((root/'masks').glob('*.npz')):
            previous_meta = json.loads((root/'segmentation_manifest.json').read_text())
            for key in ('source_sha256','checkpoint_sha256','prompt','processed_target','chunk_size'):
                if previous_meta.get(key) != provenance[key]:
                    raise ValueError(f'Cannot resume incompatible annotations: {key}')
        atomic_json(root/'segmentation_manifest.json', provenance)
        (root/'masks').mkdir(exist_ok=True)
        (root/'chunks').mkdir(exist_ok=True)
        (root/'previews').mkdir(exist_ok=True)
        completed, detected, previous, rows = 0, 0, None, []
        for start in range(0, total, args.chunk_size):
            end = min(total, start + args.chunk_size)
            chunk = root/'chunks'/f'{start:06d}'
            chunk.mkdir(exist_ok=True)
            metadata = root/'masks'/f'{start:06d}.json'
            arrays = root/'masks'/f'{start:06d}.npz'
            if metadata.exists() and arrays.exists():
                old = json.loads(metadata.read_text())
                if [r['frame_index'] for r in old] != list(range(start,end)):
                    raise ValueError(f'Incomplete saved chunk {start}')
                rows.extend(old)
                detected += sum(row['mask_area'] > 0 for row in old)
                completed += len(old)
                packed = np.load(arrays)
                if not np.array_equal(packed['frame_indices'],np.arange(start,end)):
                    raise ValueError(f'Incorrect mask indices in chunk {start}')
                previous = np.unpackbits(packed['masks'][-1], count=height*width).reshape(height,width).astype(bool)
                progress.update(completed, phase='resume', detected_frames=detected)
                continue
            for i, frame in enumerate(frames(args.video,video_meta,start=start,limit=end-start),start):
                if not cv2.imwrite(str(chunk/f'{i-start:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                    raise RuntimeError('Failed to save decoded frame')
            sid = predictor.handle_request(dict(type='start_session', resource_path=str(chunk),
                       offload_video_to_cpu=True))['session_id']
            predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text=args.prompt))
            pending = {}
            for result in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=sid,
                        start_frame_index=0,propagation_direction='forward',evict_cached_frame_outputs=True)):
                local = int(result['frame_index'])
                if not 0 <= local < end-start:
                    raise RuntimeError(f'Out-of-range model frame {local}')
                out = result['outputs']
                masks = as_numpy(out['out_binary_masks']).astype(bool)
                ids = as_numpy(out['out_obj_ids']).reshape(-1)
                scores = as_numpy(out['out_probs']).reshape(-1)
                if masks.size == 0:
                    masks = np.zeros((0,height,width),bool)
                masks = masks.reshape((-1,height,width))
                if len(ids) != len(masks) or len(scores) != len(masks):
                    raise RuntimeError('SAM output instance lengths disagree')
                union = masks.any(0) if len(masks) else np.zeros((height,width),bool)
                instances = [dict(model_id=int(oid),score=float(score),bbox=mask_box(mask),area=int(mask.sum()))
                             for oid,score,mask in zip(ids,scores,masks)]
                pending[local] = (union, instances)
                if local % 20 == 0:
                    progress.update(completed+len(pending),phase='inference',chunk_start=start,
                        gpu_memory_gb=torch.cuda.memory_allocated()/1024**3)
            predictor.handle_request(dict(type='close_session',session_id=sid))
            if set(pending) != set(range(end-start)):
                raise RuntimeError(f'Missing SAM result frames in chunk {start}')
            chunk_rows, packed_masks = [], []
            for local in range(end-start):
                mask, instances = pending[local]
                frame_i = start+local
                area = int(mask.sum())
                flags = []
                if area == 0: flags.append('no_detection')
                if len(instances)>1: flags.append('multiple_instances_review')
                overlap = None
                if previous is not None:
                    union_area = int((previous|mask).sum())
                    overlap = float((previous&mask).sum()/union_area) if union_area else None
                    if overlap is not None and overlap<0.5:flags.append('mask_change_review')
                row=dict(frame_index=frame_i,timestamp_s=frame_i/fps,object_id='garment_1',
                         mask_area=area,bbox=mask_box(mask),bbox_source='sam3_mask_extent',
                         instances=instances,chunk_start=start,adjacent_mask_iou=overlap,
                         flags=flags,status='predicted' if area else 'no_detection')
                chunk_rows.append(row)
                packed_masks.append(np.packbits(mask.reshape(-1)))
                if frame_i % 300 == 0 or frame_i == total-1:
                    frame=cv2.imread(str(chunk/f'{local:06d}.jpg'))
                    view=frame.copy();view[mask]=(view[mask]*.55+np.array([80,190,50])*.45).astype(np.uint8)
                    cv2.imwrite(str(root/'previews'/f'{frame_i:06d}.jpg'),view)
                    progress.writer.add_image(f'preview/frame_{frame_i:06d}',view[:,:,::-1].copy(),frame_i,dataformats='HWC')
                detected += int(area>0)
                previous=mask
            np.savez_compressed(arrays,masks=np.stack(packed_masks),shape=np.array([height,width]),
                                frame_indices=np.arange(start,end))
            atomic_json(metadata,chunk_rows)
            rows.extend(chunk_rows)
            completed+=len(chunk_rows)
            progress.update(completed,phase='chunk_complete',detected_frames=detected)
            # Decoded JPEGs are a cache; keep them for exact visual audits and depth preparation.
            del pending
            gc.collect();torch.cuda.empty_cache()
        if len(rows) != total: raise RuntimeError(f'Only {len(rows)}/{total} frames have SAM outputs')
        atomic_json(root/'objects.json',dict(meta=provenance,frames=rows))
        atomic_json(root/'segmentation_quality.json',dict(frames=total,detected_frames=detected,
                    missing_frames=total-detected,review_frames=sum(bool(x['flags']) for x in rows),
                    independently_verified=False))
        progress.update(total,'completed',detected_frames=detected)
    except Exception as exc:
        progress.update(locals().get('completed',0),'failed',error=str(exc))
        raise
    finally:
        progress.close()

if __name__=='__main__':main()
