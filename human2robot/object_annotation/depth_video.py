"""MoGe-3 per-frame depth. Retain deforming surfaces, never rigidify the cloth."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from human2robot.hand_mesh.video import probe, frames
from .common import Progress, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--resolution-level', type=int, default=9)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    meta = probe(args.video)
    total = min(meta['frames'], args.max_frames) if args.max_frames else meta['frames']
    if total <= 0 or not 0 <= args.resolution_level <= 9:
        raise ValueError('Invalid depth inference configuration')
    progress = Progress(root, 'depth', total)
    done = 0
    try:
        progress.update(0, phase='loading_model')
        if not torch.cuda.is_available():
            raise RuntimeError('MoGe-3 requires CUDA / FlexGEMM')
        from .flex_compat import install_dtype_itemsize
        install_dtype_itemsize()
        from moge.model.v3 import MoGeModel
        checkpoint_hash = sha256(args.checkpoint)
        if checkpoint_hash != '9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925':
            raise ValueError('MoGe-3 ViT-L checkpoint differs from pinned Hugging Face LFS hash')
        config = dict(schema='object_annotation.depth.v1', model='MoGe-3 ViT-L',
                      model_revision='184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be',
                      checkpoint_sha256=checkpoint_hash, source_sha256=sha256(args.video),
                      video=args.video, **meta, processed_target=total,
                      resolution_level=args.resolution_level, refine_steps=3,
                      depth_source='monocular_model_prediction',
                      scale_source='MoGe_learned_metric_prior_unverified',
                      scale_aligned_to_hands=False, intrinsics_source='per_frame_MoGe_estimate',
                      units='model_estimated_metres_not_calibrated',
                      coordinates='OpenCV_camera_x_right_y_down_z_forward',
                      rigid_object=False, temporal_correspondence=False,
                      training_ground_truth=False)
        manifest = root/'depth_manifest.json'
        depth_dir = root/'depth'; depth_dir.mkdir(exist_ok=True)
        if manifest.exists() and list(depth_dir.glob('*.npz')):
            if json.loads(manifest.read_text()) != config:
                raise ValueError('Cannot resume depth with changed input or model parameters')
        atomic_json(manifest, config)
        model = MoGeModel.from_pretrained(args.checkpoint).to('cuda').eval()
        for index, bgr in enumerate(frames(args.video, meta, limit=total)):
            dest = depth_dir/f'{index:06d}.npz'
            if dest.exists():
                with np.load(dest) as saved:
                    if saved['depth'].shape != (meta['height'],meta['width']) or int(saved['frame_index']) != index:
                        raise ValueError(f'Invalid saved depth frame {index}')
                done = index+1
                continue
            tensor = torch.from_numpy(bgr[:,:,::-1].copy()).permute(2,0,1).cuda().float()/255
            with torch.inference_mode():
                result = model.infer(tensor, resolution_level=args.resolution_level,
                                     refine_steps=3, use_fp16=False)
            depth = result['depth'].float().cpu().numpy()
            valid = result['mask'].cpu().numpy().astype(bool) & np.isfinite(depth) & (depth>0)
            K_normalized = result['intrinsics'].float().cpu().numpy()
            # MoGe normalized pixels use centres (u+0.5)/W, (v+0.5)/H.
            K = K_normalized.copy()
            K[0,:] *= meta['width']; K[1,:] *= meta['height']
            K[0,2] -= .5; K[1,2] -= .5
            if not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0:
                raise ValueError(f'Invalid camera intrinsics at frame {index}')
            depth[~valid] = np.nan
            temporary = dest.with_suffix('.partial.npz')
            np.savez_compressed(temporary, depth=depth.astype(np.float32),
                                valid=valid, K=K, K_normalized=K_normalized,
                                frame_index=index, timestamp_s=index/meta['fps'])
            temporary.replace(dest)
            done = index+1
            if index % 30 == 0 or done==total:
                progress.update(done, phase='inference', valid_pixel_fraction=float(valid.mean()),
                                gpu_memory_gb=torch.cuda.memory_allocated()/1024**3)
                if valid.any():
                    lo,hi = np.percentile(depth[valid],[5,95])
                    gray = (np.nan_to_num(np.clip((depth-lo)/max(float(hi-lo),1e-6),0,1))*255).astype(np.uint8)
                    preview = cv2.applyColorMap(gray,cv2.COLORMAP_TURBO)
                    preview[~valid]=0
                    progress.writer.add_image('estimated_depth',preview[:,:,::-1].copy(),index,dataformats='HWC')
        if done != total:
            raise RuntimeError(f'Depth produced only {done}/{total} frames')
        atomic_json(root/'depth_quality.json',dict(frames=done,source_frames=meta['frames'],
                    coverage='all source frames' if done==meta['frames'] else 'prefix pilot only',
                    independently_verified=False,metric_scale_verified=False,
                    temporal_consistency_verified=False))
        progress.update(done,'completed')
    except Exception as exc:
        progress.update(done,'failed',error=str(exc)); raise
    finally:
        progress.close()


if __name__=='__main__':
    main()
