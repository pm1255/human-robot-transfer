"""Validate completed SAM artifacts and attach them to the existing six-card UI."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .common import atomic_json, sha256
from human2robot.hand_mesh.video import probe, frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--web',required=True)
    p.add_argument('--manifest-copy');a=p.parse_args();root=Path(a.output);web=Path(a.web)
    data=json.loads((root/'objects.json').read_text());meta=data['meta'];rows=data['frames']
    manifest=json.loads((web/'assets/multi_manifest.json').read_text());clip=manifest['clips'][0]
    if len(rows)!=clip['frames'] or meta['processed_target']!=clip['frames']:
        raise ValueError('Partial results cannot replace full videos')
    if sha256(web/'source.mp4')!=meta['source_sha256']:
        raise ValueError('Source video hash differs from annotation source')
    for stage in [meta.get('final_segmentation_stage','sam3_adaptive'),'render']:
        status=json.loads((root/f'{stage}_status.json').read_text())
        if status['status']!='completed' or status['completed']!=clip['frames']:
            raise ValueError(f'{stage} is not complete')
    start=None;detected=0;review=0
    for i,row in enumerate(rows):
        if row['frame_index']!=i:raise ValueError('Annotation frame sequence differs')
        if row['chunk_start']!=start:
            start=row['chunk_start']
            with np.load(root/'masks'/f'{start:06d}.npz') as pack:
                masks=np.unpackbits(pack['masks'],axis=1,count=clip['height']*clip['width']).reshape(-1,clip['height'],clip['width'])
        mask=masks[i-start];y,x=np.nonzero(mask)
        box=None if not len(x) else [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
        if box!=row['bbox'] or int(mask.sum())!=row['mask_area']:
            raise ValueError(f'Mask and bounding box differ at frame {i}')
        detected+=bool(len(x));review+=bool(row['flags'])
    validations={}
    for name in ['segmentation','boundingbox']:
        video=root/f'{name}.mp4';details=probe(video)
        if details['width']!=clip['width'] or details['height']!=clip['height'] or abs(details['fps']-clip['fps'])>1e-6:
            raise ValueError(f'{name} video dimensions or FPS differ')
        count=sum(1 for _ in frames(video,details))
        if count!=clip['frames']:raise ValueError('Incomplete video decode')
        validations[name]=dict(**details,decoded_frames=count,sha256=sha256(video))
    review_path=root/'visual_review.json'
    if not review_path.exists():raise ValueError('Sample visual review record is required')
    target=web/'object_annotation';target.mkdir(exist_ok=True)
    for name in ['segmentation.mp4','boundingbox.mp4','segmentation_poster.jpg','boundingbox_poster.jpg',
                 'objects.json','segmentation_manifest.json','segmentation_quality.json','video_validation.json','visual_review.json']:
        shutil.copy2(root/name,target/name)
    for layer in clip['layers']:
        if layer['id'] not in ['sam3','bbox']:continue
        name='segmentation' if layer['id']=='sam3' else 'boundingbox'
        layer.update(video=f'object_annotation/{name}.mp4',poster=f'object_annotation/{name}_poster.jpg',
                     badge='已重做 · 待复核',badge_tone='updated',
                     assessment=f'完整 {len(rows)} 帧，{detected} 帧有物体掩码，{len(rows)-detected} 帧未检测；{review} 帧有自动复核标记。模型精度尚无独立金标验证。')
        layer['tags']=['SAM3.1 实例分割','逐帧估计','形变保留'] if layer['id']=='sam3' else ['掩码派生物体框','逐帧对应']
        layer.pop('missing_reason',None);layer.pop('empty_title',None)
    atomic_json(target/'publication_validation.json',dict(videos=validations,
                masks_and_boxes_checked=len(rows),independently_verified=False))
    atomic_json(web/'assets/multi_manifest.json',manifest)
    if a.manifest_copy:atomic_json(a.manifest_copy,manifest)
    print(json.dumps(dict(published=True,frames=len(rows),detected=detected,review=review)))


if __name__=='__main__':main()
