"""Render saved dense HaMeR JSON without pretending to rerun inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from .video import probe, frames, Writer
from .render import render_hands, composite
from .infer import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--topology',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--focal',type=float,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.predictions.read_text());meta=probe(a.video);faces=np.load(a.topology)['faces']
    if data['model']!='HaMeR' or data['num_frames']!=meta['frames']:raise ValueError('Source mismatch')
    by_frame={f['frame_index']:f for f in data['frames']}
    if set(by_frame)!=set(range(meta['frames'])):raise ValueError('Non-contiguous prediction frames')
    log=SummaryWriter(str(a.out/'events'));w=Writer(a.out/'hamer_surface.mp4',meta['width'],meta['height'],meta['fps'])
    masks=0;mesh=0
    for idx,img in enumerate(frames(a.video,meta)):
        hands=by_frame[idx]['hands'];rgb,depth,_=render_hands(hands,faces,a.focal,meta['width'],meta['height'])
        overlay=composite(img,rgb,depth);w.write(overlay);mesh+=bool(hands);masks+=int(np.isfinite(depth).sum())
        if idx in [120,600,1200,1800,2400,3000,3600,4200]:
            cv2.imwrite(str(a.out/f'frame_{idx:05d}.jpg'),np.concatenate([img,overlay],axis=1))
        if idx%120==0:
            log.add_scalar('progress/render',(idx+1)/meta['frames'],idx);log.flush();print(idx,flush=True)
    w.close();log.add_scalar('progress/render',1,meta['frames']);log.close()
    (a.out/'manifest.json').write_text(json.dumps(dict(**meta,mesh_frames=mesh,rendered_pixels=masks,
        source_predictions=str(a.predictions),prediction_sha256=sha(a.predictions),video_sha256=sha(a.video),
        renderer='triangle_z_buffer',focal=a.focal,new_inference=False,quality_status='REVIEW_NO_3D_GROUND_TRUTH'),indent=2))


if __name__=='__main__':main()
