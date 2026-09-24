"""One-time correction of OpenCV MJPEG end-of-frame timestamp reporting.

Only applies to this recorded ffmpeg uniform proxy grid. Does not shift native
source-video PTS. Keypoints and images remain untouched; correction is logged.
"""
import argparse,json
from pathlib import Path
import numpy as np

def correct(root):
    reports=[]
    for folder in sorted(Path(root).iterdir()):
        path=folder/'annotation.json'
        if not path.exists():continue
        a=json.loads(path.read_text());m=a['metadata']
        if m.get('timestamp_source')!='ffmpeg_resampled_proxy_grid':continue
        old=np.asarray(a['timestamp']);dt=float(np.median(np.diff(old)))
        if abs(old[0])<1e-8:continue
        if not np.isclose(old[0],dt,atol=1e-6):raise ValueError('Unexpected proxy offset')
        new=np.arange(len(old))*dt;a['timestamp']=new.tolist()
        correction={'old_first_timestamp':float(old[0]),'new_first_timestamp':0.,
            'reason':'OpenCV MJPEG POS_MSEC reports end of read frame; ffmpeg proxy uses zero-based uniform frame grid'}
        m['timestamp_correction']=correction;path.write_text(json.dumps(a,ensure_ascii=False,indent=2))
        p=folder/'keypoints.json';k=json.loads(p.read_text())
        for i,row in enumerate(k):row['timestamp']=float(new[i])
        p.write_text(json.dumps(k,ensure_ascii=False,indent=2))
        p=folder/'rgb_supervision.npz'
        with np.load(p) as d:arrays={key:d[key] for key in d.files}
        arrays['timestamp']=new;np.savez_compressed(p,**arrays)
        reports.append({'episode':folder.name,**correction})
    print(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();correct(a.root)
