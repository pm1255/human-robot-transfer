"""Checked streaming video I/O; works even when OpenCV lacks FFmpeg."""
import json
import subprocess
from fractions import Fraction
import numpy as np


def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_streams','-of','json',str(path)]))['streams'][0]
    return dict(width=int(data['width']),height=int(data['height']),
                fps=float(Fraction(data['avg_frame_rate'])),frames=int(data['nb_frames']))


def frames(path, meta, start=0, limit=0):
    expected=min(limit or meta['frames'],meta['frames']-start)
    proc=subprocess.Popen(['ffmpeg','-v','error','-threads','2','-ss',str(start/meta['fps']),'-i',str(path),
        '-frames:v',str(expected),'-f','rawvideo','-pix_fmt','bgr24','-'],stdout=subprocess.PIPE)
    count=0; size=meta['width']*meta['height']*3
    try:
        while True:
            data=proc.stdout.read(size)
            if not data: break
            if len(data)!=size: raise RuntimeError('Truncated decoded frame')
            count+=1
            yield np.frombuffer(data,np.uint8).reshape(meta['height'],meta['width'],3).copy()
        if proc.wait()!=0 or count!=expected:
            raise RuntimeError(f'Decode incomplete: {count}/{expected}')
    finally:
        proc.stdout.close()
        if proc.poll() is None: proc.terminate()
        proc.wait()


class Writer:
    def __init__(self,path,width,height,fps):
        self.path=path;self.count=0
        self.proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24',
            '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264',
            '-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,frame):
        self.proc.stdin.write(np.ascontiguousarray(frame,dtype=np.uint8).tobytes()); self.count+=1
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait()!=0: raise RuntimeError('Video encoder failed')
        actual=probe(self.path)
        if actual['frames']!=self.count: raise RuntimeError('Encoded frame count differs')
