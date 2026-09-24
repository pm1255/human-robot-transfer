"""Render predictions without inventing missing detections."""
import subprocess
import cv2,numpy as np
HAND=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
BODY=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
def render(video,annotation,out):
 cap=cv2.VideoCapture(str(video));w=int(cap.get(3));h=int(cap.get(4));proc=None
 try:
  if not cap.isOpened():raise ValueError('Cannot decode preview input.')
  proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','10','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
  for f in annotation['frames']:
   ok,im=cap.read()
   if not ok:raise ValueError('Video ended before annotation frames.')
   for hand in f['hands']:
    xy=(np.asarray(hand['xy'])*[w,h]).astype(int);color=(90,220,130) if hand['side']=='Left' else (240,160,75)
    for i,j in HAND:cv2.line(im,tuple(xy[i]),tuple(xy[j]),color,2,cv2.LINE_AA)
    for xy_i in xy:cv2.circle(im,tuple(xy_i),2,(255,255,255),-1)
   b=np.asarray(f.get('body',[]))
   if b.shape==(33,4):
    for i,j in BODY:
     if min(b[i,2:].min(),b[j,2:].min())>.6:cv2.line(im,tuple((b[i,:2]*[w,h]).astype(int)),tuple((b[j,:2]*[w,h]).astype(int)),(90,230,150),2)
   for o in f['objects']:
    x,y,bw,bh=o['box'];cv2.rectangle(im,(int(x*w),int(y*h)),(int((x+bw)*w),int((y+bh)*h)),(40,200,255),1)
   cv2.putText(im,'MODEL PREDICTIONS - NOT GROUND TRUTH',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
   proc.stdin.write(im.tobytes())
 finally:
  cap.release()
  if proc:
   proc.stdin.close();code=proc.wait()
   if code:raise RuntimeError('ffmpeg failed to encode annotations')
