"""Prompts from inspected source contact sheets, not segmentation ground truth."""
import json
from pathlib import Path
R=Path('/Users/panmiao/Documents/ChatGPT/paper/outputs/unified_30_20261021')
C={}
def add(sid,objects,effector,boxes,frame=0):
    guides={k:dict(frame=f,boxes=b) for k,f,b in objects}
    guides[effector]=dict(frame=frame,boxes=boxes)
    C[sid]=dict(prompts=[[k,k.replace('_',' ')]for k in guides],guides=guides)
def o(name,boxes,frame=0):return(name,frame,boxes)
add('abc130k_ep_010392',[o('cloth',[[.475,.21,.63,.61]]),o('container',[[.22,.35,.42,.60]])],'gripper',[[.21,.46,.35,.57],[.72,.48,.82,.60]])
add('abc130k_ep_014083',[o('bottle',[[.46,.42,.53,.53],[.51,.52,.58,.68],[.61,.48,.68,.59],[.56,.47,.61,.61]]),o('bin',[[.40,.32,.56,.55]])],'gripper',[[.21,.46,.36,.57],[.70,.43,.82,.54]])
add('abc130k_ep_030633',[o('plastic_bag',[[.24,.34,.40,.51]]),o('tray',[[.22,.50,.35,.63]])],'gripper',[[.20,.47,.29,.57],[.51,.39,.62,.48]])
add('abc130k_ep_044053',[o('headphones',[[.39,.44,.55,.60],[.37,.32,.49,.45]])],'gripper',[[.21,.44,.34,.53],[.69,.37,.80,.48]])
add('abc130k_ep_044783',[o('bottle',[[.40,.42,.47,.52],[.34,.44,.40,.55],[.66,.43,.72,.54]]),o('tray',[[.46,.47,.62,.57]])],'gripper',[[.30,.49,.43,.59],[.62,.48,.74,.59]])
add('abc130k_ep_082354',[o('plate',[[.28,.41,.39,.52],[.62,.44,.72,.55],[.46,.56,.59,.68]]),o('rack',[[.43,.37,.61,.55]])],'gripper',[[.27,.27,.39,.40],[.61,.34,.72,.47]])
add('abc130k_ep_096887',[o('fruit',[[.42,.47,.48,.55],[.51,.50,.56,.58],[.54,.43,.59,.52]]),o('bowl',[[.61,.42,.74,.58]])],'gripper',[[.20,.47,.29,.58],[.78,.54,.88,.63]])
add('umi_hifi_microwave',[o('microwave',[[.53,.14,.75,.37]]),o('container',[[.55,.32,.66,.47]],241)],'gripper',[[.43,.57,.54,.72],[.56,.58,.67,.75]])
add('umi_hifi_tray',[o('tray',[[.34,.25,.58,.48]]),o('cloth',[[.59,.30,.67,.41]])],'gripper',[[.32,.40,.42,.59],[.65,.34,.74,.44]])
add('umi_hifi_pillow',[o('pillow',[[.37,.35,.67,.58]]),o('cloth',[[.35,.54,.59,.74]])],'gripper',[[.31,.59,.45,.82],[.57,.55,.72,.76]])
add('umi_hifi_folder',[o('folder',[[.33,.18,.59,.27]])],'gripper',[[.33,.27,.46,.50],[.53,.25,.65,.51]])
add('umi_fast_suitcase',[o('suitcase',[[.36,.67,.61,.84]])],'gripper',[[.22,.65,.45,.87],[.55,.65,.76,.87]])
add('umi_fast_rice',[o('rice_cooker',[[.27,.60,.45,.77]]),o('bowl',[[.50,.64,.59,.75]])],'gripper',[[.26,.64,.43,.86],[.56,.65,.76,.86]])
add('umi_fast_sandwich_003103',[o('lettuce',[[.49,.74,.58,.82]]),o('bread',[[.36,.71,.43,.79]])],'gripper',[[.26,.63,.46,.86],[.55,.65,.76,.86]])
add('egosuite_p04_ep_028028',[o('bottle',[[.38,.43,.47,.64],[.45,.44,.56,.68],[.55,.43,.63,.65]])],'hand',[[.35,.51,.56,.71],[.62,.36,.76,.59]])
add('egosuite_p04_ep_021026',[o('container',[[.47,.49,.61,.59]]),o('remote_control',[[.74,.47,.84,.59]])],'hand',[[.41,.50,.49,.67],[.57,.48,.67,.67]])
add('egosuite_p04_ep_036617',[o('package',[[.45,.56,.59,.69]])],'hand',[[.41,.56,.49,.70],[.56,.55,.64,.71]])
add('egosuite_p05_ep_033793',[o('cloth',[[.44,.47,.56,.60]]),o('tray',[[.40,.27,.59,.64]])],'hand',[[.37,.50,.47,.72],[.51,.55,.68,.80]])
add('egosuite_p05_ep_023134',[o('bottle',[[.29,.27,.46,.57]],129),o('refrigerator_door',[[.42,.00,.99,.85]])],'hand',[[.29,.39,.44,.61],[.53,.43,.69,.65]])
add('egosuite_p05_ep_029284',[o('cloth',[[.48,.55,.60,.68]]),o('picture_frame',[[.52,.15,.96,.52]],71)],'hand',[[.45,.59,.54,.72],[.55,.57,.68,.73]])
add('egosuite_p06_ep_004360',[o('cup',[[.40,.50,.48,.63],[.50,.53,.57,.69],[.30,.48,.38,.64],[.32,.65,.44,.79]])],'hand',[[.28,.55,.46,.71],[.53,.52,.65,.68]])
add('egosuite_p06_ep_022668',[o('cloth',[[.37,.52,.47,.62]]),o('cabinet',[[.38,.00,.89,.51]])],'hand',[[.40,.45,.52,.62],[.50,.43,.62,.62]])
add('egosuite_p07_ep_021574',[o('curtain',[[.05,.01,.51,.70],[.52,.01,.95,.70]])],'hand',[[.40,.25,.55,.51],[.51,.55,.62,.71]])
add('egosuite_p07_ep_033567',[o('paper',[[.45,.66,.61,.85]]),o('pen_holder',[[.51,.51,.59,.62]])],'hand',[[.42,.46,.51,.62],[.52,.41,.62,.59]])
if __name__=='__main__':
 assert len(C)==24
 import cv2
 for sid,config in C.items():
  cap=cv2.VideoCapture(str(R/'samples'/sid/'cam_0.mp4'));w,h=cap.get(3),cap.get(4);cap.release();assert w and h
  scale=min(320/w,240/h);rw,rh=w*scale,h*scale;ox,oy=(320-rw)/2,(240-rh)/2
  for guide in config['guides'].values():
   for b in guide['boxes']:
    b[:]=[max(0,min(1,(b[0]*320-ox)/rw)),max(0,min(1,(b[1]*240-oy)/rh)),max(0,min(1,(b[2]*320-ox)/rw)),max(0,min(1,(b[3]*240-oy)/rh))]
    assert b[0]<b[2] and b[1]<b[3]
  config['prompt_coordinate_provenance']='320x240 letterboxed contact-sheet rectangles inverted to full source coordinates'
 # Corrections retain exact inspected frame coordinates, independent of letterboxing.
 corrected=Path(__file__).with_name('batch30_prompt_corrections.json')
 if corrected.exists():
  for sid,guides in json.loads(corrected.read_text()).items():
   C[sid]['guides'].update(guides)
   C[sid]['prompt_coordinate_provenance']='original contact-sheet inverse plus source-frame corrections in batch30_prompt_corrections.json'
 (R/'sam_config.json').write_text(json.dumps(C,indent=2))
