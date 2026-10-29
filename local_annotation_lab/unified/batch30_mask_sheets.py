from pathlib import Path
from PIL import Image,ImageDraw
R=Path('/Users/panmiao/Documents/ChatGPT/paper/outputs/unified_30_20261021')
out=R/'mask_review';out.mkdir(exist_ok=True)
for p in sorted((R/'samples').glob('*/multimodal/classes')):
 labels=sorted(p.iterdir());im=Image.new('RGB',(960,265*len(labels)),'#12232d');d=ImageDraw.Draw(im)
 for j,cls in enumerate(labels):
  for k,f in enumerate(sorted(cls.glob('preview_*.jpg'))):
   a=Image.open(f);a.thumbnail((320,240));im.paste(a,(k*320,j*265));d.text((k*320+3,j*265+242),cls.name+' '+f.stem,fill='white')
 im.save(out/(p.parent.parent.name+'.jpg'))
