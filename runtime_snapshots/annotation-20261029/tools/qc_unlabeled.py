#!/usr/bin/env python3
import argparse, json, statistics, subprocess, hashlib
from pathlib import Path
import cv2, numpy as np

def phash(gray):
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA); m = float(small.mean())
    return ''.join('1' if float(x) >= m else '0' for x in small.reshape(-1))

def probe(path):
    cmd = ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name','-of','json',path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
    st = json.loads(p.stdout).get('streams',[{}])[0]
    def frac(x):
        try:
            a,b = str(x).split('/'); return float(a)/float(b) if float(b) else 0.0
        except Exception: return float(x) if x not in (None,'N/A') else 0.0
    fps = frac(st.get('avg_frame_rate') or st.get('r_frame_rate'))
    def num(x, cast=float):
        try: return cast(x)
        except Exception: return 0 if cast is int else 0.0
    return {'width':num(st.get('width'),int),'height':num(st.get('height'),int),'fps':fps,'expected_frames':num(st.get('nb_frames'),int),'duration_sec':num(st.get('duration')),'codec':st.get('codec_name')}

def ffprobe_audio(path):
    try:
        p = subprocess.run(['ffprobe','-v','error','-select_streams','a','-show_entries','stream=index','-of','csv=p=0',path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return bool(p.stdout.strip())
    except Exception: return None

def validate_segments(path, duration):
    if not path: return {'provided': False}
    data = json.loads(Path(path).read_text()); segs = data.get('segments', []); errors=[]; warnings=[]; last_end=-1.0
    for i,s in enumerate(segs):
        start=float(s.get('start_sec',-1)); end=float(s.get('end_sec',-1))
        if not (0 <= start < end <= duration + 0.05): errors.append({'index':i,'reason':'bounds','start_sec':start,'end_sec':end})
        if start < last_end - 1e-3: errors.append({'index':i,'reason':'overlap'})
        last_end=max(last_end,end)
        if not s.get('clip_id'): errors.append({'index':i,'reason':'missing_clip_id'})
        if s.get('language',{}).get('status') != 'verified': warnings.append({'index':i,'reason':'language_unverified'})
        if s.get('review_state') != 'approved': warnings.append({'index':i,'reason':'segment_not_approved'})
    return {'provided':True,'segments':len(segs),'errors':errors,'warnings':warnings,'path':str(path)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--video',required=True); ap.add_argument('--out',required=True); ap.add_argument('--segment-manifest'); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); meta=probe(args.video); w=meta['width']; h=meta['height']; expected=meta['expected_frames']; fps=meta['fps'];
    if expected <= 0 and fps > 0 and meta['duration_sec'] > 0: expected=int(round(fps*meta['duration_sec']))
    frame_size=w*h; decoded=0; hashes=[]; means=[]; sharp=[]; dark=[]; clipped=[]; near_dup=0; exact_dup=0; prev_small=None; ffmpeg_error=''
    cmd=['ffmpeg','-v','error','-i',args.video,'-map','0:v:0','-f','rawvideo','-pix_fmt','gray','-vsync','0','-']
    try:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE); 
        while frame_size > 0:
            buf=p.stdout.read(frame_size)
            if not buf or len(buf) < frame_size: break
            gray=np.frombuffer(buf,dtype=np.uint8).reshape((h,w)); decoded += 1; small=cv2.resize(gray,(64,36),interpolation=cv2.INTER_AREA)
            if prev_small is not None and float(cv2.absdiff(small,prev_small).mean()) < 0.5: near_dup += 1
            prev_small=small; hh=phash(gray)
            if hashes and hh==hashes[-1]: exact_dup += 1
            hashes.append(hh); means.append(float(gray.mean())); sharp.append(float(cv2.Laplacian(gray,cv2.CV_64F).var())); dark.append(float((gray<=5).mean())); clipped.append(float((gray>=250).mean()))
        err=p.stderr.read().decode('utf-8','replace'); rc=p.wait(); ffmpeg_error=err.strip() if rc else ''
    except Exception as e: ffmpeg_error=repr(e)
    duration=decoded/fps if fps>0 else meta['duration_sec']; decode_ratio=decoded/expected if expected>0 else 0.0; med_sharp=statistics.median(sharp) if sharp else 0.0
    meta['decoded_frames']=decoded; meta['duration_decoded_sec']=duration
    report={'video':str(Path(args.video).resolve()),'sha256':hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),'metadata':meta,'decode':{'success_ratio':decode_ratio,'decode_failures':max(0,expected-decoded),'ffmpeg_error':ffmpeg_error},'duplicates':{'consecutive_exact_hash_ratio':exact_dup/max(1,decoded-1),'consecutive_near_duplicate_ratio':near_dup/max(1,decoded-1)},'appearance':{'mean_brightness':statistics.mean(means) if means else None,'median_laplacian_variance':med_sharp,'black_frame_ratio':sum(x>0.90 for x in dark)/max(1,decoded),'clipped_frame_ratio':sum(x>0.90 for x in clipped)/max(1,decoded)},'audio_present':ffprobe_audio(args.video),'audio_policy':'not_required_for_rgb_only_smoke_test'}
    checks={'decode_ok':decode_ratio>=0.995,'metadata_ok':w>0 and h>0 and fps>=1.0,'duration_ok':1.0<=duration<=60.0,'black_frame_ok':report['appearance']['black_frame_ratio']<=0.02,'duplicate_ok':report['duplicates']['consecutive_near_duplicate_ratio']<=0.10,'sharpness_ok':med_sharp>=10.0}; report['checks']=checks; report['segments']=validate_segments(args.segment_manifest,duration); hard_fail=not(checks['decode_ok'] and checks['metadata_ok']) or bool(report['segments'].get('errors')); report['status']='FAIL' if hard_fail else ('WARN' if not all(checks.values()) or bool(report['segments'].get('warnings')) else 'PASS')
    (out/'qc_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); md=['# Unlabeled video QC','','- Status: **'+report['status']+'**','- Video: '+report['video'],'- SHA256: '+report['sha256'],'','## Metadata','']; md += ['- '+str(k)+': '+str(v) for k,v in meta.items()]; md += ['', '## Checks','']; md += ['- '+str(k)+': '+('PASS' if v else 'WARN/FAIL') for k,v in checks.items()]; md += ['', '## Blind-input policy','', '- Only the RGB MP4 was decoded.', '- Official annotations and overlay videos were not read.', '- Audio present: '+str(report['audio_present'])+' ('+report['audio_policy']+').'];
    if args.segment_manifest: md += ['', '## Segmentation and language review','', '- Segment manifest: '+str(args.segment_manifest), '- Segments: '+str(report['segments'].get('segments',0)), '- Language warnings: '+str(len(report['segments'].get('warnings',[]))), '- Language labels remain unverified until an independent reviewer checks the clip.']
    (out/'qc_report.md').write_text('\n'.join(md)+'\n'); print(json.dumps({'status':report['status'],'decoded':decoded,'expected':expected,'duration_sec':duration,'out':str(out)},ensure_ascii=False))

if __name__=='__main__': main()
