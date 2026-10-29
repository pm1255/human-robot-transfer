"""Source-view robot composite with explicit observed cloth and estimated fills.

Preserves original source pixels outside edit masks. Cloth motion is taken from
the human video, never asserted to be robot-driven simulated cloth.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from human2robot.retarget_cloth.run_preview import Robot, table_frame, transform
from human2robot.hand_mesh.video import frames, probe, Writer
from human2robot.hand_mesh.render import rasterize
from human2robot.object_annotation.common import Progress, atomic_json
from human2robot.retarget_cloth.temporal_removal import TemporalRemoval


class Inputs:
    def __init__(self, objects, removal):
        self.objects=objects;self.removal=removal
        self.data=json.loads((objects/'objects.json').read_text());self.cache=OrderedDict()
        self.orb=cv2.ORB_create(nfeatures=1800,fastThreshold=10)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)

    def mask(self,i):
        s=self.data['frames'][i]['chunk_start'];key=f'cloth{s}'
        if key not in self.cache:
            p=np.load(self.objects/'masks'/f'{s:06d}.npz');self.cache[key]=p['masks']
        cloth=np.unpackbits(self.cache[key][i-s],count=640*480).reshape(480,640).astype(bool)
        p=self.removal/'human_masks'/f'{i:06d}.npz'
        if not p.exists():raise FileNotFoundError(f'Human removal mask not ready: {p}')
        human=np.unpackbits(np.load(p)['mask'],count=640*480).reshape(480,640).astype(bool)
        if len(self.cache)>24:self.cache.popitem(last=False)
        return cloth,human

    def rgb(self,i):
        s=self.data['frames'][i]['chunk_start']
        return cv2.imread(str(self.objects/'chunks'/f'{s:06d}'/f'{i-s:06d}.jpg'))

    def features(self,im,exclude):
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        mask=(~exclude).astype(np.uint8)*255
        return self.orb.detectAndCompute(gray,mask)

    def background(self,raw,cloth,human,i):
        # All observed pixels are restored after filling. Full union is removed
        # only in the clean-plate layer, not from the delivered visible cloth.
        union=cv2.dilate((cloth|human).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        pending=union.copy();clean=raw.copy();support=np.zeros((480,640),bool)
        k0,d0=self.features(raw,union)
        # A table homography is not valid for the floor below its front edge.
        # Infer that edge only from visible tabletop pixels; never copy warped
        # brown tabletop into pixels that belong to the floor.
        hsv=cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)
        table=(hsv[:,:,0]>=10)&(hsv[:,:,0]<=32)&(hsv[:,:,1]>65)&(hsv[:,:,2]>40)&~union
        xs=[];ys=[]
        for x in range(20,620,8):
            y=np.flatnonzero(table[220:450,x])+220
            if len(y)>25 and not human[y[-1]:min(480,y[-1]+14),x].any():xs.append(x);ys.append(y[-1])
        edge=np.full(640,400.)
        if len(xs)>12:
            xs=np.asarray(xs);ys=np.asarray(ys);fit=np.polyfit(xs,ys,1)
            for _ in range(3):
                good=np.abs(ys-np.polyval(fit,xs))<20
                if good.sum()<8:break
                fit=np.polyfit(xs[good],ys[good],1)
            edge=np.polyval(fit,np.arange(640))
        floor=np.arange(480)[:,None]>edge[None,:]+3
        donor_records=[]
        refined=getattr(self,'refined_background',False)
        accum=np.zeros_like(raw,dtype=np.float32);weights=np.zeros((480,640),np.float32)
        for shift in [-30,30,-60,60,-120,120,-240,240]:
            j=max(0,min(4833,round((i+shift)/30)*30))
            if j==i or not (self.removal/'human_masks'/f'{j:06d}.npz').exists():continue
            key=f'donor{j}'
            if key not in self.cache:
                rgb=self.rgb(j);c,h=self.mask(j);invalid=cv2.dilate((c|h).astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
                k,d=self.features(rgb,invalid);self.cache[key]=(rgb,invalid,k,d)
            rgb,invalid,k,d=self.cache[key]
            if d is None or d0 is None:continue
            matches=self.matcher.knnMatch(d,d0,k=2)
            good=[x[0] for x in matches if len(x)==2 and x[0].distance<.7*x[1].distance]
            if len(good)<12:continue
            src=np.float32([k[m.queryIdx].pt for m in good]);dst=np.float32([k0[m.trainIdx].pt for m in good])
            H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,2.5)
            if H is None or int(inliers.sum())<10:continue
            projected=cv2.perspectiveTransform(src[:,None,:],H)[:,0]
            residual=float(np.median(np.linalg.norm(projected-dst,axis=1)[inliers[:,0]>0]))
            if residual>1.8:continue
            warp=cv2.warpPerspective(rgb,H,(640,480))
            valid=cv2.warpPerspective((~invalid).astype(np.uint8),H,(640,480),flags=cv2.INTER_NEAREST)>0
            valid&=~floor
            # Reject badly aligned donors by observed-background photometry.
            observed=valid&~union
            photometric=float(np.median(np.abs(warp.astype(float)-raw)[observed])) if observed.any() else 999
            if photometric>22:continue
            if refined:
                # Correct each donor to the current exposure and blend full
                # overlapping supports. First-donor binary mosaics left sharp
                # polygons both inside and around the removed arm silhouette.
                correction=np.median(raw[observed].astype(float)-warp[observed],axis=0)
                warp=np.clip(warp.astype(float)+correction,0,255)
                feather=np.minimum(cv2.distanceTransform(valid.astype(np.uint8),cv2.DIST_L2,3)/18,1)
                weight=feather/(1+photometric/8)
                accum+=warp*weight[:,:,None];weights+=weight
            take=valid&pending
            clean[take]=warp[take];support[take]=True;pending[take]=False
            donor_records.append(dict(frame=j,inliers=int(inliers.sum()),median_registration_error_px=residual,median_color_error=photometric,filled_pixels=int(take.sum())))
            if not refined and not (pending&human).any():break
        # Unseen regions remain estimates. Use low-res Telea only where no
        # registered source background is available; never call it SVOR.
        if pending.any():
            # Scanline interpolation respects the front table edge better than
            # Telea diffusion across a large hand/forearm occlusion.
            fill=clean.copy();xx=np.arange(640)
            for y in range(480):
                known=~union[y]
                if known.sum()<2:continue
                for channel in range(3):fill[y,:,channel]=np.interp(xx,xx[known],raw[y,known,channel]).astype(np.uint8)
            clean[pending]=fill[pending]
        if refined:
            # Keep a smooth row-wise estimate as a weak prior where temporal
            # support ends, rather than a hard switch to another fill method.
            prior=raw.copy();xx=np.arange(640)
            for y in range(480):
                known=~union[y]
                if known.sum()<2:continue
                for channel in range(3):prior[y,:,channel]=np.interp(xx,xx[known],raw[y,known,channel])
            estimated=(accum+prior.astype(float)*.22)/(weights[:,:,None]+.22)
            estimated=cv2.bilateralFilter(estimated.astype(np.float32),9,22,5)
            clean[union]=np.clip(estimated[union],0,255).astype(np.uint8)
        return clean,support,donor_records


def complete_cloth(raw,cloth,human):
    """Conservative amodal estimate; all texture donors must be cloth pixels.

    This is deliberately labelled estimated: source RGB cannot reveal cloth
    hidden by the demonstrator. Never diffuse table colours into this layer.
    """
    safe_cloth=cloth&~human
    if not safe_cloth.any():return raw,np.zeros_like(cloth)
    distance,nearest=distance_transform_edt(~safe_cloth,return_indices=True)
    contours,_=cv2.findContours(cloth.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull=np.zeros_like(cloth,np.uint8)
    if contours:
        major=max(contours,key=cv2.contourArea)
        cv2.fillConvexPoly(hull,cv2.convexHull(major),1)
    closed=cv2.morphologyEx(cloth.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(101,101))).astype(bool)
    holes=human&(closed|hull.astype(bool))&(distance<50)
    # Build a cloth-only texture canvas before filling the occluded region.
    # The old inpaint(raw,human) mixed brown table and skin into white cloth.
    texture=raw[nearest[0],nearest[1]].copy()
    texture[cloth&~human]=raw[cloth&~human]
    fill=cv2.inpaint(texture,holes.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
    return fill,holes


def render_robot(robot,row,frame,K):
    R,center=frame
    rgb=np.zeros((480,640,3),np.float32);zbuf=np.full((480,640),np.inf);owner=np.full((480,640),-1,np.int32)
    for label,(side,h) in enumerate(row['hands'].items()):
        base=np.asarray(row['robot_bases'][side]) if 'robot_bases' in row else transform(Rotation.from_euler('z',np.pi/2).as_matrix(),[-.38 if side=='Left' else .38,-.42,.015])
        links=robot.links(np.asarray(h['q']),h['gripper_joint_m'])
        for link,tri,norm in robot.meshes:
            T=base@links[link];world=tri@T[:3,:3].T+T[:3,3]
            camera=world@R.T+center
            normals=(norm@T[:3,:3].T)@R.T
            good=(camera[:,:,2]>.04).all(1);camera=camera[good];normals=normals[good]
            light=np.array([-.25,-.5,-1]);light/=np.linalg.norm(light)
            # Same neutral material on both arms; color never encodes identity.
            color=np.array([162,162,162])
            if link.endswith(('1','4','6')):color=np.array([85,85,85])
            if link.endswith(('7','8')):color=np.array([185,185,185])
            shade=.50+.50*np.abs(normals@light)
            colors=shade[:,:,None]*color
            points=camera.reshape(-1,3);projected=points@K.T;uv=projected[:,:2]/projected[:,2:]
            faces=np.arange(len(points),dtype=np.int64).reshape(-1,3)
            rasterize(uv,points,colors.reshape(-1,3),faces,rgb,zbuf,owner,label)
    return np.clip(rgb,0,255).astype(np.uint8),zbuf,owner


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--objects',type=Path,required=True)
    p.add_argument('--removal',type=Path,required=True);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--indices',default='');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=4834);p.add_argument('--blend-removal',action='store_true');p.add_argument('--prefix',default='');p.add_argument('--temporal-removal',action='store_true');p.add_argument('--warmup',type=int,default=0);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    inputs=Inputs(a.objects,a.removal);data=json.loads(a.trajectory.read_text());robot=Robot(a.assets,a.root,face_budget=0)
    inputs.refined_background=a.blend_removal
    temporal=TemporalRemoval() if a.temporal_removal else None
    cv2.setNumThreads(1)
    indices=[0,30,90,300,600,696,781,1200,2400,3600,3948,4800,4833] if a.pilot else list(range(a.start,a.stop))
    if a.indices:indices=[int(x) for x in a.indices.split(',')]
    prefix=a.prefix or ('robot_repair_composite' if 'continuous' in data['meta'].get('method','') else 'source_composite')
    progress=Progress(a.objects,f'{prefix}_pilot' if a.pilot else f'{prefix}_{a.start:06d}_{a.stop:06d}',len(indices));records=[]
    writer=None if a.pilot else Writer(a.root/'robot_source_composite.mp4',640,480,30)
    meta=probe(inputs.data['meta']['video'])
    read_start=max(0,a.start-a.warmup) if temporal else a.start
    iterator=((i,inputs.rgb(i)) for i in indices) if a.pilot else enumerate(frames(inputs.data['meta']['video'],meta,start=read_start,limit=a.stop-read_start),start=read_start)
    done=0
    try:
        for i,raw in iterator:
            row=data['frames'][i];cloth,human=inputs.mask(i)
            human=cv2.dilate(human.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
            clean,supported,donors=inputs.background(raw,cloth,human,i)
            # Preserve actual visible garment pixels. The inferred completion
            # covers only holes inside its local morphological envelope.
            filled,holes=complete_cloth(raw,cloth,human)
            result=raw.copy();result[human]=clean[human]
            if holes.any():
                result[holes]=filled[holes]
            if a.blend_removal and human.any():
                # Match boundary illumination through a Poisson solve. The
                # previous binary paste retained a sharply hand-shaped seam.
                # Padding keeps masks touching the image edge inside the solve.
                pad=32;src=cv2.copyMakeBorder(result,pad,pad,pad,pad,cv2.BORDER_REFLECT)
                dst=cv2.copyMakeBorder(raw,pad,pad,pad,pad,cv2.BORDER_REFLECT)
                cm=np.pad(human.astype(np.uint8)*255,pad)
                x,y,w,h=cv2.boundingRect(cm)
                blended=cv2.seamlessClone(src,dst,cm,(x+w//2,y+h//2),cv2.NORMAL_CLONE)[pad:-pad,pad:-pad]
                result[human]=blended[human]
            removal_edit=human;temporal_record={}
            if temporal:
                result,removal_edit,temporal_record,_=temporal.apply(raw,clean,result,cloth,human,holes,i)
                if i<a.start:continue
            depth=np.load(a.objects/'depth'/f'{i:06d}.npz');D=depth['depth'];K=depth['K'];frame=table_frame(raw,D,K)
            if 'camera_K' in row:
                K=np.asarray(row['camera_K']);f=row['support_frame_camera'];frame=(np.asarray(f['R']),np.asarray(f['origin']))
            robot_mask=np.zeros((480,640),bool)
            arm_pixels={}
            if row['hands'] and frame is not None:
                rendered,z,owner=render_robot(robot,row,frame,K)
                robot_mask=np.isfinite(z)
                # Source cloth is composited using its visible depth; hidden
                # cloth completion has no reliable depth and is marked review.
                # A stabilized camera must not use fluctuating raw metric depth
                # for hard visibility decisions: that caused whole-arm pops.
                if 'camera_K' in row:
                    yy,xx=np.mgrid[:480,:640];ray=np.stack([xx,yy,np.ones_like(xx)],axis=-1)@np.linalg.inv(K).T
                    normal=frame[0][:,2];plane=float(normal@frame[1]);stable_depth=(plane+.015)/np.einsum('ijk,k->ij',ray,normal)
                    occluded=cloth&~human&(stable_depth<z-.025)
                else:occluded=cloth&~human&np.isfinite(D)&(D<z-.008)
                robot_mask&=~occluded
                result[robot_mask]=rendered[robot_mask]
                arm_pixels={side:int(((owner==j)&robot_mask).sum()) for j,side in enumerate(row['hands'])}
            edit=removal_edit|robot_mask
            untouched=bool(np.array_equal(result[~edit],raw[~edit]))
            rec=dict(frame_index=i,human_mask_pixels=int(human.sum()),cloth_mask_pixels=int(cloth.sum()),robot_pixels=int(robot_mask.sum()),arm_pixels=arm_pixels,background_donor_pixels=int((supported&human).sum()),background_inpaint_pixels=int((human&~supported&~holes).sum()),cloth_completion_pixels=int(holes.sum()),removal_boundary_blended=a.blend_removal,original_pixels_outside_edit_unchanged=untouched,donors=donors,robot_status=row['status'],physics_executed=False,training_eligible=False)
            records.append(rec)
            rec['temporal_removal']=temporal_record
            if not untouched:raise AssertionError('Pixels outside edit mask changed')
            if writer:writer.write(result)
            if a.pilot or i%120==0 or i==4833:
                mask=raw.copy();mask[human]=(mask[human]*.4+[25,25,150]).clip(0,255).astype(np.uint8)
                cv2.imwrite(str(a.root/f'composite_{i:06d}.jpg'),result)
                cv2.imwrite(str(a.root/f'clean_{i:06d}.jpg'),clean)
                compare=np.concatenate([raw,mask,clean,result],axis=1)
                cv2.imwrite(str(a.root/f'audit_{i:06d}.jpg'),compare)
                progress.writer.add_image(f'composite_{i:06d}',result[:,:,::-1].copy(),i,dataformats='HWC')
            done+=1
            if done%60==0 or a.pilot:progress.update(done,phase='source_view_compositing')
        if writer:writer.close()
        atomic_json(a.root/'composite_validation.json',dict(frames=records,summary=dict(total=len(records),unchanged_outside_edit=all(r['original_pixels_outside_edit_unchanged'] for r in records),frames_without_robot=sum(r['robot_pixels']==0 for r in records),method='SAM3.1 anchors + bidirectional optical flow; registered source background + scanline unseen fill; observed cloth + cloth-only estimated completion; original-view URDF render',cloth_physically_simulated=False,hidden_content_verified=False,training_eligible=False)))
        progress.update(done,'completed')
    except Exception as exc:progress.update(done,'failed',error=str(exc));raise
    finally:progress.close()


if __name__=='__main__':main()
