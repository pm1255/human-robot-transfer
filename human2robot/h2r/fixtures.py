"""Procedural stress fixtures with independent saved generating trajectories.

These are explicitly synthetic tests, never evidence of human-video accuracy.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import transform, invert, apply
from .schema import Episode

CASES=[
 ("clean_pinch","干净捏取","连续的夹爪参考轨迹，检验坐标与 IK。"),
 ("camera_motion","相机移动","相机移动造成的表观位移，应在统一世界坐标后抵消。"),
 ("jitter","轻微抖动","在有观测的位置上做稳健平滑，并与合成真值比较。"),
 ("spike","单帧跳点","一帧误差注入 10 厘米，检查修复有没有偏离邻帧。"),
 ("short_gap","短暂遮挡","同一身份、同一阶段内的短缺失可插值，保留修复标志。"),
 ("long_gap","长时间遮挡","长缺失保留为空，禁止生成有效动作窗口。"),
 ("event_boundary","接触边界缺失","即使缺失很短，也禁止跨动作阶段插值。"),
 ("identity_switch","身份切换","身份边界两侧不能平滑连接或组成训练窗口。"),
 ("unreachable","机器人够不到","人体标注可用，但目标超出机器人可达范围。"),
 ("unknown_scale","单目尺度未知","保留人体监督，拒绝米制机器人动作导出。"),
 ("wide_grip","开口过大","人手的捏合间距超出示例夹爪开口，不静默截断。"),
 ("rotating_grasp","抓取方向变化","位置近似不变而朝向变化，检验旋转跟踪。"),
]


def hand_landmarks(T,width):
    local=np.zeros((21,3));local[0]=[0,0,-0.10]
    # Approximate anatomical landmarks for an analytic pinch fixture.
    for finger in range(5):
        for k in range(4):local[1+finger*4+k]=[(finger-2)*.016,0,-.06+.015*k]
    local[4]=[-width/2,0,0];local[8]=[width/2,0,0]
    local[5]=[.02,0,-.04];local[17]=[-.02,0,-.04]
    return apply(T,local)


def make_case(case,robot,seed=17,n=61):
    rng=np.random.default_rng(seed);t=np.arange(n)/10
    qs=np.tile([0,-.7,1.1,0,-.4,0.],(n,1)).astype(float)
    qs[:,0]=.15*np.sin(t*.6);qs[:,1]+=.06*np.sin(t*.8);qs[:,2]+=.06*np.sin(t*.7)
    if case=="rotating_grasp":qs[:,5]=.35*np.sin(t*.8)
    truth=np.stack([robot.fk(q) for q in qs]);camera=np.repeat(np.eye(4)[None],n,axis=0)
    if case=="camera_motion":
        for i in range(n):camera[i]=transform(Rotation.from_euler('y',.15*np.sin(t[i])).as_matrix(),[.12*np.sin(t[i]),0,0])
    width=.11 if case=="wide_grip" else .04
    points=np.stack([apply(invert(camera[i]),hand_landmarks(truth[i],width)) for i in range(n)])
    observed=np.ones(n,bool);confidence=np.full(n,.95);track=np.zeros(n,int);phase=np.full(n,"holding",dtype="U16")
    if case in {"jitter","spike"}:
        points+=rng.normal(0,.007,(n,1,3))
    if case=="spike":points[n//2]+=[0,.1,0]
    if case in {"short_gap","long_gap","event_boundary"}:
        a=n//2;b=a+(1 if case!="long_gap" else 10);points[a:b]=np.nan;observed[a:b]=False;confidence[a:b]=0
    if case=="event_boundary":phase[n//2:]= "release"
    if case=="identity_switch":track[n//2:]=1;points[n//2:,:,1]+=.10
    if case=="unreachable":points[:,:,0]+=1.0
    metadata={"units":"m","scale_status":"unknown" if case=="unknown_scale" else "synthetic_metric",
              "source_kind":"synthetic","source_uri":f"procedural://{case}","split_group":f"synthetic-{seed}",
              "camera_status":"synthetic_truth","mirror_status":"not_mirrored","grasp_status":"synthetic_fixture",
              "license_status":"self_generated","seed":seed,"task":"synthetic pinch reference",
              "task_source":"procedural_fixture"}
    return Episode(case,t,points,observed,confidence,camera,track,phase,metadata),truth
