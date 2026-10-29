"""URDF serial-chain FK and bounded numerical IK; no dynamics claim."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import transform, rotation_error, check_transform


def vector(text, default):
    return np.fromstring(text, sep=" ") if text else np.asarray(default,float)


@dataclass
class Joint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


class SerialRobot:
    def __init__(self, urdf, base="base", tip="tcp"):
        self.urdf=Path(urdf); tree=ET.parse(self.urdf).getroot()
        children={j.find("child").get("link"):j for j in tree.findall("joint")}
        chain=[];node=tip;seen=set()
        while node!=base:
            if node in seen or node not in children:raise ValueError("URDF chain is missing/cyclic")
            seen.add(node);j=children[node]
            if j.find("mimic") is not None:raise ValueError("Mimic joints need an explicit coupling adapter")
            kind=j.get("type")
            if kind not in {"fixed","revolute","prismatic","continuous"}:raise ValueError(f"Unsupported joint {kind}")
            origin=j.find("origin");xyz=vector(origin.get("xyz") if origin is not None else None,[0,0,0]);rpy=vector(origin.get("rpy") if origin is not None else None,[0,0,0])
            axis=j.find("axis");ax=vector(axis.get("xyz") if axis is not None else None,[1,0,0])
            if ax.shape!=(3,) or not np.isfinite(ax).all() or np.linalg.norm(ax)<1e-9:raise ValueError('Invalid URDF joint axis')
            ax/=np.linalg.norm(ax)
            limit=j.find("limit")
            lower=float(limit.get("lower","-3.14159265")) if limit is not None else -np.pi
            upper=float(limit.get("upper","3.14159265")) if limit is not None else np.pi
            velocity=float(limit.get("velocity","2")) if limit is not None else 2
            if kind!='fixed' and (lower>=upper or velocity<=0 or not np.isfinite([lower,upper,velocity]).all()):
                raise ValueError('Invalid URDF limits')
            chain.append(Joint(j.get("name"),kind,transform(Rotation.from_euler("xyz",rpy).as_matrix(),xyz),ax,lower,upper,velocity))
            node=j.find("parent").get("link")
        self.chain=list(reversed(chain));active=[j for j in self.chain if j.kind!="fixed"]
        self.names=[j.name for j in active];self.lower=np.array([j.lower for j in active]);self.upper=np.array([j.upper for j in active]);self.velocity=np.array([j.velocity for j in active])
        self.n=len(active);self.name=tree.get("name");self.base=base;self.tip=tip

    def fk(self,q,return_points=False):
        q=np.asarray(q,float)
        if q.shape!=(self.n,) or not np.isfinite(q).all():raise ValueError("Invalid joint vector")
        T=np.eye(4);points=[T[:3,3].copy()];i=0
        for j in self.chain:
            T=T@j.origin
            if j.kind!="fixed":
                if j.kind=="prismatic":T=T@transform(translation=j.axis*q[i])
                else:T=T@transform(Rotation.from_rotvec(j.axis*q[i]).as_matrix())
                i+=1
            points.append(T[:3,3].copy())
        return (T,np.asarray(points)) if return_points else T

    def collision_penalties(self,q,obstacles=(),ground_z=None,radius=0.015):
        """Conservative link-capsule vs sphere/plane proxy, not full mesh collision."""
        _,points=self.fk(q,True);penalties=[]
        for a,b in zip(points[1:-1],points[2:]):
            if ground_z is not None:penalties.append(max(0,ground_z+radius-min(a[2],b[2])))
            for obs in obstacles:
                c=np.asarray(obs["center"]);v=b-a
                s=np.clip(np.dot(c-a,v)/max(np.dot(v,v),1e-12),0,1)
                penalties.append(max(0,float(obs["radius"])+radius-np.linalg.norm(a+s*v-c)))
        return np.asarray(penalties)

    def ik(self,target,seed=None,previous=None,dt=None,obstacles=(),ground_z=None,
           position_tolerance=0.012,rotation_tolerance=0.15,max_nfev=100):
        target=check_transform(target)
        seed=np.zeros(self.n) if seed is None else np.asarray(seed,float)
        lo=self.lower.copy();hi=self.upper.copy()
        if previous is not None:
            if dt is None or dt<=0:raise ValueError("Velocity constraints need positive dt")
            lo=np.maximum(lo,previous-self.velocity*dt);hi=np.minimum(hi,previous+self.velocity*dt)
        if np.any(lo>=hi):raise ValueError("Empty joint feasible interval")
        seed=np.clip(seed,lo+1e-9,hi-1e-9)
        def residual(q):
            T=self.fk(q)
            continuity=(q-seed)*0.01
            c=self.collision_penalties(q,obstacles,ground_z)
            return np.r_[(T[:3,3]-target[:3,3])/0.01,
                         rotation_error(target[:3,:3],T[:3,:3])/0.1,continuity,c/0.002]
        result=least_squares(residual,seed,bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        actual,points=self.fk(result.x,True)
        position_error=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
        angle_error=float(np.linalg.norm(rotation_error(target[:3,:3],actual[:3,:3])))
        collisions=self.collision_penalties(result.x,obstacles,ground_z)
        collision=float(collisions.max(initial=0))
        ok=position_error<=position_tolerance and angle_error<=rotation_tolerance and collision<1e-4
        return {"q":result.x,"actual":actual,"link_points":points,"position_error_m":position_error,
                "rotation_error_rad":angle_error,"collision_proxy_m":collision,"valid":bool(ok),
                "solver_status":int(result.status),"evaluations":int(result.nfev)}
