"""Unilateral compliant four-tug plant using original KVLCC2 MMG hydrodynamics.

This replaces the old bilateral constrained DAE for the v4 experiments only.
The ship's 3-DOF hydrodynamic equations are those in model.ship.Ship.
Each tug is a planar rigid body with body-fixed surge/sway/yaw velocities,
quadratic drag, rotational damping and an azimuthing stern thruster.
Push fenders compress but do not pull. Towlines stretch but do not push.
Equal/opposite forces and their moments act on ship and tug.
This is a research simulation, not a validated contact mechanics solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from model.ship import Ship
from model.tug import Tug

def rot(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s],[s,c]])
def wrap(a):
    return (a+np.pi)%(2*np.pi)-np.pi
def cross2(a,b):
    return float(a[0]*b[1]-a[1]*b[0])

@dataclass
class ContactConfig:
    rope_length: float=40.
    rope_stiffness: float=140_000.
    rope_damping: float=500_000.
    push_stiffness: float=850_000.
    push_damping: float=850_000.
    # Dashpot engages continuously across the first compression/stretch band.
    # These regularization distances are assumed, not measured parameters.
    push_damping_transition_m: float=.15
    rope_damping_transition_m: float=.20
    push_friction_stiffness: float=70_000.
    push_friction_damping: float=160_000.
    friction_coeff: float=.22
    bumper_m: float=4.5
    max_contact_force: float=2.8e6
    max_tow_force: float=2.0e6
    tug_drag_long: float=270_000.
    tug_drag_lat: float=510_000.
    tug_yaw_damping: float=26_000_000.
    twin_azimuth_max_yaw_moment_Nm: float=4_000_000.
    push_twin_azimuth_max_yaw_moment_Nm: float=5_000_000.
    quay_stiffness: float=1.5e6
    quay_damping: float=5e6

class UnilateralPlant:
    def __init__(self, ship:Ship, tug:Tug, hull, side:int, contact:ContactConfig|None=None,
                 wind=0.,wind_direction_deg=0.,current=0.,current_direction_deg=0.):
        self.ship,self.tug=ship,tug
        self.hull=hull
        self.side=int(side)
        self.c=contact or ContactConfig()
        self.env=(float(wind),np.deg2rad(wind_direction_deg),float(current),np.deg2rad(current_direction_deg))
        self.ap=float(tug.api)
        self.at=float(tug.ati)
        self.max_force=float(tug.command_force_limit_N)
        self.Mt=np.array([tug.mi+tug.mxi,tug.mi+tug.myi,tug.IzGi+tug.Jzi])
        self._ship_points=np.asarray(hull.points,dtype=float)
        self.attach_x=np.array([-.30*ship.Lpp,.30*ship.Lpp,-.50*ship.Lpp,.50*ship.Lpp])
        self.attach_local=np.array([ [x,self._side_y(x)] for x in self.attach_x])
        self.last_force=np.zeros(4)

    def _side_y(self,x):
        val=np.asarray(self.hull.get_y(float(x)),float)
        if not len(val):raise ValueError('No hull section at '+str(x))
        return float(val[:,1].max() if self.side==1 else val[:,1].min())

    def initial(self,x0=0.,y0=0.,psi0_deg=40.):
        ship=np.array([x0,y0,np.deg2rad(psi0_deg),0.,0.,0.],float)
        tugs=np.zeros((4,6),float)
        R=rot(ship[2]); n=R@np.array([0.,self.side])
        for i,a in enumerate(self.attach_local):
            # BOTH modes keep their bows facing the ship. A towing tug
            # pulls astern (signed NEGATIVE thrust), not by turning around.
            heading=ship[2]-self.side*np.pi/2
            distance=(self.ap+self.c.bumper_m if i<2 else self.c.rope_length+self.at)
            tugs[i,:2]=ship[:2]+R@a+n*distance
            tugs[i,2]=wrap(heading)
        return ship,tugs

    def attachment(self,ship):
        return ship[:2]+self.attach_local@rot(ship[2]).T

    @staticmethod
    def velocity_at(pose, r_local):
        v=rot(pose[2])@pose[3:5]
        arm=rot(pose[2])@np.asarray(r_local)
        return v+pose[5]*np.array([-arm[1],arm[0]])

    def forces(self,ship,tugs,thrust,azimuth,quay_y=None,yaw_moment=None):
        """Returns generalized ship/tug forces, diagnostics. No state mutation."""
        fship=np.zeros(3);ftug=np.zeros((4,3));measure={}
        if yaw_moment is None:yaw_moment=np.zeros(4)
        R=rot(ship[2]);n=R@np.array([0.,self.side]);t=R@np.array([1.,0.])
        ship_pts=self.attachment(ship)
        for i in range(4):
            tug=tugs[i];Rt=rot(tug[2]);bow_or_stern=(np.array([self.ap,0.]) if i<2 else np.array([self.at,0.]))
            q=tug[:2]+Rt@bow_or_stern; a=ship_pts[i]
            va=self.velocity_at(ship,self.attach_local[i])
            vq=self.velocity_at(tug,bow_or_stern)
            if i<2:
                # Bumper is ahead of the tug bow; compression starts at gap < bumper.
                signed_gap=float((q-a)@n)
                penetration=self.c.bumper_m-signed_gap
                closing=-float((vq-va)@n)
                # Regularize viscous force at first contact: a raw conditional
                # dashpot adds a finite force jump exactly at zero penetration.
                damp_gate=np.clip(penetration/max(1e-8,self.c.push_damping_transition_m),0.,1.)
                mag=np.clip(self.c.push_stiffness*max(0.,penetration)
                   +self.c.push_damping*max(0.,closing)*damp_gate if penetration>0 else 0.,
                   0,self.c.max_contact_force)
                # Normal force on ship points INTO the hull. Equal/opposite on tug.
                F_ship=-mag*n
                tang_disp=float((q-a)@t)
                slip=float((vq-va)@t)
                friction=np.clip(self.c.push_friction_stiffness*tang_disp
                       +self.c.push_friction_damping*slip,-self.c.friction_coeff*mag,
                         self.c.friction_coeff*mag) if mag>0 else 0.
                F_ship+=friction*t
                measure[f'push{i+1}_normal_N']=float(mag)
                measure[f'push{i+1}_tangent_N']=float(friction)
                measure[f'push{i+1}_gap_m']=signed_gap
                measure[f'tow{i+1}_tension_N']=0.
            else:
                delta=q-a;dist=float(np.linalg.norm(delta));line=delta/max(dist,1e-12)
                stretch=dist-self.c.rope_length
                extension_rate=float((vq-va)@line)
                # An extended line only resists extension: no negative tension.
                damp_gate=np.clip(stretch/max(1e-8,self.c.rope_damping_transition_m),0.,1.)
                mag=np.clip(self.c.rope_stiffness*max(0.,stretch)+
                   self.c.rope_damping*max(0.,extension_rate)*damp_gate if stretch>0 else 0.,
                    0,self.c.max_tow_force)
                F_ship=mag*line
                measure[f'tow{i+1}_tension_N']=float(mag)
                measure[f'tow{i+1}_length_m']=dist
                measure[f'push{i+1}_normal_N']=0.
                measure[f'push{i+1}_tangent_N']=0.
            F_tug=-F_ship
            ship_arm=a-ship[:2];tug_arm=q-tug[:2]
            fship[:2]+=F_ship
            fship[2]+=cross2(ship_arm,F_ship)
            ftug[i,:2]+=F_tug
            ftug[i,2]+=cross2(tug_arm,F_tug)
            # The propulsion force is separate from towline/contact reaction.
            # Signed thrust: T1,T2 >= 0 (ahead/push), T3,T4 <= 0
            # (astern/tow). The azimuth is relative to tug BOW heading.
            # Negative thrust points opposite to the tug bow, not a 180deg
            # change of the tug's heading. xTi is stern thruster coordinate.
            Fprop=Rt@ (float(thrust[i])*np.array([np.cos(azimuth[i]),np.sin(azimuth[i])]))
            ftug[i,:2]+=Fprop
            ftug[i,2]+=cross2(Rt@np.array([self.tug.xTi,0.]),Fprop)
            # Differential twin-azimuth yaw moment: finite physical actuator,
            # not a direct overwrite of the tug angular state.
            allowed=(self.c.push_twin_azimuth_max_yaw_moment_Nm if i<2
                     else self.c.twin_azimuth_max_yaw_moment_Nm)
            ftug[i,2]+=float(np.clip(yaw_moment[i],-allowed,allowed))
        if quay_y is not None:
            # Face along earth X, quay is BELOW the target vessel. Fender contact
            # based on exact transformed ship outline, not the vessel centre.
            world=self._ship_points@R.T+ship[:2]
            low=world[np.argmin(world[:,1])]
            pen=max(0.,float(quay_y-low[1]))
            if pen>0:
                vy=float(self.velocity_at(ship,R.T@(low-ship[:2]))[1])
                fn=max(0.,self.c.quay_stiffness*pen+self.c.quay_damping*max(0.,-vy))
                fn=min(fn,15e6)
                fship[1]+=fn
                fship[2]+=cross2(low-ship[:2],np.array([0.,fn]))
                measure['fender_force_N']=fn
            else:measure['fender_force_N']=0.
            measure['fender_gap_m']=float(low[1]-quay_y)
        return fship,ftug,measure

    def derivative(self,ship,tugs,thrust,azimuth,quay_y=None,yaw_moment=None):
        fs,ft,diags=self.forces(ship,tugs,thrust,azimuth,quay_y,yaw_moment)
        # Physical force -> body generalized force in original KVLCC2 MMG model.
        tau_body=rot(ship[2]).T@fs[:2]
        self.ship.external_tau=np.array([*tau_body,fs[2]])
        A,C=self.ship.ship_matrices(ship[3],ship[4],ship[5],ship[2],*self.env)
        deriv=np.linalg.solve(A,C.flatten())
        ds=np.array([deriv[3],deriv[4],ship[5],*deriv[:3]])
        dt=np.zeros_like(tugs)
        for i,tug in enumerate(tugs):
            R=rot(tug[2]);vx,vy=tug[3:5];r=tug[5]
            F=R.T@ft[i,:2]
            Fx=F[0]-self.c.tug_drag_long*vx*abs(vx)+(self.tug.mi+self.tug.myi)*vy*r
            Fy=F[1]-self.c.tug_drag_lat*vy*abs(vy)-(self.tug.mi+self.tug.mxi)*vx*r
            N=ft[i,2]-self.c.tug_yaw_damping*r
            dt[i,:2]=R@[vx,vy]
            dt[i,2]=r
            dt[i,3:]=[Fx/self.Mt[0],Fy/self.Mt[1],N/self.Mt[2]]
        return ds,dt,diags

    def step(self,ship,tugs,thrust,azimuth,dt,quay_y=None,yaw_moment=None):
        # Semi-implicit symplectic Euler for coupled masses (force evaluated
        # before step, position using updated velocity). dt<=0.1 advised.
        ds,dts,diagnostics=self.derivative(ship,tugs,thrust,azimuth,quay_y,yaw_moment)
        ship2=ship.copy();tugs2=tugs.copy()
        ship2[3:]+=dt*ds[3:]
        ship2[2]=wrap(ship[2]+dt*ship2[5])
        ship2[:2]+=dt*(rot(ship2[2])@ship2[3:5])
        tugs2[:,3:]+=dt*dts[:,3:]
        tugs2[:,2]=wrap(tugs[:,2]+dt*tugs2[:,5])
        for i in range(4):tugs2[i,:2]+=dt*(rot(tugs2[i,2])@tugs2[i,3:5])
        return ship2,tugs2,diagnostics

    def geometry(self,ship,tugs):
        p=self.attachment(ship)
        out={}
        for i in range(4):
            # Towline is attached to towing tug BOW in astern-towing mode.
            a=tugs[i,:2]+rot(tugs[i,2])@np.array([self.ap if i<2 else self.at,0.])
            idx=i+1
            out.update({f'tug{idx}_x_m':tugs[i,0],f'tug{idx}_y_m':tugs[i,1],
                f'tug{idx}_psi_deg':np.rad2deg(tugs[i,2]),
                f'tug{idx}_contact_x_m':p[i,0],f'tug{idx}_contact_y_m':p[i,1],
                f'tug{idx}_rope_anchor_x_m':a[0],f'tug{idx}_rope_anchor_y_m':a[1]})
        return out

    def geometry_safety(self,ship,tugs):
        from shapely.geometry import Polygon
        hull=Polygon(self._ship_points@rot(ship[2]).T+ship[:2]).buffer(0)
        from decentralized_game.tow_safety import tug_corners
        R=rot(ship[2]);normal=R@np.array([0.,self.side]);p=self.attachment(ship)
        worst_margin=np.inf;worst_overlap=0.
        result=[]
        for i,tug in enumerate(tugs):
            poly=Polygon(tug_corners(tug[:2],tug[2]));clear=hull.distance(poly)
            overlapping=hull.intersects(poly)
            margin=float((tug[:2]-p[i])@normal)
            worst_margin=min(worst_margin,margin)
            worst_overlap=max(worst_overlap,float(hull.intersection(poly).area))
            result.append(dict(clearance_m=float(clear),overlap=bool(overlapping),
                               overlap_area_m2=float(hull.intersection(poly).area),
                               free_side_distance_m=margin))
        return result
