"""One-tug contact-point NMPC with no access to the vessel's 3DOF state.

IMPORTANT: this is a reduced-order *point-tracking* NMPC, not the original
full-vessel MMG NMPC with secretly reconstructed global ship state. Other
agents' actions and the unknown hull acceleration enter as a bounded,
locally estimated disturbance. Finite-horizon feasibility and quay collision
safety cannot be certified from one point alone.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .sensors import ContactObservation
from physical_v4.plant import rot, wrap


@dataclass(frozen=True)
class ContactNMPCConfig:
    horizon_s:float=75.
    prediction_dt_s:float=15.
    max_iterations:int=30
    max_fraction_step:float=.38
    max_offset_step_deg:float=24.
    push_fraction_limit:float=.52
    tow_fraction_limit:float=.65
    # Effective point acceleration at full nominal fraction, NOT known MMG
    # parameters: calibrate on measured local input-output trajectories.
    effective_acceleration_mps2:float=.0010
    observer_alpha:float=.28
    observer_clip_mps2:float=.0015
    velocity_reference_mps:float=.090
    precision_speed_mps:float=.022
    precision_radius_m:float=25.
    stopping_acceleration_mps2:float=.00016
    slew_weight:float=.12
    effort_weight:float=.03
    safety_weight:float=22.


class ContactNMPC:
    """One independent 2-knot receding-horizon controller, four instances.

    Only `ContactObservation` is accepted by `act`. Teacher global state does
    not occur in any method signature or optimization routine.
    """
    def __init__(self,agent_id:int,cfg:ContactNMPCConfig|None=None):
        if agent_id not in range(4):raise ValueError('agent id must be 0..3')
        self.i=int(agent_id)
        self.cfg=cfg or ContactNMPCConfig()
        c=self.cfg
        if min(c.horizon_s,c.prediction_dt_s,c.effective_acceleration_mps2,
               c.stopping_acceleration_mps2)<=0:
            raise ValueError('horizon/dt/accelerations must be positive')
        self.station_xy=None
        self.quay_y=None
        self.hull_xy=None
        self.reset()

    def configure_berth(self, station_xy, quay_y, hull_xy):
        """Offline static geometry, not online vessel pose or peer data."""
        a=np.asarray(station_xy,float);h=np.asarray(hull_xy,float)
        if a.shape!=(2,) or h.ndim!=2 or h.shape[1]!=2 or not np.isfinite(h).all():
            raise ValueError('invalid offline berth geometry')
        self.station_xy=a.copy()
        self.quay_y=float(quay_y)
        self.hull_xy=h.copy()

    def _local_quay_risk(self, obs):
        if self.hull_xy is None:return 0.,0.
        # Heuristic hull orientation proxy from own bow-inboard heading.
        # This is not a measured or identified ship heading; under relative
        # tug yaw misalignment it can be wrong. Used for risk reduction only.
        psi=wrap(float(obs.own[2])+np.pi/2)
        rotm=rot(psi)
        relative=self.hull_xy-self.station_xy[None,:]
        low=float(np.min((relative@rotm.T)[:,1]))
        gap=float(obs.point_xy[1]+low-self.quay_y)
        return gap,abs(float(np.rad2deg(psi)))

    def reset(self):
        self.last=np.zeros(2,float)
        self.plan=np.zeros((2,2),float)
        self.last_velocity=None
        self.last_time=None
        self.last_model_accel=np.zeros(2,float)
        self.disturbance=np.zeros(2,float)
        self.n_optimizations=0
        self._cached_risk=(0.,0.)

    @staticmethod
    def _normal(obs):
        d=obs.relative_tip_xy
        norm=float(np.linalg.norm(d))
        if norm>1.e-4:return d/norm
        # Avoid assuming hidden vessel heading when tracker geometry degenerates.
        return -np.array([np.cos(obs.own[2]),np.sin(obs.own[2])])

    def _action_accel(self,fraction,offset_deg,normal,role):
        # Nominal point acceleration along tug-induced contact force. The
        # actual acceleration will differ from the coupled five-body plant.
        c=self.cfg
        phi=np.deg2rad(offset_deg)
        transverse=np.array([-normal[1],normal[0]])
        direction=(-normal if role=='push' else normal)*np.cos(phi)+transverse*np.sin(phi)
        return c.effective_acceleration_mps2*float(fraction)*direction

    def _estimate_disturbance(self,obs,normal):
        if self.last_time is None:return
        elapsed=obs.time_s-self.last_time
        if elapsed<=1.e-7:raise ValueError('observation timestamps must increase')
        if elapsed>max(120.,3.*self.cfg.horizon_s):
            self.disturbance[:]=0.
            return
        accel=(obs.point_velocity_xy-self.last_velocity)/elapsed
        # Command from the preceding sample is a nominal effective wrench;
        # all remaining unknown terms are absorbed in a bounded local residual.
        innovation=np.clip(accel-self.last_model_accel,
                    -self.cfg.observer_clip_mps2,self.cfg.observer_clip_mps2)
        a=self.cfg.observer_alpha
        self.disturbance=(1.-a)*self.disturbance+a*innovation

    def _reference_velocity(self,point,goal):
        c=self.cfg
        e=goal-point
        dist=float(np.linalg.norm(e))
        if dist<1.e-8:return np.zeros(2)
        speed=min(c.velocity_reference_mps,
                  np.sqrt(2.*c.stopping_acceleration_mps2*dist))
        if dist<c.precision_radius_m:
            # Smooth interpolation: avoid a discontinuous reference at 25 m.
            mix=np.clip(dist/max(c.precision_radius_m,1.e-8),0.,1.)
            speed=min(speed,c.precision_speed_mps+
                      (c.velocity_reference_mps-c.precision_speed_mps)*mix)
        speed=min(speed,max(.004,dist/170.))
        return (speed/dist)*e

    def _cost(self,x,obs,normal,record=False):
        c=self.cfg
        fractions=x[[0,2]]
        offsets=x[[1,3]]
        p=obs.point_xy.copy()
        v=obs.point_velocity_xy.copy()
        goal=obs.goal_point_xy
        quay_gap,heading_proxy_deg=self._cached_risk
        n=max(2,int(np.ceil(c.horizon_s/c.prediction_dt_s)))
        dt=c.horizon_s/n
        total=0.;path=[]
        for k in range(n):
            j=0 if k<n//2 else 1
            acceleration=self._action_accel(fractions[j],offsets[j],normal,obs.role)+self.disturbance
            acceleration=np.clip(acceleration,-.003,.003)
            # Constant-acceleration point kinematics. There is no artificial
            # vessel yaw or surrogate ship state in this predictive model.
            p=p+v*dt+.5*acceleration*dt**2
            v=v+acceleration*dt
            error=goal-p
            dist=float(np.linalg.norm(error))
            vref=self._reference_velocity(p,goal)
            # Speed envelope: high velocity near one's assigned point target is
            # penalized. These are *soft* local constraints, not global safety.
            safe_speed=min(.09,.043+dist/650.)
            speed_excess=max(0.,np.linalg.norm(v)-safe_speed)
            # Velocity approaching the point faster than the available braking
            # distance is penalized, avoiding overshoot despite weak drag.
            inward=np.dot(v,error/max(dist,1.e-8)) if dist>1.e-8 else 0.
            stop_v=np.sqrt(2*c.stopping_acceleration_mps2*dist)
            unsafe_approach=max(0.,inward-stop_v)
            stage=(2.0*(dist/70.)**2+
                   7.0*np.sum(((v-vref)/.09)**2)+
                   .35*np.sum((v/.12)**2)*(dist<12.)+
                   c.safety_weight*((speed_excess/.06)**2+(unsafe_approach/.05)**2)+
                   c.effort_weight*fractions[j]**2)
            # The aft point is often closest to the quay while a highly
            # rotated ship is still yawing. Avoid advancing that point into
            # danger before the bow has rotated. Only own station, own tug
            # attitude and local point tracking are used. This is NOT an
            # invariant-safety guarantee under partial observability.
            if self.i==0 and heading_proxy_deg>9.:
                estimated_gap=quay_gap+(p[1]-obs.point_xy[1])
                danger=max(0.,(22.-estimated_gap)/22.)
                stage+=95.*danger**2*fractions[j]**2
                stage+=40.*danger**2*(max(0.,-v[1])/.06)**2
            total+=stage/n
            if record:path.append((float(p[0]),float(p[1]),float(v[0]),float(v[1])))
        final_err=float(np.linalg.norm(goal-p))
        total+=1.6*(final_err/60.)**2+1.3*np.sum((v/.09)**2)
        # Slew limit and regularization suppress oscillations of local agents.
        total+=c.slew_weight*(((fractions[0]-self.last[0])/.25)**2+
            .15*((offsets[0]-self.last[1])/15.)**2*max(.05,fractions[0])+
            .6*((fractions[1]-fractions[0])/.25)**2+
            .1*((offsets[1]-offsets[0])/15.)**2*max(.05,fractions[1]))
        return (float(total),path) if record else float(total)

    def act(self,obs:ContactObservation):
        if not isinstance(obs,ContactObservation):
            raise TypeError('ContactNMPC accepts ContactObservation only (no env/ship truth)')
        if obs.role!=('push' if self.i<2 else 'tow'):
            raise ValueError('contact observation belongs to wrong tug role')
        normal=self._normal(obs)
        self._cached_risk=self._local_quay_risk(obs)
        self._estimate_disturbance(obs,normal)
        limit=self.cfg.push_fraction_limit if self.i<2 else self.cfg.tow_fraction_limit
        prev=self.last
        # First knot rate bound; second knot rate bounds enforced with smooth
        # penalties, rather than incorrect bounds relative to an old future knot.
        l=max(0.,prev[0]-self.cfg.max_fraction_step)
        h=min(limit,prev[0]+self.cfg.max_fraction_step)
        # First offset knot is rate limited against last realized request;
        # future knot is unconstrained within physical steering command range.
        offset_lo=max(-20.,prev[1]-self.cfg.max_offset_step_deg)
        offset_hi=min(20.,prev[1]+self.cfg.max_offset_step_deg)
        bounds=[(l,h),(offset_lo,offset_hi),(0.,limit),(-20.,20.)]
        warm=self.plan.ravel().copy()
        warm=np.array([np.clip(z,*b) for z,b in zip(warm,bounds)])
        fn=lambda x:self._cost(x,obs,normal)
        candidates=[warm]
        for fraction in (0.,min(.25,limit),limit):
            for offset in (-15.,0.,15.):
                x=np.array([np.clip(fraction,l,h),np.clip(offset,offset_lo,offset_hi),fraction,offset])
                candidates.append(x)
        warm=min(candidates,key=fn)
        result=minimize(fn,warm,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':self.cfg.max_iterations,'ftol':1.e-5,'maxls':10})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and fn(result.x)<=fn(warm)+1.e-9:
            optimum=result.x
        else:optimum=warm
        fraction,offset=float(optimum[0]),float(optimum[1])
        self.plan=np.asarray(optimum,float).reshape((2,2)).copy()
        self.last=np.array([fraction,offset])
        self.last_velocity=obs.point_velocity_xy.copy()
        self.last_time=float(obs.time_s)
        self.last_model_accel=self._action_accel(fraction,offset,normal,obs.role)
        self.n_optimizations+=1
        info={'agent':self.i+1,'observation_model':'contact_point_only',
              'objective':fn(optimum),'fraction':fraction,'offset_deg':offset,
              'point_error_m':float(np.linalg.norm(obs.goal_point_xy-obs.point_xy)),
              'point_speed_mps':float(np.linalg.norm(obs.point_velocity_xy)),
              'local_disturbance_accel_mps2':float(np.linalg.norm(self.disturbance)),
              'solver_success':bool(result.success),'solver_iterations':int(result.nit),
              'success_solver':bool(result.success),'iterations':int(result.nit),
              'observation_time_s':float(obs.time_s),
              'local_quay_proxy_gap_m':float(self._cached_risk[0]),
              'local_tug_heading_proxy_deg':float(self._cached_risk[1]),
              'quay_risk_model':'static_hull_and_own_tug_heading_heuristic'}
        return np.array([2.*fraction-1.,offset/20.],np.float32),info
