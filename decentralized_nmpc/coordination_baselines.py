"""Paper CONTROL baselines on the same physical five-vessel simulation.

These are *newly implemented comparator algorithms*, NOT reproductions of
Chen (2025), Huynh & Kim (2024), or Du et al. (2022).

central: one 16-variable shared-ship nonlinear MPC jointly chooses both
         planning knots of the four tug commands from full tug observations.
admm:  consensus-ADMM across four identical joint-control variable copies;
         each agent has its own objective component and local dual state,
         exchanging decision vectors to agree on a joint feasible plan.
         This prototype uses a shared nonlinear prediction model; it is NOT
         the multi-layer ship/tug collision-avoidance algorithm of Du et al.

The unilateral push/tow rules, real nonlinear coupled 5-vessel plant,
actuator dynamics, geometry, fenders and BerthingMonitor remain unmodified.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize
from physical_v4.plant import rot,wrap
from model.ship import Ship
from .controller import NMPCConfig,LocalNMPC,common_ship_observation


@dataclass(frozen=True)
class CoordinationSettings:
    method: str = 'central'
    admm_rounds: int = 3
    admm_rho: float = 0.5
    admm_inner_iterations: int = 4
    admm_tolerance: float = 0.03


class JointNMPC:
    """Joint plan controller or consensus-ADMM comparison, with physical inputs.

    All centralized/coordinated methods deliberately have access to the four
    tug states. The *original* decentralized LocalNMPC does NOT.
    """
    def __init__(self, env, cfg: NMPCConfig, settings: CoordinationSettings):
        if settings.method not in ('central','admm'):
            raise ValueError('method must be central or admm')
        self.cfg=cfg
        self.settings=settings
        self.model=Ship()
        self.goal=np.asarray(env.goal,float).copy()
        self.hull=np.asarray(env.hull.points,float).copy()
        self.side=int(env.side)
        self.local=[LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],
                env.side,env.plant.max_force,cfg) for i in range(4)]
        self.plan=np.zeros((4,2,2),float)
        self.last=np.zeros((4,2),float)
        self.xcopies=np.zeros((4,16),float)
        self.duals=np.zeros((4,16),float)
        self.last_ship=None
        self.last_time=None
        self.last_wrench=np.zeros(3)
        self.bias=np.zeros(3)
        self.history=[]
        self.last_report={}
        self.n_decisions=0
        self.communication_bytes_total=0

    def _update_model(self, env, measure_environment=True):
        # Shared ship measurements and full tug configuration intentionally
        # available to these *communication-enabled* comparison methods.
        obs=common_ship_observation(env,0,measure_environment=measure_environment)
        ship=np.asarray(obs.ship,float)
        self.model.external_tau=np.zeros(3)
        a,c=self.model.ship_matrices(ship[3],ship[4],ship[5],ship[2],
                Vw=obs.wind_mps,psi_w=obs.wind_to_rad,
                Vc=obs.current_mps,psi_c=obs.current_to_rad)
        hydro=np.linalg.solve(a,c.flatten())[:3]
        invmass=np.linalg.inv(a[:3,:3])
        if self.last_ship is not None and env.time > self.last_time+0.1:
            measured=(ship[3:6]-self.last_ship[3:6])/(env.time-self.last_time)
            innovation=measured-hydro-invmass@self.last_wrench
            innovation=np.clip(innovation,[-.0009,-.0009,-2e-5],[.0009,.0009,2e-5])
            alpha=self.cfg.residual_alpha
            self.bias=(1-alpha)*self.bias+alpha*innovation
        for controller in self.local:
            controller._update_stall_estimate(ship,float(env.time))
            if controller.mission_start is None:
                controller.mission_start=ship[:2].copy()
        return ship,hydro,invmass

    def _bounds(self,env):
        bounds=[]
        for i in range(4):
            own=np.asarray(env.tugs[i],float)
            ship=np.asarray(env.ship,float)
            heading_error=float(abs(np.rad2deg(wrap(own[2]-ship[2]+self.side*np.pi/2))))
            limit=.52 if i<2 else .65
            if i<2 and heading_error>22.:
                limit=min(limit,max(.08,.52*(30.-heading_error)/8.))
            for knot in range(2):
                prev=self.last[i] if knot==0 else self.plan[i,1]
                fmin=max(0.,float(prev[0])-self.cfg.max_fraction_step)
                fmax=min(limit,float(prev[0])+self.cfg.max_fraction_step)
                # Guard numerical degeneracy of a noisy physical heading.
                fmin=min(fmin,fmax)
                omin=max(-20.,float(prev[1])-self.cfg.max_offset_step_deg)
                omax=min(20.,float(prev[1])+self.cfg.max_offset_step_deg)
                bounds.extend([(fmin,fmax),(omin,omax)])
        return bounds

    def _joint_wrench(self,plan,knot):
        return sum((self.local[i]._force_signature(plan[i,knot,0],
                  plan[i,knot,1]) for i in range(4)),np.zeros(3,float))

    def _cost(self,x,ship,hydro,invmass):
        c=self.cfg
        plan=np.reshape(x,(4,2,2))
        state=ship.copy()
        dt=c.prediction_dt_s
        n=max(2,int(round(c.horizon_s/dt)))
        initial_dist=float(np.linalg.norm(ship[:2]-self.goal[:2]))
        total=0.
        for k in range(n):
            seg=0 if k<n//2 else 1
            wrench=self._joint_wrench(plan,seg)
            a=hydro+invmass@wrench+self.bias
            a[:2]+=-.001*(state[3:5]-ship[3:5])
            a[2]+=-.007*(state[5]-ship[5])
            state[3:6]+=dt*a
            state[5]=np.clip(state[5],-.007,.007)
            state[2]=wrap(state[2]+dt*state[5])
            velocity=rot(state[2])@state[3:5]
            state[:2]+=dt*velocity
            ref,yaw_ref,dist=(self.local[0]._mission_velocity_reference(state)
                if c.mission_guidance else self.local[0]._guidance(state))
            psi_err=wrap(state[2]-self.goal[2])
            rel=(state[:2]-self.goal[:2])@rot(-self.goal[2]).T
            phase=max(0.,min(1.,(80.-dist)/80.))
            head_weight=c.heading_weight*(1+c.near_berth_heading_boost*phase)
            motion_weight=c.motion_weight*(1+.55*phase)
            stall=max(loc.stall_factor for loc in self.local)
            stage=(c.position_weight*(1+c.anti_stall_gain*stall)*(dist/100.)**2+
                   head_weight*(psi_err/.35)**2+
                   motion_weight*(1+.7*stall)*np.sum(((velocity-ref)/.08)**2)+
                   c.yaw_rate_weight*((state[5]-yaw_ref)/.00065)**2+
                   c.cross_track_weight*(rel[1]/25.)**2+
                   c.along_track_weight*(rel[0]/35.)**2+
                   c.effort_weight*np.sum(plan[:,seg,0]**2)/4.)
            lo=self.hull@rot(state[2]).T
            clearance=float(np.min(lo[:,1])+state[1]-self.local[0].quay_y)
            approach=max(0.,-velocity[1])
            bad=c.constraint_weight*(max(0.,8.-clearance)/8.)**2
            if clearance<45.:
                bad+=c.constraint_weight*(max(0.,approach-.045)/.05)**2
                stopping_short=max(0.,approach*75.-max(0.,clearance-6.))
                bad+=.35*c.constraint_weight*(stopping_short/20.)**2
            progress_short=max(0.,dist-(initial_dist-.25*k))
            total+=(stage+bad+.08*(progress_short/25.)**2)/n
            if c.mission_guidance:
                total+=self.local[0]._mission_cost(state,velocity,dist,initial_dist)/n
        for i in range(4):
            fr=plan[i,:,0];off=plan[i,:,1];init=self.last[i]
            slew=((fr[0]-init[0])/.25)**2+((off[0]-init[1])/14.)**2
            slew+=.5*((fr[1]-fr[0])/.25)**2+.5*((off[1]-off[0])/14.)**2
            total+=c.slew_weight*slew/4.
        dist=float(np.linalg.norm(state[:2]-self.goal[:2]))
        heading=wrap(state[2]-self.goal[2])
        if c.mission_guidance:
            total+=.65*self.local[0]._mission_cost(state,velocity,dist,initial_dist)
        total+=self.local[0]._terminal_position_cost(dist)
        total+=c.terminal_heading_weight*(heading/.35)**2
        total+=c.terminal_speed_weight*np.sum((velocity/.05)**2)
        return float(total)

    def _initial_plan(self,bounds,fun):
        warm=self.plan.flatten()
        warm=np.array([np.clip(a,*bd) for a,bd in zip(warm,bounds)],float)
        # Physically feasible nonzero seeds avoid a saddle at zero force.
        seeds=[warm]
        for offset in (-12.,0.,12.):
            seed=warm.copy()
            for i in range(4):
                for seg in range(2):
                    f=4*i+2*seg
                    seed[f]=np.clip(max(seed[f],.12),*bounds[f])
                    seed[f+1]=np.clip(offset if i<2 else -offset,*bounds[f+1])
            seeds.append(seed)
        return min(seeds,key=fun)

    def _central(self,warm,bounds,fn):
        result=minimize(fn,warm,method='L-BFGS-B',bounds=bounds,
                options={'maxiter':self.cfg.max_iterations,'maxls':8,'ftol':1e-4})
        if np.all(np.isfinite(result.x)) and np.isfinite(result.fun) and result.fun<=fn(warm)+1e-8:
            return result.x,dict(converged=bool(result.success),iterations=int(result.nit),
                                 objective=float(result.fun),primal_residual=0.,dual_residual=0.,
                                 admm_rounds=0,communication_bytes=0)
        return warm,dict(converged=False,iterations=int(result.nit),objective=float(fn(warm)),
                         primal_residual=0.,dual_residual=0.,admm_rounds=0,communication_bytes=0)

    def _admm(self,warm,bounds,fn):
        # Scaled consensus ADMM: local x_i, average joint plan z, scaled dual u_i.
        # Each node carries a distinct own-effort regularizer; the common
        # nonlinear prediction objective is split into four equal shares.
        rho=float(self.settings.admm_rho)
        if rho<=0:raise ValueError('admm_rho must be > 0')
        z=warm.copy()
        u=self.duals.copy()
        x=self.xcopies.copy()
        if not self.n_decisions:
            x[:]=warm
        total_inner=0
        primal=dual=float('inf')
        rounds=0
        for q in range(self.settings.admm_rounds):
            for i in range(4):
                def local_cost(v):
                    pr=np.reshape(v,(4,2,2))
                    # Only node i's local effort is additionally priced.
                    individual=.04*np.sum(pr[i,:,0]**2)
                    return .25*fn(v)+individual+(.5*rho)*float(np.sum((v-z+u[i])**2))
                res=minimize(local_cost,np.clip(x[i],np.asarray(bounds)[:,0],np.asarray(bounds)[:,1]),
                         method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':self.settings.admm_inner_iterations,'ftol':1e-4,'maxls':6})
                x[i]=res.x if np.all(np.isfinite(res.x)) else np.clip(x[i],np.asarray(bounds)[:,0],np.asarray(bounds)[:,1])
                total_inner+=int(res.nit)
            old=z.copy()
            z=np.mean(x+u,axis=0)
            z=np.array([np.clip(a,*bd) for a,bd in zip(z,bounds)])
            u+=x-z
            primal=float(np.linalg.norm(x-z))
            dual=float(rho*np.sqrt(4)*np.linalg.norm(z-old))
            rounds=q+1
            if max(primal,dual)<self.settings.admm_tolerance:
                break
        self.duals=u
        self.xcopies=x
        # Each iteration: four 16-D plan uplinks and four 16-D consensus
        # downlinks, float64; excludes networking/protocol overhead.
        bytes_sent=rounds*4*2*16*8
        self.communication_bytes_total+=bytes_sent
        if not np.isfinite(fn(z)) or fn(z)>fn(warm)+1e-8:
            z=warm.copy()
        return z,dict(converged=(primal<self.settings.admm_tolerance and dual<self.settings.admm_tolerance),
             iterations=total_inner,objective=fn(z),primal_residual=primal,
             dual_residual=dual,admm_rounds=rounds,communication_bytes=bytes_sent)

    def act(self,env,measure_environment=True):
        ship,hydro,invmass=self._update_model(env,measure_environment)
        bounds=self._bounds(env)
        objective=lambda x:self._cost(x,ship,hydro,invmass)
        warm=self._initial_plan(bounds,objective)
        if self.settings.method=='central':
            solution,details=self._central(warm,bounds,objective)
        else:
            solution,details=self._admm(warm,bounds,objective)
        plan=solution.reshape((4,2,2)).copy()
        self.plan=plan
        self.last=plan[:,0,:].copy()
        self.last_wrench=self._joint_wrench(plan,0)
        self.last_ship=ship.copy()
        self.last_time=float(env.time)
        self.n_decisions+=1
        actions=np.column_stack([2*self.last[:,0]-1,self.last[:,1]/20.]).astype(np.float32)
        details=dict(details,method=self.settings.method,observation_time_s=float(env.time))
        self.last_report=details
        reports=[]
        for i in range(4):
            reports.append(dict(agent=i+1,success_solver=details['converged'],iterations=details['iterations'],
                objective=details['objective'],fraction=float(self.last[i,0]),
                offset_deg=float(self.last[i,1]),residual_acceleration_norm=float(np.linalg.norm(self.bias)),
                own_heading_error_deg=float(np.rad2deg(abs(wrap(env.tugs[i,2]-ship[2]+self.side*np.pi/2)))),
                own_thrust_N=float(env.act.thrust[i]),
                wind_sensor_mps=float(env.plant.env[0]) if measure_environment else 0.,
                current_sensor_mps=float(env.plant.env[2]) if measure_environment else 0.,
                disturbance_mode=self.cfg.disturbance_mode,admm_rounds=details['admm_rounds'],
                primal_residual=details['primal_residual'],dual_residual=details['dual_residual'],
                communication_bytes=details['communication_bytes'] if i==0 else 0))
        return actions,reports
