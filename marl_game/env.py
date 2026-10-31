"""Research Dec-POMDP / cooperative partially observable Markov game.

Four local role-conditioned controllers. Peer tug states/actions and wind/current
are absent from observations. The simulator exposes complete state only to the
offline CTDE critic and debugging. The compliant v5.3 physical plant is reused.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from geom.constraints import Constraints
from model.ship import Ship
from model.tug import Tug
from physical_v4.plant import UnilateralPlant,ContactConfig,rot,wrap
from physical_v4.game import Action,TugHelmsman
from physical_v4.push_contact import PushContactController
from physical_v4.push_heading_guard import PushHeadingGuard
from physical_v4.tow_station import TowStationController
from physical_v4.actuators import ActuatorBank,ActuatorConfig
from physical_v4.termination import BerthingMonitor,BerthingCriteria
from decentralized_game.tow_safety import free_side_from_quay

@dataclass
class EnvConfig:
    dt:float=.2
    decision_seconds:float=10.
    max_seconds:float=2800.
    target_x:float=50.
    target_y:float=-100.
    target_psi_deg:float=0.
    initial_psi_deg:float=20.
    eval_start_x_m:float=0.
    eval_start_y_m:float=0.
    initial_psi_jitter_deg:float=4.
    eval_heading_jitter_deg:float=0.
    eval_position_jitter_m:float=0.
    position_jitter_m:float=2.
    wind_max_mps:float=0.
    current_max_mps:float=0.
    observation_noise:float=.015
    observation_dropout:float=0.
    train:bool=True
    eval_wind_mps:float=0.
    eval_wind_direction_deg:float=0.
    eval_current_mps:float=0.
    eval_current_direction_deg:float=0.
    stop_on_unsafe:bool=True
    use_actuators:bool=True
    hold_s:float=5.
    post_success_seconds:float=0.  # Evaluation only: continue without changing controller/plant.
    push_heading_guard:bool=True
    seed:int=123
    reward_shaping:bool=True
    contact_only:bool=False  # local policy + local helm; full truth retained only for plant/evaluation

class BerthingGameEnv:
    n_agents=4
    action_dim=2
    def __init__(self,cfg:EnvConfig|None=None):
        self.cfg=cfg or EnvConfig()
        if not 0<self.cfg.dt<=.2:raise ValueError('dt must be in (0,0.2]')
        ratio=self.cfg.decision_seconds/self.cfg.dt
        if abs(ratio-round(ratio))>1e-6:raise ValueError('decision_seconds must be an integer multiple of dt')
        self.substeps=int(round(ratio));self.rng=np.random.default_rng(self.cfg.seed)
        self.hull=Constraints(str(Path(__file__).resolve().parents[1]/'geom'/'waterline_ship.dxf'))
        self.goal=np.array([self.cfg.target_x,self.cfg.target_y,np.deg2rad(self.cfg.target_psi_deg)])
        self.side=free_side_from_quay(self.goal[2],np.array([0.,-1.]))
        target_hull=np.asarray(self.hull.points)@rot(self.goal[2]).T+self.goal[:2]
        self.quay_y=float(target_hull[:,1].min()-6.)
        self.ship_model=Ship(); self.tug_model=Tug()
        self.tug_model.bollard_pull_N=100.*9806.65
        self.tug_model.command_force_limit_N=self.tug_model.bollard_pull_N*.9
        self.plant=UnilateralPlant(self.ship_model,self.tug_model,self.hull,self.side,ContactConfig())
        self.helms=[TugHelmsman(i,self.side) for i in range(4)]
        self.push=[PushContactController(i,self.side,self.plant) for i in range(2)]
        self.tow=[TowStationController(i,self.side,self.plant) for i in (2,3)]
        self.obs_dim=19 if self.cfg.contact_only else 26
        if self.cfg.contact_only:
            from contact_local.actuation import ContactOnlyHelm
            self.local_helms=[ContactOnlyHelm(i,self.plant.max_force,
                self.plant.c.bumper_m,self.plant.c.rope_length) for i in range(4)]
        self.state_dim=3+3+4*6+4*3+7 # ship pose, ship velocity, tugs, realized actuators, env/mission
        self.monitor=BerthingMonitor(BerthingCriteria(hold_s=self.cfg.hold_s))
        self.replay_snapshots=[]
        self.replay_probability=0.
        self.reset()

    def reset(self, snapshot=None):
        c=self.cfg
        if snapshot is None and c.train and self.replay_snapshots and self.rng.random()<self.replay_probability:
            snapshot=self.replay_snapshots[int(self.rng.integers(len(self.replay_snapshots)))]
        self.monitor=BerthingMonitor(BerthingCriteria(hold_s=c.hold_s))
        if c.contact_only:
            for helm in self.local_helms:
                helm.reset()
        self.first_success_ship = None
        self.first_success_hull_gap_m = None
        if c.train:
            wind=float(self.rng.uniform(0,c.wind_max_mps))
            current=float(self.rng.uniform(0,c.current_max_mps))
            wd=float(self.rng.uniform(-np.pi,np.pi));cd=float(self.rng.uniform(-np.pi,np.pi))
        else:
            wind=float(c.eval_wind_mps);current=float(c.eval_current_mps)
            wd=np.deg2rad(c.eval_wind_direction_deg);cd=np.deg2rad(c.eval_current_direction_deg)
        self.plant.env=(wind,wd,current,cd)
        self.act=ActuatorBank(self.plant.max_force,ActuatorConfig())
        if snapshot is not None:
            self.ship=np.asarray(snapshot['ship'],dtype=float).copy()
            self.tugs=np.asarray(snapshot['tugs'],dtype=float).copy()
            self.time=float(snapshot.get('time',0.))
            for key,target in [('act_thrust','thrust'),('act_azimuth','azimuth'),('act_moment','moment')]:
                if key in snapshot:setattr(self.act,target,np.asarray(snapshot[key],float).copy())
        else:
            position_noise=c.position_jitter_m if c.train else c.eval_position_jitter_m
            heading_noise=c.initial_psi_jitter_deg if c.train else c.eval_heading_jitter_deg
            x=float(c.eval_start_x_m if not c.train else 0.)+float(self.rng.uniform(-position_noise,position_noise))
            y=float(c.eval_start_y_m if not c.train else 0.)+float(self.rng.uniform(-position_noise,position_noise))
            psi=c.initial_psi_deg+float(self.rng.uniform(-heading_noise,heading_noise))
            self.ship,self.tugs=self.plant.initial(x,y,psi)
            self.time=0.
        self.heading_guards=[PushHeadingGuard(self.side,self.plant.max_force,
            self.plant.tug.xTi,self.plant.c.push_twin_azimuth_max_yaw_moment_Nm) for _ in range(2)]
        self.last_actions=np.zeros((4,2),np.float32)
        self.last_actions[:,0]=-1. # 0 fraction
        if snapshot is not None and 'last_actions' in snapshot:
            self.last_actions=np.asarray(snapshot['last_actions'],np.float32).copy()
        self.old_potential=self.potential()
        return self.observe(),self.critic_state()

    def snapshot(self):
        return {'ship':self.ship.copy(),'tugs':self.tugs.copy(),'time':float(self.time),
                'act_thrust':self.act.thrust.copy(), 'act_azimuth':self.act.azimuth.copy(),
                'act_moment':self.act.moment.copy(),'last_actions':self.last_actions.copy()}

    def hull_gap(self):
        ship_pts=self.plant._ship_points@rot(self.ship[2]).T+self.ship[:2]
        return float(ship_pts[:,1].min()-self.quay_y)

    def potential(self):
        sh=self.ship
        e=np.linalg.norm((sh[:2]-self.goal[:2])/100.)
        epsi=abs(wrap(sh[2]-self.goal[2]))/np.deg2rad(30.)
        v=np.linalg.norm(sh[3:5])/.12
        yaw=abs(sh[5])/np.deg2rad(.10)
        # Stage value; bounded for safe, physically realistic states.
        return float(-(.75*e+ .40*epsi+.26*v+.20*yaw))

    def observe(self):
        """Local noisy sensors only; an agent cannot read any peer tug state.

        Ship measurements are common AIS/vision-like signals, not true hidden forces.
        Noise is applied ONLY to policy observations, not to dynamics/critic.
        """
        if self.cfg.contact_only:
            from contact_local.sensors import contact_observation
            return np.stack([contact_observation(self,i).policy_vector() for i in range(4)])
        sh=self.ship;c=self.cfg
        R=rot(sh[2]);v_earth=R@sh[3:5]
        e_global=sh[:2]-self.goal[:2]
        goal_in_ship=R.T@e_global
        gap=self.hull_gap()
        obs=[]
        for i in range(4):
            tug=self.tugs[i];rel=R.T@(tug[:2]-sh[:2]);angle=wrap(tug[2]-sh[2])
            own_v=tug[3:5];own_omega=tug[5]
            # Anchors are own physical geometry; other tug geometry is never observed.
            a=self.plant.attachment(sh)[i]
            bow=tug[:2]+rot(tug[2])@np.array([self.plant.ap if i<2 else self.plant.at,0.])
            ship_v=self.plant.velocity_at(sh,self.plant.attach_local[i])
            tug_v=self.plant.velocity_at(tug,[self.plant.ap if i<2 else self.plant.at,0.])
            local_delta=R.T@(bow-a);rel_speed=R.T@(tug_v-ship_v)
            pair=[1. if i<2 else 0.,-1. if i in (0,2) else 1.]
            x=np.array([
                goal_in_ship[0]/200.,goal_in_ship[1]/200.,
                np.sin(wrap(sh[2]-self.goal[2])),np.cos(wrap(sh[2]-self.goal[2])),
                sh[3]/.3,sh[4]/.3,sh[5]/np.deg2rad(.2),
                gap/150.,v_earth[1]/.3,
                rel[0]/250.,rel[1]/100.,np.sin(angle),np.cos(angle),
                own_v[0]/.6,own_v[1]/.6,own_omega/np.deg2rad(2.),
                local_delta[0]/80.,local_delta[1]/80.,
                rel_speed[0]/.5,rel_speed[1]/.5,
                self.last_actions[i,0],self.last_actions[i,1],
                *pair,self.time/max(c.max_seconds,1.),
                1. if i>=2 else 0.
            ],dtype=np.float32)
            # Deliberately no peer state, other actions, wind/current or generalized forces.
            if c.train and c.observation_noise>0:
                noisy=x.copy(); noisy[:20]+=self.rng.normal(0,c.observation_noise,20)
                if c.observation_dropout>0:
                    drop=self.rng.random(20)<c.observation_dropout
                    noisy[:20][drop]=0.
                x=noisy
            obs.append(np.clip(x,-10.,10.))
        return np.stack(obs)

    def critic_state(self):
        sh=self.ship
        a=self.act
        env=self.plant.env
        st=np.r_[[(sh[0]-self.goal[0])/200.,(sh[1]-self.goal[1])/200.,wrap(sh[2]-self.goal[2])/np.pi],
                  sh[3:5]/.5,sh[5]/np.deg2rad(.5),
                  np.column_stack(((self.tugs[:,:2]-self.ship[:2])/250.,
                           self.tugs[:,2:3]/np.pi,self.tugs[:,3:5]/.7,
                           self.tugs[:,5:6]/np.deg2rad(2.))).ravel(),
                  a.thrust/self.plant.max_force,a.azimuth/.6,a.moment/5e6,
                  [env[0]/10.,np.sin(env[1]),np.cos(env[1]),env[2]/2.,np.sin(env[3]),np.cos(env[3]),self.time/max(self.cfg.max_seconds,1.)]]
        assert st.size==self.state_dim,(st.size,self.state_dim)
        return np.clip(st.astype(np.float32),-20,20)

    def step(self, actions):
        """Actions normalized to [-1,1] per agent: physical [fraction,offset]."""
        actions=np.asarray(actions,dtype=float)
        if actions.shape!=(4,2) or not np.isfinite(actions).all():raise ValueError('actions require finite (4,2)')
        actions=np.clip(actions,-1,1)
        chosen=[Action(float((a[0]+1)*.5),float(a[1]*20)) for a in actions]
        self.last_actions=actions.astype(np.float32)
        prepot=self.old_potential
        previous_distance=float(np.linalg.norm(self.ship[:2]-self.goal[:2]))
        previous_heading_deg=float(abs(np.rad2deg(wrap(self.ship[2]-self.goal[2]))))
        start_time=self.time
        min_gap=1e10;peak_near_speed=0.;peak_force=0.;bad=False;reason=''
        energy=0.;success=False;diags={};safety_checks=[]
        dt=self.cfg.dt
        for k in range(self.substeps):
            ramp=np.clip(self.time/15.,0.,1.)
            thrust=np.array([a.fraction*self.plant.max_force*(1 if i<2 else -1) for i,a in enumerate(chosen)])*ramp
            if self.cfg.contact_only:
                # Sensor driver reads simulator truth solely to synthesize THIS
                # attachment-point measurement. Helms see no ship pose, ship
                # velocities, peer data or simulator object.
                from contact_local.sensors import contact_observation
                commands=[self.local_helms[i].command(contact_observation(self,i),actions[i])
                          for i in range(4)]
                thrust=np.array([cmd[0] for cmd in commands],float)*ramp
                az=np.array([cmd[1] for cmd in commands],float)
                yaw=np.array([cmd[2] for cmd in commands],float)
            else:
                az=np.array([self.helms[i].steer(self.ship,self.tugs[i],a) for i,a in enumerate(chosen)])
                yaw=np.array([self.helms[i].yaw_moment(self.ship,self.tugs[i],a) for i,a in enumerate(chosen)])
            for i in ([] if self.cfg.contact_only else range(2)):
                thrust[i],az[i],_=self.push[i].command(self.ship,self.tugs[i],chosen[i]);thrust[i]*=ramp
                nominal=self.ship[2]-self.side*np.pi/2
                deviation=wrap(nominal-self.tugs[i,2]);rate=self.tugs[i,5]-self.ship[5]
                prop_yaw=self.plant.tug.xTi*thrust[i]*np.sin(az[i])
                correction=float(np.clip((abs(deviation)-np.deg2rad(12.))/np.deg2rad(9.),0.,1.))
                protective=np.clip(22e6*deviation-28e6*rate-prop_yaw,
                   -self.plant.c.push_twin_azimuth_max_yaw_moment_Nm,
                    self.plant.c.push_twin_azimuth_max_yaw_moment_Nm)
                yaw[i]=(1-correction)*yaw[i]+correction*protective
            for i in (() if self.cfg.contact_only else (2,3)):
                thrust[i],az[i],_,_=self.tow[i-2].command(self.ship,self.tugs[i],chosen[i],thrust[i],az[i]);thrust[i]*=ramp
            if self.cfg.push_heading_guard and not self.cfg.contact_only:
                for i in range(2):
                    thrust[i],az[i],yaw[i],_=self.heading_guards[i].filter(
                        self.ship,self.tugs[i],thrust[i],az[i],yaw[i])
            if self.cfg.use_actuators:
                thrust,az,yaw=self.act.step(thrust,az,yaw,dt)
            else:
                self.act.thrust=thrust.copy();self.act.azimuth=az.copy();self.act.moment=yaw.copy()
            self.ship,self.tugs,diags=self.plant.step(self.ship,self.tugs,thrust,az,dt,self.quay_y,yaw)
            self.time+=dt
            energy+=float(np.mean(np.abs(thrust))/self.plant.max_force)*dt
            peak_force=max(peak_force,max(diags['push1_normal_N'],diags['push2_normal_N'],diags['tow3_tension_N'],diags['tow4_tension_N']))
            if not np.isfinite(self.ship).all() or not np.isfinite(self.tugs).all() or np.max(abs(self.tugs))>1e6:
                bad=True;reason='NONFINITE_OR_DIVERGENCE';break
            gap=float(diags.get('fender_gap_m',1000))
            min_gap=min(min_gap,gap)
            vy=float((rot(self.ship[2])@self.ship[3:5])[1]);close=max(0.,-vy)
            if gap<45:peak_near_speed=max(peak_near_speed,close)
            if gap<-.02 or (gap<45 and close>.095):bad=True;reason='QUAY_PENETRATION_OR_SPEED'
            for i in range(2):
                if self.push[i].observe(self.ship,self.tugs[i])['gap']>self.plant.c.bumper_m+2.5:
                    bad=True;reason='PUSH_DETACH'
                if abs(wrap(self.tugs[i,2]-self.ship[2]+self.side*np.pi/2))>np.deg2rad(30):
                    bad=True;reason='PUSH_HEADING'
            if k==self.substeps-1 or bad:
                # shapely polygon audit once per decision (costly)
                safety_checks=self.plant.geometry_safety(self.ship,self.tugs)
                if any(q['overlap_area_m2']>.25 or q['free_side_distance_m']<0 for q in safety_checks):
                    bad=True;reason='HULL_COLLISION_OR_CROSSING'
            distance=float(np.linalg.norm(self.ship[:2]-self.goal[:2]))
            speed=float(np.linalg.norm(self.ship[3:5]))
            success_now=self.monitor.update(t=self.time,dt=dt,position_error_m=distance,
                      heading_error_deg=np.rad2deg(wrap(self.ship[2]-self.goal[2])),speed_mps=speed,
                      yaw_rate_deg_s=abs(np.rad2deg(self.ship[5])),safe=not bad,
                      hull_gap_m=gap)
            if success_now:
                self.first_success_ship = self.ship.copy()
                self.first_success_hull_gap_m = float(gap)
            success=success or success_now
            # The 5-second initial-success monitor remains unchanged. Only evaluation
            # rollouts with an explicit follow-up duration continue past it.
            hold_done = (self.cfg.post_success_seconds > 0 and
                         self.monitor.first_success_time_s is not None and
                         self.time >= self.monitor.first_success_time_s + self.cfg.post_success_seconds - 1e-8)
            if (bad and self.cfg.stop_on_unsafe) or (success and self.cfg.post_success_seconds <= 0) or hold_done:
                break
        done_unsafe=bool(bad or not np.isfinite(self.ship).all())
        hold_done = bool(self.cfg.post_success_seconds > 0 and
                         self.monitor.first_success_time_s is not None and
                         self.time >= self.monitor.first_success_time_s + self.cfg.post_success_seconds - 1e-8)
        terminated=bool((success and self.cfg.post_success_seconds <= 0) or hold_done or
                        (done_unsafe and self.cfg.stop_on_unsafe))
        truncated=bool(self.time>=self.cfg.max_seconds and not terminated)
        actualpot=self.potential() if np.isfinite(self.ship).all() else -100.
        self.old_potential=actualpot
        # Progress-based task reward. The old discounted negative potential gave
        # a *positive* idle bonus, while an accelerating teacher could be
        # penalized by the velocity component. That reward was exploitable by
        # simply issuing zero thrust indefinitely (see diagnosis report).
        # Here every idle decision has a strictly negative time cost.
        # This progress reward is NOT policy-invariant potential shaping;
        # evaluation must use actual safe berthing rather than return alone.
        current_distance=float(np.linalg.norm(self.ship[:2]-self.goal[:2])) if not done_unsafe else previous_distance
        current_heading_deg=float(abs(np.rad2deg(wrap(self.ship[2]-self.goal[2])))) if not done_unsafe else previous_heading_deg
        elapsed_fraction=(self.time-start_time)/max(self.cfg.decision_seconds,1e-8)
        reward=-.018*elapsed_fraction-.0015*energy/max(self.cfg.decision_seconds,1e-8)
        if self.cfg.reward_shaping:
            # Heading and position must improve together: a 1-deg heading
            # correction matters about as much as 1.5 m of centre progress.
            # Per-decision progress avoids the spurious stationary bonus of
            # discounted negative potentials in earlier versions.
            reward+=.08*(previous_distance-current_distance)
            reward+=.12*(previous_heading_deg-current_heading_deg)
            # Penalize uncorrected heading over the *remaining* approach;
            # this is intentionally not policy-invariant potential shaping.
            reward-=.0012*max(0.,current_heading_deg-3.)*elapsed_fraction
            # Slow down near the quay; allow sensible higher speeds offshore.
            speed=float(np.linalg.norm(self.ship[3:5]))
            if current_distance<50.:
                speed_limit=.045+(.10-.045)*np.clip(current_distance/50.,0.,1.)
                reward-=.035*(max(0.,speed-speed_limit)/.06)**2*elapsed_fraction
            if self.hull_gap()<50.:
                closing=max(0.,-(rot(self.ship[2])@self.ship[3:5])[1])
                reward-=.04*(max(0.,closing-.045)/.05)**2*elapsed_fraction
        if done_unsafe:reward-=16.
        if success:reward+=35.
        obs=self.observe() if np.isfinite(self.ship).all() else np.zeros((4,self.obs_dim),np.float32)
        state=self.critic_state() if np.isfinite(self.ship).all() else np.zeros(self.state_dim,np.float32)
        info=dict(success=bool(success),unsafe=bool(done_unsafe),reason=reason,
             first_success_recorded=self.monitor.first_success_time_s is not None,
             stationkeeping_complete=hold_done,
             time_s=self.time,position_error_m=float(np.linalg.norm(self.ship[:2]-self.goal[:2])),
             heading_error_deg=float(abs(np.rad2deg(wrap(self.ship[2]-self.goal[2])))),
             hull_gap_m=float(self.hull_gap()),max_approach_speed_mps=peak_near_speed,
             max_interaction_N=peak_force,applied_thrust_N=self.act.thrust.tolist(),
             push_normal_N=[float(diags.get(f'push{i}_normal_N',0)) for i in (1,2)],
             tow_tension_N=[float(diags.get(f'tow{i}_tension_N',0)) for i in (3,4)],
             push_heading_guard_interventions=[g.n_interventions for g in self.heading_guards],
             max_push_heading_error_deg=[g.max_abs_error_deg for g in self.heading_guards])
        return obs,state,float(reward),terminated,truncated,info
