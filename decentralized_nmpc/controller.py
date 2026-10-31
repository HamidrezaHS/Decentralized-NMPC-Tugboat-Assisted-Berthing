"""Decentralized receding-horizon MPC for tug-assisted KVLCC2 berthing.

Each instance is owned by ONE tug. Inputs contain public vessel measurements,
that tug's own state and actuator history, and immutable mission/attachment
geometry. No peer-tug observations, actions, messages, game equilibria, joint
force allocator or central optimizer are used at runtime.

The optimizer uses a nonlinear, reduced-order ship-prediction model (body
mass/drag linearized at the measurement, rigid-body coordinate transforms,
yaw moment at THIS tug's designated contact station, nonlinear collision and
speed constraints). The independently integrated high-fidelity coupled plant
remains the validation simulator; this is not a full-state 5-body NMPC.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize
from physical_v4.plant import rot, wrap
from model.ship import Ship


@dataclass(frozen=True)
class NMPCConfig:
    horizon_s: float = 90.0
    prediction_dt_s: float = 15.0
    max_iterations: int = 10
    max_fraction_step: float = 0.38
    max_offset_step_deg: float = 24.0
    # Parameter counts are fixed to two control knots (current + future).
    position_weight: float = 0.24
    heading_weight: float = 0.72
    lateral_speed_weight: float = 6.0
    yaw_rate_weight: float = 6.4
    motion_weight: float = 5.2
    effort_weight: float = 0.035
    slew_weight: float = 0.10
    constraint_weight: float = 22.0
    model_force_gain: float = 0.72
    residual_alpha: float = 0.25
    use_bias_estimator: bool = True
    disturbance_mode: str = "measured"  # none, residual, measured
    precision_radius_m: float = 25.0
    precision_speed_cap_mps: float = 0.030
    terminal_position_weight: float = 0.45
    terminal_counterforce_weight: float = 24.0  # local penalty for opposing needed berth-normal correction
    terminal_counterforce_near_boost: float = 3.0  # maintain terminal correction through the 2.5-m hold zone
    terminal_heading_weight: float = 0.08
    terminal_position_scale_m: float = 10.0
    terminal_position_gate_m: float = 25.0
    berth_position_tolerance_m: float = 2.5
    terminal_speed_weight: float = 0.06
    cross_track_weight: float = 0.34
    along_track_weight: float = 0.16
    near_berth_heading_boost: float = 2.1
    # Ground-track targets should never be corrected by subtracting ambient
    # wind/current velocities: hydrodynamics and the disturbance observer
    # already account for the corresponding accelerations.
    disturbance_feedforward_current: float = 0.0  # legacy, intentionally unused
    disturbance_feedforward_wind: float = 0.0     # legacy, intentionally unused
    anti_stall_window_s: float = 200.0
    anti_stall_min_progress_m: float = 1.5
    anti_stall_gain: float = 1.3
    guidance_close_radius_m: float = 38.0
    guidance_safe_lateral_speed_mps: float = 0.035
    far_desired_speed_mps: float = 0.085
    min_desired_speed_mps: float = 0.008
    # Explicit, mission-independent look-ahead and quay-tangent corridor.
    # Enabled ONLY for the decentralized proposed method, not the v2 baselines.
    mission_guidance: bool = False
    braking_acceleration_mps2: float = 0.000055
    along_corridor_margin_m: float = 26.0
    corridor_weight: float = 12.0
    predictive_stop_weight: float = 8.0
    overshoot_velocity_weight: float = 9.0
    along_velocity_weight: float = 7.0
    # Optional physical-actuator reserve for adverse measured environmental loads.
    # This lifts only the old *software* cap on push fractions (0.52), not
    # the physical propulsion limit (fraction 1.0) or contact safety guard.
    environmental_thrust_reserve: bool = False
    nominal_push_fraction_cap: float = 0.52
    max_push_fraction_cap: float = 0.90
    reserve_start_ratio: float = 0.65
    reserve_full_ratio: float = 1.00


@dataclass(frozen=True)
class TugObservation:
    """Precisely one public ship state plus own tug state and own actuation."""
    ship: np.ndarray          # x,y,heading,u,v,r: shared ship sensors
    own: np.ndarray           # x,y,heading,u,v,r: tug navigation system
    own_thrust: float         # measured own applied thruster force
    own_azimuth: float        # measured own applied azimuth
    time_s: float
    wind_mps: float = 0.0
    wind_to_rad: float = 0.0
    current_mps: float = 0.0
    current_to_rad: float = 0.0


def common_ship_observation(env, agent_id: int, measure_environment: bool = True,
                            wind_sensor_std_mps: float = 0., current_sensor_std_mps: float = 0.,
                            rng=None) -> TugObservation:
    i=int(agent_id)
    if not 0<=i<4: raise ValueError('agent index must be in 0..3')
    # These are emulated sensors (anemometer and current meter) on THIS tug,
    # not other tugs' states or commands. Read-only environmental field is
    # uniform in the simulator. Zero noise is the ideal-sensor assumption.
    if measure_environment:
        vw,dw,vc,dc=(float(v) for v in env.plant.env)
        if rng is not None:
            wx=np.array([vw*np.cos(dw),vw*np.sin(dw)])
            cx=np.array([vc*np.cos(dc),vc*np.sin(dc)])
            wx+=rng.normal(0,wind_sensor_std_mps,2)
            cx+=rng.normal(0,current_sensor_std_mps,2)
            vw=float(np.linalg.norm(wx));dw=float(np.arctan2(wx[1],wx[0]))
            vc=float(np.linalg.norm(cx));dc=float(np.arctan2(cx[1],cx[0]))
    else:
        vw=dw=vc=dc=0.
    return TugObservation(env.ship.copy(),env.tugs[i].copy(),
                          float(env.act.thrust[i]),float(env.act.azimuth[i]),float(env.time),
                          vw,dw,vc,dc)


class LocalNMPC:
    def __init__(self, agent_id: int, goal: np.ndarray, hull_body: np.ndarray,
                 station_xy_body: np.ndarray, free_side: int, max_force: float,
                 cfg: NMPCConfig | None = None):
        if not 0<=agent_id<4:raise ValueError('one of 4 local controllers required')
        self.i=int(agent_id)
        self.goal=np.asarray(goal,dtype=float).copy()
        self.hull=np.asarray(hull_body,dtype=float).copy()
        self.station=np.asarray(station_xy_body,dtype=float).copy()
        self.side=int(free_side)
        self.max_force=float(max_force)
        self.cfg=cfg or NMPCConfig()
        if self.cfg.disturbance_mode not in ('none','residual','measured'):
            raise ValueError('disturbance_mode must be none, residual or measured')
        self.model=Ship()
        self.last=np.array([0.,0.],float)  # physical fraction, offset degrees
        self.plan=np.zeros((2,2),float)
        self.last_ship=None
        self.last_time=None
        self.last_own_force=np.zeros(3)
        self.bias=np.zeros(3)
        self.last_report={}
        self.n_optimizations=0
        self.progress_anchor=None
        self.progress_time=None
        self.stall_factor=0.0
        self.environmental_load_away_N=0.0
        self.push_fraction_cap=self.cfg.nominal_push_fraction_cap
        # The mission corridor is inferred from public ship pose at first act.
        # No central route or peer observation is required.
        self.mission_start=None
        self.mission_axis=rot(-self.goal[2])
        self.mission_goal_along=float((self.mission_axis@self.goal[:2])[0])
        # This is an immutable berth reference, NOT a force allocator.
        self.quay_y=self.goal[1]+float(np.min((self.hull@rot(self.goal[2]).T)[:,1]))-6.

    def reset(self):
        self.last[:]=0.
        self.plan[:]=0.
        self.last_ship=None
        self.last_time=None
        self.last_own_force[:]=0.
        self.bias[:]=0.
        self.last_report={}
        self.progress_anchor=None
        self.progress_time=None
        self.stall_factor=0.0
        self.environmental_load_away_N=0.0
        self.push_fraction_cap=self.cfg.nominal_push_fraction_cap
        self.mission_start=None

    def _force_signature(self, fraction, offset_deg):
        """Estimated THIS tug's ship-body wrench, no peer force prediction."""
        normal=(-self.side if self.i<2 else self.side)
        theta=normal*np.pi/2.+np.deg2rad(offset_deg)
        F=self.max_force*self.cfg.model_force_gain*float(fraction)
        fx=F*.65*math.cos(theta)
        fy=F*math.sin(theta)
        # Local attachment yaw moment is not a generalized-force allocation.
        moment=self.station[0]*fy-self.station[1]*fx
        return np.array([fx,fy,moment])

    def _environmental_push_cap(self, ship, environment_wrench_body):
        """Individual push-tug authority bound from shared ship/environment data.

        Positive projected force points *away* from the globally y-low quay.
        The force difference is computed from the exact same hydrodynamic and
        wind force model as the measured-hydrodynamic prediction. It does NOT
        use another tug's telemetry, decisions or forces. This is a reserve
        allocator only in the sense of a local actuator ceiling, NOT a global
        force-allocation algorithm: the individual NMPC still selects its own
        force, and the physical actuator/controller constraints remain intact.
        """
        c=self.cfg
        self.environmental_load_away_N=0.0
        self.push_fraction_cap=float(c.nominal_push_fraction_cap)
        if not c.environmental_thrust_reserve or self.i>=2:
            return self.push_fraction_cap
        body=np.asarray(environment_wrench_body,dtype=float).reshape(3)
        inertial=rot(float(ship[2]))@body[:2]
        normal=rot(float(self.goal[2]))@np.array([0.,1.])
        away=float(max(0.,inertial@normal))
        self.environmental_load_away_N=away
        nominal_total=2.0*self.max_force*max(1e-6,c.nominal_push_fraction_cap)
        ratio=away/nominal_total
        denom=max(1.e-6,c.reserve_full_ratio-c.reserve_start_ratio)
        blend=float(np.clip((ratio-c.reserve_start_ratio)/denom,0.,1.))
        # Smoothstep avoids erratic on-off cap changes due to sensor noise.
        blend=blend*blend*(3.-2.*blend)
        self.push_fraction_cap=float(np.clip(
            c.nominal_push_fraction_cap+(c.max_push_fraction_cap-c.nominal_push_fraction_cap)*blend,
            0.,1.))
        return self.push_fraction_cap

    def _residual_observer(self, obs: TugObservation, invmass, hydroacc):
        """Infer an aggregate unmodeled acceleration from *ship sensors only*.

        Missing action information cannot be recovered separately; this is a
        bounded lumped disturbance estimate, not access to other controllers.
        """
        if not self.cfg.use_bias_estimator or self.cfg.disturbance_mode=='none' or self.last_ship is None:
            return
        dt=obs.time_s-self.last_time
        if dt<=0.1:return
        measured=(np.asarray(obs.ship)[3:6]-self.last_ship[3:6])/dt
        innovation=measured-hydroacc-invmass@self.last_own_force
        limit=np.array([.0009,.0009,2.0e-5])
        innovation=np.clip(innovation,-limit,limit)
        self.bias=(1-self.cfg.residual_alpha)*self.bias+self.cfg.residual_alpha*innovation

    def _update_stall_estimate(self, ship, time_s):
        """Detect lack of progress from public ship positioning alone.

        This modifies cost weights, not the commanded action or safety guards.
        Only past and current public vessel positions are used.
        """
        dist=float(np.linalg.norm(self.goal[:2]-ship[:2]))
        if self.progress_anchor is None:
            self.progress_anchor=dist
            self.progress_time=float(time_s)
            return
        elapsed=float(time_s)-self.progress_time
        if elapsed >= self.cfg.anti_stall_window_s:
            achieved=self.progress_anchor-dist
            # Detect low-progress equilibria also INSIDE the near-berth region.
            # The minimum required improvement is distance-scaled so gentle
            # final approaches are not spuriously treated as stalled.
            threshold=min(self.cfg.anti_stall_min_progress_m, max(.25,.02*dist))
            # Near the berth a VLCC can crawl at just 1--3 mm/s, improving
            # position by >0.25 m in 200 s while still being effectively
            # stuck 8--20 m short of its destination. Compare progress with
            # a conservative fraction of the *ground-referenced* local speed
            # request. Activate ONLY after measured low progress, when heading
            # and along-quay position are already aligned; no peer telemetry.
            heading_aligned=abs(wrap(ship[2]-self.goal[2]))<np.deg2rad(2.)
            along_near=abs(float((ship[:2]-self.goal[:2])@rot(self.goal[2])[:,0]))<7.
            if 3.5<dist<25. and heading_aligned and along_near:
                expected_speed=min(self.cfg.precision_speed_cap_mps,dist/720.)
                threshold=max(threshold,min(1.25,.38*expected_speed*elapsed))
            if dist>1.2*self.cfg.berth_position_tolerance_m and achieved<threshold:
                self.stall_factor=min(1.,self.stall_factor+0.40)
            else:
                self.stall_factor=max(0.,self.stall_factor-0.25)
            self.progress_anchor=dist
            self.progress_time=float(time_s)

    def _guidance(self, ship, obs: TugObservation | None = None):
        """Plan a GROUND-REFERENCED approach, independent of wind/current velocity.

        Wind is an air velocity, not a desired ground-track correction.
        Current effects are handled by hydrodynamic prediction and the public
        ship-motion residual observer. Velocity reference is inertial/earth-fixed.
        """
        e=self.goal[:2]-ship[:2]
        dist=float(np.linalg.norm(e))
        c=self.cfg
        desired_speed=min(c.far_desired_speed_mps, max(c.min_desired_speed_mps,dist/720.0))
        if c.precision_radius_m > 0 and dist < c.precision_radius_m:
            desired_speed=min(desired_speed,c.precision_speed_cap_mps)
        # Keep the approach under the safe low-speed range near the berth;
        # this is a reference, with the physical safety monitor still active.
        if dist < c.guidance_close_radius_m:
            desired_speed=min(desired_speed,c.guidance_safe_lateral_speed_mps)
        target_v=desired_speed*e/max(dist,1e-9)
        heading_error=wrap(self.goal[2]-ship[2])
        target_yawrate=float(np.clip(heading_error/420.0,-.00075,.00075))
        return target_v,target_yawrate,dist

    def _mission_geometry(self, ship):
        """Along-berth coordinate, target error, inertial velocity and corridor.

        For a rotated target heading these quantities rotate with the berth;
        the existing physical plant still assumes the quay is globally y-low.
        """
        xy=self.mission_axis@ship[:2]
        v_earth=rot(ship[2])@ship[3:5]
        v_axis=self.mission_axis@v_earth
        target_axis=self.mission_axis@self.goal[:2]
        err=float(target_axis[0]-xy[0])
        if self.mission_start is None:
            start=float(xy[0])
        else:
            start=float((self.mission_axis@self.mission_start)[0])
        lo=min(start,float(target_axis[0]))-self.cfg.along_corridor_margin_m
        hi=max(start,float(target_axis[0]))+self.cfg.along_corridor_margin_m
        return xy,v_axis,err,lo,hi

    def _mission_velocity_reference(self, ship):
        """Braking-aware GROUND velocity toward the berth coordinates.

        This reference does not cancel ocean currents from the ground track;
        the water-current forces enter through the dynamic prediction model.
        """
        c=self.cfg
        vref,desired_r,dist=self._guidance(ship)
        xyz,vaxis,err,lo,hi=self._mission_geometry(ship)
        # Smooth stopping-speed envelope: v^2 <= 2*a*distance.
        brake_speed=math.sqrt(max(0.,2.*c.braking_acceleration_mps2*abs(err)))
        along_ref=math.copysign(min(c.far_desired_speed_mps,brake_speed),err) if abs(err)>1e-8 else 0.
        # Blend inertial path guidance with braking-aware along-quay control.
        ref_axis=self.mission_axis@vref
        ref_axis[0]=float(np.clip(along_ref,-.085,.085))
        return self.mission_axis.T@ref_axis,desired_r,dist

    def _mission_cost(self, state, velocity, goal_distance, initial_dist):
        """Penalty for exiting a start--goal corridor or crossing the target.

        Cost is predictive, not a post-hoc unsafe termination condition.
        It cannot certify collision avoidance or guaranteed convergence.
        """
        c=self.cfg
        along,v_axis,err,lo,hi=self._mission_geometry(state)
        x=float(along[0]); vx=float(v_axis[0])
        # Predict inertial coast including physically slow VLCC stopping.
        speed=abs(vx)
        lookahead=min(500.,speed/max(c.braking_acceleration_mps2,1e-8))
        coast_x=x+.5*vx*lookahead
        out=max(0.,lo-x,x-hi)
        future_out=max(0.,lo-coast_x,coast_x-hi)
        # Signed velocity is essential: copysign(vx,err) discards vx's sign.
        # Penalise BOTH running away from the target and approaching faster
        # than the available stopping distance permits.
        toward=max(0.,math.copysign(1.,err)*vx) if abs(err)>1e-6 else 0.
        outward=max(0.,-math.copysign(1.,err)*vx) if abs(err)>1e-6 else abs(vx)
        effective_dist=max(0.,abs(err))
        refbound=math.sqrt(2.*c.braking_acceleration_mps2*effective_dist)
        overspeed=max(0.,toward-refbound)
        overshoot=max(0.,-err*vx) if abs(err)>1e-8 else speed*abs(err)
        return (c.corridor_weight*(out/20.)**2+
                c.predictive_stop_weight*(future_out/20.)**2+
                c.overshoot_velocity_weight*(overspeed/.035)**2+
                0.5*c.overshoot_velocity_weight*(outward/.035)**2+
                c.along_velocity_weight*(max(0.,overshoot)/1.5)**2)

    def _terminal_position_cost(self, distance):
        """Numerically relevant berth-scale end penalty with smooth far-field gate.

        Historically (distance/100 m)^2 became almost zero at 5--15 m,
        allowing a stable but incorrect near-berth equilibrium. A gate
        avoids exerting excessive influence during distant turning.
        """
        c=self.cfg
        d=float(distance)
        if c.terminal_position_weight<=0.:
            return 0.
        scale=max(c.terminal_position_scale_m,1e-3)
        gate_radius=max(c.terminal_position_gate_m,1e-3)
        gate=1./(1.+(d/gate_radius)**4)
        return c.terminal_position_weight*(d/scale)**2*gate

    def _terminal_counterforce_gate(self, ship):
        """Ship-measurement-only gate for persistent, low-speed terminal stall.

        No other tug states, commands or predicted actions are observed. The
        correction is dormant during turning, far-field travel, and ordinary
        closing. The tug is discouraged from counteracting berth-normal error
        only when progress has demonstrably stalled near its target.
        """
        c=self.cfg
        e=self.goal[:2]-ship[:2]
        berth_normal=rot(self.goal[2])[:,1]
        cross=float(e@berth_normal)
        along=float(e@rot(self.goal[2])[:,0])
        # v4.4 had an abrupt OFF switch for abs(cross) <= tolerance + 0.5 m.
        # The switch was at 3.0 m for a 2.5-m target: the controller stopped
        # discouraging counteracting force before the success band was reached,
        # and repeatedly moved back and forth across this threshold.
        # Fade the penalty out only near *zero* residual cross-track error.
        # A smooth gate stays active through the whole success tolerance.
        deadband=max(0.25,0.4*c.berth_position_tolerance_m)
        if abs(cross)<=deadband or abs(along)>7.0:
            return 0., np.zeros(2)
        heading_error=abs(wrap(self.goal[2]-ship[2]))
        near=np.clip((22.-abs(cross))/10.,0.,1.)
        aligned=np.clip((np.deg2rad(2.)-heading_error)/np.deg2rad(1.),0.,1.)
        # Smoothstep avoids an abrupt on/off transition at the deadband.
        alpha=float(np.clip((abs(cross)-deadband)/max(.5, c.berth_position_tolerance_m-deadband),0.,1.))
        error_gate=alpha*alpha*(3.-2.*alpha)
        ground_vel=rot(ship[2])@ship[3:5]
        moving_toward=ground_vel@berth_normal*np.sign(cross)
        slow=np.clip((.012-moving_toward)/.009,0.,1.)
        gate=float(near*aligned*slow*self.stall_factor*error_gate)
        return gate, berth_normal*np.sign(cross)

    def _predict_cost(self, opt, ship, initial, invmass, hydroacc, obs, record=False):
        c=self.cfg
        # 2 planning intervals: first command held for half horizon, then second.
        fractions=opt[[0,2]]
        offsets=opt[[1,3]]
        state=ship.copy()
        dt=c.prediction_dt_s
        n=max(2,int(round(c.horizon_s/dt)))
        guide_v,guide_r,initial_dist=self._guidance(ship, obs)
        total=0.0
        safety=0.0
        trajectory=[]
        for k in range(n):
            seg=0 if k<n//2 else 1
            f=self._force_signature(fractions[seg],offsets[seg])
            # Linearized MMG acceleration and nonlinear pose kinematics.
            a=hydroacc+invmass@f+self.bias
            a[:2]+=-.0010*(state[3:5]-ship[3:5])
            a[2]+=-.007*(state[5]-ship[5])
            state[3:6]+=dt*a
            state[5]=np.clip(state[5],-.007,.007)
            state[2]=wrap(state[2]+dt*state[5])
            vw=rot(state[2])@state[3:5]
            state[:2]+=dt*vw
            e=self.goal[:2]-state[:2]
            dist=float(np.linalg.norm(e))
            speed_ref,yaw_ref,_=(self._mission_velocity_reference(state)
                if c.mission_guidance else self._guidance(state, obs))
            psi_err=wrap(state[2]-self.goal[2])
            rel_goal=(state[:2]-self.goal[:2])@rot(-self.goal[2]).T
            along_err=float(rel_goal[0]); cross_err=float(rel_goal[1])
            phase=max(0., min(1., (80.-dist)/80.0))
            heading_gain=c.heading_weight*(1.+c.near_berth_heading_boost*phase)
            motion_gain=c.motion_weight*(1.+0.55*phase)
            stage=(c.position_weight*(1.+c.anti_stall_gain*self.stall_factor)*(dist/100.)**2+
                heading_gain*(psi_err/.35)**2+
                motion_gain*(1.+0.7*self.stall_factor)*np.sum(((vw-speed_ref)/.08)**2)+
                c.yaw_rate_weight*((state[5]-yaw_ref)/.00065)**2+
                c.cross_track_weight*(cross_err/25.)**2+
                c.along_track_weight*(along_err/35.)**2+
                c.effort_weight*fractions[seg]**2)
            # Physical hull geometry -> conservative low-point collision test.
            lo=self.hull@rot(state[2]).T
            clearance=float(np.min(lo[:,1])+state[1]-self.quay_y)
            approach=max(0.,-vw[1])
            safe_speed=.045 if clearance<45 else .095
            gap_short=max(0.,8.-clearance)
            # Reserve stopping margin (nonnegative, never penalize safe slowness).
            stopping_short=max(0.,approach*75.-max(0.,clearance-6.))
            bad=c.constraint_weight*(gap_short/8.)**2
            if clearance<45:
                bad+=c.constraint_weight*(max(0.,approach-safe_speed)/.05)**2
                bad+=.35*c.constraint_weight*(stopping_short/20.)**2
            safety+=bad
            progress_short=max(0., dist-(initial_dist-0.25*k))
            total+=(stage+bad+0.08*(progress_short/25.)**2)/n
            if c.mission_guidance:
                total+=self._mission_cost(state,vw,dist,initial_dist)/n
            if record:trajectory.append((float(state[0]),float(state[1]),float(state[2]),float(clearance)))
        # In a prolonged low-speed terminal stall, independent controllers
        # may otherwise simultaneously saturate in opposing normal directions.
        # Penalize ONLY this tug's force projected away from the observed
        # berth-normal correction. No centralized allocation is introduced.
        gate,needed_dir=self._terminal_counterforce_gate(ship)
        if gate>0.:
            counterforce_cost=0.
            for fraction,offset in zip(fractions,offsets):
                own_world=rot(ship[2])@self._force_signature(fraction,offset)[:2]
                oppose=max(0.,-float(own_world@needed_dir))
                norm_force=max(1.,self.max_force*c.model_force_gain)
                counterforce_cost+=(oppose/norm_force)**2
            # Extra influence is confined to the last few metres, where a
            # counteracting force can otherwise reverse the final closing
            # motion before the monitor completes its 5-second hold. The
            # geometric correction uses only this tug's estimated wrench.
            cross_error=abs(float((self.goal[:2]-ship[:2])@rot(self.goal[2])[:,1]))
            terminal_proximity=float(np.clip((6.-cross_error)/3.,0.,1.))
            local_weight=c.terminal_counterforce_weight*(1.+
                (c.terminal_counterforce_near_boost-1.)*terminal_proximity)
            total+=local_weight*gate*0.5*counterforce_cost
        # Azimuth reference is not a meaningful force direction at near-zero
        # thrust. Penalising its change at full strength creates an artificial
        # zero-thrust local optimum that can persist even as ship error grows.
        # The physical azimuth actuators below remain RATE-LIMITED by the
        # unchanged simulator; this weighting only affects NMPC cost.
        azimuth_activity_0=max(.02,(max(fractions[0],initial[0])/.52)**2)
        azimuth_activity_1=max(.02,(max(fractions[0],fractions[1])/.52)**2)
        slew=((fractions[0]-initial[0])/.25)**2+azimuth_activity_0*((offsets[0]-initial[1])/14.)**2
        slew+=.5*((fractions[1]-fractions[0])/.25)**2+.5*azimuth_activity_1*((offsets[1]-offsets[0])/14.)**2
        total+=c.slew_weight*slew
        # Terminal correction matters near berth, where average stage errors
        # otherwise underweight a small final offset or heading error.
        term_dist=float(np.linalg.norm(state[:2]-self.goal[:2]))
        if c.mission_guidance:
            total+=0.65*self._mission_cost(state,vw,term_dist,initial_dist)
        term_heading=wrap(state[2]-self.goal[2])
        goal_frame_vel=rot(-self.goal[2])@vw
        total+=self._terminal_position_cost(term_dist)
        total+=c.terminal_heading_weight*(term_heading/.35)**2
        total+=c.terminal_speed_weight*np.sum((goal_frame_vel/0.05)**2)
        if record:return total,safety,trajectory
        return float(total)

    def act(self, observation: TugObservation) -> tuple[np.ndarray,dict]:
        """Solve ONE local bounded NMPC problem; action for this tug alone."""
        ship=np.asarray(observation.ship,dtype=float).copy()
        own=np.asarray(observation.own,dtype=float)
        if ship.shape!=(6,) or own.shape!=(6,):raise ValueError('Expected 3DOF ship and own tug states')
        if not np.all(np.isfinite(ship)) or not np.all(np.isfinite(own)):
            raise ValueError('Nonfinite sensor observation')
        # MMG ship acceleration with no externally applied force, from public
        # ship measurement. Does not read private peer states or thrusts.
        self.model.external_tau=np.zeros(3)
        # Feed-forward environmental dynamics only when THIS tug carries an
        # environmental sensor; no access to peer states or true simulator
        # wind/current through the optimizer.
        vw=float(observation.wind_mps) if self.cfg.disturbance_mode=='measured' else 0.
        dw=float(observation.wind_to_rad) if self.cfg.disturbance_mode=='measured' else 0.
        vc=float(observation.current_mps) if self.cfg.disturbance_mode=='measured' else 0.
        dc=float(observation.current_to_rad) if self.cfg.disturbance_mode=='measured' else 0.
        A,C=self.model.ship_matrices(ship[3],ship[4],ship[5],ship[2],Vw=vw,psi_w=dw,Vc=vc,psi_c=dc)
        full=np.linalg.solve(A,C.flatten())
        hydroacc=full[:3]
        invmass=np.linalg.inv(A[:3,:3])
        # Disturbance load estimate relative to a no-environment ship state.
        # The actual ship model and physical thrust are unchanged.
        if self.i<2 and self.cfg.environmental_thrust_reserve and (vw>0. or vc>0.):
            _,calm_C=self.model.ship_matrices(ship[3],ship[4],ship[5],ship[2])
            environment_wrench=(C.flatten()-calm_C.flatten())[:3]
        else:
            environment_wrench=np.zeros(3)
        self._environmental_push_cap(ship,environment_wrench)
        self._residual_observer(observation,invmass,hydroacc)
        self._update_stall_estimate(ship,observation.time_s)
        if self.mission_start is None:
            self.mission_start=ship[:2].copy()
        dt=self.cfg.prediction_dt_s
        # The own-tug heading sensor creates a purely LOCAL guard against
        # push-hull misalignment; no other tug is observed or queried.
        own_heading_error=abs(wrap(own[2]-ship[2]+self.side*np.pi/2))
        heading_deg=float(np.rad2deg(own_heading_error))
        role_limit=(self.push_fraction_cap if self.i<2 else .65)
        if self.i<2 and heading_deg>22.:
            role_limit=min(role_limit,max(.08,role_limit*(30.-heading_deg)/8.))
        bounds=[]
        for knot in range(2):
            prev=self.last if knot==0 else self.plan[1]
            bounds.extend([(min(role_limit,max(0.,prev[0]-self.cfg.max_fraction_step)),
                            min(role_limit,prev[0]+self.cfg.max_fraction_step)),
                           # When the thrust command is effectively zero,
                           # do not lock the virtual direction to its old
                           # angle. The actuator enforces physical rate limits.
                           # A future knot uses slew cost relative to the
                           # optimized first knot, not the OLD future plan.
                           ((-20.,20.) if (knot>0 or prev[0]<.06) else
                            (max(-20.,prev[1]-self.cfg.max_offset_step_deg),
                             min(20.,prev[1]+self.cfg.max_offset_step_deg)))])
        initial=self.last.copy()
        # Each controller retains its own warm-start trajectory.
        warm=self.plan.flatten().copy()
        warm=np.array([float(np.clip(v,*b)) for v,b in zip(warm,bounds)])
        fn=lambda x:self._predict_cost(x,ship,initial,invmass,hydroacc,observation)
        # Multiplicative fraction/azimuth dynamics create a stationary saddle
        # at exactly zero thrust. Multistart bounded warm-starts avoid a false
        # local optimum that permanently switches a tug off.
        seeds=[warm]
        for off in (-17.,0.,17.):
            seed=warm.copy()
            seed[0]=float(np.clip(max(.18,seed[0]),*bounds[0]))
            seed[1]=float(np.clip(off,*bounds[1]))
            seed[2]=float(np.clip(seed[0],*bounds[2]))
            seed[3]=float(np.clip(off,*bounds[3]))
            seeds.append(seed)
        warm=min(seeds,key=fn)
        res=minimize(fn,warm,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':self.cfg.max_iterations,'ftol':1e-4,'maxls':8})
        solution=res.x if np.all(np.isfinite(res.x)) and np.isfinite(res.fun) else warm
        cost=float(fn(solution))
        # Reject an out-of-bounds or worse optimizer iterate; safe fallback.
        if cost>float(fn(warm))+1e-8:solution=warm;cost=float(fn(warm))
        command=np.array([float(solution[0]),float(solution[1])])
        self.plan=np.reshape(solution,(2,2)).copy()
        self.last=command.copy()
        self.last_ship=ship.copy()
        self.last_time=float(observation.time_s)
        self.last_own_force=self._force_signature(command[0],command[1])
        self.n_optimizations+=1
        info={'agent':self.i+1,'success_solver':bool(res.success),'iterations':int(res.nit),
              'objective':cost,'fraction':command[0],'offset_deg':command[1],
              'residual_acceleration_norm':float(np.linalg.norm(self.bias)),
              'observation_time_s':observation.time_s,
              'own_heading_error_deg':heading_deg,
              'own_thrust_N':float(observation.own_thrust),
              'wind_sensor_mps':vw,'current_sensor_mps':vc,
              'disturbance_mode':self.cfg.disturbance_mode,
              'stall_factor':float(self.stall_factor),
              'estimated_away_environment_force_N':float(self.environmental_load_away_N),
              'push_fraction_cap':float(self.push_fraction_cap),
              'terminal_counterforce_gate':float(self._terminal_counterforce_gate(ship)[0]),
              'mission_guidance':bool(self.cfg.mission_guidance),
              'ground_speed_reference_mps':float(np.linalg.norm(self._guidance(ship)[0]))}
        self.last_report=info
        return np.array([2*command[0]-1,command[1]/20.],dtype=np.float32),info
