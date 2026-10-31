"""Decentralized partially observed dynamic-game controller (no RL, no allocator).

Each tug is a separate object receiving ONLY a TugObservation and its own action
history. The ship-motion residual (smoothed numerical derivative) serves as a
belief about the aggregate effect of unobserved other controllers and disturbances.

A local 1-ship/1-tug coupled DAE provides counterfactual *marginal* accelerations
for a finite set of own feasible controls. The agent performs a myopic receding-
horizon best response, conditional on its own disturbance belief. All evaluations
are private; there is NO exchange of other agents' actions or states.

This is an approximate, belief-based game response. It is NOT an equilibrium
solver or proven exact potential game; safety is NOT formally guaranteed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from model.ship import Ship
from model.tug import Tug
from simulation.simulator import Sim
from .types import TugObservation, TugAction, wrap
from .tow_safety import lead_error, heading_error, predictive_lead_errors


@dataclass
class AgentConfig:
    decision_period_s: float = 10.
    lookahead_s: float = 90.
    delta_rate_deg_s: float = .35
    max_azimuth_deviation_deg: float = 35.
    force_fraction_grid: tuple[float, ...] = (0., .15, .35, .55, .75, 1.)
    azimuth_offset_grid_deg: tuple[float, ...] = (-30., -15., 0., 15., 30.)
    accel_smoothing_s: float = 8.
    desired_speed_mps: float = .17
    safety_near_gap_m: float = 45.
    max_near_speed_mps: float = .095
    effort_cost: float = .03
    action_change_cost: float = .014
    improvement_tolerance: float = 1e-8
    tow_rope_lead_limit_deg: float = 50.
    tow_heading_limit_deg: float = 75.
    tow_prediction_seconds: float = 25.
    tow_lead_cost: float = 45.
    tow_heading_cost: float = 10.
    tow_angular_rate_cost: float = 1.5
    tow_recovery_steering_cost: float = 80.
    tow_recovery_rate_horizon_s: float = 22.


class LocalGameAgent:
    """One independent player in a partially observed general-sum dynamic game."""
    def __init__(self, agent_id: int, mode: str, force_limit_N: float,
                 config: AgentConfig | None = None, env=(0., 0., 0., 0.),
                 free_side: int = 1):
        if mode not in ('push', 'tow'): raise ValueError('mode must be push or tow')
        if force_limit_N <= 0: raise ValueError('force limit must be positive')
        if free_side not in (-1, 1): raise ValueError('free_side must be +1 or -1')
        self.free_side = free_side
        self.id = int(agent_id)
        self.mode = mode
        self.max_force = float(force_limit_N)
        self.cfg = config or AgentConfig()
        self.env = tuple(float(x) for x in env)
        self.base_azimuth = 0. if mode == 'push' else np.pi
        self.action = TugAction(0., self.base_azimuth)
        self.last_decision_time = -np.inf
        self.last_observed_time = None
        self.last_observed_earth = None
        self.last_observed_yaw = None
        self.measured_acc = np.zeros(3)
        self.proposals_evaluated = 0
        self.decision_count = 0
        self.last_cost = float('nan')
        self.last_predicted_approach_speed = float('nan')
        self.last_tow_lead_error_deg = float('nan')
        self.rejected_tow_actions = 0
        self.tow_no_feasible_action = False
        # Crucial: this surrogate contains ONE tug, *never* the four-tug plant.
        self.solo_ship = Ship()
        self.solo_tug = Tug()
        self.solo_sim = Sim(dt=self.cfg.decision_period_s, MainShip=self.solo_ship,
             push_Tugs=[self.solo_tug] if mode=='push' else [],
             tow_Tugs=[self.solo_tug] if mode=='tow' else [],
             delta_rate=np.deg2rad(self.cfg.delta_rate_deg_s),
             enable_tug_wind_shadow=True)

    def observe(self, obs: TugObservation) -> None:
        """Update own ship acceleration estimate from two ship-only observations."""
        vel = np.asarray(obs.ship_earth_velocity, dtype=float)
        yaw = obs.ship_body_velocity[2]
        if self.last_observed_time is not None:
            dt = obs.time - self.last_observed_time
            if dt <= 0: raise ValueError('Time must increase for independent observer')
            derivative = np.array([*( (vel - self.last_observed_earth) / dt),
                                    (yaw - self.last_observed_yaw) / dt])
            # Low-pass filtering of observable kinematics (no plant-force access).
            alpha = 1. - np.exp(-dt/max(.01, self.cfg.accel_smoothing_s))
            self.measured_acc += alpha * (np.clip(derivative,[-.1,-.1,-.002],[.1,.1,.002])-self.measured_acc)
        self.last_observed_time = obs.time
        self.last_observed_earth = vel.copy()
        self.last_observed_yaw = yaw

    def _solo_response(self, obs: TugObservation, action: TugAction):
        u,v,r = obs.ship_body_velocity
        psi = obs.ship_pose[2]
        own = np.asarray(obs.own_tug_state, dtype=float)
        args = dict(ship_states=[u,v,r,psi,*self.env],
                    push_tug_states=[own] if self.mode=='push' else [],
                    tow_tug_states=[own] if self.mode=='tow' else [],
                    control_inputs=[(action.thrust,action.azimuth)],
                    delta_list=[obs.own_applied_azimuth])
        solution, _, sizes, _ = self.solo_sim.solve_one_step(**args)
        ship_block, pb, tb = Sim.split_solution(solution,sizes,1 if self.mode=='push' else 0)
        # ship block [u_dot,v_dot,r_dot,earth_x_ddot,earth_y_ddot]
        return np.array([ship_block[3],ship_block[4],ship_block[2]],dtype=float), (
            None if self.mode=='push' else (float(tb[0][2]),float(tb[0][5])))

    def _solo_acceleration(self, obs: TugObservation, action: TugAction) -> np.ndarray:
        return self._solo_response(obs,action)[0]

    def _tow_candidate(self, obs, response, baseline_yaw_acc):
        """Predict local rope heading. No other-tug observation is used.

        A hard action-set screen is imposed at a short prediction horizon.
        This is NOT a mathematically certified forward-invariant safety filter.
        """
        if self.mode != 'tow': return True, 0.
        own=obs.own_tug_state
        ship_h=obs.ship_pose[2]
        ship_r=obs.ship_body_velocity[2]
        tug_h=own[5];tug_r=own[4]
        theta_dot,theta=own[6],own[7]
        tug_rdot,theta_ddot=response
        lead_now=lead_error(ship_h,tug_h,theta,self.free_side)
        head_now=heading_error(ship_h,tug_h,self.free_side)
        self.last_tow_lead_error_deg=np.rad2deg(lead_now)
        lead_end,head_end=predictive_lead_errors(
            ship_h,ship_r,tug_h,tug_r,theta,theta_dot,tug_rdot,theta_ddot,
            baseline_yaw_acc,self.free_side,self.cfg.tow_prediction_seconds)
        lead_limit=np.deg2rad(self.cfg.tow_rope_lead_limit_deg)
        head_limit=np.deg2rad(self.cfg.tow_heading_limit_deg)
        feasible=(abs(lead_end)<=lead_limit and abs(head_end)<=head_limit)
        # If already outside domain, allow only candidates predicting a
        # strictly decreasing violation, to retain a possible recovery action.
        violation_now=max(0.,abs(lead_now)-lead_limit)+max(0.,abs(head_now)-head_limit)
        violation_end=max(0.,abs(lead_end)-lead_limit)+max(0.,abs(head_end)-head_limit)
        if violation_now>0 and violation_end < violation_now-1e-6:feasible=True
        extra=(self.cfg.tow_lead_cost*(lead_end/lead_limit)**2
              +self.cfg.tow_heading_cost*(head_end/head_limit)**2
              +self.cfg.tow_angular_rate_cost*
               ((tug_r+theta_dot-ship_r)/np.deg2rad(.15))**2)
        return feasible,float(extra)

    def _cost(self, obs: TugObservation, action: TugAction,
              baseline_acc: np.ndarray, candidate_acc: np.ndarray,
              current_acc: np.ndarray) -> tuple[float,float]:
        c = self.cfg
        H = c.lookahead_s
        x,y,psi = obs.ship_pose
        ue,ve = obs.ship_earth_velocity
        r = obs.ship_body_velocity[2]
        goal = np.asarray(obs.target_pose[:2])
        err = goal-np.array([x,y])
        remaining=float(np.linalg.norm(err))
        # Belief = observed aggregate acceleration + marginal acceleration
        # of *own* proposed action relative to currently applied own action.
        acc = baseline_acc + np.clip(candidate_acc-current_acc,
                    [-.006,-.006,-.00008],[.006,.006,.00008])
        pred_velocity = np.array([ue,ve])+H*acc[:2]
        pred_position = np.array([x,y])+H*np.array([ue,ve])+.5*H**2*acc[:2]
        pred_r = r+H*acc[2]
        pred_psi = wrap(psi+H*r+.5*H**2*acc[2])
        desired_velocity = err/max(180.,remaining/max(c.desired_speed_mps,1e-7))
        if remaining < 20.:
            desired_velocity *= min(1.,remaining/30.)
        psi_error = wrap(obs.target_pose[2]-psi)
        r_desired = float(np.clip(psi_error/180.,-np.deg2rad(.1),np.deg2rad(.1)))
        objective = (np.sum(((pred_position-goal)/55.)**2)
                   + 4.0*np.sum(((pred_velocity-desired_velocity)/.13)**2)
                   + 2.0*(wrap(pred_psi-obs.target_pose[2])/np.deg2rad(8))**2
                   + 3.0*((pred_r-r_desired)/np.deg2rad(.13))**2)
        # Berth normal is set by desired target displacement and ship heading.
        norm=np.array([-np.sin(obs.target_pose[2]),np.cos(obs.target_pose[2])]);
        if float((goal-np.array([x,y]))@norm)<0: norm=-norm
        approach=float(pred_velocity@norm)
        # Quay proximity uses gap observation. It may be unsafe to treat as exact.
        allowed = c.max_near_speed_mps if obs.near_wall_gap_m<c.safety_near_gap_m else c.desired_speed_mps
        objective += 35.*(max(0.,approach-allowed)/.08)**2
        if obs.near_wall_gap_m<25.:
            objective += 30.*(max(0.,-obs.near_wall_gap_m)/5.)**2
        objective += c.effort_cost*(action.thrust/self.max_force)**2
        objective += c.action_change_cost*((action.thrust-self.action.thrust)/self.max_force)**2
        objective += .004*(wrap(action.azimuth-self.base_azimuth)/np.deg2rad(35.))**2
        return float(objective),approach

    def decide(self, obs: TugObservation) -> TugAction:
        """Locally computed approximate best response; no reference to other agents."""
        if obs.time-self.last_decision_time<self.cfg.decision_period_s-1e-7:
            return self.action
        current_action = TugAction(obs.own_applied_thrust,obs.own_applied_azimuth)
        try:
            current_acc = self._solo_acceleration(obs,current_action)
        except (np.linalg.LinAlgError,ValueError,FloatingPointError):
            self.last_decision_time = obs.time
            self.action = TugAction(0.,obs.own_applied_azimuth)
            self.last_cost = np.inf
            return self.action
        # Predict change on top of observed acceleration. At t=0, ship acceleration
        # is unknown: use solo baseline as conservative initial estimate.
        base_acc = self.measured_acc if self.last_observed_time is not None and obs.time>0 else current_acc
        force_candidates = np.unique(np.clip([*self.cfg.force_fraction_grid,
                     self.action.thrust/self.max_force],0.,1.)) * self.max_force
        angles=np.deg2rad(self.cfg.azimuth_offset_grid_deg)+self.base_azimuth
        angles=np.unique(np.r_[angles,self.action.azimuth])
        limit=np.deg2rad(self.cfg.max_azimuth_deviation_deg)
        best = self.action
        best_cost = np.inf
        best_approach = float('nan')
        self.tow_no_feasible_action=False
        for T in force_candidates:
            for angle in angles:
                if abs(wrap(angle-self.base_azimuth))>limit+1e-8:continue
                candidate=TugAction(float(T),float(angle))
                try:
                    candidate_acc,tow_response=self._solo_response(obs,candidate)
                except (np.linalg.LinAlgError,ValueError,FloatingPointError):continue
                self.proposals_evaluated+=1
                feasible,tow_penalty=self._tow_candidate(obs,tow_response,base_acc[2])
                if not feasible:
                    self.rejected_tow_actions+=1
                    continue
                cost, approach=self._cost(obs,candidate,base_acc,candidate_acc,current_acc)
                cost+=tow_penalty
                if self.mode=='tow':
                    own=obs.own_tug_state
                    h=heading_error(obs.ship_pose[2],own[5],self.free_side)
                    l=lead_error(obs.ship_pose[2],own[5],own[7],self.free_side)
                    relative_yaw=own[4]-obs.ship_body_velocity[2]
                    # Corrective propeller azimuth: tug bows face the ship;
                    # delta above pi produces NEGATIVE yaw moment in model.
                    steer=np.clip(h+relative_yaw*self.cfg.tow_recovery_rate_horizon_s+.35*l,
                                  -np.deg2rad(30),np.deg2rad(30))
                    needed=wrap(self.base_azimuth+steer-candidate.azimuth)
                    # Regulate heading via azimuth rather than plant-state hacks.
                    cost+=self.cfg.tow_recovery_steering_cost*(needed/np.deg2rad(30))**2
                    if abs(steer)>np.deg2rad(8):
                        cost+=25.*max(0.,.15-candidate.thrust/self.max_force)**2
                if cost<best_cost-self.cfg.improvement_tolerance:
                    best_cost=cost;best=candidate;best_approach=approach
        if not np.isfinite(best_cost):
            # Do not silently accept a known-infeasible action. Zero thrust is
            # a failure indication, not a guarantee that momentum will vanish.
            self.tow_no_feasible_action=True
            best=TugAction(0.,obs.own_applied_azimuth)
        self.action=best
        self.last_cost=best_cost
        self.last_predicted_approach_speed=best_approach
        self.last_decision_time=obs.time
        self.decision_count+=1
        return best

    def prepare_equilibrium_bids(self, obs: TugObservation, public_acc: np.ndarray):
        """Privately compute unilateral action catalog for ONE frozen stage game.

        Outbound protocol subsequently shares only the *selected* bid, never
        own tug state, DAE parameters or another controller's action list.
        """
        from .equilibrium import ActionBid
        observed_action=TugAction(obs.own_applied_thrust,obs.own_applied_azimuth)
        current_acc=self._solo_acceleration(obs,observed_action)
        forces=np.unique(np.clip([*self.cfg.force_fraction_grid,
            self.action.thrust/self.max_force],0.,1.))*self.max_force
        angles=np.unique(np.r_[np.deg2rad(self.cfg.azimuth_offset_grid_deg)
                                +self.base_azimuth,self.action.azimuth,
                                obs.own_applied_azimuth])
        limit=np.deg2rad(self.cfg.max_azimuth_deviation_deg)
        bids=[]
        for T in forces:
            for angle in angles:
                if abs(wrap(angle-self.base_azimuth))>limit+1e-8:continue
                a=TugAction(float(T),float(angle))
                try:
                    acc,response=self._solo_response(obs,a)
                except (np.linalg.LinAlgError,ValueError,FloatingPointError):
                    continue
                self.proposals_evaluated+=1
                ok,penalty=self._tow_candidate(obs,response,public_acc[2])
                if not ok:
                    self.rejected_tow_actions+=1
                    continue
                # Same bounded marginal acceleration surrogate as v2.
                marginal=np.clip(acc-current_acc,[-.006,-.006,-.00008],
                                                  [.006,.006,.00008])
                local=(self.cfg.effort_cost*(a.thrust/self.max_force)**2
                       +self.cfg.action_change_cost*
                         ((a.thrust-self.action.thrust)/self.max_force)**2
                       +.004*(wrap(a.azimuth-self.base_azimuth)
                                       /np.deg2rad(35.))**2
                       +penalty)
                if self.mode=='tow':
                    own=obs.own_tug_state
                    h=heading_error(obs.ship_pose[2],own[5],self.free_side)
                    l=lead_error(obs.ship_pose[2],own[5],own[7],self.free_side)
                    relative_yaw=own[4]-obs.ship_body_velocity[2]
                    steer=np.clip(h+relative_yaw*self.cfg.tow_recovery_rate_horizon_s
                                  +.35*l,-np.deg2rad(30),np.deg2rad(30))
                    needed=wrap(self.base_azimuth+steer-a.azimuth)
                    local+=self.cfg.tow_recovery_steering_cost*(needed/np.deg2rad(30))**2
                    if abs(steer)>np.deg2rad(8):
                        local+=25.*max(0.,.15-a.thrust/self.max_force)**2
                bids.append(ActionBid(a,tuple(map(float,marginal)),float(local)))
        # The closed-loop state may be outside every local admissible envelope.
        # Do NOT invent a feasible action or falsely claim Nash equilibrium.
        self.tow_no_feasible_action=(len(bids)==0)
        return bids
