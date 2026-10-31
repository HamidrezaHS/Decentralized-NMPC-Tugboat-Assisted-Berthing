"""Independent, high-rate, physically actuated push-tug heading supervisor.

Guard is called once per PHYSICAL INTEGRATION substep (not once per NMPC
planning interval). It reads only shared ship yaw/heading, own tug yaw/heading,
and this tug's intended propulsion and yaw commands. No peer information or
state overwrites. Actuator rate limits and the original hard safety monitor
remain in effect. This is NOT a mathematical guarantee of collision avoidance
or heading invariance. If actuator authority is insufficient, the hard monitor
still stops with UNSAFE and the failed trajectory must be reported.
"""
from dataclasses import dataclass
import numpy as np
from .plant import wrap


@dataclass(frozen=True)
class PushHeadingGuardConfig:
    engage_deg: float = 11.0
    full_deg: float = 19.0
    thrust_cutback_deg: float = 24.0
    hard_monitor_deg: float = 30.0
    lookahead_s: float = 3.0
    gain_p_Nm_rad: float = 24e6
    gain_d_Nm_s_rad: float = 32e6
    max_assist_azimuth_deg: float = 28.0
    min_thrust_fraction: float = 0.11
    high_risk_max_thrust_fraction: float = 0.30


class PushHeadingGuard:
    def __init__(self, side, max_thrust, stern_thruster_x_m, max_yaw_moment_Nm,
                 cfg: PushHeadingGuardConfig | None = None):
        self.side=int(side)
        self.max_thrust=float(max_thrust)
        self.stern_x=float(stern_thruster_x_m)
        self.max_yaw=float(max_yaw_moment_Nm)
        self.cfg=cfg or PushHeadingGuardConfig()
        self.n_interventions=0
        self.max_abs_error_deg=0.0
        self.max_abs_predicted_error_deg=0.0

    def filter(self, ship, own, thrust, azimuth, yaw_moment):
        """Return feasible requested setpoints for the existing actuator bank.

        The angular error is signed (desired bow-inboard tug heading minus
        current tug heading); future error uses the measured relative yaw rate.
        """
        c=self.cfg
        desired=ship[2]-self.side*np.pi/2
        error=wrap(desired-own[2])
        error_rate=float(ship[5]-own[5])
        predicted=error+c.lookahead_s*error_rate
        current_deg=float(abs(np.rad2deg(error)))
        predicted_deg=float(abs(np.rad2deg(predicted)))
        self.max_abs_error_deg=max(self.max_abs_error_deg,current_deg)
        self.max_abs_predicted_error_deg=max(self.max_abs_predicted_error_deg,predicted_deg)
        risk=max(current_deg,predicted_deg)
        if risk < c.engage_deg:
            return float(thrust),float(azimuth),float(yaw_moment),False
        blend=float(np.clip((risk-c.engage_deg)/(c.full_deg-c.engage_deg),0,1))
        # Correct in the direction of the predicted error when it is diverging;
        # otherwise bring heading and yaw rate back with conventional PD action.
        desired_yaw=(c.gain_p_Nm_rad*error+c.gain_d_Nm_s_rad*error_rate)
        if abs(predicted)>abs(error) and np.sign(predicted)!=np.sign(desired_yaw):
            desired_yaw+=0.25*c.gain_p_Nm_rad*(predicted-error)
        # Stern thruster x < 0: positive yaw assistance needs negative azimuth.
        desired_steer=(-np.sign(desired_yaw)*np.deg2rad(c.max_assist_azimuth_deg)
                       if abs(desired_yaw)>1e-6 else float(azimuth))
        steer=(1-blend)*float(azimuth)+blend*desired_steer
        steer=float(np.clip(steer,-np.deg2rad(35.),np.deg2rad(35.)))
        demanded_thrust=max(0.,float(thrust))
        if risk>=c.thrust_cutback_deg:
            # Reduce destabilizing contact/propulsion yaw moment; never reverse
            # a push tug or artificially erase the physical contact force.
            frac=float(np.clip((c.hard_monitor_deg-risk)/
                    (c.hard_monitor_deg-c.thrust_cutback_deg),0,1))
            limit=self.max_thrust*(c.min_thrust_fraction+
                         (c.high_risk_max_thrust_fraction-c.min_thrust_fraction)*frac)
            demanded_thrust=min(demanded_thrust,limit)
        # Propulsion yaw moment is applied at the stern and competes with the
        # limited differential-azimuth yaw actuator. Compensate its contribution.
        propulsion_yaw=self.stern_x*demanded_thrust*np.sin(steer)
        corrective=float(np.clip(desired_yaw-propulsion_yaw,-self.max_yaw,self.max_yaw))
        yaw=float(np.clip((1-blend)*float(yaw_moment)+blend*corrective,
                          -self.max_yaw,self.max_yaw))
        self.n_interventions+=1
        return demanded_thrust,steer,yaw,True
