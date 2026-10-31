"""Sensor-facing contract: one hull attachment point + own tug navigation.

The simulator uses hidden vessel truth ONLY to synthesize a point-tracker
measurement. This is not a complete-vessel-state observation. At deployment a
real point tracker and tug navigation system must provide these measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from physical_v4.plant import rot


def _v2(x, name):
    v = np.asarray(x, dtype=float).copy()
    if v.shape != (2,) or not np.isfinite(v).all():
        raise ValueError(f'{name} must be a finite two-dimensional vector')
    v.setflags(write=False)
    return v


def _v6(x, name):
    v = np.asarray(x, dtype=float).copy()
    if v.shape != (6,) or not np.isfinite(v).all():
        raise ValueError(f'{name} must be a finite six-dimensional vector')
    v.setflags(write=False)
    return v


@dataclass(frozen=True)
class ContactObservation:
    """Only information available to an individual tug online.

    point_xy/point_velocity_xy: earth-referenced position/velocity of this
        tug's assigned vessel attachment point measured by its own tracker.
    own: tug x/y/heading/surge/sway/yaw rate (its own IMU/GNSS).
    tip_xy/tip_velocity_xy: tug-side attachment point from own navigation.
    own_thrust_N/own_azimuth_rad: *realized* own actuator states.
    goal_point_xy: fixed mission configuration, computed before deployment.

    No full vessel pose, vessel heading, other agents, ship center, hull outline,
    environmental truth or simulation handle is in this object.
    """
    point_xy: np.ndarray
    point_velocity_xy: np.ndarray
    own: np.ndarray
    tip_xy: np.ndarray
    tip_velocity_xy: np.ndarray
    own_thrust_N: float
    own_azimuth_rad: float
    goal_point_xy: np.ndarray
    time_s: float
    role: str

    def __post_init__(self):
        for k in ('point_xy','point_velocity_xy','tip_xy','tip_velocity_xy','goal_point_xy'):
            object.__setattr__(self,k,_v2(getattr(self,k),k))
        object.__setattr__(self,'own',_v6(self.own,'own'))
        if self.role not in ('push','tow'):
            raise ValueError('role must be push or tow')
        for k in ('own_thrust_N','own_azimuth_rad','time_s'):
            if not np.isfinite(getattr(self,k)):
                raise ValueError(f'{k} must be finite')

    @property
    def relative_tip_xy(self):
        return self.tip_xy-self.point_xy

    @property
    def relative_tip_velocity_xy(self):
        return self.tip_velocity_xy-self.point_velocity_xy

    def policy_vector(self) -> np.ndarray:
        """A fixed 19-component vector suitable for future IL policies.

        World frame is fixed by the quay/mission; no vessel attitude is included.
        Role is one-hot. Does not include or depend on other agents.
        """
        vec=np.r_[(self.goal_point_xy-self.point_xy)/200.,
                  self.point_velocity_xy/.25,
                  self.own[:2]/250.,
                  np.sin(self.own[2]),np.cos(self.own[2]),
                  self.own[3:5]/.7,self.own[5]/.03,
                  self.relative_tip_xy/50.,self.relative_tip_velocity_xy/.5,
                  self.own_thrust_N/1.e6,self.own_azimuth_rad/.7,
                  1. if self.role=='push' else 0.,1. if self.role=='tow' else 0.]
        assert vec.shape==(19,),vec.shape
        return np.clip(vec,-20.,20.).astype(np.float32)


def contact_goal(goal_xypsi: np.ndarray, local_station_xy: np.ndarray) -> np.ndarray:
    """Offline mission design: desired absolute position of ONE hull point.

    This preconfigured coordinate is NOT a live ship pose measurement.
    """
    g=np.asarray(goal_xypsi,float);r=np.asarray(local_station_xy,float)
    if g.shape!=(3,) or r.shape!=(2,) or not np.isfinite(g).all() or not np.isfinite(r).all():
        raise ValueError('goal must be 3D pose and station must be 2D geometry')
    return g[:2]+rot(g[2])@r


def contact_observation(env, agent_id: int, *,
                        position_noise_m=0., velocity_noise_mps=0., rng=None):
    """Simulator-only sensor driver, where vessel state is used *internally*.

    Returned object carries no vessel pose/velocity. Noise is independent per
    tug and is applied only to the sensed contact point, never to plant truth.
    Do NOT pass `env` to a controller; pass only this returned dataclass.
    """
    i=int(agent_id)
    if i not in (0,1,2,3):raise ValueError('agent id must be 0..3')
    if position_noise_m<0 or velocity_noise_mps<0:raise ValueError('noise std must be nonnegative')
    ship=env.ship
    tug=env.tugs[i]
    station=env.plant.attach_local[i]
    point=ship[:2]+rot(ship[2])@station
    speed=env.plant.velocity_at(ship,station)
    tip_station=np.array([env.plant.ap if i<2 else env.plant.at,0.])
    tip=tug[:2]+rot(tug[2])@tip_station
    tip_speed=env.plant.velocity_at(tug,tip_station)
    if position_noise_m or velocity_noise_mps:
        if rng is None:raise ValueError('an independent sensor rng is required when noise is enabled')
        point=point+rng.normal(0.,position_noise_m,2)
        speed=speed+rng.normal(0.,velocity_noise_mps,2)
    return ContactObservation(point, speed, tug, tip, tip_speed,
                   float(env.act.thrust[i]),float(env.act.azimuth[i]),
                   contact_goal(env.goal,station), float(env.time),
                   'push' if i<2 else 'tow')
