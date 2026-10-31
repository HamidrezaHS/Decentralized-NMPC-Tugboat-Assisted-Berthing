"""Strict observation API. No global state or co-player controls are available to an agent."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TugObservation:
    time: float
    ship_pose: tuple[float, float, float]          # x/y/heading, shared navigation aid
    ship_body_velocity: tuple[float, float, float] # u/v/r, ship telemetry
    ship_earth_velocity: tuple[float, float]       # inertial motion estimate
    own_tug_state: tuple[float, ...]               # own local rigid/rope state
    own_applied_thrust: float
    own_applied_azimuth: float
    target_pose: tuple[float, float, float]        # berth mission (shared in advance)
    near_wall_gap_m: float                        # local chart+ship estimate, not force

    def __post_init__(self):
        if len(self.ship_pose) != 3 or len(self.ship_body_velocity) != 3:
            raise ValueError('Invalid ship observation')
        if len(self.own_tug_state) not in (6, 8):
            raise ValueError('Own tug state must be 6 push or 8 tow coordinates')
        fields = (*self.ship_pose, *self.ship_body_velocity, *self.ship_earth_velocity,
                  *self.own_tug_state, self.own_applied_thrust, self.own_applied_azimuth,
                  *self.target_pose, self.near_wall_gap_m)
        if not np.all(np.isfinite(fields)):
            raise ValueError('Observation has non-finite values')


@dataclass(frozen=True)
class TugAction:
    thrust: float
    azimuth: float


def wrap(angle: float) -> float:
    return (float(angle) + np.pi) % (2. * np.pi) - np.pi
