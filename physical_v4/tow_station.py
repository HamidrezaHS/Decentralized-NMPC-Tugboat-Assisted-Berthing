"""Local station-keeping of a slack towline; no shared state or force allocation.

When a game player selects zero tow thrust, the tug must still navigate to
preserve a usable outboard towing station. Under a slack unilateral rope the
maintenance thrust transmits NO force to the ship; this makes the physical
assumption explicit instead of leaving a motionless tug to be overrun.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .plant import rot

@dataclass
class TowStationConfig:
    desired_rope_length_offset_m:float= -1.0
    max_outboard_speed_mps:float=.24
    max_inboard_speed_mps:float=.13
    station_velocity_gain:float=420_000.
    station_position_gain:float=.13
    transverse_speed_limit_mps:float=.20
    transverse_position_gain:float=.095
    transverse_velocity_gain:float=220_000.
    max_azimuth_deg:float=35.
    engage_station_when_slack_below_m:float= -0.40

class TowStationController:
    def __init__(self,index,side,plant,config=None):
        assert index in (2,3)
        self.i=index;self.side=int(side);self.plant=plant
        self.c=config or TowStationConfig()

    def command(self,ship,tug,action,base_thrust,base_azimuth):
        p=self.plant;cfg=self.c
        a=p.attachment(ship)[self.i]
        # Bow-mounted towline while bow faces the ship. Propulsion is astern.
        q=tug[:2]+rot(tug[2])@np.array([p.at,0.])
        vq=p.velocity_at(tug,[p.at,0.]);va=p.velocity_at(ship,p.attach_local[self.i])
        delta=q-a;dist=float(np.linalg.norm(delta))
        R=rot(ship[2]);n=R@np.array([0.,self.side]);t=R@np.array([1.,0.])
        line=delta/max(1e-8,dist)
        rel=vq-va
        radial_rate=float(rel@line)
        # Only when slack, point to the local outboard station. On taut towline
        # avoid altering a selected equilibrium tow force.
        need_recover=(dist<p.c.rope_length+cfg.engage_station_when_slack_below_m
             or float((q-a)@n)<.65*p.c.rope_length)
        if not need_recover:return base_thrust,base_azimuth,'TOWING',dist
        target=p.c.rope_length+cfg.desired_rope_length_offset_m
        vref=float(np.clip(cfg.station_position_gain*(target-dist),
                   -cfg.max_inboard_speed_mps,cfg.max_outboard_speed_mps))
        Fnormal=max(0.,cfg.station_velocity_gain*(vref-radial_rate))
        vt=float(rel@t);position_t=float((q-a)@t)
        vref_t=float(np.clip(-cfg.transverse_position_gain*position_t,
                  -cfg.transverse_speed_limit_mps,cfg.transverse_speed_limit_mps))
        Ft=cfg.transverse_velocity_gain*(vref_t-vt)
        request=Fnormal*n+Ft*t
        b=rot(tug[2]).T@request
        # Astern thrust: the requested OUTBOARD world force has NEGATIVE surge
        # in a bow-inboard tug frame. Reverse signed thrust instead of rotating
        # the hull by pi. A forward-only station request is not permitted in
        # this towing mode, to avoid accidentally pushing toward the ship.
        if b[0]>=0:return float(base_thrust),float(base_azimuth),'STATION_UNAVAILABLE',dist
        reverse=-b
        az=float(np.clip(np.arctan2(reverse[1],reverse[0]),
                     -np.deg2rad(cfg.max_azimuth_deg),np.deg2rad(cfg.max_azimuth_deg)))
        chosen=np.array([np.cos(az),np.sin(az)])
        command=-float(np.clip(reverse@chosen,0.,p.max_force))
        # More negative means stronger astern pull. The game command remains
        # the base while its magnitude is already adequate.
        if command<base_thrust:return command,az,'RECOVER_SLACK',dist
        return float(base_thrust),float(base_azimuth),'GAME_DRIVE',dist
