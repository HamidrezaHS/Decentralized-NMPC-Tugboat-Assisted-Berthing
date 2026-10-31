"""Independent, body-fixed push-station/autonomous contact-keeping loop.

It consumes ONLY public ship pose/velocity, this tug's state, this tug's
attachment and the stage-game thrust request.  It never reads peer actions.

A compliant push fender is still a unilateral force in the plant: this
controller commands *thruster force*, never manipulates state/contact forces.
The stage game's requested normal force is realized only when in contact.

This is a preliminary engineering controller, not a verified safety filter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .plant import rot, wrap

@dataclass
class PushStationConfig:
    engaged_gap_m: float = .55       # permissible positive clearance from bumper
    recovery_gap_m: float = 1.20     # begin explicit re-approach
    max_gap_m: float = 2.50          # instrumentation safety threshold
    approach_speed_mps: float = .28
    recess_speed_mps: float = .15
    gap_velocity_gain: float = .14
    normal_velocity_gain: float = 400_000.
    tangent_position_gain: float = 30_000.
    tangent_velocity_gain: float = 240_000.
    tangent_speed_mps: float = .22
    max_azimuth_deg: float = 35.
    action_contact_threshold: float = .08

class PushContactController:
    """Per-tug autonomous contact/standby supervisor with local feedback.

    Action semantics: an equilibrium action sets desired force once engaging;
    while detached, local contact recovery takes precedence and the equilibrium
    request is ramped in near the hull. No central allocator or hidden force.
    """
    def __init__(self,index:int,side:int,plant,config:PushStationConfig|None=None):
        assert index in (0,1)
        self.index=index;self.side=int(side);self.plant=plant
        self.c=config or PushStationConfig()
        self.mode='STANDBY'
        self.last_command=0.

    def observe(self,ship,tug):
        p=self.plant
        R=rot(ship[2]);n=R@np.array([0.,self.side]);t=R@np.array([1.,0.])
        anchor=p.attachment(ship)[self.index]
        tip=tug[:2]+rot(tug[2])@np.array([p.ap,0.])
        va=p.velocity_at(ship,p.attach_local[self.index])
        vtip=p.velocity_at(tug,np.array([p.ap,0.]))
        d=tip-anchor;dv=vtip-va
        return dict(gap=float(d@n),gap_rate=float(dv@n),
                    tangential=float(d@t),tangential_rate=float(dv@t),
                    n=n,t=t,tip=tip,anchor=anchor)

    def command(self,ship,tug,action):
        o=self.observe(ship,tug);c=self.c;p=self.plant
        engaged=action.fraction>c.action_contact_threshold
        target_gap=p.c.bumper_m-(.12 if engaged else -.25)
        excess=o['gap']-p.c.bumper_m
        if excess>c.recovery_gap_m:
            self.mode='REAPPROACH'
        elif excess>c.engaged_gap_m:
            self.mode='APPROACH' if engaged else 'STANDBY'
        elif engaged:
            self.mode='MAINTAIN'
        else:
            self.mode='STANDBY'
        # Relative normal velocity setpoint. Positive gap is OUTBOARD.
        vref=float(np.clip(-c.gap_velocity_gain*(o['gap']-target_gap),
                 -c.approach_speed_mps,c.recess_speed_mps))
        # When the tug is away from the hull, divert its available thrust from
        # the game-requested push to re-approach; do not rush into contact.
        game_request=float(action.fraction*p.max_force)
        contact_blend=float(np.clip((c.recovery_gap_m-excess)/c.recovery_gap_m,0.,1.))
        normal_force=contact_blend*game_request+c.normal_velocity_gain*(o['gap_rate']-vref)
        normal_force=max(0.,normal_force)
        # Keep tip in its own hull station to avoid tangential slide/drift.
        vref_t=float(np.clip(-.12*o['tangential'],-c.tangent_speed_mps,c.tangent_speed_mps))
        lateral_force=-c.tangent_velocity_gain*(o['tangential_rate']-vref_t)
        # A stern azimuth unit can only exert +/- 35 degrees relative to bow.
        # Projection below accounts for the true feasible propulsion vector.
        force_world=-normal_force*o['n']+lateral_force*o['t']
        world_unit=rot(tug[2])@np.array([1.,0.])
        desired_body=rot(tug[2]).T@force_world
        if desired_body[0]<=0:
            steer=0.;thrust=0.
        else:
            steer=float(np.clip(np.arctan2(desired_body[1],desired_body[0]),
                   -np.deg2rad(c.max_azimuth_deg),np.deg2rad(c.max_azimuth_deg)))
            available=np.array([np.cos(steer),np.sin(steer)])
            thrust=float(np.clip(desired_body@available,0.,p.max_force))
        self.last_command=thrust
        return thrust,steer,dict(mode=self.mode,gap_m=o['gap'],gap_rate_mps=o['gap_rate'],
            tangential_error_m=o['tangential'],requested_game_force_N=game_request,
            actual_thruster_N=thrust,desired_gap_m=target_gap)
