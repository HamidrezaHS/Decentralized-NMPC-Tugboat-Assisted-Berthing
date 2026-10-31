"""Low-level propulsion and contact/rope station management without ship state.

Each helm sees ONLY own contact-point sensor and its own requested action.
All ship-dependent contact mechanics stay inside the physical PLANT.
"""
from __future__ import annotations
import numpy as np
from physical_v4.plant import rot,wrap
from .sensors import ContactObservation


class ContactOnlyHelm:
    def __init__(self, agent_id:int, max_force:float, bumper_m:float=4.5,
                 rope_length_m:float=40.):
        if agent_id not in range(4):raise ValueError('agent id must be 0..3')
        self.i=int(agent_id)
        self.max_force=float(max_force)
        self.bumper=float(bumper_m)
        self.rope=float(rope_length_m)
        self.last_mode='INIT'
        # Local installation calibration: world-space outboard unit is configured
        # before the mission. No live vessel orientation is read.
        self._initial_outboard=None
        self._initial_error=None
        # This plant explicitly restricts the final quay to an earth-fixed
        # east-west line; +Y is the outboard free side for both push/tow.
        # For a rotated berth this vector must be a mission parameter.
        self._berth_outboard=np.array([0.,1.],dtype=float)

    def reset(self):
        """Reset episodic local calibration without consulting vessel state."""
        self.last_mode='INIT'
        self._initial_outboard=None
        self._initial_error=None

    def command(self, obs:ContactObservation, action):
        """Return signed thrust, local azimuth, yaw moment; no vessel inputs."""
        if obs.role!=('push' if self.i<2 else 'tow'):
            raise ValueError('role mismatch')
        action=np.asarray(action,float)
        if action.shape!=(2,) or not np.isfinite(action).all():
            raise ValueError('action must be finite (fraction, offset_deg)')
        fraction=float(np.clip((action[0]+1.)*.5,0.,1.))
        offset=float(np.clip(action[1]*20.,-20.,20.))
        delta=obs.relative_tip_xy
        distance=float(np.linalg.norm(delta))
        # Mission-calibrated outboard orientation, not the instantaneous
        # rope direction. Using delta / |delta| for the tangent makes
        # delta dot tangent == 0 identically and destroys station keeping.
        point_error=float(np.linalg.norm(obs.goal_point_xy-obs.point_xy))
        if self._initial_outboard is None:
            out=delta/distance if distance>1.e-4 else -rot(obs.own[2])@np.array([1.,0.])
            self._initial_outboard=out.copy()
            self._initial_error=max(point_error,1.)
        progress=float(np.clip(1.-point_error/self._initial_error,0.,1.))
        outward=(1.-progress)*self._initial_outboard+progress*self._berth_outboard
        outward=outward/max(float(np.linalg.norm(outward)),1.e-8)
        inward=-outward
        tangent=np.array([-outward[1],outward[0]])
        relative_v=obs.relative_tip_velocity_xy
        radial_v=float(relative_v@outward)
        tangential_v=float(relative_v@tangent)
        tangential=float(delta@tangent)
        desired_heading=float(np.arctan2(inward[1],inward[0]))
        heading_error=float(wrap(desired_heading-obs.own[2]))
        # Exactly-local yaw PD. Yaw-moment actuator is physical/rate-limited.
        # Feed back only own IMU heading and yaw rate. Stronger anti-drift
        # authority prevents a loaded push tug from turning broadside.
        yaw_limit=5.0e6 if self.i<2 else 4.0e6
        yaw=float(np.clip(24.0e6*heading_error-35.0e6*obs.own[5],
                          -yaw_limit,yaw_limit))
        if self.i<2:
            # Positive compression occurs at distance < bumper. Allow robust
            # re-engagement from detachment without needing ship heading.
            target=self.bumper-(.12 if fraction>.08 else -.20)
            vref=float(np.clip(-.15*(distance-target),-.25,.14))
            normal=max(0.,fraction*self.max_force+4.e5*(radial_v-vref))
            tangent_force=float(np.clip(-1.2e5*tangential_v-4.5e4*tangential,-.55*self.max_force,.55*self.max_force))
            desired_world=(-normal*np.cos(np.deg2rad(offset))*outward+
                           (normal*np.sin(np.deg2rad(offset))+tangent_force)*tangent)
            body=rot(-obs.own[2])@desired_world
            if body[0]<=0:
                thrust=0.;az=0.
            else:
                az=float(np.clip(np.arctan2(body[1],body[0]),-.61,.61))
                thrust=float(np.clip(body@np.array([np.cos(az),np.sin(az)]),0.,self.max_force))
            self.last_mode='CONTACT' if distance<=self.bumper+.55 else 'REAPPROACH'
        else:
            # Astern propulsion, tension-only line; maintain station if slack.
            target=self.rope-.5
            vref=float(np.clip(.15*(target-distance),-.12,.23))
            requested=fraction*self.max_force
            if distance<self.rope-.4:
                requested=max(requested,4.e5*(vref-radial_v))
                self.last_mode='RECOVER_SLACK'
            else:self.last_mode='TOWING'
            tangent_force=float(np.clip(-1.2e5*tangential_v-4.5e4*tangential,-.65*self.max_force,.65*self.max_force))
            desired_world=(requested*np.cos(np.deg2rad(offset))*outward+
                           (requested*np.sin(np.deg2rad(offset))+tangent_force)*tangent)
            body=-rot(-obs.own[2])@desired_world
            if body[0]<=0:
                thrust=0.;az=0.
            else:
                az=float(np.clip(np.arctan2(body[1],body[0]),-.61,.61))
                thrust=-float(np.clip(body@np.array([np.cos(az),np.sin(az)]),0.,self.max_force))
        # Feedback of *own* heading and yaw rate counters stern-thruster yaw.
        # The requested offset affects thrust direction, not desired hull pose.
        # The bounded yaw actuator, rather than azimuth saturation, is
        # responsible for keeping the hull heading aligned with the station.
        steering=float(np.clip(-1.1*heading_error+4.0*obs.own[5],-.48,.48))
        if self.i>=2:steering=-steering
        az=float(np.clip(az+steering,-.61,.61))
        return thrust,az,yaw
