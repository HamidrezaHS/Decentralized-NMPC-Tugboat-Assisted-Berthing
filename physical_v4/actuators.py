"""First-order rate-limited physical actuator states for the archived v5.3 runner.

Each tug retains independent thrust/azimuth/yaw states; no coupling/allocator.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class ActuatorConfig:
    thrust_time_constant_s: float = 4.
    thrust_max_fraction_per_s: float = .06
    azimuth_time_constant_s: float = 3.
    azimuth_max_rate_deg_s: float = 2.
    yaw_time_constant_s: float = 3.
    yaw_max_rate_MNm_per_s: float = .6

class ActuatorBank:
    def __init__(self,max_force,config=None):
        self.max_force=float(max_force)
        self.c=config or ActuatorConfig()
        self.thrust=np.zeros(4)
        self.azimuth=np.zeros(4)
        self.moment=np.zeros(4)

    def step(self,thrust,azimuth,moment,dt):
        dt=float(dt)
        if not 0 < dt <= .2: raise ValueError('physical actuator dt must be in (0, .2]')
        thrust=np.asarray(thrust,float);azimuth=np.asarray(azimuth,float);moment=np.asarray(moment,float)
        if any(v.shape!=(4,) for v in (thrust,azimuth,moment)):raise ValueError('four actuator commands are required')
        c=self.c
        alpha_t=min(1.,dt/max(c.thrust_time_constant_s,1e-6))
        alpha_a=min(1.,dt/max(c.azimuth_time_constant_s,1e-6))
        alpha_m=min(1.,dt/max(c.yaw_time_constant_s,1e-6))
        t_change=np.clip(alpha_t*(thrust-self.thrust),-c.thrust_max_fraction_per_s*self.max_force*dt,c.thrust_max_fraction_per_s*self.max_force*dt)
        a_change=np.clip(alpha_a*(azimuth-self.azimuth),-np.deg2rad(c.azimuth_max_rate_deg_s)*dt,np.deg2rad(c.azimuth_max_rate_deg_s)*dt)
        m_change=np.clip(alpha_m*(moment-self.moment),-c.yaw_max_rate_MNm_per_s*1e6*dt,c.yaw_max_rate_MNm_per_s*1e6*dt)
        self.thrust=np.clip(self.thrust+t_change,-self.max_force,self.max_force)
        self.azimuth=np.clip(self.azimuth+a_change,-np.deg2rad(35),np.deg2rad(35))
        self.moment=np.clip(self.moment+m_change,-5e6,5e6)
        return self.thrust.copy(),self.azimuth.copy(),self.moment.copy()
