"""Information barrier: public 3-DOF vessel state + private own tug data.

Only this sensor adapter is allowed to read the simulator. NMPC controllers
receive immutable snapshots and cannot query other tug states or commands.
The physical plant naturally couples all four tugs; that is NOT information
exchange between policies. Local environmental observations are idealized by
default (optional Gaussian noise is supported by the original sensor adapter).
"""
from __future__ import annotations
import numpy as np
from decentralized_nmpc.controller import TugObservation, common_ship_observation

SHIP_STATE_VECTOR_COLUMNS = (
    'ship_x_m','ship_y_m','ship_psi_rad','ship_u_mps','ship_v_mps','ship_r_radps',
    'own_x_m','own_y_m','own_psi_rad','own_u_mps','own_v_mps','own_r_radps',
    'own_thrust_N','own_azimuth_rad',
    'goal_x_m','goal_y_m','goal_psi_rad',
    'wind_mps','wind_to_rad','current_mps','current_to_rad',
)

def _readonly(vector, size):
    array=np.array(vector,dtype=np.float64,copy=True)
    if array.shape!=(size,) or not np.all(np.isfinite(array)):
        raise ValueError(f'Expected finite ({size},) sensor vector')
    array.setflags(write=False)
    return array

class ShipStateLocalSensor:
    """One independent instance per tug; no peer data are returned.

    env must be passed ONLY to read(), on the simulator side. The controller
    receives exclusively a TugObservation, not the environment or other tugs.
    """
    def __init__(self,agent_id:int,measure_environment=True,
                 wind_sensor_std_mps=0.,current_sensor_std_mps=0.,seed=123):
        if int(agent_id)!=agent_id or not 0<=agent_id<4:
            raise ValueError('agent_id must be an integer 0..3')
        self.agent_id=int(agent_id)
        self.measure_environment=bool(measure_environment)
        self.wind_sensor_std_mps=float(wind_sensor_std_mps)
        self.current_sensor_std_mps=float(current_sensor_std_mps)
        if min(self.wind_sensor_std_mps,self.current_sensor_std_mps)<0:
            raise ValueError('sensor std must be nonnegative')
        self.rng=np.random.default_rng(seed+10001+37*self.agent_id)

    def read(self,env)->TugObservation:
        obs=common_ship_observation(
            env,self.agent_id,measure_environment=self.measure_environment,
            wind_sensor_std_mps=self.wind_sensor_std_mps,
            current_sensor_std_mps=self.current_sensor_std_mps,
            rng=self.rng if self.measure_environment else None)
        return TugObservation(
            ship=_readonly(obs.ship,6), own=_readonly(obs.own,6),
            own_thrust=float(obs.own_thrust),own_azimuth=float(obs.own_azimuth),
            time_s=float(obs.time_s),wind_mps=float(obs.wind_mps),
            wind_to_rad=float(obs.wind_to_rad),current_mps=float(obs.current_mps),
            current_to_rad=float(obs.current_to_rad))

def ship_state_policy_vector(obs:TugObservation,goal)->np.ndarray:
    """21 raw features, fixed order, EXACTLY the observation available to NMPC.

    Agent identity is not included: each agent has its own actor/checkpoint.
    Normalize on TRAINING data only; do not normalize using evaluation data.
    """
    goal=np.asarray(goal,dtype=float)
    if goal.shape!=(3,) or not np.all(np.isfinite(goal)):
        raise ValueError('Expected finite goal pose (x, y, psi)')
    x=np.concatenate((obs.ship,obs.own,
            [obs.own_thrust,obs.own_azimuth],goal,
            [obs.wind_mps,obs.wind_to_rad,obs.current_mps,obs.current_to_rad]))
    assert len(x)==len(SHIP_STATE_VECTOR_COLUMNS)
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite observation')
    return x.copy()
