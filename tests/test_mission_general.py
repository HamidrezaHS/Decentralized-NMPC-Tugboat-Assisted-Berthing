"""Mission-local formulation and geometry regression tests."""
import numpy as np
from marl_game.env import BerthingGameEnv, EnvConfig
from decentralized_nmpc import LocalNMPC, NMPCConfig, common_ship_observation
from run_nmpc import rollout


def make_controller(goal=(50.,-100.,0.), start=(0.,0.), index=0):
    env=BerthingGameEnv(EnvConfig(train=False,initial_psi_deg=20.,
                eval_start_x_m=start[0],eval_start_y_m=start[1],
                target_x=goal[0],target_y=goal[1],target_psi_deg=goal[2]))
    c=LocalNMPC(index,env.goal,env.hull.points,env.plant.attach_local[index],
                env.side,env.plant.max_force,NMPCConfig(mission_guidance=True,max_iterations=2))
    c.mission_start=np.array(start,float)
    return env,c


def test_signed_velocity_reference_brakes_and_reverses_with_target():
    env,c=make_controller(goal=(50.,-100.,0.))
    sh=np.array([49.,-100.,0.,.08,0.,0.])
    v,_,_=c._mission_velocity_reference(sh)
    assert 0 < v[0] < .011  # a=5.5e-5; d=1 m; vmax~0.0105
    sh[0]=51.
    v,_,_=c._mission_velocity_reference(sh)
    assert -.011 < v[0] < 0
    sh[0]=50.
    v,_,_=c._mission_velocity_reference(sh)
    assert abs(v[0])<1e-9


def test_cost_penalizes_runaway_in_either_along_direction():
    for gx in (50.,-50.):
        env,c=make_controller(goal=(gx,-100.,0.))
        vx=0.06 if gx>0 else -0.06
        safe=np.array([gx,-70.,0.,0.,0.,0.])
        runaway=np.array([gx+np.sign(gx)*90.,-70.,0.,vx,0.,0.])
        assert c._mission_cost(runaway,None,0,0)>c._mission_cost(safe,None,0,0)+20.


def test_mission_model_ignores_peer_tug_telemetry():
    env=BerthingGameEnv(EnvConfig(train=False,initial_psi_deg=40.))
    obs=common_ship_observation(env,0)
    env.tugs[1:,:]+=np.array([10.,20.,0.,0.,0.,0.])
    obs2=common_ship_observation(env,0)
    c1=LocalNMPC(0,env.goal,env.hull.points,env.plant.attach_local[0],env.side,
                 env.plant.max_force,NMPCConfig(mission_guidance=True,max_iterations=2))
    c2=LocalNMPC(0,env.goal,env.hull.points,env.plant.attach_local[0],env.side,
                 env.plant.max_force,NMPCConfig(mission_guidance=True,max_iterations=2))
    a,_=c1.act(obs)
    b,_=c2.act(obs2)
    assert np.allclose(a,b)


def test_different_start_and_berth_coordinates_are_saved(tmp_path):
    metrics=rollout(out=tmp_path,psi0=-20.,start_x=10.,start_y=5.,
           target_x=75.,target_y=-120.,max_seconds=10.,maxiter=2,
           guidance_mode='mission',render=False,verbose=False,save_every_seconds=0.)
    assert metrics['guidance_mode']=='mission'
    assert metrics['start_x_m']==10.
    assert metrics['start_y_m']==5.
    assert np.isfinite(metrics['path_length_ratio'])


def test_reject_unsupported_rotated_quay(tmp_path):
    import pytest
    with pytest.raises(ValueError,match='east-west quay'):
        rollout(out=tmp_path,target_psi_deg=15.,max_seconds=10.,render=False)
