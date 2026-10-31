"""Regression checks for distributed information boundary and NMPC toolchain."""
from pathlib import Path
import tempfile
import numpy as np
from marl_game.env import BerthingGameEnv, EnvConfig
from decentralized_nmpc import LocalNMPC, NMPCConfig, common_ship_observation
from visualization.marine_renderer import get_tug_hull
from run_nmpc import rollout


def _make(env,i):
    return LocalNMPC(i,env.goal,env.hull.points,env.plant.attach_local[i],env.side,
                     env.plant.max_force,NMPCConfig(max_iterations=4))


def test_observation_excludes_peer_tugs_and_actuators():
    env=BerthingGameEnv(EnvConfig(train=False,max_seconds=80.))
    obs=common_ship_observation(env,0)
    env.tugs[1:,0:2]+=100.  # unrelated peer positions are never observed
    env.act.thrust[1:]+=1000.
    obs2=common_ship_observation(env,0)
    assert np.array_equal(obs.ship,obs2.ship)
    assert np.array_equal(obs.own,obs2.own)
    assert obs.own_thrust==obs2.own_thrust
    a,_=_make(env,0).act(obs)
    b,_=_make(env,0).act(obs2)
    assert np.allclose(a,b,atol=1e-6)


def test_bounds_and_future_cost_nonnegative():
    env=BerthingGameEnv(EnvConfig(train=False,max_seconds=80.))
    for i in range(4):
        c=_make(env,i)
        a,info=c.act(common_ship_observation(env,i))
        assert a.shape==(2,)
        assert -1.00001<=a[0]<=1.00001
        assert -1.00001<=a[1]<=1.00001
        assert 0<=info['fraction']<=(.52 if i<2 else .65)+1e-6
        assert np.isfinite(info['objective'])
        assert np.isfinite(info['own_heading_error_deg'])


def test_tug_outline_is_from_project_dxf():
    pts=get_tug_hull()
    assert len(pts)>300
    assert 20<float(np.ptp(pts[:,0]))<25
    assert 7<float(np.ptp(pts[:,1]))<10


def test_run_exports_logged_physics_and_local_optimizations():
    with tempfile.TemporaryDirectory() as temp:
        metrics=rollout(out=temp,psi0=20.,max_seconds=30.,dt=.2,decision_seconds=10.,
                        maxiter=4,verbose=False,render=False)
        folder=Path(temp)
        assert (folder/'trajectory.csv').exists()
        assert (folder/'local_nmpc_decisions.csv').exists()
        assert (folder/'configuration.json').exists()
        assert metrics['seconds']>0
        assert metrics['status'] in ('TIMEOUT','UNSAFE','SUCCESS')
