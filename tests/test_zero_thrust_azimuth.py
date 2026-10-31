"""Ensure an idle tug can reposition its virtual azimuth without deadlocking.

All safety monitor, unilateral interactions, and physical actuator limits are
unchanged from the audited base model.
"""
import numpy as np
from types import SimpleNamespace
from marl_game.env import BerthingGameEnv, EnvConfig
from decentralized_nmpc import LocalNMPC, NMPCConfig, common_ship_observation
from physical_v4.actuators import ActuatorBank
import decentralized_nmpc.controller as mod


def test_idle_azimuth_slew_has_negligible_artificial_penalty():
    env = BerthingGameEnv(EnvConfig(train=False, initial_psi_deg=20.))
    env.reset()
    c = LocalNMPC(0,env.goal,env.hull.points,env.plant.attach_local[0],env.side,
                  env.plant.max_force,NMPCConfig(mission_guidance=True))
    ship=env.ship.copy()
    obs=common_ship_observation(env,0)
    initial=np.array([0.,-17.])
    A,C=c.model.ship_matrices(*ship[3:6],ship[2],Vw=0.,psi_w=0.,Vc=0.,psi_c=0.)
    hydroacc=np.linalg.solve(A,C.flatten())[:3]
    invmass=np.linalg.inv(A[:3,:3])
    stale=c._predict_cost(np.array([0.,-17.,0.,-17.]),ship,initial,invmass,hydroacc,obs)
    rotated=c._predict_cost(np.array([0.,17.,0.,17.]),ship,initial,invmass,hydroacc,obs)
    assert rotated-stale < 0.03


def test_idle_virtual_azimuth_can_reorient_but_active_knot_remains_bounded(monkeypatch):
    env=BerthingGameEnv(EnvConfig(train=False,initial_psi_deg=20.));env.reset()
    c=LocalNMPC(0,env.goal,env.hull.points,env.plant.attach_local[0],env.side,
                env.plant.max_force,NMPCConfig(mission_guidance=True,max_iterations=1))
    captured=[]
    def fake_minimize(fun,x0,method,bounds,options):
        captured.append(bounds)
        return SimpleNamespace(x=np.array(x0),fun=fun(np.array(x0)),success=True,nit=0)
    monkeypatch.setattr(mod,'minimize',fake_minimize)
    c.last=np.array([0.,-17.]);c.plan=np.array([[0.,-17.],[0.,-17.]])
    c.act(common_ship_observation(env,0))
    assert captured[0][1]==(-20.,20.)
    assert captured[0][3]==(-20.,20.)
    c.last=np.array([.3,-20.]);c.plan=np.array([[.3,-20.],[.3,-20.]])
    c.act(common_ship_observation(env,0))
    assert captured[1][1]==(-20.,4.)


def test_physical_azimuth_rate_limit_still_applies():
    bank=ActuatorBank(max_force=1.e6)
    _,az,_=bank.step(np.zeros(4),np.deg2rad(np.ones(4)*20.),np.zeros(4),dt=.2)
    assert np.max(az)<=np.deg2rad(.4)+1e-9
