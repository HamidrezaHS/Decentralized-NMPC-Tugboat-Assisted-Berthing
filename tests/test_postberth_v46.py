"""Post-berthing audit and seed variation tests; no changes to the physical criteria."""
import json
import math
import pytest
from marl_game.env import BerthingGameEnv, EnvConfig
from physical_v4.termination import BerthingMonitor, BerthingCriteria
from run_nmpc import rollout


def _sample(m,t,e=2.0,heading=.1,speed=.01,yaw=.001,safe=True):
    return m.update(t=t,dt=.2,position_error_m=e,heading_error_deg=heading,
                    speed_mps=speed,yaw_rate_deg_s=yaw,safe=safe,hull_gap_m=8.0)


def test_post_success_audit_substep_detection_and_first_event():
    m=BerthingMonitor(BerthingCriteria(hold_s=.4))
    assert not _sample(m,.2)
    assert _sample(m,.4)
    assert m.first_success_time_s==pytest.approx(.4)
    assert m.post_success_samples==0
    assert not _sample(m,.6)
    assert not _sample(m,.8,e=2.7)
    assert m.post_success_elapsed_s==pytest.approx(.4)
    assert m.post_success_in_limits_s==pytest.approx(.2)
    assert m.post_success_violations==1
    assert m.post_success_max_position_error_m==pytest.approx(2.7)
    assert m.post_success_min_hull_gap_m==pytest.approx(8.)


def test_environment_continue_flag_stops_on_audit_end():
    env=BerthingGameEnv(EnvConfig(train=False,max_seconds=80.,post_success_seconds=.4))
    env.monitor.first_success_time_s=0.0  # inject known success event to test termination mechanics
    _,_,_,term,trunc,info=env.step([[0.,0.]]*4)
    assert term and not trunc
    assert info['stationkeeping_complete']
    assert env.time==pytest.approx(.4)
    assert env.monitor.post_success_samples==2


def test_zero_followup_preserves_standard_monitor_end():
    c=EnvConfig(train=False)
    assert c.post_success_seconds==0.
    env=BerthingGameEnv(c)
    assert not env.cfg.post_success_seconds


def test_seed_dependent_eval_initial_condition():
    kw=dict(train=False,eval_position_jitter_m=1.,eval_heading_jitter_deg=2.,initial_psi_deg=20.)
    e1=BerthingGameEnv(EnvConfig(seed=123,**kw))
    e2=BerthingGameEnv(EnvConfig(seed=124,**kw))
    e3=BerthingGameEnv(EnvConfig(seed=123,**kw))
    assert (e1.ship[:3]!=e2.ship[:3]).any()
    assert (e1.ship[:3]==e3.ship[:3]).all()
    assert all(abs(v)<=1 for v in e1.ship[:2])
    assert abs(math.degrees(e1.ship[2])-20)<=2


def test_short_timeout_reports_no_hold_success(tmp_path):
    m=rollout(out=tmp_path,max_seconds=20.,post_success_seconds=600.,
              maxiter=2,render=False,verbose=False,save_every_seconds=0.)
    assert m['status']=='TIMEOUT'
    assert not m['initial_berthing_success']
    assert not m['stationkeeping_passed']
    assert m['post_success_observed_s']==0.
    assert m['first_success_time_s'] is None
    cfg=json.loads((tmp_path/'configuration.json').read_text())
    assert cfg['post_success_seconds']==600.


def test_invalid_hold_and_jitter(tmp_path):
    with pytest.raises(ValueError,match='post_success_seconds'):
        rollout(out=tmp_path,post_success_seconds=-1.,render=False)
    with pytest.raises(ValueError,match='jitter'):
        rollout(out=tmp_path,eval_heading_jitter_deg=-1.,render=False)
