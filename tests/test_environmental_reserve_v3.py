import numpy as np
from decentralized_nmpc.controller import LocalNMPC, NMPCConfig
from ENV_DIRECTION_SWEEP import build_cases


def make_agent(agent=0,enabled=True):
    return LocalNMPC(agent,np.array([50.,-100.,0.]),np.array([[-160.,-30.],[160.,30.]]),
                     np.array([-96.,30.]),1,882598.5,
                     NMPCConfig(environmental_thrust_reserve=enabled))


def test_reserve_none_and_favorable_direction():
    a=make_agent()
    ship=np.array([0.,0.,0.,0.,0.,0.])
    assert np.isclose(a._environmental_push_cap(ship,[0,0,0]),0.52)
    assert np.isclose(a._environmental_push_cap(ship,[0,-2e6,0]),0.52)


def test_adverse_force_expands_software_cap_but_not_physical_capacity():
    a=make_agent()
    ship=np.array([0.,0.,0.,0.,0.,0.])
    assert .52<a._environmental_push_cap(ship,[0,854000,0])<=.9
    assert np.isclose(a._environmental_push_cap(ship,[0,9e6,0]),.9)
    assert a._environmental_push_cap(ship,[0,854000,0])<=1.
    assert a.environmental_load_away_N==854000.


def test_no_environmental_reserve_in_legacy_or_towing_roles():
    ship=np.array([0.,0.,0.,0.,0.,0.])
    for agent,enabled in [(0,False),(2,True),(3,True)]:
        a=make_agent(agent,enabled)
        assert a._environmental_push_cap(ship,[0,2e6,0])==.52


def test_direction_matrix_covers_independently_rotated_wind_and_current():
    cases=build_cases()
    assert len(cases)==25
    combined=[x for x in cases if x.wind_mps>0 and x.current_mps>0]
    assert len(combined)==16
    assert len({x.name for x in cases})==len(cases)
    assert {(x.wind_to_deg,x.current_to_deg) for x in combined}=={(a,b) for a in (0,90,180,270) for b in (0,90,180,270)}
