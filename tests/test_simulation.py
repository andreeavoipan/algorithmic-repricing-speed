import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.agents import AgentState, MyopicBestResponseAgent
from repricing.simulation import DuopolyConfig, DuopolyResult, run_duopoly

A=np.array([2.,2.])

def br(i,j,threshold=0.):
    return MyopicBestResponseAgent(i,j,np.array([1.,1.]),A,.25,0.,1.,3.,threshold)


def test_effective_speed_counts_reprices_per_period_not_only_intervals():
    result=DuopolyResult(
        price_history=np.zeros((3000,2)), cost_history=np.zeros((3000,2)),
        reprice_events=[(1,0),(2,0)], profit_history=np.zeros((3000,2)),
        decision_log=[], burn_in=0)
    assert np.isclose(result.effective_speed(0),2/3000)
    assert np.isclose(result.inter_reprice_speed(0),1.0)


def test_burn_in_excludes_early_events():
    result=DuopolyResult(np.zeros((100,2)),np.zeros((100,2)),[(1,0),(50,0)],np.zeros((100,2)),[],20)
    assert np.isclose(result.effective_speed(0),1/80)


def test_conditional_rate_is_bounded_and_tech_rate_caps_realized_rate():
    r=run_duopoly(br(0,1),br(1,0),DuopolyConfig(T=600,burn_in=100,tech_freq_0=.25,seed=3,cost_shock_std=.01,cost_shock_std_1=.01))
    assert 0 <= r.conditional_repricing_rate(0) <= 1
    # finite-sample realised repricing rate cannot exceed realised technical opportunities
    opp=sum(x["can_recompute"] for x in r.decision_log if x["firm"]==0 and x["period"]>=100)/500
    assert r.effective_speed(0) <= opp + 1e-12


def test_both_agents_decide_from_same_start_of_period_prices():
    class Spy:
        def __init__(self,i):
            self.firm_index=i; self.a=A; self.mu=.25; self.a0=0.; self.seen=[]
        def decide(self,state):
            self.seen.append((state.own_price,state.last_observed_rival_price))
            return True, state.own_price + .1
    x,y=Spy(0),Spy(1)
    run_duopoly(x,y,DuopolyConfig(T=1,init_price_0=1.5,init_price_1=1.7,seed=0))
    assert x.seen[0] == (1.5,1.7)
    assert y.seen[0] == (1.7,1.5)


def test_mean_reverting_costs_are_market_state_and_stay_bounded():
    r=run_duopoly(br(0,1),br(1,0),DuopolyConfig(T=1000,cost_shock_std=.03,cost_shock_std_1=.03,cost_rho=.95,seed=1))
    assert np.all(r.cost_history >= .5) and np.all(r.cost_history <= 1.5)
    assert abs(r.cost_history[:,0].mean()-1.) < .2
