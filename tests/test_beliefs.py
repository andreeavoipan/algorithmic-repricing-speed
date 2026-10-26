import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.agents import AgentState, MyopicBestResponseAgent
from repricing.beliefs import UncertainBeliefAgent
from repricing.simulation import DuopolyConfig, run_duopoly

KW = dict(costs=np.array([1., 1.]), a=np.array([2., 2.]), mu=.25, a0=0., price_floor=1., price_ceiling=3.)


def test_uncertain_belief_reduces_to_benchmark_at_age_zero():
    base = MyopicBestResponseAgent(0, 1, cost_threshold=0., **KW)
    unc = UncertainBeliefAgent(0, 1, cost_threshold=0., belief_sd=.05, **KW)
    s = AgentState(1.6, 1.4, True, own_cost=1., market_costs=np.array([1., 1.]), obs_age=0)
    d0, d1 = base.decision_details(s), unc.decision_details(s)
    assert abs(d0["candidate_price"] - d1["candidate_price"]) < 1e-3
    assert abs(d0["expected_gain"] - d1["expected_gain"]) < 1e-6


def test_uncertain_belief_changes_evaluation_when_information_is_stale():
    unc = UncertainBeliefAgent(0, 1, cost_threshold=0., belief_sd=.05, **KW)
    fresh = AgentState(1.6, 1.4, True, own_cost=1., market_costs=np.array([1., 1.]), obs_age=0)
    stale = AgentState(1.6, 1.4, True, own_cost=1., market_costs=np.array([1., 1.]), obs_age=25)
    assert abs(unc.decision_details(fresh)["expected_gain"] - unc.decision_details(stale)["expected_gain"]) > 1e-6


def test_observation_age_counter_in_simulation():
    seen = []

    class Probe(MyopicBestResponseAgent):
        def decide(self, state):
            seen.append(state.obs_age)
            return super().decide(state)

    a = Probe(0, 1, cost_threshold=0., **KW)
    b = MyopicBestResponseAgent(1, 0, cost_threshold=0., **KW)
    run_duopoly(a, b, DuopolyConfig(T=30, obs_freq_0=0., seed=1, cost_shock_std=.01, cost_shock_std_1=.01, record_decision_log=False))
    assert seen[:5] == [0, 1, 2, 3, 4] and seen[-1] == 29
