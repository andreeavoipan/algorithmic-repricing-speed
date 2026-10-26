import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.agents import AgentState, FixedRuleAgent, MyopicBestResponseAgent, AnticipatoryBestResponseAgent


def make_br_agent(cost_threshold=0.0):
    return MyopicBestResponseAgent(0,1,np.array([1.,1.]),np.array([2.,2.]),.25,0.,1.,3.,cost_threshold)


def test_fixed_rule_undercuts_when_rival_cheaper():
    a=FixedRuleAgent(0,1.,3.)
    move,p=a.decide(AgentState(1.5,1.4,True))
    assert move and p < 1.4


def test_fixed_rule_does_not_reprice_if_cannot_recompute():
    a=FixedRuleAgent(0,1.,3.)
    move,p=a.decide(AgentState(1.5,1.4,False))
    assert not move and p == 1.5


def test_zero_threshold_moves_toward_best_response():
    a=make_br_agent(0.)
    move,p=a.decide(AgentState(1.,1.8,True))
    assert move and p != 1.


def test_high_threshold_blocks_small_gains():
    a=make_br_agent(10.)
    move,p=a.decide(AgentState(1.5,1.55,True))
    assert not move and p == 1.5


def test_decision_uses_belief_not_hidden_true_rival_price():
    """There is deliberately no true_rival_price/oracle field in AgentState."""
    assert "true_rival_price" not in AgentState.__dataclass_fields__
    a=make_br_agent(0.)
    s1=AgentState(1.5,1.2,True,own_cost=1.,market_costs=np.array([1.,1.]))
    s2=AgentState(1.5,1.8,True,own_cost=1.,market_costs=np.array([1.,1.]))
    assert not np.isclose(a.decision_details(s1)["candidate_price"], a.decision_details(s2)["candidate_price"])


def test_market_costs_in_state_override_agent_copy():
    a=make_br_agent(0.)
    s_low=AgentState(1.5,1.5,True,own_cost=.6,market_costs=np.array([.6,1.]))
    s_high=AgentState(1.5,1.5,True,own_cost=1.4,market_costs=np.array([1.4,1.]))
    p_low=a.decision_details(s_low)["candidate_price"]
    p_high=a.decision_details(s_high)["candidate_price"]
    assert p_high > p_low


def test_anticipatory_agent_changes_expected_gain_with_reaction_probability():
    base=dict(firm_index=0,rival_index=1,costs=np.array([1.,1.]),a=np.array([2.,2.]),mu=.25,a0=0.,price_floor=1.,price_ceiling=3.,cost_threshold=0.)
    s=AgentState(1.3,1.7,True,own_cost=1.,market_costs=np.array([1.,1.]))
    no_react=AnticipatoryBestResponseAgent(**base,reaction_probability=0.)
    react=AnticipatoryBestResponseAgent(**base,reaction_probability=1.)
    assert not np.isclose(no_react.decision_details(s)["expected_gain"], react.decision_details(s)["expected_gain"])
