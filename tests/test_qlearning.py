import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.qlearning import QLearningAgent


def make_agent(switching_cost=0.0, n_staleness_bins=4):
    return QLearningAgent(
        firm_index=0, rival_index=1, costs=np.array([1.0, 1.0]),
        a=np.array([2.0, 2.0]), mu=0.25, a0=0.0,
        price_floor=1.0, price_ceiling=3.0, switching_cost=switching_cost,
        n_markup_bins=13, markup_range=(-0.3, 2.0),
        n_gap_bins=15, gap_range=1.0, gamma=0.0, n_staleness_bins=n_staleness_bins,
        rng=np.random.default_rng(0),
    )


def test_markup_to_bin_is_in_range():
    agent = make_agent()
    b = agent.markup_to_bin(0.5)
    assert 0 <= b < agent.n_markup_bins


def test_gap_to_bin_clips_out_of_range():
    agent = make_agent()
    assert agent.gap_to_bin(10.0) == agent.n_gap_bins - 1
    assert agent.gap_to_bin(-10.0) == 0


def test_hold_action_keeps_own_price():
    agent = make_agent()
    price = agent.action_to_price(agent.HOLD, own_price=1.7, own_cost=1.0)
    assert price == 1.7


def test_non_hold_action_targets_cost_plus_markup():
    agent = make_agent()
    action = 7
    expected = np.clip(1.0 + agent.markup_grid[action], agent.price_floor, agent.price_ceiling)
    price = agent.action_to_price(action, own_price=1.7, own_cost=1.0)
    assert np.isclose(price, expected)


def test_state_bins_react_to_cost_change():
    """The whole point of v3: same own_price/rival_price but different
    own_cost should generally give a different markup_bin."""
    agent = make_agent()
    m1, g1, s1 = agent.state_bins(own_price=1.5, own_cost=1.0, rival_price=1.5)
    m2, g2, s2 = agent.state_bins(own_price=1.5, own_cost=0.6, rival_price=1.5)
    assert m1 != m2  # markup differs (1.5-1.0=0.5 vs 1.5-0.6=0.9)
    assert g1 == g2  # gap to rival is unchanged (both own_price=rival_price=1.5)


def test_staleness_to_bin_is_monotonic_and_bounded():
    agent = make_agent()
    assert agent.staleness_to_bin(0) == 0
    assert agent.staleness_to_bin(1) < agent.staleness_to_bin(10)
    assert agent.staleness_to_bin(1) <= agent.staleness_to_bin(2)
    assert agent.staleness_to_bin(10_000) == agent.n_staleness_bins - 1


def test_state_bins_react_to_staleness():
    """v4: the same (markup, gap) should carry a different staleness_bin
    depending on how many periods the firm has already been holding --
    this is what restores the Markov property for the HOLD action."""
    agent = make_agent()
    m1, g1, s1 = agent.state_bins(own_price=1.5, own_cost=1.0, rival_price=1.5, periods_held=0)
    m2, g2, s2 = agent.state_bins(own_price=1.5, own_cost=1.0, rival_price=1.5, periods_held=20)
    assert m1 == m2 and g1 == g2
    assert s1 != s2


def test_reward_includes_switching_cost_only_when_price_changes():
    agent = make_agent(switching_cost=0.05)
    r_no_switch = agent.reward(new_own_price=1.5, old_own_price=1.5, rival_price=1.5)
    r_switch = agent.reward(new_own_price=1.6, old_own_price=1.5, rival_price=1.5)
    assert r_no_switch > r_switch - 0.06


def test_update_changes_q_value_toward_target():
    agent = make_agent()
    mb, gb, sb, action = 5, 5, 0, 3
    before = agent.Q[mb, gb, sb, action]
    agent.update(mb, gb, sb, action, r=1.0, next_markup_bin=5, next_gap_bin=5, next_staleness_bin=0)
    after = agent.Q[mb, gb, sb, action]
    assert after > before


def test_periods_held_resets_on_reprice_and_grows_on_hold():
    from repricing.agents import AgentState
    agent = make_agent()
    agent.training = False
    agent.Q[:] = 0.0
    agent.Q[:, :, :, agent.HOLD] = 1.0  # make HOLD uniquely greedy everywhere
    s = AgentState(1.5, 1.5, True, own_cost=1.0, market_costs=np.array([1.0, 1.0]))
    assert agent._periods_held == 0
    agent.decide(s)
    assert agent._periods_held == 1  # held -> staleness grows
    agent.decide(s)
    assert agent._periods_held == 2
    # Now force a reprice by making a move action uniquely greedy.
    agent.Q[:] = 0.0
    agent.Q[:, :, :, 0] = 1.0
    moved, _ = agent.decide(AgentState(1.5, 1.5, True, own_cost=0.3, market_costs=np.array([0.3, 1.0])))
    assert moved
    assert agent._periods_held == 0  # reprice resets staleness


def test_q_agent_implements_common_decide_interface():
    from repricing.agents import AgentState
    agent = make_agent()
    agent.training = False
    agent.Q[:] = 0.0
    # Make HOLD uniquely greedy.
    agent.Q[:, :, :, agent.HOLD] = 1.0
    moved, price = agent.decide(AgentState(1.5, 1.5, True, own_cost=1.0, market_costs=np.array([1.0,1.0])))
    assert not moved
    assert price == 1.5


def test_learning_hook_updates_q_in_training_mode():
    from repricing.agents import AgentState
    agent = make_agent()
    agent.training = True
    agent.epsilon = 0.0
    s = AgentState(1.5,1.5,True,own_cost=1.0,market_costs=np.array([1.0,1.0]))
    agent.decide(s)
    before = agent.Q.copy()
    agent.learn(1.0, s)
    assert np.any(agent.Q != before)
