import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from repricing.demand import logit_quantities, profits, profit_of_firm


def test_quantities_sum_below_one():
    q = logit_quantities(np.array([1.5, 1.5]), a=np.array([2.0, 2.0]), mu=0.25)
    assert q.sum() < 1.0
    assert np.all(q > 0)


def test_symmetric_prices_give_symmetric_quantities():
    q = logit_quantities(np.array([1.5, 1.5]), a=np.array([2.0, 2.0]), mu=0.25)
    assert np.isclose(q[0], q[1])


def test_lower_price_gives_higher_quantity():
    q = logit_quantities(np.array([1.2, 1.8]), a=np.array([2.0, 2.0]), mu=0.25)
    assert q[0] > q[1]


def test_profit_of_firm_matches_profits_vector():
    prices = np.array([1.4, 1.6])
    costs = np.array([1.0, 1.0])
    a = np.array([2.0, 2.0])
    mu = 0.25
    full = profits(prices, costs, a, mu)
    single = profit_of_firm(1.4, firm_index=0, rival_prices=prices, costs=costs, a=a, mu=mu)
    assert np.isclose(full[0], single)
