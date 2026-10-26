"""Robustness check for non-myopic Q-learning.

This experiment is intentionally separate from the apples-to-apples gamma=0 run.
It asks whether the learned repricing-rate pattern survives when the agent values
future rewards.  Switching costs are kept on the same per-period payoff scale, so
results should be interpreted as a sensitivity check rather than as an identical
parameterization.
"""
from __future__ import annotations
import copy, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.agents import MyopicBestResponseAgent
from repricing.qlearning import QLearningAgent
from repricing.simulation import DuopolyConfig, run_duopoly

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results"); os.makedirs(RESULTS_DIR,exist_ok=True)
A=np.array([2.,2.]); MU=.25; A0=0.; FLOOR,CEIL=1.,3.
GAMMAS=[0.,.5,.9,.95]; SWITCHING_COSTS=[0.,.002,.008]
N_TRAIN=300_000; N_TRAIN_SEEDS=4; N_EVAL=5000; BURN_IN=500; N_EVAL_SEEDS=4

def rival():
    return MyopicBestResponseAgent(1,0,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,0.)

def train(gamma,cost,seed):
    q=QLearningAgent(0,1,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,
        switching_cost=cost,alpha=.1,gamma=gamma,epsilon=1.,epsilon_min=.05,
        epsilon_decay_steps=int(.6*N_TRAIN),training=True,rng=np.random.default_rng(seed))
    run_duopoly(q,rival(),DuopolyConfig(T=N_TRAIN,seed=seed,cost_shock_std=.01,
        cost_shock_std_1=.01,cost_rho=.98,record_decision_log=False))
    q.training=False; q.epsilon=0.
    return q

def evaluate(q,seed):
    z=copy.deepcopy(q); z.training=False; z.epsilon=0.
    r=run_duopoly(z,rival(),DuopolyConfig(T=N_EVAL,burn_in=BURN_IN,seed=50_000+seed,
        cost_shock_std=.01,cost_shock_std_1=.01,cost_rho=.98,record_decision_log=False))
    return r.effective_speed(0)

def main():
    rows=[]
    for gamma in GAMMAS:
        for cost in SWITCHING_COSTS:
            vals=[]
            for ts in range(N_TRAIN_SEEDS):
                q=train(gamma,cost,ts)
                vals += [evaluate(q,s) for s in range(N_EVAL_SEEDS)]
            rows.append(dict(gamma=gamma,switching_cost=cost,reprice_rate_mean=np.mean(vals),
                             reprice_rate_std=np.std(vals),n_samples=len(vals)))
            print(gamma,cost,np.mean(vals))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR,"qlearning_gamma_robustness.csv"),index=False)

if __name__=="__main__": main()
