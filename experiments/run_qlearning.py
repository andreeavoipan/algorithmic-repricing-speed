"""Q-learning robustness using the same simulator/timing as analytical agents."""
from __future__ import annotations
import copy, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from repricing.agents import MyopicBestResponseAgent
from repricing.qlearning import QLearningAgent
from repricing.simulation import DuopolyConfig, run_duopoly

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results"); os.makedirs(RESULTS_DIR,exist_ok=True)
A=np.array([2.,2.]); MU=.25; A0=0.; FLOOR,CEIL=1.,3.
SWITCHING_COSTS=[0.,.0005,.002,.008]
N_TRAIN=500_000; N_TRAIN_SEEDS=8; N_EVAL=5000; BURN_IN=500; N_EVAL_SEEDS=8


def rival():
    return MyopicBestResponseAgent(1,0,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,0.)


def train(cost, seed, gamma=0.0):
    q=QLearningAgent(0,1,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,
                     switching_cost=cost,alpha=.1,gamma=gamma,epsilon=1.,epsilon_min=.05,
                     epsilon_decay_steps=int(.6*N_TRAIN),training=True,
                     rng=np.random.default_rng(seed))
    run_duopoly(q,rival(),DuopolyConfig(T=N_TRAIN,burn_in=0,seed=seed,
        cost_shock_std=.01,cost_shock_std_1=.01,cost_rho=.98,record_decision_log=False))
    q.training=False; q.epsilon=0.
    return q


def evaluate(q, seed):
    agent=copy.deepcopy(q); agent.training=False; agent.epsilon=0.
    r=run_duopoly(agent,rival(),DuopolyConfig(T=N_EVAL,burn_in=BURN_IN,seed=10_000+seed,
        cost_shock_std=.01,cost_shock_std_1=.01,cost_rho=.98,record_decision_log=False))
    return r.effective_speed(0)


def main():
    rows=[]; seed_rows=[]
    for cost in SWITCHING_COSTS:
        vals=[]; seed_means=[]
        for ts in range(N_TRAIN_SEEDS):
            q=train(cost,ts)
            x=[evaluate(q,s) for s in range(N_EVAL_SEEDS)]; vals.extend(x); seed_means.append(float(np.mean(x)))
            seed_rows.append(dict(switching_cost=cost,train_seed=ts,reprice_rate_mean=float(np.mean(x)),reprice_rate_std=float(np.std(x))))
            print(f"cost={cost} train_seed={ts} rate={np.mean(x):.4f}",flush=True)
        sm=np.array(seed_means)
        rows.append(dict(switching_cost=cost,reprice_rate_mean=np.mean(vals),reprice_rate_std=np.std(vals),n_samples=len(vals),
                         seed_mean=sm.mean(),seed_median=float(np.median(sm)),seed_std=sm.std(ddof=1),
                         seed_ci95=1.96*sm.std(ddof=1)/np.sqrt(len(sm)),n_train_seeds=len(sm)))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS_DIR,"qlearning_grid.csv"),index=False)
    sd=pd.DataFrame(seed_rows); sd.to_csv(os.path.join(RESULTS_DIR,"qlearning_seeds.csv"),index=False)
    # Per-training-seed scatter: the effective sample size is the number of training seeds,
    # and the distribution can be bimodal, so means over seeds are shown next to the points.
    plt.figure(figsize=(7,5)); xs=range(len(SWITCHING_COSTS)); rng=np.random.default_rng(0)
    for k,cost in enumerate(SWITCHING_COSTS):
        y=sd[sd.switching_cost==cost].reprice_rate_mean.values
        plt.scatter(k+rng.uniform(-.12,.12,len(y)),y,s=28,alpha=.8)
        plt.plot([k-.25,k+.25],[np.median(y)]*2,color="k",lw=1.5)
    plt.xticks(list(xs),[str(c) for c in SWITCHING_COSTS]); plt.ylim(-.05,1.05)
    plt.xlabel("switching_cost"); plt.ylabel("Reprices / period (mean over 8 evaluations)")
    plt.title("Q-learning: one point per training seed (bar = median)")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"qlearning_effective_speed.png"),dpi=150)


if __name__=="__main__": main()
