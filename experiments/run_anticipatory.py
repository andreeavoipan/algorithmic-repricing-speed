"""Endogenous strategic-reaction experiment.

Unlike cost_threshold, reaction_probability is not an adjustment cost.  It changes
how much the tested firm internalises the rival's anticipated best response.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","src"))
from repricing.agents import AnticipatoryBestResponseAgent, MyopicBestResponseAgent
from repricing.simulation import DuopolyConfig, run_duopoly

RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results"); os.makedirs(RESULTS_DIR,exist_ok=True)
A=np.array([2.,2.]); MU=.25; A0=0.; FLOOR,CEIL=1.,3.
REACTION_PROBS=[0.,.25,.5,.75,1.]; TECH_FREQS=np.linspace(.1,1.,10); N_SEEDS=20; T=5000; BURN_IN=500


def main():
    rows=[]
    for rho in REACTION_PROBS:
        for tech in TECH_FREQS:
            vals=[]
            for seed in range(N_SEEDS):
                a0=AnticipatoryBestResponseAgent(0,1,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,
                                                  cost_threshold=0.,reaction_probability=rho)
                a1=MyopicBestResponseAgent(1,0,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,0.)
                r=run_duopoly(a0,a1,DuopolyConfig(T=T,burn_in=BURN_IN,tech_freq_0=tech,
                    seed=seed,cost_shock_std=.01,cost_shock_std_1=.01,cost_rho=.98))
                vals.append(r.effective_speed(0))
            rows.append(dict(reaction_probability=rho,tech_freq=tech,
                             reprice_rate_mean=np.mean(vals),reprice_rate_std=np.std(vals)))
    df=pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS_DIR,"anticipatory_grid.csv"),index=False)
    plt.figure(figsize=(7,5))
    for rho in REACTION_PROBS:
        s=df[df.reaction_probability==rho]
        plt.plot(s.tech_freq,s.reprice_rate_mean,marker="o",label=f"reaction probability={rho}")
    plt.xlabel("Technical recomputation probability"); plt.ylabel("Reprices / period")
    plt.title("Endogenous strategic anticipation and repricing speed"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"anticipatory_speed.png"),dpi=150)


if __name__=="__main__": main()
