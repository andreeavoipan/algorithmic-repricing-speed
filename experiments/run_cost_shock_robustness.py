"""Robustness of repricing-rate result to cost-shock intensity."""
from __future__ import annotations
import os,sys
import numpy as np
import pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
from repricing.agents import MyopicBestResponseAgent
from repricing.simulation import DuopolyConfig,run_duopoly
A=np.array([2.,2.]); MU=.25; A0=0.; FLOOR,CEIL=1.,3.
SIGMAS=[.0025,.005,.01,.02]; THRESHOLDS=[0.,.0005,.002,.008]; N_SEEDS=20; T=5000; BURN_IN=500
RESULTS_DIR=os.path.join(os.path.dirname(__file__),"..","results"); os.makedirs(RESULTS_DIR,exist_ok=True)

def main():
    rows=[]
    for sigma in SIGMAS:
        for th in THRESHOLDS:
            vals=[]
            for seed in range(N_SEEDS):
                x=MyopicBestResponseAgent(0,1,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,th)
                y=MyopicBestResponseAgent(1,0,np.array([1.,1.]),A,MU,A0,FLOOR,CEIL,0.)
                r=run_duopoly(x,y,DuopolyConfig(T=T,burn_in=BURN_IN,seed=seed,cost_shock_std=sigma,cost_shock_std_1=sigma,cost_rho=.98))
                vals.append(r.effective_speed(0))
            rows.append(dict(sigma=sigma,cost_threshold=th,reprice_rate_mean=np.mean(vals),reprice_rate_std=np.std(vals)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR,"cost_shock_robustness.csv"),index=False)

if __name__=="__main__": main()
