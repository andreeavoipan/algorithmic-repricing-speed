import sys,os,json,numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); SRC=os.path.join(HERE,'..','..','src')
sys.path.insert(0,SRC)
from repricing.agents import MyopicBestResponseAgent,AnticipatoryBestResponseAgent
from repricing.simulation import DuopolyConfig,run_duopoly
OUT=os.path.join(HERE,'..','..','results','extra'); os.makedirs(OUT,exist_ok=True)

def run_one(spec):
    """spec: dict(kind, C, rho, tau, omega, seed, a, mu, a0, sigma, belief_sd, T)"""
    a=np.array(spec.get('a',[2.,2.])); mu=spec.get('mu',.25); a0=spec.get('a0',0.)
    common=dict(costs=np.array([1.,1.]),a=a,mu=mu,a0=a0,price_floor=1.,price_ceiling=3.)
    C=spec.get('C',0.)
    kind=spec.get('kind','myopic')
    if kind=='anticip':
        x=AnticipatoryBestResponseAgent(0,1,cost_threshold=C,reaction_probability=spec['rho'],**common)
    elif kind=='belief':
        from repricing.beliefs import UncertainBeliefAgent
        x=UncertainBeliefAgent(0,1,cost_threshold=C,belief_sd=spec['belief_sd'],**common)
    else:
        x=MyopicBestResponseAgent(0,1,cost_threshold=C,**common)
    y=MyopicBestResponseAgent(1,0,cost_threshold=0.,**common)
    sg=spec.get('sigma',.01)
    r=run_duopoly(x,y,DuopolyConfig(T=spec.get('T',5000),burn_in=500,tech_freq_0=spec.get('tau',1.),tech_freq_1=1.,
        obs_freq_0=spec.get('omega',1.),obs_freq_1=1.,seed=spec['seed'],cost_shock_std=sg,cost_shock_std_1=sg,cost_rho=.98,
        record_decision_log=False,**({'init_price_0':spec['init'],'init_price_1':spec['init']} if 'init' in spec else {})))
    ph=r.price_history[500:];pr=r.profit_history[500:]
    sp=r.effective_speed(0)
    return dict(spec=spec,speed0=sp,speed1=r.effective_speed(1),price0=float(ph[:,0].mean()),price1=float(ph[:,1].mean()),
                profit0=float(pr[:,0].mean()),profit1=float(pr[:,1].mean()),net0=float(pr[:,0].mean()-C*sp))

def key(s): return json.dumps(s,sort_keys=True)
def _w(s): return run_one(s)
def run_all(specs,fname,procs=2):
    f=f'{OUT}/{fname}.jsonl'; done=set()
    if os.path.exists(f):
        for l in open(f):
            try: d=json.loads(l); done.add(key(d['spec']))
            except: pass
    todo=[s for s in specs if key(s) not in done]
    print(f'{fname}: {len(specs)} total, {len(todo)} to do',flush=True)
    with Pool(procs) as p, open(f,'a') as fh:
        for i,r in enumerate(p.imap_unordered(_w,todo,chunksize=1)):
            fh.write(json.dumps(r)+'\n'); fh.flush()
            if i%20==0: print(fname,i,'/',len(todo),flush=True)
    print(fname,'DONE',flush=True)
