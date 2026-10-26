from common import *
N=12
specs=[]
for init in [1.2,1.5,1.8]:
    for s in range(N):
        specs.append(dict(kind='myopic',C=.008,tau=1.,seed=s,init=init))
        specs.append(dict(kind='myopic',C=0.,tau=1.,seed=s,init=init))
        specs.append(dict(kind='anticip',C=0.,rho=1.,tau=1.,seed=s,init=init))
if __name__=='__main__': run_all(specs,'exp2b')
