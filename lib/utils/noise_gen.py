import numpy as np
from scipy.special import beta as beta_func

def beta_exp_dist(x, n, alpha,beta, e):
    return n*(x**(alpha-1)*(1-x)**(beta-1))/beta_func(alpha,beta)*(1-np.exp(-e*x))

def sample(func_name, params, size, n_prop_samples=int(1e4), seed=None):
    rng = np.random.default_rng(seed)
    if(func_name == 'beta_exp'):
        x_range = np.linspace(0.0001,0.9999,n_prop_samples)
        p = beta_exp_dist(x_range, *params)
        p /= np.sum(p)
        return rng.choice(x_range, size=size, p=p)
    elif(func_name == 'normal'):
        return rng.normal(*params, size=size)
    elif(func_name == 'uniform'):
        return rng.uniform(*params, size=size)
    else:
        raise ValueError(f"No such noise generator: '{func_name}'")