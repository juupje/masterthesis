import matplotlib.pyplot as plt
import os, sys
import numpy as np
import utils
import h5py
from scipy.optimize import curve_fit
from scipy.special import beta as beta_func, gamma as gamma_func

def gamma_dist(x, n, k, theta, e):
    return n*(x**(k-1)*np.exp(-x/theta))/(gamma_func(k)*theta**k)*(1-np.exp(-e*x))

def beta_dist(x, n, alpha,beta, e):
    return n*(x**(alpha-1)*(1-x)**(beta-1))/beta_func(alpha,beta)*(1-np.exp(-e*x))

PLOT_DIR = os.path.join(os.getenv("HOME"), "Analysis", "plots", "pt_900-hists")
DATA_DIR = "/scratch/work/geuskens/data/lhco/new/pt900"
file_bg = h5py.File(os.path.join(DATA_DIR, "N100-bg.h5"))
file_sn = h5py.File(os.path.join(DATA_DIR, "N100-sn.h5"))

n_events = 50000
deltaR_j1_bg = np.array(file_bg["jet1/features"][:n_events, :30, -1])
deltaR_j1_sn = np.array(file_sn["jet1/features"][:n_events, :30, -1])
deltaR_j2_bg = np.array(file_bg["jet2/features"][:n_events, :30, -1])
deltaR_j2_sn = np.array(file_sn["jet2/features"][:n_events, :30, -1])

mean = np.mean(deltaR_j1_bg)
var = np.var(deltaR_j1_bg)
print("Mean: ", mean)
print("Var: ", var)

mu_hat = mean/(1-mean)
beta = 2.5#(mu_hat/(var*(1+mu_hat)**2)-1)/(mu_hat+1)
alpha = 1.02
print(alpha, beta)
'''samples = np.random.beta(alpha, beta,  size=int(1e6))
print("Mean poisson:", np.mean(samples), " - ", alpha/(alpha+beta))
print("Var poisson:", np.var(samples), " - ", alpha*beta/((alpha+beta)**2*(alpha+beta+1)))
print("Max poisson:", np.max(samples))
print("Min poisson:", np.min(samples))'''

fig, (axes,axes2) = plt.subplots(2,2, figsize=(8,10))
#axes2[0].hist(samples, bins=1000, density=True)
for col, bg, sn in zip((0,1), (deltaR_j1_bg, deltaR_j2_bg), (deltaR_j1_sn, deltaR_j2_sn)):
    y,bins,_ = axes[col].hist(bg[bg!=0].flatten(), bins=1000, histtype='step', density=True, label="Background")
    if(col == 0):
        x = (bins[1:]+bins[:-1])/2
        idx = x < 1
        idx_fit = (x > 0.005) & (x < 1)
        print("BETA:")
        popt, pcov = curve_fit(beta_dist, x[idx_fit],y[idx_fit], p0=(1,0.6,3, 10), bounds=[(0.1,0,0, 0), (100, 5, 50, 100)])
        print(f"{popt=}, {pcov=}")
        axes[col].plot(x[idx], beta_dist(x[idx], *popt), label='beta fit')

        x_range = np.linspace(0.0001,0.9999,int(1e4))
        p = beta_dist(x_range, *popt)
        p /= np.sum(p)
        samples = np.random.choice(x_range, size=int(1e6), p=p)
        axes[col].hist(samples, bins=1000, density=True, label='samples')
    axes[col].hist(sn[sn!=0].flatten(), bins=1000, histtype='step', density=True, label="Signal")
    axes[col].set_title(rf"$\Delta R_{{j{col+1}}}$")
    axes[col].set_xlabel(r"$\Delta R$")
    axes[col].set_ylabel("freq.")
    axes[col].set_xlim(0,1)
axes[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "deltaR_hist.png"), dpi=250)