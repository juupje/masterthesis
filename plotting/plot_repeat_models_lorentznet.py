import numpy as np
import h5py, os, re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import utils, steganologger

dirs = {"N10": "N10", "N20": "N20", "N30": "N30", "N40": "N40"}
REPEATS = 10
old_settings = np.seterr(divide='ignore')
fig, ax = plt.subplots()
coloriter = utils.ColorIter()
counter = 0
for key in dirs:
    inv_eps_bs = []
    tpr = None
    aucs = []
    times = []
    color = next(coloriter)
    for repeat in range(REPEATS):
        file = h5py.File(os.path.join(dirs[key], str(repeat), "pred_best.h5"))
        pred = np.array(file["pred"][:,1])
        label = np.array(file["label"])
        fpr_p, tpr_p, _ = roc_curve(label, pred)
        #ax.plot(tpr_p, 1/fpr_p, color=color, alpha=0.2, linewidth=1)
        aucs.append(roc_auc_score(label, pred))
        if(tpr is None):
            tpr = tpr_p
            fpr = fpr_p
        else:
            fpr = np.interp(tpr, tpr_p, fpr_p)
        inv_eps_b = 1/fpr
        inv_eps_b[np.isinf(inv_eps_b)] = np.nan
        inv_eps_bs.append(inv_eps_b)
        file.close()
        with open(os.path.join(dirs[key], str(repeat), 'runtime.txt'), 'r') as f:
            s = f.read()
            mat = re.search(r"Train time: (\d+(\.\d+)?)", s)
            if(mat):
                times.append(float(mat.group(1)))

    stack = np.vstack(inv_eps_bs)
    idx = np.any(np.isnan(stack),axis=0)
    stack = stack[:,~idx]
    tpr = tpr[~idx]
    avg = np.mean(stack,axis=0)
    std = np.std(stack, axis=0)
    print(np.where(std>avg))
    ax.plot(tpr, avg, color=color, label=key)
    ax.fill_between(tpr, avg-std, avg+std, color=color, alpha=0.3)
    ax.text(0.5,0.7-counter*0.05, f"AUROC={np.mean(aucs):.4f}, T={np.mean(times)/60:.2f}m", color=color, transform=ax.transAxes)
    counter += 1
np.seterr(**old_settings)
x = np.linspace(0.001,1,100)
ax.plot(x, 1/x, 'k--')

# plot reference!
pred = np.load(os.path.join(os.getenv("HOME"), "gits", "LorentzNet-release", "logs", "top", "TestN20_2", "score.npy"))
fpr, tpr, thesholds = roc_curve(pred[...,0], pred[...,2])
#tpr is signal efficiency
#tnr is background efficiency
eps_s = tpr
eps_b = fpr
color = next(coloriter)
ax.plot(eps_s, 1/eps_b, color=color,label="PyTorch, N=20")
dfpr = fpr[1:]-fpr[:-1]
auroc = roc_auc_score(pred[...,0],pred[...,2])
ax.text(0.5,0.7-counter*0.05, f"AUROC={auroc:.4f}", color=color, transform=ax.transAxes)

ax.set_xlabel(r"$\epsilon_S$")
ax.set_xlabel(r"$1/\epsilon_B$")
ax.grid()
ax.set_yscale('log')

fig.legend()
fig.suptitle("LorentzNet, 60k events")
fig.savefig("rocs.png", dpi=250, bbox_inches='tight')
steganologger.encode("rocs.png", data={"repeats": REPEATS, "error_band": "std", "run": os.path.dirname(__file__)}, overwrite=True)