import numpy as np
import h5py, os, re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import steganologger
from utils.colors import ColorIter
from utils.calc import extract_at_value
from jlogger import JLogger

logger = JLogger(comment="Plotting ROCs for different S/B ratios")
ratios = [0.5,0.1,0.05,0.01] #assumes that there are runs with names "SR={ratio:.3f}"


old_settings = np.seterr(divide='ignore')
coloriter = ColorIter()
counter = 0
inv_eps_b_at_30 = []
inv_eps_b_at_50 = []
eps_s_at_cut = []
fig, ax = plt.subplots()
for ratio in ratios:
    color = next(coloriter)
    file = h5py.File(os.path.join(f"SR={ratio:.3f}", "pred_best.h5"),'r')
    pred = np.array(file["pred"][:,1])
    label = np.array(file["label"])
    fpr, tpr, _ = roc_curve(label, pred)
    auroc = roc_auc_score(label, pred)
    file.close()
    print(extract_at_value(tpr, 0.3, fpr))
    inv_eps_b_at_30.append(extract_at_value(tpr, 0.3, 1/fpr)[1])
    inv_eps_b_at_50.append(extract_at_value(tpr, 0.5, 1/fpr)[1])
    eps_s_at_cut.append(extract_at_value(fpr, 1e-4, tpr)[1])
    time = 0
    with open(os.path.join(f"SR={ratio:.3f}", 'runtime.txt'), 'r') as f:
        s = f.read()
        mat = re.search(r"Train time: (\d+(\.\d+)?)", s)
        if(mat):
            time = float(mat.group(1))
    inv_fpr = 1/fpr
    inv_fpr[np.isinf(inv_fpr)] = np.nan
    ax.plot(tpr, inv_fpr, color=color, label=f"$S/B={ratio:.2f}$")
    ax.text(0.5,0.7-counter*0.05, f"AUROC={auroc:.4f}" + (f" T={time/60:.2f}m" if time>0 else ""), color=color, transform=ax.transAxes)
    counter += 1
np.seterr(**old_settings)
x = np.linspace(0.001,1,100)
ax.plot(x, 1/x, 'k--')

ax.set_xlabel(r"$\epsilon_S$")
ax.set_xlabel(r"$1/\epsilon_B$")
ax.grid()
ax.set_yscale('log')

fig.legend()
fig.suptitle("LorentzNet, 30k BG events")
logger.log_figure(fig, "rocs.png", data_file=", ".join([os.path.join(os.path.dirname(__file__), f"SR={ratio:.3f}", "pred_best.h5")]),
                         dpi=250, bbox_inches='tight')
data = {f"S/B={ratios[i]:.2f}":{"1/eps_b@eps_s=0.30": inv_eps_b_at_30[i], "1/eps_b@eps_s=0.50": inv_eps_b_at_50[i], "eps_s@eps_b=0.0001": eps_s_at_cut[i]} for i in range(len(ratios))}
steganologger.encode("rocs.png", data={"repeats": None, "run": os.path.dirname(__file__), "results":data}, overwrite=True)