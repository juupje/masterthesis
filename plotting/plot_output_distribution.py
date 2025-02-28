#Plots the distributions of the value of the output nodes of the model
#Meant to be copied into the working directory of a job without a grid run
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec 
import os
import numpy as np
import h5py
import re
import config
import glob
import jlogger as jl
from sklearn.metrics import roc_curve, roc_auc_score

logger = jl.JLogger()
processed_file = re.sub("^.*/data", os.getenv("DATA_DIR"), config.PROCESSED_FILE)
if(os.path.isfile("test-results.h5")):
    file = h5py.File("test-results.h5", 'r')
    do_test = False
else:
    import tensorflow as tf
    file = h5py.File("test-results.h5", 'w')
    do_test = True
    data = h5py.File(processed_file, mode='r')

model_names = glob.glob("model-checkpoints/*.h5")
name_seq = re.compile("^.*/(.*).h5$")
x = np.linspace(0.001,1,100)
inv_x = 1/x

aurocs = []
epochs = []
for model_name in model_names:
    name = name_seq.search(model_name).group(1)
    print(model_name)
    if(do_test):
        group = file.create_group(name)
        l = ["coords1", "features1", "mask1", "coords2", "features2", "mask2"]
        bg = [np.array(data["x_test_bg"][x]) for x in l]
        sn = [np.array(data["x_test_sn"][x]) for x in l]
        model = tf.keras.models.load_model(model_name)
        p_bg = model.predict(bg, batch_size=config.BATCH_SIZE)
        p_sn = model.predict(sn, batch_size=config.BATCH_SIZE)
        group.create_dataset("p_bg", data=p_bg, compression='gzip')
        group.create_dataset("p_sn", data=p_sn, compression='gzip')
    else:
        p_bg = np.array(file[name]["p_bg"])
        p_sn = np.array(file[name]["p_sn"])
    #plot the probabilities
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 1:3])
    ax1.hist(p_bg[:,0], bins=100, histtype="step", label="Correct (0)")
    ax1.hist(p_bg[:,1], bins=100, histtype="step", label="Incorrect (1)")
    ax2.hist(p_sn[:,0], bins=100, histtype="step", label="Incorrect (0)")
    ax2.hist(p_sn[:,1], bins=100, histtype="step", label="Correct (1)")
    ax1.set_title("Background")
    ax2.set_title("Signal")
    for ax in (ax1,ax2):
        ax.set_xlabel("Score")
        ax.set_ylabel("Rel. freq.")
        ax.set_xlim(0,1)
        ax.set_yscale("log")
        ax.set_ylim(1, 10**4)
        ax.grid()
        ax.legend(loc='upper right')

    y_true = np.concatenate([np.zeros(p_bg.shape[0]), np.ones(p_sn.shape[0])])
    y_pred = np.concatenate([p_bg[:,1], p_sn[:,1]])
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    auroc = roc_auc_score(y_true, y_pred)
    eps_s = tpr
    eps_b = fpr
    ax3.plot(eps_s, 1/eps_b,label=f"AUC={auroc:.4f}")
    ax3.plot(x,inv_x, color='k', linestyle='--')
    ax3.set_xlabel(r"$\epsilon_S$")
    ax3.set_ylabel(r"$1/\epsilon_B$")
    ax3.set_title("ROC curve")
    ax3.set_yscale("log")
    ax3.grid()
    ax3.legend()

    m = re.search(r"(\d+)\.h5", model_name)
    if(m):
        epochs.append(int(m.group(1)))
        aurocs.append(auroc)
    elif("final" in name):
        epochs.append(50)
        aurocs.append(auroc)

    fig.suptitle(name.replace("-", " "))
    #fig.tight_layout()
    fig.savefig(f"score_dist-{name:s}.png", dpi=200)

#plot aurocs

if(do_test):
    logger.log_data(os.path.abspath("test-results.h5"),
        comment="Output of the network after different number of epochs",
        data_used=processed_file)
    logger.log_figure(None, "score-dist-*.png")
    data.close()
