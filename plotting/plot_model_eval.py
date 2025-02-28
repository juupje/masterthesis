import matplotlib.pyplot as plt
import h5py
import os,sys
import numpy as np
import argparse
import re
import importlib
from utils import ColorIter

if __name__=="__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import jlogger as jl
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.colors as mcolors

    visible_devices = tf.config.list_physical_devices("GPU")
    assert len(visible_devices)==1, "visible devices "+str(visible_devices)
    print("Using visible devices: ", visible_devices)
    colors = ColorIter()

    parser = argparse.ArgumentParser("Plot training and evaluation of collection of models")
    parser.add_argument("tag_run", help="Tag and run name", type=str)

    args = vars(parser.parse_args())
    OUT_DIR = os.path.join(os.path.dirname(__file__), args["tag_run"])
    sys.path.append(OUT_DIR)
    config = importlib.import_module("pn_tt_config")
    
    runs = []
    with open(os.path.join(OUT_DIR, "jobs.lst"), 'r') as f:
        for line in f.readlines():
            runs.append(re.search('\/([^\/]*)\/[^\/]+$', line).group(1))
    print(runs)
    
    #### TRAINING ######
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, sharex=True, sharey='row',figsize=(10,12))
    for run in runs:
        file = h5py.File(f"{OUT_DIR}/{run}/training_stats.h5", 'r')
        loss = np.array(file["loss"])
        val_loss = np.array(file["val_loss"])
        acc = np.array(file["accuracy"])
        val_acc = np.array(file["val_accuracy"])

        x = np.arange(1,len(loss)+1)
        color = next(colors)
        ax1.plot(x,loss, color=color,label=run)
        ax2.plot(x,val_loss, color=color)
        ax3.plot(x,acc, color=color)
        ax4.plot(x,val_acc, color=color)
        file.close()
    
    for ax in (ax1,ax2,ax3,ax4):
        ax.set_xlabel("Epoch")
        ax.grid()
    ax1.set_ylabel("Loss (cat. cross-entropy)")
    ax2.set_ylabel("Loss (cat. cross-entropy)")
    ax3.set_ylabel("Accuracy")
    ax4.set_ylabel("Accuracy")
    ax1.set_title("Training loss")
    ax2.set_title("Validation loss")
    ax3.set_title("Training accuracy")
    ax4.set_title("Validation accuracy")
    fig.legend()
    fig.suptitle("Training of ParticleNet-Lite on $t$-tagging")
    path = os.path.join(OUT_DIR,"training.png")
    print("Saving plot: " + path)
    fig.savefig(path, dpi=250)
    
    colors.rewind()
    ##### ROC ######
    print("Computing ROC curves...")
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True, sharex=True, figsize=(8,6))
    if(os.path.isfile(os.path.join(OUT_DIR,"test-results.h5"))):
        file = h5py.File(os.path.join(OUT_DIR,"test-results.h5"), 'r')
        do_test = False
    else:
        file = h5py.File(os.path.join(OUT_DIR,"test-results.h5"), 'w')
        do_test = True
    for run in runs:
        if(do_test):
            run_config = importlib.import_module(f"{run}.pn_tt_config")
            print(run)
            data_file = os.path.join(run_config.DATA_DIR, run_config.PREPROCESSED_TEST_DATA)
            print(data_file)
            data = h5py.File(data_file, mode='r')
            group = file.create_group(run)
            group_final = group.create_group("final")
            group_best = group.create_group("best")
            l = ["coords", "features", "mask"]
            bg = [np.array(data["background"][x]) for x in l]
            sn = [np.array(data["signal"][x]) for x in l]
            model = tf.keras.models.load_model(f"{OUT_DIR}/{run}/model-checkpoints/model-final.h5")
            p_bg = model.predict(bg, batch_size=run_config.BATCH_SIZE)
            p_sn = model.predict(sn, batch_size=run_config.BATCH_SIZE)
            group_final.create_dataset("p_bg", data=p_bg, compression='gzip')
            group_final.create_dataset("p_sn", data=p_sn, compression='gzip')
            data.close()
        else:
            p_bg = np.array(file[run]["final/p_bg"])
            p_sn = np.array(file[run]["final/p_sn"])
        #plot the ROC curve
        color = next(colors)
        y_true = np.concatenate([np.zeros(p_bg.shape[0]), np.ones(p_sn.shape[0])])
        y_pred = np.concatenate([p_bg[:,1], p_sn[:,1]])

        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        auroc = roc_auc_score(y_true, y_pred)
        #tpr is signal efficiency
        #tnr is background efficiency
        eps_s = tpr
        eps_b = fpr

        #extract training time
        with open(os.path.join(OUT_DIR, run, 'jobid.txt'), 'r') as f:
            txt = "".join(f.readlines())
            match = re.search(r"Train time: (\d+\.\d+)$", txt, re.MULTILINE)
            if(match):
                time = int(float(match.group(1)))
                h = time//3600
                time = f"T={h:d}h{(time-3600*h)//60:02d}"
            else:
                time = ""
        with np.errstate(divide='ignore'):
            ax1.plot(eps_s, 1/eps_b, color=color,label=re.sub("000$", "k", run) + f" $^{{AUC={auroc:.3f}}}_{{ {time:s} }}$")

        if(do_test):
            model = tf.keras.models.load_model(f"{OUT_DIR}/{run}/model-checkpoints/model-checkpoint.h5")
            p_bg = model.predict(bg, batch_size=run_config.BATCH_SIZE)
            p_sn = model.predict(sn, batch_size=run_config.BATCH_SIZE)
            group_best.create_dataset("p_bg", data=p_bg)
            group_best.create_dataset("p_sn", data=p_sn)
        else:
            p_bg = np.array(file[run]["best/p_bg"])
            p_sn = np.array(file[run]["best/p_sn"])
        y_true = np.concatenate([np.zeros(p_bg.shape[0]), np.ones(p_sn.shape[0])])
        y_pred = np.concatenate([p_bg[:,1], p_sn[:,1]])
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        #tpr is signal efficiency
        #tnr is background efficiency
        eps_s = tpr
        eps_b = fpr
        with np.errstate(divide='ignore'):
            ax2.plot(eps_s, 1/eps_b, color=color)

        #ax.text(0.8,0.8-i*0.05, f"AUROC={auroc:.2f}", color=color, transform=ax.transAxes)
        # random classifier
    file.close()
    x = np.linspace(0.001,1,100)
    for ax in (ax1,ax2):
        ax.plot(x, 1/x, 'k--')
        ax.set_yscale('log')
        ax.set_xlim(0,1)
        ax.set_ylim(1,None)
        ax.set_xlabel(r"$\epsilon_S$")
        ax.set_ylabel(r"$1/\epsilon_B$")
        ax.grid()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                        box.width, box.height * 0.8])
    ax1.set_title("Final model")
    ax2.set_title("Best model (val. loss.)")
    fig.suptitle("ROC curve for ParticleNet-Lite")
    fig.legend(loc='lower center', bbox_to_anchor=(0.5,0), ncol=min(3, len(runs)//2))
    path = os.path.join(OUT_DIR,"roc.png")
    fig.savefig(path, dpi=250)