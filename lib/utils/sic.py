import numpy as np
import numpy.typing as npt

def sic_curve(n_background, tpr:npt.NDArray, fpr:npt.NDArray=None, inv_fpr:npt.NDArray=None, max_err=0.2):
    if(inv_fpr is None):
        if(fpr is None): raise ValueError("Both fpr and inv_fpr are None")
        inv_fpr = 1/(fpr+1e-30)
        inv_fpr[fpr<=1e-29] = np.nan
    inds = inv_fpr < n_background*max_err**2
    return tpr[inds], tpr[inds]*np.sqrt(inv_fpr[inds])
