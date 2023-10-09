import numpy as np
from typing import Tuple

def calculate_bins(values:np.ndarray, nbins:int, low:float=None, high:float=None) -> np.ndarray:
    start = low if low is not None else np.min(values)
    end = high if high is not None else np.max(values)
    step = max(1,np.round((end-start)/nbins))
    return np.arange(start,end+step,step)

def create_chunks(chunksize:int=512,start:int=0,total_size:int=1000000) -> list:
    chunks = int(np.ceil(total_size/chunksize))
    return [(i*chunksize+start,min(total_size,(i+1)*chunksize)+start) for i in range(chunks)]

def get_topK(particles:np.ndarray, k:int, n_feat:int, feat_idx:int) -> Tuple[np.ndarray, np.ndarray]:
    particles = particles.reshape(particles.shape[0], particles.shape[1]//n_feat,n_feat) #(N,P,F) #split into particles
    idx = np.argsort(particles[:,:,feat_idx],axis=1)[:,-k:] #(N,K) take top k of the feat_idx'th feature
    #invert order
    idx = idx[:,::-1]
    b_idx = np.tile(np.reshape(np.arange(idx.shape[0]), (-1,1)), (1,idx.shape[1])) #(N,K) [[0,0..],[1,1..],...]
    topk = particles[b_idx,idx] #(N,K,F)
    topk = topk.reshape((topk.shape[0],-1)) #(N,K*F)
    return topk, idx

def get_nparticles(array:np.ndarray) -> int:
    """
    Uses a binary search to find the number of particles in the array.
    The array has to be formatted as [pt0, eta0, phi0, pt1, eta1, phi1,...]
        and should contain at least 1 particle and no pt equal to zero.
    """
    start = 0
    end = N = len(array)//3
    idx = (start+end)//2
    while idx>=0 and idx < N:
        if(array[idx*3]==0):
            if(array[(idx-1)*3]!=0):
                return idx
            end = idx
            idx = min(idx-1,(start+idx)//2)
        else:
            start = idx
            idx = max(idx+1,(idx+end)//2)
    return idx

def extract_at_value(arr:np.ndarray, value:float, others:list|np.ndarray, thres:float=1e-3):
    if(isinstance(others, (list,tuple))):
        for other in others:
            if(other.shape!=arr.shape):
                raise ValueError(f"Lengths of arrays do not match {len(other):d} was expected to be {len(arr):d}")
    diff = np.abs(arr-value)
    idx = np.argmin(diff)
    if(diff[idx]<thres):
        return idx, ([other[idx] for other in others] if isinstance(others,(list,tuple)) else others[idx])
    else:
        return None
    
def linreg(x,y,e_y=None, e_x=None): 
    if(e_y is not None):
        w_sum = np.sum(1./e_y**2)
        w_sum_x = np.sum(x/e_y**2)
        w_sum_y = np.sum(y/e_y**2)
        w_sum_xx = np.sum(x**2/e_y**2)
        w_sum_xy = np.sum(x*y/e_y**2)
        delta = w_sum*w_sum_xx-w_sum_x**2
        b = (w_sum_xx*w_sum_y-w_sum_x*w_sum_xy)/delta
        a = (w_sum*w_sum_xy-w_sum_x*w_sum_y)/delta
        eb = np.sqrt(w_sum_xx/delta)
        ea = np.sqrt(w_sum/delta)
        cov = -w_sum_x/delta
        corr = cov/(ea*eb)
        chiq = np.sum(((y-a*x-b)/e_y)**2)
        
        if(e_x is not None):
            import scipy.odr as odr
            def f(B,x):
                return B[0]*x+B[1]
            model = odr.Model(f)
            data = odr.RealData(x,y, sx=e_x, sy=e_y)
            odrfunc = odr.ODR(data, model, beta0=[a,b])
            output = odrfunc.run()
            ndof = len(x)-2
            chiq = output.res_var*ndof
            corr = output.cov_beta[0,1]/np.sqrt(output.cov_beta[0,0]*output.cov_beta[1,1])
            return output.beta[0], np.sqrt(output.cov_beta[0,0]), output.beta[1], np.sqrt(output.cov_beta[1,1]), chiq, corr
            
        return (a, ea, b, eb, chiq, corr)
    else:
        x_avg = np.mean(x)
        y_avg = np.mean(y)
        sxy = np.sum([(x[i]-x_avg)*(y[i]-y_avg) for i in range(len(x))])
        sxx = np.sum((x-x_avg)**2)
        syy = np.sum((y-y_avg)**2)
        a = sxy/sxx
        b = y_avg - a*x_avg
        corr = sxy/np.sqrt(sxx*syy)
        return a, b, corr