import numpy as np

def PtEtaPhi_to_XYZE(pt, eta, phi):
    return pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta), pt*np.cosh(eta)

def PtEtaPhi_to_EXYZ(pt, eta, phi):
    return pt*np.cosh(eta), pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)

def XYZ_to_PtEtaPhi(px,py,pz):
    pt = np.sqrt(np.square(px)+np.square(py))
    x = np.sqrt(np.square(px)+np.square(py)+np.square(pz))
    x[np.isclose(x,0)]=1e-20 #regularizer
    eta = np.arctanh(pz/x)
    phi =  np.arctan2(py,px)+np.pi #0-2pi
    return pt, eta, phi
