import numpy as np

def PtEtaPhi_to_XYZE(pt, eta, phi):
    return pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta), pt*np.cosh(eta)

def PtEtaPhi_to_EXYZ(pt, eta, phi):
    return pt*np.cosh(eta), pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)

def XYZ_to_PtEtaPhi(x,y,z):
    pt = np.sqrt(np.square(x)+np.square(y))
    eta = np.arctanh(z/np.sqrt(np.square(x)+np.square(y)+np.square(z)))
    phi =  np.arctan2(x,y)+np.pi #0-2pi
    return pt, eta, phi