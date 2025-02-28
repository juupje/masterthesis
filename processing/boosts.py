import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils
coloriter = utils.colors.ColorIter()
#first, try a sample jet
t = np.linspace(0.1,2*np.pi+0.1, 20, endpoint=False)
eta = 0.6*np.sin(t)
phi = 0.6*np.cos(t)
pt = 300*t

#load a real jet
file = h5py.File("/hpcwork/kd106458/data/toptagging/particlenet/test.processed-30.h5", 'r')
jet = np.array(file["background/features"][0,:,:3])
jet_total = np.sum(np.stack(utils.coords.PtEtaPhi_to_EXYZ(np.exp(jet[:,2]), jet[:,0], jet[:,1]),axis=1).T,axis=1)
_, eta_jet, phi_jet = utils.coords.XYZ_to_PtEtaPhi(jet_total[1], jet_total[2], jet_total[3])
jet[:,0] -= eta_jet
jet[:,1] -= phi_jet
jet[jet[:,1]>1,1] -= np.pi
jet[jet[:,1]<-1,1] += np.pi
jet = np.stack(utils.coords.PtEtaPhi_to_EXYZ(np.exp(jet[:,2]), jet[:,0], jet[:,1]),axis=1).T

p = np.stack(utils.coords.PtEtaPhi_to_EXYZ(pt, eta, phi),axis=1)
p = p.T

def plot(p, name):
    c = next(coloriter)
    pt, eta, phi = utils.coords.XYZ_to_PtEtaPhi(p[1], p[2], p[3])
    phi -= np.pi
    fig,ax = plt.subplots(figsize=(3,3))
    ax.scatter(eta, phi, s=pt/10, facecolors=c, edgecolors=None, alpha=0.4)
    ax.scatter(eta, phi, s=pt/10, facecolors='none', edgecolors=c)
    ax.set_xlabel(r"$\Delta\eta$")
    ax.set_ylabel(r"$\Delta\phi$")
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    fig.savefig(f"boosts/{name}.png", dpi=250, bbox_inches='tight')

plot(p, "initial")
plot(jet, "initial_jet")
#Apply y-z rotation
alpha = -np.pi/2
R = np.array([[1,0,0,0],[0,1,0,0], [0,0,np.cos(alpha), -np.sin(alpha)], [0,0,np.sin(alpha), np.cos(alpha)]])
Rp = R@p
plot(Rp, "rotate_y-z")
plot(R@jet, "rotate_y-z_jet")

#Apply x-t boost
beta = 0.4
L = np.array([[np.cosh(beta), np.sinh(beta), 0, 0], [np.sinh(beta), np.cosh(beta), 0, 0], [0,0,1,0], [0,0,0,1]])
Lp = L@p
plot(Lp, "boost_x-t")
plot(L@jet, "boost_x-t_jet")

# z–t boost with x–z rotation
beta = 0.2
zero = np.array(utils.coords.PtEtaPhi_to_EXYZ(pt[-1], 0, 0)).reshape(4,1)
L = np.array([[np.cosh(beta), 0, 0, np.sinh(beta)], [0,1,0,0], [0,0,1,0], [np.sinh(beta), 0, 0, np.cosh(beta)]])
r = L@zero
alpha = -np.arctan2(r[3,0], r[1,0])
alpha = 0.1
R = np.array([[1,0,0,0], [0, np.cos(alpha), 0, -np.sin(alpha)], [0,0,1,0], [0, np.sin(alpha), 0, np.cos(alpha)]])
tiltz_p = R@L@p
print("Zero check: ", np.allclose((R@r)[2:],0))
plot(tiltz_p, "tilt-z")
plot(R@L@jet, "tilt-z_jet")

# y–t boost with y–z rotation.
zero = np.array(utils.coords.PtEtaPhi_to_EXYZ(pt[-1], 0, 0)).reshape(4,1)
L = np.array([[np.cosh(beta), 0, np.sinh(beta), 0], [0,1,0,0], [np.sinh(beta), 0, np.cosh(beta), 0], [0,0,0,1]])
r = L@zero
print(r)
#alpha = -np.arctan2(r[2,0], r[3,0])
alpha = 0.1
R = np.array([[1,0,0,0], [0,1,0,0], [0,0,np.cos(alpha), -np.sin(alpha)], [0,0,np.sin(alpha), np.cos(alpha)]])
tilty_p = R@L@p
print("Zero check: ", np.allclose((R@r)[2:],0))
print(R@r)
plot(tilty_p, "tilt-y")
plot(R@L@jet, "tilt-y_jet")
