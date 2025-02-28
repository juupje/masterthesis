# %%
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0,"/home/joep/Documents/Gits/thesis_hpc/lib")
import utils
import numpy as np
from models import lorentznetV2 as lorentznet
from models.feature_layers import AdjacencyMatrix
from models.CustomModels import SupervisedModel

# %%
config = utils.parse_config("config.py")
config["BATCH_SIZE"] = 256

# %%
from keras import ops as ko
input_shapes = dict(coordinates=(config["N"],4), mask=(config["N"],1))
if(config["N_FEATURES"]!=0): #add scalar features
    input_shapes["scalars"] = (config["N"],config["N_FEATURES"]+(config.get("NOISE_FEATURES") or 0))
model = lorentznet.lorentz_net(input_shapes=input_shapes, ln_params=config["LN_PARAMS"],
            mlp_params=dict(phi_e=config["LN_PHI_E"], phi_x=config["LN_PHI_X"], phi_h=config["LN_PHI_H"], phi_m=config["LN_PHI_M"]),
            no_scalars=config["N_FEATURES"]==0, model_class=SupervisedModel, **config.get("MODEL_ARGS", {}))

# %%
from main import get_data
train_gen, val_gen = get_data(config)

# %%
x,y = train_gen[8]
print([xi.shape for xi in x],y.shape)
n_particles = np.sum(x[1], axis=(1,2))
print(n_particles[-10:])

# %%
mask = ko.squeeze(ko.not_equal(x[1], 0),axis=2)
node_mask, edge_mask = mask, lorentznet.create_edge_mask(mask)
edges, seg_ids = AdjacencyMatrix(ko.shape(x[0])[1])(node_mask, edge_mask)

# %%
i,j = edges
seg_i, seg_j, idx_down, idx_up = seg_ids
print(i)
print(seg_i)
print(j)
print(seg_j)
print(idx_down)
print(idx_up)

# %%
coords = ko.reshape(x[0], (-1, *x[0].shape[2:]))
ko.take(ko.segment_sum(ko.take(coords,seg_j,axis=0), segment_ids=seg_i, num_segments=None, sorted=True), idx_down,axis=0)

# %%
print(model(x))


