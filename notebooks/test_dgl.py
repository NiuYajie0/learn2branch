#%%
import os
os.chdir("../")

os.environ["DGLBACKEND"] = "pytorch"

#%%
# import numpy as np
# import tensorflow as tf
# from dgl.nn import GATConv
# %%
import dgl
import numpy as np
import torch as th
from dgl.nn import GATConv

# %%
# with tf.device("CPU:0"):
g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
g = dgl.add_self_loop(g)
feat = th.ones(6, 10)
gatconv = GATConv(10, 2, num_heads=3)
res = gatconv(g, feat)
res
# %%
