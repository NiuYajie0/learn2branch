
#%% import
import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip

import tensorflow as tf
# import tensorflow.contrib.eager as tfe

import utilities
from utilities import log

# %%
from utilities_tf import load_batch_gcnn
from S03_train_gcnn import load_batch_tf

# %%
load_batch_gcnn('data\\samples\\facilities\\100_100_5\\train\\sample_1.pkl')
# %%
