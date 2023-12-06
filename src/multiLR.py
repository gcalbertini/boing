## Standard libraries
import os
import json
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg", "pdf")  # For export
import seaborn as sns

sns.set()

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/"


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True  #  bool that, if True, causes cuDNN to only use deterministic convolution algorithms.
torch.backends.cudnn.benchmark = False  # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# Fetching the device that will be used throughout
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

'''
Some notes:
epoch = 1 forward and backward pass of ALL train instances
batch size = number of training instances in one forward & backward pass
number of iters = number of passes, each pass using [batch size] num samples
100 samples, BS of 20 --> 100/20 = 5 iters/epoch;
if had 90/20 with drop_last as True in dataloader would get 4 iter/ep for consistent batch size
'''

class load_data:
    def __init__(self):
        X_train = np.loadtxt(
            "./data/py_training_X.csv", delimiter=",", dtype=np.float64
        )  # float64 used in nb
        X_test = np.loadtxt(
            "./data/py_training_X.csv", delimiter=",", dtype=np.float64
        )
        X_features = np.loadtxt(
            "./data/py_X_ft_names.csv", delimiter=",", dtype=np.str_
        )
        y_raw = np.loadtxt("./data/py_y_train.csv", delimiter=",", dtype=np.str_)
        y_enc = np.loadtxt(
            "./data/py_y_enc_train.csv", delimiter=",", dtype=np.float64
        )
        y_enc_features = np.loadtxt(
            "./data/py_y_enc_ft_names.csv", delimiter=",", dtype=np.str_
        )

        self.y_enc_names = y_enc_features
        self.X_features = X_features
        
        # Make into tensors
        self.train = torch.from_numpy(X_train)
        self.test = torch.from_numpy(X_test)
        self.listingID = np.loadtxt(
            "./data/py_test_id.csv", delimiter=",", dtype="i8"
        )  # 64-bit signed integer used in nb
        self.n_samples = X_train.shape[0]

        # Will later map encoded values to originals -- could've used native Pytorch for this...
        self.target_not_enc = y_raw
        # n_samples x 30; format is [[a], [b], [c]]
        self.target_enc = torch.from_numpy(y_enc[:, 0:])

    def __tensors__(self, encoded=True):
        if encoded:
            return self.train, self.target_enc
        else:
            return self.train, self.target_not_enc
                      
    def __len__(self):
        return self.n_samples


data = load_data()
n_samples = len(data)
features, enc_labels = data.__tensors__(encoded=True)
_, labels = data.__tensors__(False)
print('lol')

