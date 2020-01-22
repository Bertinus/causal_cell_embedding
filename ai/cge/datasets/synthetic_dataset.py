from torch.utils.data import Dataset
import os
import sys
import numpy as np
from SyntheticDataGenerator.structuredgraph import StructuredGraph
from SyntheticDataGenerator.structural_equation import lin_hidden_max_obs_generator
from SyntheticDataGenerator.obs_subgraph_generator import random_obs_subgraph_generator
import matplotlib.pyplot as plt
import time


########################################################################################################################
# Synthetic dataset
########################################################################################################################


class SyntheticDataset(Dataset):

    def __init__(self, n_hidden=15, n_observations=978, seed=0):
        np.random.seed(seed)
        self.graph = StructuredGraph(n_hidden=n_hidden, n_observations=n_observations)
        return

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        return
