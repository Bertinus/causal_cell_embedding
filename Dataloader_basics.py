#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.parse import parse
import matplotlib.pyplot as plt
import os
from os.path import abspath

path_to_data = "/home/user/Documents/CausalGene_Data/GSE70138_Broad_LINCS/Level5_COMPZ_n118050x12328_2017-03-06.gctx"
path_to_siginfo = "/home/user/Documents/CausalGene_Data/GSE70138_Broad_LINCS/sig_info_2017-03-06.txt"
sig_info =  pd.read_csv(path_to_siginfo, sep="\t")

class data_loader:
    def __init__(self, cell_ids, pert_types, batch_size, path_to_data, path_to_siginfo, sig_info):
        self.cell_ids = cell_ids
        self.pert_types = pert_types
        self.batch_size = batch_size
        self.path_to_data = path_to_data
        self.path_to_siginfo = path_to_siginfo
        self.sig_info = sig_info

    def get_env_specific_data(self):
        given_env = self.sig_info["sig_id"][(self.sig_info["pert_id"] == self.pert_types[0]) & (self.sig_info["cell_id"] == self.cell_ids[0])]
        data_in_given_env = parse(self.path_to_data, cid = given_env)
        return data_in_given_env.data_df

    def get_all_env_ids(self):
        all_enviromnents = self.sig_info.pert_id.unique().tolist()
        all_cells = self.sig_info.cell_id.unique().tolist()
        return all_enviromnents, all_cells


if __name__ == "__main__":
    
    cell_ids = ["A375"]
    pert_types = ["DMSO"]
    batch_size = '64'
    
    loader = data_loader(cell_ids, pert_types, batch_size, path_to_data, path_to_siginfo, sig_info)
    dataset = loader.get_env_specific_data()
    #dataset.normalize_by_gene("standard_scale")
    all_enviromnents, all_cells = loader.get_all_env_ids()

    #To see the results
    print('All cell-ids present in L1000 Dataset: ' ,all_cells)
    print('All environments present in L1000 Dataset: ' ,all_enviromnents)
    print('Data corresponding to given cell id and perturbagent: ',dataset)

