from torch.utils.data import Dataset
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch
from random import shuffle
from tqdm import tqdm
import numpy as np
import os


class SyntheticDataset(Dataset):

    def __init__(self):
        return

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        return


class L1000Dataset(Dataset):
    """
    Information on the dataset can be found here:
    https://docs.google.com/document/d/1q2gciWRhVCAAnlvF2iRLuJ7whrGP6QjpsCMq1yWz7dU/edit#heading=h.usef9o7fuux3
    """

    def __init__(self,
                 path_to_data="Data/L1000_PhaseII/GSE70138_Broad_LINCS/Level5_COMPZ_n118050x12328_2017-03-06.gctx",
                 path_to_sig_info="Data/L1000_PhaseII/GSE70138_Broad_LINCS/sig_info_2017-03-06.txt",
                 path_to_gene_info="Data/L1000_PhaseII/GSE70138_Broad_LINCS/gene_info_2017-03-06.txt"):

        # Data path
        self.path_to_data = path_to_data

        # Read metadata
        self.sig_info = pd.read_csv(path_to_sig_info, sep="\t", index_col="sig_id")
        self.gene_info = pd.read_csv(path_to_gene_info, sep="\t")
        self.landmark_gene_list = self.gene_info[self.gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str)

        # Load all data
        self.data = parse(self.path_to_data, rid=self.landmark_gene_list).data_df.T

    def get_non_empty_env_dict(self):
        """
        :return: dict with (pert, cell) keys corresponding to non empty environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        """
        # if the dict has been saved previously, load it
        dict_path = "Data/L1000_PhaseII/non_empty_env_dict.npy"
        if os.path.isfile(dict_path):
            return np.load(dict_path, allow_pickle='TRUE').item()

        print("Building dict of all environments, only happens the first time...")
        env_dict = {}
        for index, row in self.data.iterrows():
            pert = self.sig_info.pert_id.loc[index]
            cell = self.sig_info.cell_id.loc[index]
            if (pert, cell) in env_dict.keys():
                env_dict[(pert, cell)].append(index)
            else:
                env_dict[(pert, cell)] = [index]
        np.save(dict_path, env_dict)
        return env_dict

    def __len__(self):
        return len(self.sig_info.shape[0])

    def __getitem__(self, sig_id):
        return self.data.loc[sig_id].to_numpy()


class L1000Sampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments

    Loop over all environments in the keys of the dict. Once in a given environment,
    yields all samples from that environment

    """

    def __init__(self, env_dict, batchsize, seed=0):
        """
        :param env_dict: dict with (pert, cell) keys corresponding to non empty environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        :param batchsize:
        """
        self.env_dict = env_dict
        self.batchsize = batchsize
        self.seed = seed

    def iterator(self):
        np.random.seed(42)
        perm_indices = np.random.permutation(len(self.env_dict.keys()))
        keys = list(self.env_dict.keys())
        # Loop over environments
        for idx in perm_indices:
            pert, cell = keys[idx]
            # Loop over samples in a given environment
            for sig_id in self.env_dict[(pert, cell)]:
                yield sig_id

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return len(self.dataset)


def l1000_dataloader(batch_size=64):
    dataset = L1000Dataset()
    sampler = L1000Sampler(dataset.get_non_empty_env_dict(), batch_size)

    return DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)


if __name__ == "__main__":

    dataloader = l1000_dataloader()
    for i in dataloader:
        print(i.shape)
