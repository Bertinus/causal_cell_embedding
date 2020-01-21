from torch.utils.data import Dataset
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os


########################################################################################################################
# L1000 dataset
########################################################################################################################


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
        return len(self.sig_info)

    def __getitem__(self, sig_id):
        return self.data.loc[sig_id].to_numpy(), \
               self.sig_info.pert_id.loc[sig_id], \
               self.sig_info.cell_id.loc[sig_id]


########################################################################################################################
# L1000 sampler
########################################################################################################################


class BasicL1000Sampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments

    Loop over all environments in the keys of the dict. Once in a given environment,
    yields all samples from that environment

    """

    def __init__(self, env_dict, length, batch_size, seed=0, restrict_to_envs_longer_than=None):
        """
        :param env_dict: dict with (pert, cell) keys corresponding to desired environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        """
        self.env_dict = env_dict
        self.length = length
        self.seed = seed
        self.batch_size = batch_size

        if restrict_to_envs_longer_than is not None:
            self.restrict_env_dict(restrict_to_envs_longer_than)

    def iterator(self):
        np.random.seed(self.seed)
        keys = np.random.permutation(list(self.env_dict.keys()))
        batch = []
        # Loop over environments
        for pert, cell in keys:
            # Loop over samples in a given environment
            for sig_id in np.random.permutation(self.env_dict[(pert, cell)]):
                batch.append(sig_id)
                if len(batch) == self.batch_size:  # If batch is full, yield it
                    yield batch
                    batch = []
            # Before moving to a new environment, yield current batch even if it is not full
            if batch:
                yield batch
                batch = []

    def restrict_env_dict(self, restrict_to_envs_longer_than):
        """
        Remove environments that do not contain enough samples from the dictionary
        """
        for k in list(self.env_dict.keys()):
            if len(self.env_dict[k]) < restrict_to_envs_longer_than:
                del self.env_dict[k]

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return self.length


class EnvironmentL1000Sampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments

    Each batch contains samples from randomly drawn n_env_per_batch distinct environments

    """

    def __init__(self, env_dict, length, batch_size, n_env_per_batch=1, seed=0, restrict_to_envs_longer_than=None):
        """
        :param env_dict: dict with (pert, cell) keys corresponding to desired environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        :param n_env_per_batch: Number of environments per batch
        """
        self.env_dict = env_dict
        self.length = length
        self.batch_size = batch_size
        self.n_env_per_batch = n_env_per_batch
        self.seed = seed

        if restrict_to_envs_longer_than is not None:
            self.restrict_env_dict(restrict_to_envs_longer_than)

    def iterator(self):
        np.random.seed(self.seed)
        keys = np.random.permutation(list(self.env_dict.keys()))
        env_cpt = 0
        while True:
            if env_cpt >= len(keys):
                break
            batch = []
            # choose indices specific to several environments and add them to the batch list
            for env in range(self.n_env_per_batch):
                if env_cpt >= len(keys):
                    break
                pert, cell = keys[env_cpt]
                env_cpt += 1
                batch.extend(self.env_dict[(pert, cell)])

            # Choose batch_size elements at random in the batch list
            yield np.random.permutation(batch)[:self.batch_size]

    def restrict_env_dict(self, restrict_to_envs_longer_than):
        """
        Remove environments that do not contain enough samples from the dictionary
        """
        for k in list(self.env_dict.keys()):
            if len(self.env_dict[k]) < restrict_to_envs_longer_than:
                del self.env_dict[k]

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return self.length

########################################################################################################################
# L1000 dataloaders
########################################################################################################################


def basic_l1000_dataloader(batch_size=16, restrict_to_envs_longer_than=None):
    dataset = L1000Dataset()
    sampler = BasicL1000Sampler(dataset.get_non_empty_env_dict(), len(dataset), batch_size=batch_size,
                                restrict_to_envs_longer_than=restrict_to_envs_longer_than)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


def environment_l1000_dataloader(batch_size=16, n_env_per_batch=3, restrict_to_envs_longer_than=None):
    dataset = L1000Dataset()
    sampler = EnvironmentL1000Sampler(dataset.get_non_empty_env_dict(), len(dataset), batch_size=batch_size,
                                      n_env_per_batch=n_env_per_batch,
                                      restrict_to_envs_longer_than=restrict_to_envs_longer_than)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


if __name__ == "__main__":

    dataloader = environment_l1000_dataloader(restrict_to_envs_longer_than=10)
    for x, pert, cell in dataloader:
        print(x.shape, len(pert), len(cell))