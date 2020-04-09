from torch.utils.data import Dataset
from cmapPy.pandasGEXpress.parse import parse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import ai.causalcell.utils.register as register
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
import random
import time

paths_to_L1000_files = {
    "phase1": {
        "path_to_data": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328"
                        ".gctx",
        "path_to_sig_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_sig_info.txt",
        "path_to_gene_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_gene_info.txt",
        "dict_path": "Data/L1000_PhaseI/non_empty_env_dict.npy",
        "path_to_pert_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_pert_info.txt"
    }, "phase2": {
        "path_to_data": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/Level5_COMPZ_n118050x12328_2017-03-06.gctx",
        "path_to_sig_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/sig_info_2017-03-06.txt",
        "path_to_gene_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/gene_info_2017-03-06.txt",
        "dict_path": "Data/L1000_PhaseII/non_empty_env_dict.npy",
        "path_to_pert_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/pert_info_2017-03-06.txt"
    }}

"""Dict of l1000 datasets that will be initialized with the relevant dataset objects if necessary so that all 
dataloaders use the same dataset object when possible, in order to instantiate only one."""
dict_of_l1000_datasets = {}


########################################################################################################################
# L1000 dataset
########################################################################################################################

def get_fingerprint(smile, radius, nBits):
    try:
        if smile == "-666" or smile == "restricted":
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, nBits))
    except:
        return None


class L1000Dataset(Dataset):
    """
    Information on the dataset can be found here:
    https://docs.google.com/document/d/1q2gciWRhVCAAnlvF2iRLuJ7whrGP6QjpsCMq1yWz7dU/edit#heading=h.usef9o7fuux3
    """

    def __init__(self, phase="phase2", radius=2, nBits=1024, remove_null_fingerprint_envs=True):
        """
        :param phase: phase 1 or 2 of the dataset
        :param radius: parameter for fingerprints https://www.macinchem.org/reviews/clustering/clustering.php
        :param nBits: desired length of fingerprints
        """

        seconds = time.time()
        print(phase, radius, nBits, remove_null_fingerprint_envs)

        assert phase in ["phase1", "phase2"]
        self.phase = phase

        # Data path
        self.path_to_data = paths_to_L1000_files[self.phase]["path_to_data"]

        # Read metadata
        self.sig_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_sig_info"], sep="\t", index_col="sig_id",
                                    usecols=["sig_id", "pert_id", "cell_id"])
        self.pert_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_pert_info"], sep="\t",
                                     index_col="pert_id")

        print("read metadata took", time.time() - seconds, "seconds")
        seconds = time.time()

        # Get list of landmark genes
        gene_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_gene_info"], sep="\t")
        self.landmark_gene_list = gene_info[gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str)

        print("get list landmark genes took", time.time() - seconds, "seconds")
        seconds = time.time()

        # Compute fingerprints
        # TODO: precompute fingerprints? 26 seconds with phase 1
        self.pert_info["fps"] = self.pert_info.apply(lambda row: get_fingerprint(row["canonical_smiles"],
                                                                                 radius, nBits), axis=1)

        print("compute fingerprint took", time.time() - seconds, "seconds")
        seconds = time.time()

        # Load all data
        # TODO: save data as a dataframe directly? 64 seconds with phase 1
        self.data = parse(self.path_to_data, rid=self.landmark_gene_list).data_df.T

        print("load all data took", time.time() - seconds, "seconds")
        seconds = time.time()

        # Get dictionary of non empty environments
        self.env_dict = self.get_non_empty_env_dict()

        print("load dictionary took", time.time() - seconds, "seconds")
        seconds = time.time()

        # TODO: save dictionary? 6 seconds with phase 1
        if remove_null_fingerprint_envs:
            self.remove_null_fingerprint_envs()

        print("removing null fingerprints took", time.time() - seconds, "seconds")

    def get_non_empty_env_dict(self):
        """
        :return: dict with (pert, cell) keys corresponding to non empty environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        """
        # if the dict has been saved previously, load it
        dict_path = paths_to_L1000_files[self.phase]["dict_path"]
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

    def remove_null_fingerprint_envs(self):
        """
        Remove from the dictionary the environments that have no fingerprint
        """
        for k in list(self.env_dict.keys()):
            try:
                if type(self.pert_info["fps"].loc[k[0]]) != np.ndarray:
                    # The perturbation fingerprint is not an array. We remove the environment from dict
                    del self.env_dict[k]
            except:
                # The perturbation is not in the pert_info dataframe! We remove the environment from dict
                del self.env_dict[k]

    def __len__(self):
        return len(self.sig_info)

    def __getitem__(self, sig_id):
        """
        :param sig_id: index of the line in the self.data dataframe
        :return: a 4 tuple (x, fingerprint, pert_id, cell_id)
        """
        return self.data.loc[sig_id].to_numpy(), \
               self.pert_info.fps.loc[self.sig_info.pert_id.loc[sig_id]], \
               self.sig_info.pert_id.loc[sig_id], \
               self.sig_info.cell_id.loc[sig_id]


########################################################################################################################
# L1000 sampler
########################################################################################################################

class L1000Sampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments
    """

    def __init__(self, env_dict, length, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1), seed=0):
        """
        :param env_dict: dict with (pert, cell) keys corresponding to desired environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        """

        seconds = time.time()

        self.env_dict = env_dict
        self.length = length
        self.batch_size = batch_size

        if restrict_to_envs_longer_than is not None:
            self.restrict_to_large_envs(restrict_to_envs_longer_than)

        self.perform_split(split, train_val_test_prop, seed=seed)

        print("initialize sampler took", time.time() - seconds, "seconds")

    def restrict_to_large_envs(self, restrict_to_envs_longer_than):
        """
        Remove environments that do not contain enough samples from the dictionary
        """
        for k in list(self.env_dict.keys()):
            if len(self.env_dict[k]) < restrict_to_envs_longer_than:
                del self.env_dict[k]

    def perform_split(self, split, train_val_test_prop, seed):
        """
        Performs split according to the required proportions. Spliting is performed at the environment level,
        so that all the examples from a given environment end up in the same split
        Updates the env_dict accordingly
        :param train_val_test_prop:
        :param split: 'train', 'val' or 'test'
        """
        assert split in ['train', 'valid', 'test']
        assert len(train_val_test_prop) == 3
        keys = list(self.env_dict.keys())
        total_length = sum([len(v) for v in self.env_dict.values()])

        keys_with_lenght = [(k, len(self.env_dict[k])) for k in keys]
        keys_with_lenght.sort()  # To make sure the order is always the same
        # We shuffle with always the same seed, without affecting the global seed of the random package
        random.Random(seed).shuffle(keys_with_lenght)

        train_length, valid_length, test_length = 0, 0, 0
        train_keys, valid_keys, test_keys = [], [], []

        # Fill each split with environemnts until it is full.
        # Unconventional way because each environment has a different weight
        idx = 0
        while train_length < train_val_test_prop[0] * total_length:
            train_keys.append(keys_with_lenght[idx][0])
            train_length += keys_with_lenght[idx][1]
            idx += 1
        while valid_length < train_val_test_prop[1] * total_length:
            valid_keys.append(keys_with_lenght[idx][0])
            valid_length += keys_with_lenght[idx][1]
            idx += 1
        while idx < len(keys_with_lenght):
            test_keys.append(keys_with_lenght[idx][0])
            test_length += keys_with_lenght[idx][1]
            idx += 1

        # Select the keys corresponding to the desired split
        keys = [train_keys, valid_keys, test_keys][['train', 'valid', 'test'].index(split)]

        self.env_dict = {key: self.env_dict[key] for key in set(keys)}

        print(split + " split of size",
              [train_length, valid_length, test_length][['train', 'valid', 'test'].index(split)],
              "with number of environments", len(self.env_dict.keys()))

    def iterator(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return self.length


class IIDL1000Sampler(L1000Sampler):
    """
    IID sampler
    """

    def __init__(self, env_dict, length, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        self.split_values = None
        super().__init__(env_dict, length, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

    def perform_split(self, split, train_val_test_prop, seed):
        """
        Override the perform_split method to enable splitting without taking into account environments at all
        """
        assert split in ['train', 'valid', 'test']
        assert len(train_val_test_prop) == 3

        # Get all values available
        all_values = []
        for v in self.env_dict.values():
            all_values.extend(v)

        all_values.sort()  # To make sure the order is always the same
        # We shuffle with always the same seed, without affecting the global seed of the random package
        random.Random(seed).shuffle(all_values)

        # Perform train, valid, test split
        train_length = int(train_val_test_prop[0] * len(all_values))
        valid_length = int(train_val_test_prop[1] * len(all_values))
        test_length = len(all_values) - train_length - valid_length
        all_values_train = all_values[:train_length]
        all_values_valid = all_values[train_length: train_length+valid_length]
        all_values_test = all_values[-test_length:]

        # Select the values corresponding to the desired split
        self.split_values = \
            [all_values_train, all_values_valid, all_values_test][['train', 'valid', 'test'].index(split)]

        print(split + " split of size", len(self.split_values), "without taking environment into account")

    def iterator(self):
        # Randomize
        self.split_values = np.random.permutation(self.split_values)
        # Create batches
        batch = []
        # Loop over values
        for v in self.split_values:
            batch.append(v)
            if len(batch) == self.batch_size:  # If batch is full, yield it
                yield batch
                batch = []


class IIDEnvSplitL1000Sampler(L1000Sampler):
    """
    Sampler with splitting per environment, but the data is then shuffled within each split
    """

    def __init__(self, env_dict, length, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(env_dict, length, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

    def iterator(self):
        # Convert list of lists to list
        merged_values = []
        for el in self.env_dict.values():
            merged_values.extend(el)

        # Randomize
        merged_values = np.random.permutation(sorted(list(merged_values)))
        # Create batches
        batch = []
        # Loop over values
        for v in merged_values:
            batch.append(v)
            if len(batch) == self.batch_size:  # If batch is full, yield it
                yield batch
                batch = []


class LoopOverEnvsL1000Sampler(L1000Sampler):
    """
    Loop over all environments in the keys of the dict. Once in a given environment,
    yields all samples from that environment
    """

    def __init__(self, env_dict, length, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(env_dict, length, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

    def iterator(self):
        keys = np.random.permutation(sorted(list(self.env_dict.keys())))
        batch = []
        # Loop over environments
        for pert, cell in keys:
            # Loop over samples in a given environment
            for sig_id in np.random.permutation(sorted(self.env_dict[(pert, cell)])):
                batch.append(sig_id)
                if len(batch) == self.batch_size:  # If batch is full, yield it
                    print(pert, cell, batch)
                    yield batch
                    batch = []
            # Before moving to a new environment, yield current batch even if it is not full
            if batch:
                yield batch
                batch = []


class NEnvPerBatchL1000Sampler(L1000Sampler):
    """
    Each batch contains samples from randomly drawn n_env_per_batch distinct environments
    """

    def __init__(self, env_dict, length, batch_size, n_env_per_batch=1, restrict_to_envs_longer_than=None,
                 split='train', train_val_test_prop=(0.7, 0.2, 0.1)):
        """
        :param n_env_per_batch: Number of environments per batch
        """
        self.n_env_per_batch = n_env_per_batch
        self.min_size_of_envs = restrict_to_envs_longer_than
        super().__init__(env_dict, length, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

    def iterator(self):
        keys = np.random.permutation(sorted(list(self.env_dict.keys())))
        env_cpt = 0
        while True:
            batch = []
            # Note that an epoch will correspond to seeing all environments once but not all samples
            if env_cpt >= len(keys):
                break
            # choose indices specific to several environments and add them to the batch list
            for env in range(self.n_env_per_batch):
                if env_cpt >= len(keys):
                    break
                pert, cell = keys[env_cpt]  # Select the next environment to be added to the batch list
                env_cpt += 1
                # Add self.min_size_of_envs examples to the batch list
                batch.extend(np.random.permutation(sorted(self.env_dict[(pert, cell)])[:self.min_size_of_envs]))
            # Choose batch_size elements at random in the batch list
            yield np.random.permutation(batch)[:self.batch_size]


########################################################################################################################
# L1000 dataloaders
########################################################################################################################

def get_dataset(dataset_args):
    # If the dataset has already been instantiated, use it
    if tuple(dataset_args) in dict_of_l1000_datasets:
        dataset = dict_of_l1000_datasets[tuple(dataset_args)]
    else:  # Otherwise instantiate it and save it in the dictionary of datasets
        dataset = L1000Dataset(*dataset_args)
        dict_of_l1000_datasets[tuple(dataset_args)] = dataset

    return dataset


@register.setdataloadername("L1000_iid")
def iid_l1000_dataloader(phase="phase2", batch_size=16, restrict_to_envs_longer_than=None, split='train',
                         train_val_test_prop=(0.7, 0.2, 0.1), remove_null_fingerprint_envs=True, radius=2,
                         nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits, remove_null_fingerprint_envs]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, len(dataset), batch_size, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = IIDL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("L1000_iid_env_split")
def iid_env_split_l1000_dataloader(phase="phase2", batch_size=16, restrict_to_envs_longer_than=None, split='train',
                                   train_val_test_prop=(0.7, 0.2, 0.1), remove_null_fingerprint_envs=True, radius=2,
                                   nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits, remove_null_fingerprint_envs]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, len(dataset), batch_size, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = IIDEnvSplitL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("L1000_loop_over_envs")
def loop_over_envs_l1000_dataloader(phase="phase2", batch_size=16, restrict_to_envs_longer_than=None, split='train',
                                    train_val_test_prop=(0.7, 0.2, 0.1), remove_null_fingerprint_envs=True, radius=2,
                                    nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits, remove_null_fingerprint_envs]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, len(dataset), batch_size, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = LoopOverEnvsL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("L1000_n_env_per_batch")
def n_env_per_batch_l1000_dataloader(phase="phase2", batch_size=16, n_env_per_batch=3,
                                     restrict_to_envs_longer_than=None, split='train',
                                     train_val_test_prop=(0.7, 0.2, 0.1), remove_null_fingerprint_envs=True,
                                     radius=2, nBits=1024):

    # Initialize dataset
    dataset_args = [phase, radius, nBits, remove_null_fingerprint_envs]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, len(dataset), batch_size, n_env_per_batch, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = NEnvPerBatchL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


if __name__ == "__main__":
    dataloader = n_env_per_batch_l1000_dataloader(phase="phase2", restrict_to_envs_longer_than=10, split="train")
    print(len(dataloader))
    for data in dataloader:
        print(data[0].shape)
