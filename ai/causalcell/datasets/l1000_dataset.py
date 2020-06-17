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
import pickle
import time

paths_to_L1000_files = {
    "phase1": {
        "path_to_dir": "Data/L1000_PhaseI",
        "path_to_data": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_"
                        "n473647x12328.gctx",
        "path_to_sig_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_sig_info.txt",
        "path_to_gene_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_gene_info.txt",
        "path_to_pert_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_pert_info.txt",
        "path_to_cell_info": "Data/L1000_PhaseI/GSE92742_Broad_LINCS/GSE92742_Broad_LINCS_cell_info.txt"
    }, "phase2": {
        "path_to_dir": "Data/L1000_PhaseII",
        "path_to_data": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/Level5_COMPZ_n118050x12328_2017-03-06.gctx",
        "path_to_sig_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/sig_info_2017-03-06.txt",
        "path_to_gene_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/gene_info_2017-03-06.txt",
        "path_to_pert_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/pert_info_2017-03-06.txt",
        "path_to_cell_info": "Data/L1000_PhaseII/GSE70138_Broad_LINCS/cell_info_2017-04-28.txt"
    }}

"""Dict of l1000 datasets that will be initialized with the relevant dataset objects if necessary so that all 
dataloaders use the same dataset object when possible, in order to instantiate only one."""
dict_of_l1000_datasets = {}


def get_fingerprint(smile, radius, nBits):
    try:
        if smile == "-666" or smile == "restricted":
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, nBits))
    except:
        return None


def get_concentration(s):
    if s.endswith('µM') or s.endswith('um'):
        return float(s[:-3])
    if s.endswith('nM'):
        return 0.001 * float(s[:-3])
    return -1


def get_time(s):
    return float(s[:-2])


########################################################################################################################
# L1000 dataset
########################################################################################################################


class L1000Dataset(Dataset):
    """
    Information on the dataset can be found here:
    https://docs.google.com/document/d/1q2gciWRhVCAAnlvF2iRLuJ7whrGP6QjpsCMq1yWz7dU/edit#heading=h.usef9o7fuux3
    """

    def __init__(self, phase="phase2", radius=2, nBits=1024):
        """
        :param phase: phase 1 or 2 of the dataset
        :param radius: parameter for fingerprints https://www.macinchem.org/reviews/clustering/clustering.php
        :param nBits: desired length of fingerprints
        """
        assert phase in ["phase1", "phase2", "both"]
        self.both = (phase == "both")
        if phase == "both":
            self.phase = "phase1"
        else:
            self.phase = phase

        # fingerprint parameters
        self.radius = radius
        self.nBits = nBits

        # Load metadata
        self.cell_info, self.landmark_gene_list, self.pert_info = self.load_metadata()
        self.sig_info = self.build_environment_representation()

        # Load data
        if self.both:
            self.data = pd.concat([self.load_data("phase1"), self.load_data("phase2")], sort=False)
        else:
            self.data = self.load_data(phase)

        # Get dictionary of non empty environments
        self.env_dict = self.get_env_dict()

    def load_data(self, phase):
        """
        We store the data as a single column dataframe containing numpy arrays.
        It allows a considerable speedup when accessing rows during training
        """
        path_to_data = paths_to_L1000_files[phase]["path_to_data"]
        df_path = os.path.join(paths_to_L1000_files[phase]["path_to_dir"], "dataframe.pkl")
        if os.path.isfile(df_path):
            print("Loading data of", phase)
            pickle_in = open(df_path, "rb")
            data = pickle.load(pickle_in)
        else:  # If the data has not been saved yet, parse the original file and save dataframe
            print("Parsing original data, only happens the first time...")
            data = parse(path_to_data, rid=self.landmark_gene_list).data_df.T
            # Remove rows that are not in sig_info
            data = data[data.index.isin(self.sig_info.index)]

            # Transform data so that it only has one column
            d = pd.DataFrame(index=data.index, columns=["gene_expr"])
            d["gene_expr"] = list(data.to_numpy())
            data = d

            # Save data
            pickle_out = open(df_path, "wb")
            pickle.dump(data, pickle_out, protocol=2)
            pickle_out.close()
        return data

    def get_env_dict(self):
        """
        :return: dict with (pert, cell) keys corresponding to non empty environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        """
        dict_path = os.path.join(paths_to_L1000_files[self.phase]["path_to_dir"], "dict_" + str(self.both) + ".npy")
        if os.path.isfile(dict_path):  # if the dict has been saved previously, load it
            env_dict = np.load(dict_path, allow_pickle='TRUE').item()
        else:
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

    def load_metadata(self):
        # cell_info and gene_info files are the same for both phases
        cell_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_cell_info"],
                                sep="\t", index_col="cell_id")
        cell_info['cell_id'] = cell_info.index  # Store cell_id in a column

        # Get list of landmark genes
        gene_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_gene_info"], sep="\t")
        landmark_gene_list = gene_info[gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str)

        # Load pert_info
        pert_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_pert_info"], sep="\t",
                                index_col="pert_id", usecols=["pert_id", "canonical_smiles"])
        if self.both:  # If we want both phases, load the other phase as well
            pert_info_2 = pd.read_csv(paths_to_L1000_files["phase2"]["path_to_pert_info"], sep="\t",
                                      index_col="pert_id", usecols=["pert_id", "canonical_smiles"])
            pert_info = pd.concat([pert_info, pert_info_2])
            # Remove duplicate indices
            pert_info = pert_info.loc[~pert_info.index.duplicated(keep='first')]

        # Load fingerprints
        fps_path = os.path.join(paths_to_L1000_files[self.phase]["path_to_dir"], "fingerprints_" + str(self.radius)
                                + "_" + str(self.nBits) + "_" + str(self.both) + ".pkl")
        if os.path.isfile(fps_path):
            fps = pd.read_pickle(fps_path)
        else:  # If fingerprints have not been saved yet, compute them
            print("Computing fingerprints, only happens the first time...")
            fps = pert_info.apply(lambda row: get_fingerprint(row["canonical_smiles"],
                                                              self.radius, self.nBits), axis=1)
            fps.to_pickle(fps_path)  # Save fingerprints

        # Add fingerprints to the pert_info file
        pert_info["fps"] = fps

        return cell_info, landmark_gene_list, pert_info

    def build_environment_representation(self):
        """
        We store the environment representation in a single column containing numpy arrays.
        It allows a considerable speedup when accessing rows during training
        """
        sig_info_path = os.path.join(paths_to_L1000_files[self.phase]["path_to_dir"], "sig_info_" + str(self.radius)
                                     + "_" + str(self.nBits) + "_" + str(self.both) + ".pkl")
        if os.path.isfile(sig_info_path):
            pickle_in = open(sig_info_path, "rb")
            sig_info = pickle.load(pickle_in)
        else:
            print("Building environment representations, only happens the first time...")
            # Load original file
            sig_info = pd.read_csv(paths_to_L1000_files[self.phase]["path_to_sig_info"], sep="\t",
                                   index_col="sig_id",
                                   usecols=["sig_id", "pert_id", "cell_id", "pert_idose", "pert_itime"])
            if self.both:  # If we want both phases, load the other phase as well
                sig_info_2 = pd.read_csv(paths_to_L1000_files["phase2"]["path_to_sig_info"], sep="\t",
                                         index_col="sig_id",
                                         usecols=["sig_id", "pert_id", "cell_id", "pert_idose", "pert_itime"])
                sig_info = pd.concat([sig_info, sig_info_2])

            # Convert time to float and add to sig_info
            sig_info['pert_itime_value'] = sig_info['pert_itime'].apply(get_time)
            # Convert concentrations to float and add to sig_info
            sig_info['pert_idose_value'] = sig_info['pert_idose'].apply(get_concentration)

            # Add fingerprints to the sig_info file
            pert_id_of_sig = sig_info['pert_id']  # Get all sig_info pert_ids
            # Drop rows that are not in pert_info
            pert_id_of_sig = pert_id_of_sig.drop(sig_info[~sig_info['pert_id'].isin(self.pert_info.index)]
                                                 .index)
            sig_fps = self.pert_info.loc[pert_id_of_sig]['fps']
            sig_fps.index = pert_id_of_sig.index
            sig_fps = sig_fps.reindex(sig_info.index, fill_value=None)
            # Add the corresponding fingerprints to sig_info
            sig_info['fps'] = sig_fps

            # Drop rows that do not have fingerprints
            sig_info = sig_info.dropna()
            # Create multiple columns that will store one component of the fps per column
            fps_df = list(sig_info['fps'])
            fps_df = pd.DataFrame(np.array([list(fp) for fp in fps_df]))
            fps_df.index = sig_info.index
            sig_info = pd.concat([sig_info, fps_df], axis=1, sort=False)
            sig_info = sig_info.drop(["fps"], axis=1)

            # Add information about the cells in the sig_info file in the form of one hot encoding
            cell_id_of_sig = sig_info['cell_id']
            # Drop rows that are not in cell_info
            cell_id_of_sig = cell_id_of_sig.drop(sig_info[~sig_info['cell_id'].isin(self.cell_info.index)]
                                                 .index)
            sig_info_of_cell = self.cell_info.loc[cell_id_of_sig][['primary_site', 'original_growth_pattern',
                                                                   'subtype', 'sample_type', 'cell_type', 'cell_id']]
            sig_info_of_cell = pd.get_dummies(sig_info_of_cell)  # Get one hot encodings
            sig_info_of_cell.index = cell_id_of_sig.index
            sig_info_of_cell = sig_info_of_cell.reindex(sig_info.index, fill_value=0)
            sig_info = pd.concat([sig_info, sig_info_of_cell], axis=1, sort=False)

            # Transform sig_info so that env_repr is stored in a single column as a numpy array
            env_repr = pd.DataFrame(index=sig_info.index, columns=["env_repr"])
            env_repr["env_repr"] = list(sig_info[sig_info.columns[4:]].to_numpy().astype(np.float32))
            sig_info = pd.concat([sig_info[sig_info.columns[:4]], env_repr], axis=1, sort=False)

            # Save data
            pickle_out = open(sig_info_path, "wb")
            pickle.dump(sig_info, pickle_out, protocol=2)
            pickle_out.close()

        return sig_info

    def __len__(self):
        all_values = []
        for v in self.env_dict.values():
            all_values.extend(v)
        return len(all_values)

    def __getitem__(self, sig_id):
        """
        :param sig_id: index of the line in the self.data dataframe
        :return: a 4 tuple (x, fingerprint, pert_id, cell_id)
        """
        # Get env_representation
        env_representation = self.sig_info.at[sig_id, "env_repr"]
        pert_id = self.sig_info.at[sig_id, "pert_id"]
        cell_id = self.sig_info.at[sig_id, "cell_id"]

        return self.data.at[sig_id, 'gene_expr'], env_representation, pert_id, cell_id


########################################################################################################################
# L1000 sampler
########################################################################################################################

class L1000Sampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments
    """

    def __init__(self, env_dict, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1), seed=0):
        """
        :param env_dict: dict with (pert, cell) keys corresponding to desired environments
                dict[(pert, cell)] contains the list of all corresponding sig_ids
        """
        self.env_dict = env_dict
        self.batch_size = batch_size
        self.n_samples = 0
        self.n_envs_in_split = 0

        if restrict_to_envs_longer_than is not None:
            self.restrict_to_large_envs(restrict_to_envs_longer_than)

        self.perform_split(split, train_val_test_prop, seed=seed)

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

        self.n_samples = [train_length, valid_length, test_length][['train', 'valid', 'test'].index(split)]
        self.n_envs_in_split = len(self.env_dict.keys())

        print(split + " split of size", self.n_samples, "with number of environments", self.n_envs_in_split)

    def iterator(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return self.n_samples // self.batch_size


class IIDL1000Sampler(L1000Sampler):
    """
    IID sampler
    """

    def __init__(self, env_dict, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        self.split_values = None
        super().__init__(env_dict, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

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
        all_values_valid = all_values[train_length: train_length + valid_length]
        all_values_test = all_values[-test_length:]

        # Select the values corresponding to the desired split
        self.split_values = \
            [all_values_train, all_values_valid, all_values_test][['train', 'valid', 'test'].index(split)]

        self.n_samples = len(self.split_values)
        self.n_envs_in_split = len(self.env_dict.keys())

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

    def __init__(self, env_dict, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(env_dict, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

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

    def __init__(self, env_dict, batch_size, restrict_to_envs_longer_than=None, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(env_dict, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

    def iterator(self):
        keys = np.random.permutation(sorted(list(self.env_dict.keys())))
        batch = []
        # Loop over environments
        for pert, cell in keys:
            # Loop over samples in a given environment
            for sig_id in np.random.permutation(sorted(self.env_dict[(pert, cell)])):
                batch.append(sig_id)
                if len(batch) == self.batch_size:  # If batch is full, yield it
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

    def __init__(self, env_dict, batch_size, n_env_per_batch=3, restrict_to_envs_longer_than=None,
                 split='train', train_val_test_prop=(0.7, 0.2, 0.1)):
        """
        :param n_env_per_batch: Number of environments per batch
        """
        self.n_env_per_batch = n_env_per_batch
        self.min_size_of_envs = restrict_to_envs_longer_than
        super().__init__(env_dict, batch_size, restrict_to_envs_longer_than, split, train_val_test_prop)

    def iterator(self):
        keys = np.random.permutation(sorted(list(self.env_dict.keys())))
        env_cpt = 0
        while env_cpt < len(keys):
            # Note that an epoch will correspond to seeing all environments once but not all samples
            batch = []
            # choose indices specific to several environments and add them to the batch list
            for _ in range(self.n_env_per_batch):
                if env_cpt >= len(keys):
                    break
                pert, cell = keys[env_cpt]  # Select the next environment to be added to the batch list
                env_cpt += 1
                # Add self.min_size_of_envs examples to the batch list
                batch.extend(np.random.permutation(sorted(self.env_dict[(pert, cell)]))[:self.min_size_of_envs])
            # Choose batch_size elements at random in the batch list
            yield np.random.permutation(batch)[:self.batch_size]


########################################################################################################################
# L1000 dataloaders
########################################################################################################################

def get_dataset(dataset_args):
    # If the dataset has already been instantiated, use it
    if tuple(dataset_args) in dict_of_l1000_datasets:
        print("Dataset already instanciated, loading it")
        dataset = dict_of_l1000_datasets[tuple(dataset_args)]
    else:  # Otherwise instantiate it and save it in the dictionary of datasets
        dataset = L1000Dataset(*dataset_args)
        dict_of_l1000_datasets[tuple(dataset_args)] = dataset

    return dataset


@register.setdataloadername("L1000_iid")
def iid_l1000_dataloader(phase="phase2", batch_size=16, restrict_to_envs_longer_than=None, split='train',
                         train_val_test_prop=(0.7, 0.2, 0.1), radius=2,
                         nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, batch_size, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = IIDL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("L1000_iid_env_split")
def iid_env_split_l1000_dataloader(phase="phase2", batch_size=16, restrict_to_envs_longer_than=None, split='train',
                                   train_val_test_prop=(0.7, 0.2, 0.1), radius=2,
                                   nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, batch_size, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = IIDEnvSplitL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("L1000_loop_over_envs")
def loop_over_envs_l1000_dataloader(phase="phase2", batch_size=16, restrict_to_envs_longer_than=None, split='train',
                                    train_val_test_prop=(0.7, 0.2, 0.1), radius=2,
                                    nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, batch_size, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = LoopOverEnvsL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("L1000_n_env_per_batch")
def n_env_per_batch_l1000_dataloader(phase="phase2", batch_size=16, n_env_per_batch=3,
                                     restrict_to_envs_longer_than=None, split='train',
                                     train_val_test_prop=(0.7, 0.2, 0.1),
                                     radius=2, nBits=1024):
    # Initialize dataset
    dataset_args = [phase, radius, nBits]
    dataset = get_dataset(dataset_args)

    # Initialize sampler
    sampler_args = [dataset.env_dict, batch_size, n_env_per_batch, restrict_to_envs_longer_than,
                    split, train_val_test_prop]
    sampler = NEnvPerBatchL1000Sampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


if __name__ == "__main__":
    dataloader = iid_env_split_l1000_dataloader(phase="both", restrict_to_envs_longer_than=None, split="train",
                                                radius=8, nBits=1024, train_val_test_prop=(1., 0., 0.))
    print(len(dataloader))
    cpt = 0
    t = time.time()
    for data in dataloader:
        cpt += 1
        print(data[0].shape, data[1].shape)
    print("epoch loop in", time.time() - t)
