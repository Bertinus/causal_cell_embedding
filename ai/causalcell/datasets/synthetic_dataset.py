from torch.utils.data import Dataset
from SyntheticDataGenerator.structuredgraph import StructuredGraph
from torch.utils.data import DataLoader
from SyntheticDataGenerator.structural_equation import binary_noisy_lin_generator, NoisyLinear, \
    noisy_lin_hidden_lin_obs_generator, noisy_lin_hidden_neural_net_obs_generator
from SyntheticDataGenerator.dag_generator import multi_gn_graph_generator, gn_graph_generator, empty_graph_generator
from SyntheticDataGenerator.obs_subgraph_generator import random_obs_subgraph_generator
import numpy as np
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import ai.causalcell.utils.register as register
import random

# """Dict of synthetic datasets that will be initialized with the relevant dataset objects if necessary so that all
# dataloaders use the same dataset object when possible, in order to instantiate only one."""
# dict_of_synthetic_datasets = {}

dag_generator_dict = {"multi_gn": multi_gn_graph_generator, "gn": gn_graph_generator, "empty": empty_graph_generator}
struct_eq_generator_dict = {"noisy_lin_hidden_lin_obs": noisy_lin_hidden_lin_obs_generator,
                            "noisy_lin_hidden_neural_net_obs_generator": noisy_lin_hidden_neural_net_obs_generator}

global_graph = None  # We want all datasets to use the same graph
global_graph_params = None


########################################################################################################################
# Synthetic dataset
########################################################################################################################


class SyntheticDataset(Dataset):

    def __init__(self, n_hidden=15, n_observations=978, n_examples_per_env=1000, n_envs=100, attach_proba=0.2,
                 dag_generator="multi_gn", struct_eq_gen="noisy_lin_hidden_lin_obs"):

        assert dag_generator in dag_generator_dict.keys()
        assert struct_eq_gen in struct_eq_generator_dict.keys()

        # We will use a global graph shared between datasets
        global global_graph
        global global_graph_params

        self.data = None
        self.envs = None

        # Create graph if necessary
        if global_graph is None:
            global_graph = StructuredGraph(n_hidden=n_hidden, n_observations=n_observations,
                                           structural_equation_generator=struct_eq_generator_dict[struct_eq_gen],
                                           directed_acyclic_graph_generator=dag_generator_dict[dag_generator],
                                           obs_subgraph_generator=lambda g, n_obs: random_obs_subgraph_generator(
                                               g, n_obs, proba=attach_proba),
                                           mean=0.0, var=1.0)
            global_graph_params = str((n_hidden, n_observations, attach_proba, dag_generator, struct_eq_gen))
            print("global graph created with parameters", global_graph_params)
        else:
            print("global graph already instantiated")
            assert global_graph_params == str((n_hidden, n_observations, attach_proba, dag_generator, struct_eq_gen)), \
                'Graph parameters must be the same for all datasets'

        # Create data corresponding to different environments
        for _ in range(n_envs):
            inter_node = np.random.randint(n_hidden)  # Choose a variable to intervene on
            inter_shift = np.random.normal(loc=0.0, scale=5.0)  # Choose shift applied to the intervened variable

            # Intervene on the graph
            global_graph.set_soft_intervention(inter_node, function=NoisyLinear(
                lambda size: np.random.normal(loc=inter_shift, scale=1.0, size=size)))

            # Generate data
            global_graph.generate(n_examples=n_examples_per_env)
            if self.data is None:
                self.data = global_graph.get_observations()
            else:
                self.data = np.concatenate((self.data, global_graph.get_observations()), axis=0)

            # Compute representation of the environment
            env_representation = np.zeros((n_examples_per_env, n_hidden))
            env_representation[:, inter_node] = inter_shift
            if self.envs is None:
                self.envs = env_representation
            else:
                self.envs = np.concatenate((self.envs, env_representation), axis=0)

        self.data = self.data.astype(np.float32)
        self.envs = self.envs.astype(np.float32)

        # Put back graph in default mode
        global_graph.to_default_state()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        We format all datasets to return 4-tuples
        :return: a 4 tuple (x, fingerprint, pert_id, cell_id)
        """
        return self.data[idx], self.envs[idx], 0, 0


########################################################################################################################
# Samplers
########################################################################################################################


class SyntheticSampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments
    """

    def __init__(self, batch_size, n_examples_per_env=1000, n_envs=100, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        assert split in ['train', 'valid', 'test']
        assert len(train_val_test_prop) == 3
        self.batch_size = batch_size
        self.n_examples_per_env = n_examples_per_env
        self.n_envs = n_envs
        self.split = split
        self.train_val_test_prop = train_val_test_prop
        self.split_idx = None
        self.n_envs_in_split = 0
        self.n_samples = 0

        self.perform_split()

    def perform_split(self):
        """
        Performs split according to the required proportions. Spliting is performed at the environment level,
        so that all the examples from a given environment end up in the same split
        Updates the env_dict accordingly
        """
        list_of_idx = list(range(self.n_examples_per_env * self.n_envs))
        train_length, valid_length, test_length = self.get_split_lengths(list_of_idx)
        self.split_list_idx(list_of_idx, train_length, valid_length, test_length)

    def get_split_lengths(self, list_of_idx):
        """
        Computes length of splits at the environment level
        """
        n = self.n_examples_per_env
        train_length = int(len(list_of_idx) * self.train_val_test_prop[0] // n * n)
        valid_length = int(len(list_of_idx) * self.train_val_test_prop[1] // n * n)
        test_length = int(len(list_of_idx) - train_length - valid_length)

        return train_length, valid_length, test_length

    def split_list_idx(self, list_of_idx, train_length, valid_length, test_length):
        """
        Splits list of indices
        """
        # Split the list of indices
        train_idx = list_of_idx[:train_length]
        valid_idx = list_of_idx[train_length: train_length + valid_length]
        test_idx = list_of_idx[-test_length:]

        self.split_idx = [train_idx, valid_idx, test_idx][['train', 'valid', 'test'].index(self.split)]

        self.n_envs_in_split = len(self.split_idx) // self.n_examples_per_env
        if len(self.split_idx) % self.n_examples_per_env != 0:
            self.n_envs_in_split += 1
        self.n_samples = len(self.split_idx)

        print(self.split + " split of size", self.n_samples, "with number of environments", self.n_envs_in_split)

    def iterator(self):
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return len(self.split_idx) // self.batch_size


class IIDSyntheticSampler(SyntheticSampler):
    """
    IID sampler
    """

    def __init__(self, batch_size, n_examples_per_env=1000, n_envs=100, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(batch_size, n_examples_per_env, n_envs, split, train_val_test_prop)

    def perform_split(self):
        """
        Override the perform_split method to enable splitting without taking into account environments at all
        """
        list_of_idx = list(range(self.n_examples_per_env * self.n_envs))
        random.Random(0).shuffle(list_of_idx)  # We shuffle everything to make it IID
        train_length, valid_length, test_length = self.get_split_lengths(list_of_idx)
        self.split_list_idx(list_of_idx, train_length, valid_length, test_length)

    def get_split_lengths(self, list_of_idx):
        """
        Computes length of splits without taking environment into account
        """
        train_length = int(len(list_of_idx) * self.train_val_test_prop[0])
        valid_length = int(len(list_of_idx) * self.train_val_test_prop[1])
        test_length = int(len(list_of_idx) - train_length - valid_length)

        return train_length, valid_length, test_length

    def iterator(self):
        # Randomize
        self.split_idx = np.random.permutation(self.split_idx)
        # Create batches
        batch = []
        # Loop over values
        for v in self.split_idx:
            batch.append(v)
            if len(batch) == self.batch_size:  # If batch is full, yield it
                yield batch
                batch = []


class IIDEnvSplitSyntheticSampler(SyntheticSampler):
    """
    Sampler with splitting per environment, but the data is then shuffled within each split
    """

    def __init__(self, batch_size, n_examples_per_env=1000, n_envs=100, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(batch_size, n_examples_per_env, n_envs, split, train_val_test_prop)

    def iterator(self):
        # Randomize
        self.split_idx = np.random.permutation(self.split_idx)
        # Create batches
        batch = []
        # Loop over values
        for v in self.split_idx:
            batch.append(v)
            if len(batch) == self.batch_size:  # If batch is full, yield it
                yield batch
                batch = []


class LoopOverEnvsSyntheticSampler(SyntheticSampler):
    """
    Loop over all environments in the keys of the dict. Once in a given environment,
    yields all samples from that environment
    """

    def __init__(self, batch_size, n_examples_per_env=1000, n_envs=100, split='train',
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        super().__init__(batch_size, n_examples_per_env, n_envs, split, train_val_test_prop)

    def iterator(self):
        env_loop = np.random.permutation(range(len(self.split_idx) // self.n_examples_per_env))
        batch = []
        # Loop over environments
        for env in env_loop:
            # Loop over samples in a given environment
            for idx in np.random.permutation(range(self.n_examples_per_env)):
                batch.append(self.split_idx[self.n_examples_per_env * env + idx])
                if len(batch) == self.batch_size:  # If batch is full, yield it
                    yield batch
                    batch = []
            # Before moving to a new environment, yield current batch even if it is not full
            if batch:
                yield batch
                batch = []


class NEnvPerBatchSyntheticSampler(SyntheticSampler):
    """
    Each batch contains samples from randomly drawn n_env_per_batch distinct environments
    """

    def __init__(self, batch_size, n_examples_per_env=1000, n_envs=100, split='train', n_env_per_batch=3,
                 train_val_test_prop=(0.7, 0.2, 0.1)):
        """
        :param batch_size: Expected to be a multiple of n_env_per_batch
        :param n_env_per_batch: Number of environments per batch
        """
        self.n_env_per_batch = n_env_per_batch
        super().__init__(batch_size, n_examples_per_env, n_envs, split, train_val_test_prop)

    def iterator(self):
        env_loop = np.random.permutation(range(len(self.split_idx) // self.n_examples_per_env))
        env_cpt = 0
        while env_cpt < len(env_loop):
            # Note that an epoch will correspond to seeing all environments once but not all samples
            batch = []
            # Loop over environments
            for _ in range(self.n_env_per_batch):
                if env_cpt >= len(env_loop):
                    break
                env = env_loop[env_cpt]
                env_cpt += 1
                # Loop over samples in a given environment
                for idx in np.random.permutation(np.arange(self.n_examples_per_env))[:self.batch_size //
                                                                                      self.n_env_per_batch]:
                    batch.append(self.split_idx[self.n_examples_per_env * env + idx])
            yield batch


########################################################################################################################
# Dataloaders
########################################################################################################################

#
# def get_dataset(dataset_args):
#     # If the dataset has already been instantiated, use it
#     if tuple(dataset_args) in dict_of_synthetic_datasets:
#         dataset = dict_of_synthetic_datasets[tuple(dataset_args)]
#     else:  # Otherwise instantiate it and save it in the dictionary of datasets
#         dataset = SyntheticDataset(*dataset_args)
#         dict_of_synthetic_datasets[tuple(dataset_args)] = dataset
#         print('A new dataset has been created. Number of datasets: ', len(dict_of_synthetic_datasets))
#
#     return dataset


@register.setdataloadername("synthetic_iid")
def iid_synthetic_dataloader(n_hidden=15, n_observations=978, n_examples_per_env=1000, n_envs=100, attach_proba=0.2,
                             batch_size=16, split='train', train_val_test_prop=(0.7, 0.2, 0.1),
                             dag_generator="multi_gn", struct_eq_gen="noisy_lin_hidden_lin_obs"):
    # Initialize dataset
    dataset_args = [n_hidden, n_observations, n_examples_per_env, n_envs, attach_proba, dag_generator, struct_eq_gen]
    dataset = SyntheticDataset(*dataset_args)

    # Initialize sampler
    sampler_args = [batch_size, n_examples_per_env, n_envs, split, train_val_test_prop]
    sampler = IIDSyntheticSampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("synthetic_iid_env_split")
def iid_env_split_synthetic_dataloader(n_hidden=15, n_observations=978, n_examples_per_env=1000, n_envs=100,
                                       attach_proba=0.2, batch_size=16, split='train',
                                       train_val_test_prop=(0.7, 0.2, 0.1), dag_generator="multi_gn",
                                       struct_eq_gen="noisy_lin_hidden_lin_obs"):
    # Initialize dataset
    dataset_args = [n_hidden, n_observations, n_examples_per_env, n_envs, attach_proba, dag_generator, struct_eq_gen]
    dataset = SyntheticDataset(*dataset_args)

    # Initialize sampler
    sampler_args = [batch_size, n_examples_per_env, n_envs, split, train_val_test_prop]
    sampler = IIDEnvSplitSyntheticSampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("synthetic_loop_over_envs")
def loop_over_envs_synthetic_dataloader(n_hidden=15, n_observations=978, n_examples_per_env=1000, n_envs=100,
                                        attach_proba=0.2, batch_size=16, split='train',
                                        train_val_test_prop=(0.7, 0.2, 0.1), dag_generator="multi_gn",
                                        struct_eq_gen="noisy_lin_hidden_lin_obs"):
    # Initialize dataset
    dataset_args = [n_hidden, n_observations, n_examples_per_env, n_envs, attach_proba, dag_generator, struct_eq_gen]
    dataset = SyntheticDataset(*dataset_args)

    # Initialize sampler
    sampler_args = [batch_size, n_examples_per_env, n_envs, split, train_val_test_prop]
    sampler = LoopOverEnvsSyntheticSampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


@register.setdataloadername("synthetic_n_env_per_batch")
def n_env_per_batch_synthetic_dataloader(n_hidden=15, n_observations=978, n_examples_per_env=1000, n_envs=100,
                                         attach_proba=0.2, batch_size=16, split='train', n_env_per_batch=3,
                                         train_val_test_prop=(0.7, 0.2, 0.1), dag_generator="multi_gn",
                                         struct_eq_gen="noisy_lin_hidden_lin_obs"):
    # Initialize dataset
    dataset_args = [n_hidden, n_observations, n_examples_per_env, n_envs, attach_proba, dag_generator, struct_eq_gen]
    dataset = SyntheticDataset(*dataset_args)

    # Initialize sampler
    sampler_args = [batch_size, n_examples_per_env, n_envs, split, n_env_per_batch, train_val_test_prop]
    sampler = NEnvPerBatchSyntheticSampler(*sampler_args)

    return DataLoader(dataset=dataset, batch_sampler=sampler)


if __name__ == "__main__":
    dl = loop_over_envs_synthetic_dataloader(batch_size=16, n_hidden=5, n_observations=10, n_examples_per_env=20,
                                             n_envs=40, attach_proba=0.3, split="valid",
                                             train_val_test_prop=(0.5, 0.25, 0.25))
    global_graph.draw(show_node_name=True, show_values=False, show_eq=False, show_weights=True, colorbar=False)
    plt.show()
    cpt = 0
    for i in dl:
        cpt += 1
