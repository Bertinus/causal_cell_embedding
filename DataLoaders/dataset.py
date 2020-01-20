from torch.utils.data import Dataset
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch
from random import shuffle


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

        # Read metadata
        self.sig_info = pd.read_csv(path_to_sig_info, sep="\t")
        self.gene_info = pd.read_csv(path_to_gene_info, sep="\t")

        self.all_perts = self.sig_info.pert_id.unique().tolist()  # All perturbations
        self.all_cells = self.sig_info.cell_id.unique().tolist()
        self.landmark_gene_list = self.gene_info[self.gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str)

        # Initialize environment variables
        self.env_pert_idx = None
        self.env_cell_idx = None
        self.env_pert_name = None
        self.env_cell_name = None
        self.env_data = None

        # Data path
        self.path_to_data = path_to_data
        # Load data of the first cell line first perturbation
        self.select_env(pert=0, cell=0)

    def select_env(self, pert, cell):
        """
        :param pert: index of perturbation type
        :param cell: index of cell id
        Loads corresponding environment in the env_data variable
        """
        # Set environment variables
        self.env_pert_idx = pert
        self.env_cell_idx = cell
        self.env_pert_name = self.all_perts[pert]
        self.env_cell_name = self.all_cells[cell]
        # Load env data
        env_subset = self.sig_info["sig_id"][(self.sig_info["pert_id"] == self.env_pert_name) &
                                             (self.sig_info["cell_id"] == self.env_cell_name)]
        if env_subset.empty:
            raise IndexError
        self.env_data = parse(self.path_to_data, cid=env_subset, rid=self.landmark_gene_list).data_df

    def envlen(self):
        """
        :return: Length of the environment currently selected
        """
        return len(self.env_data.T)

    def __len__(self):
        return len(self.sig_info.shape[0])

    def __getitem__(self, idx_env):
        idx, pert, cell = idx_env
        if self.env_pert_idx != pert or self.env_cell_idx != cell:
            self.select_env(pert, cell)
            # print("Changing env")
        else:
            pass
            # print("Environment remains the same")
        return self.env_data[self.env_data.columns[idx]].to_numpy()


class L1000Sampler(Sampler):
    """
    Sampler class used to loop over the dataset while taking into account environments

    Loop over all environments in the list self.environments. Once in a given environment,
    yields all samples from that environment

    """

    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        self.batchsize = batchsize
        # List of environments to loop over
        # TODO : too slow, precompute a list with non empty envs and their length
        self.environments = [(a, b)
                             for a in range(len(self.dataset.all_perts))
                             for b in range(len(self.dataset.all_cells))]
        shuffle(self.environments)

    def iterator(self):
        for pert, cell in self.environments:
            try:
                self.dataset.select_env(pert, cell)
                # print("New environment of length", self.dataset.envlen())
                for idx in torch.randperm(self.dataset.envlen()):
                    yield (idx, pert, cell)
            except IndexError:
                pass
                # print("Empty environment")

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return len(self.dataset)


def l1000_dataloader(batch_size=64):
    dataset = L1000Dataset()
    sampler = L1000Sampler(dataset, batch_size)

    return DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)


if __name__ == "__main__":

    dataloader = l1000_dataloader()
    for i in dataloader:
        print(i.shape)
