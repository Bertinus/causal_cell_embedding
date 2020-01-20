from torch.utils.data import Dataset
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse


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
                 path_to_gene_info="Data/L1000_PhaseII/GSE70138_Broad_LINCS/gene_info_2017-03-06.txt",
                 batch_size=64):

        self.batch_size = batch_size

        # Read metadata
        self.sig_info = pd.read_csv(path_to_sig_info, sep="\t")
        self.gene_info = pd.read_csv(path_to_gene_info, sep="\t")

        self.all_perts = self.sig_info.pert_id.unique().tolist()  # All perturbations
        self.all_cells = self.sig_info.cell_id.unique().tolist()
        self.landmark_gene_list = self.gene_info[self.gene_info['pr_is_lm'] == 1]["pr_gene_id"].astype(str)

        # Initialize environment variables
        self.env_pert_type = None
        self.env_cell_id = None
        self.env_data = None

        # Data path
        self.path_to_data = path_to_data
        # Load data of the first cell line first perturbation
        self.select_env(env_cell_id=0, env_pert_type=0)

    def select_env(self, env_pert_type, env_cell_id):
        """
        :param env_pert_type: index of perturbation type
        :param env_cell_id: index of cell id
        Loads corresponding environment in the env_data variable
        """
        # Set environment variables
        self.env_pert_type = self.all_perts[env_pert_type]
        self.env_cell_id = self.all_cells[env_cell_id]
        # Load env data
        env_subset = self.sig_info["sig_id"][(self.sig_info["pert_id"] == self.env_pert_type) &
                                             (self.sig_info["cell_id"] == self.env_cell_id)]
        self.env_data = parse(self.path_to_data, cid=env_subset, rid=self.landmark_gene_list).data_df

    def envlen(self):
        """
        :return: Length of the environment currently selected
        """
        return len(self.env_data)

    def __len__(self):
        return len(self.sig_info.shape[0])

    def __getitem__(self, idx):
        return 0


if __name__ == "__main__":
    dataset = L1000Dataset()
