from torch.utils.data import Dataset
from SyntheticDataGenerator.structuredgraph import StructuredGraph
from torch.utils.data import DataLoader


########################################################################################################################
# Synthetic dataset
########################################################################################################################


class SyntheticDataset(Dataset):

    def __init__(self, n_hidden=15, n_observations=978, n_examples=1000):
        graph = StructuredGraph(n_hidden=n_hidden, n_observations=n_observations)
        graph.generate(n_examples=n_examples)
        self.data = graph.get_observations()

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We format all datasets to return 4-tuples
        return self.data[idx], 0, 0, 0


def synthetic_dataloader(n_hidden=15, n_observations=978, n_examples=1000, batch_size=16,
                         split='train', train_val_test_prop=(0.7, 0.2, 0.1)):
    dataset = SyntheticDataset(n_hidden=n_hidden, n_observations=n_observations, n_examples=n_examples)
    return DataLoader(dataset, batch_size=batch_size)


if __name__ == "__main__":
    dl = synthetic_dataloader()
    for data in dl:
        print(data)
        break