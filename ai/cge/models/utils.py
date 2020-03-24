import math
import torch
import numpy as np

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def generate_feedforward_layers(n_init, n_layers, shrink_factor, min_size=1):
    """
    Given three parameters:
        n_initial: the size of the first hidden layer of the network.
        n_layers: the number of layers in the feedforward network.
        shrink_factor: the multiplier applied to layer l to determine the
                       size of layer n+1
    Return a python list defining the hidden layer sizes of the feedforward
    network.
    """
    architecture = []

    assert n_layers > 0

    # Ensures that the number of neurons in a layer is never smaller than
    # min_size neurons.
    for i in range(n_layers):
        architecture.append(max(min_size, math.ceil(n_init * shrink_factor**i)))

    return(architecture)


class Flatten(torch.nn.Module):
    """
    Helper class that allows you to perform a flattening op inside
    nn.Sequential.
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Dummy(torch.nn.Module):
    """
    Helper class that allows you to do nothing. Replaces `lambda x: x`.
    """
    def forward(self, x):
        return(x)


def shuffle_batch(data):
    """
    Rotate the features by one through the batch index so they are
    all in a unique new location in the batch.
    """
    data_perm = torch.clone(data)
    batch_idx = torch.arange(data.shape[0])
    batch_idx_rotate = torch.clone(batch_idx)
    batch_idx_rotate[1:] = batch_idx[:-1]
    batch_idx_rotate[0] = batch_idx[-1]
    data_perm = data_perm[batch_idx_rotate]

    return(data_perm)


def shuffle_features(data):
    """
    Rotate the features by one through the batch index so they are
    all in a unique new location in the batch.
    """
    # Shuffle features for each item in the batch independently.
    for i in range(data.shape[0]):
        data[i, :] = data[i, torch.randperm(data.shape[1])]

    return(data)


def get_mi(X, Y, mi_net, mode='linear'):
    """
    We rotate the features in Y over the batch dimension to marginalize. In mode
    'linear', we assume X and Y are 2D matrices. In mode 'conv' we assume X
    is 4D ([batch_size, channels, dim_x, dim_y]). Y is assumed to be 2D
    ([batch_size, channels]). In this approach we are computing the MI between
    the channel dimension of each, for each location of the feature map in X.

    For more details, see fig 6. in "Learning Deep Representations by
    Maximizing Mutual Information Estimation and Maximization Hjelm et al 2019",
    for the "encode and dot-product" Local DIM method.
    """
    assert mode in ['linear', 'conv']

    if mode == 'linear':
        joint = torch.cat([X, Y], dim=1)
    elif mode == 'conv':
        batch, n_ch, dim_x, dim_y = X.shape
        joint = torch.bmm(X.view(batch, n_ch, -1).permute(0, 2, 1),
                          Y.unsqueeze(-1).repeat(1, 1, dim_x * dim_y))
        joint = torch.diagonal(joint, dim1=1, dim2=2)

    Y_perm = shuffle_batch(Y)  # Marginalize Y over X.

    if mode == 'linear':
        marginal = torch.cat([X, Y_perm], dim=1)
    elif mode == 'conv':
        marginal = torch.bmm(X.view(batch, n_ch, -1).permute(0, 2, 1),
                             Y_perm.unsqueeze(-1).repeat(1, 1, dim_x * dim_y))
        marginal = torch.diagonal(marginal, dim1=1, dim2=2)

    mi = mi_net(joint, marginal)

    return mi


def scale_data(self, data):
    """Scales data per batch-instance to [0 SCALE]. TODO: DEPRICATE."""
    EPS = 10e-6  # For numerical stability.
    SCALE = 255  # I believe this will make things safer during resize.
    mins = np.min(data, axis=(1,2))[:, np.newaxis, np.newaxis]
    maxs = np.max(data, axis=(1,2))[:, np.newaxis, np.newaxis]
    data = ((data-mins) / (maxs-mins+EPS)) * SCALE

    return data

if __name__  == "__main__":
    assert generate_feedforward_layers(100, 1, 0.5) == [100]
    assert generate_feedforward_layers(100, 4, 0.5) == [100, 50, 25, 13]
    assert generate_feedforward_layers(20, 4, 0.5) == [20, 10, 10, 10]
