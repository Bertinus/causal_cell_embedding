from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn


class Dummy(torch.nn.Module):
    """
    Helper class that allows you to do nothing. Replaces `lambda x: x`.
    """

    def forward(self, x):
        return x


class KeepMax(torch.nn.Module):
    """
    Keep only the max and apply zero elsewhere
    """
    def forward(self, x):
        output = torch.zeros(x.shape)
        output[torch.arange(output.size(0)), torch.max(x, dim=1)[1]] =  torch.max(x, dim=1)[0]
        return output


class LinearBatch(nn.Module):
    """A linear layer with batch normalization."""

    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearBatch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out = self.linear(x)
        return self.batch_norm(out)


class LinearSpec(nn.Module):
    """A linear layer with spectral normalization."""

    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSpec, self).__init__()
        self.linear = spectral_norm(nn.Linear(input_dim, output_dim, bias=bias))

    def forward(self, x):
        return self.linear(x)


class LinearLayers(nn.Module):
    """
    A basic feed forward network with configurable dropout and layer-wise
    normalization.
    """

    def __init__(self, layers, dropout=0, norm='none', modulelist=False,
                 bias=True, activate_final=True):
        super(LinearLayers, self).__init__()

        norm_dict = {
            'batch': LinearBatch, 'spectral': LinearSpec, 'none': nn.Linear}

        assert 0 <= dropout < 1
        assert norm in norm_dict.keys()

        # Set up the linear layers.
        self.linear = norm_dict[norm]
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        arch = []
        n_layers = len(layers)
        for i in range(n_layers - 1):
            arch.append(self.linear(layers[i], layers[i + 1], bias=bias))

            # Add activation to all layers, except the final one optionally.
            if activate_final or i + 1 < n_layers - 1:
                arch.append(self.activation)

                # Only dropout the activated layers.
                if dropout > 0:
                    arch.append(self.dropout)

        # Allows for the linear layer to be a passthrough in the case that
        # the number of layers requested is empty or a single value.
        if len(arch) > 0:
            if modulelist:
                self.model = nn.ModuleList(arch)
            else:
                self.model = nn.Sequential(*arch)
        else:
            self.model = Dummy()

    def embed(self, x):
        return self.model(x)

    def forward(self, x):
        return self.embed(x)
