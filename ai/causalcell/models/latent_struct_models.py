import ai.causalcell.utils.register as register
from ai.causalcell.models.utils import *
from ai.causalcell.models.autoencoder import BasicModel
import ai.causalcell.utils.configuration as configuration
import torch
import torch.optim.adam


@register.setmodelname('struct_AE')
class StructAutoEncoder(BasicModel):

    def __init__(self, enc_layers, latent_layers, dec_layers, optimizer_params, dropout=0, norm='none'):
        super(StructAutoEncoder, self).__init__()

        self.latent_scm = LateralLinearLayers(layers=latent_layers, dropout=dropout, norm=norm, activate_final=False)
        self.encoder = LinearLayers(layers=enc_layers, dropout=dropout, norm=norm, activate_final=False)
        self.decoder = LinearLayers(layers=dec_layers, dropout=dropout,
                                    norm=norm, activate_final=False)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.rec_optimizer = configuration.setup_optimizer(optimizer_params['rec_optimizer'])(self.parameters())

    def embed(self, enc_input):
        return self.encoder(enc_input)

    def decod(self, z, fingerprint):
        c = self.latent_scm(z, fingerprint)
        x_prime = self.decoder(c)

        return c, x_prime

    def forward(self, x, fingerprint=0, compound=0, line=0):
        z = self.embed(torch.cat((x, fingerprint), dim=1))
        c, x_prime = self.decod(z, fingerprint)
        return {'z': z, 'c': c, 'x_prime': x_prime, 'x': x}

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        return {'recon_loss': recon_loss}


@register.setmodelname('basic_struct_AE')
class BasicStructAutoEncoder(nn.Module):

    def __init__(self, enc_layers, latent_layers, dec_layers, optimizer_params, dropout=0, norm='none'):
        super(BasicStructAutoEncoder, self).__init__()

        self.encoder = LinearLayers(layers=enc_layers, dropout=dropout, norm=norm, activate_final=False)
        self.decoder = LinearLayers(layers=dec_layers, dropout=dropout,
                                    norm=norm, activate_final=False)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.rec_optimizer = configuration.setup_optimizer(optimizer_params['rec_optimizer'])(self.parameters())

    def embed(self, enc_input):
        return self.encoder(enc_input)

    def decod(self, z, fingerprint):
        c = z + fingerprint
        x_prime = self.decoder(c)

        return c, x_prime

    def forward(self, x, fingerprint=0, compound=0, line=0):
        z = self.embed(torch.cat((x, fingerprint), dim=1))
        c, x_prime = self.decod(z, fingerprint)
        return {'z': z, 'c': c, 'x_prime': x_prime, 'x': x}

    def forward_backward_update(self, x, fingerprint=0, compound=0, line=0, device='cpu'):

        outputs = self.forward(x, fingerprint, compound, line)
        losses = self.loss(outputs)
        loss = sum(losses.values())

        self.rec_optimizer.zero_grad()
        loss.backward()
        self.rec_optimizer.step()

        return loss, losses

    def forward_loss(self, x, fingerprint=0, compound=0, line=0, device='cpu'):
        outputs = self.forward(x, fingerprint, compound, line)
        losses = self.loss(outputs)
        loss = sum(losses.values())

        return loss, losses

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        return {'recon_loss': recon_loss}


@register.setmodelname('struct_VAE')
class StructAutoEncoder(nn.Module):
    """
    Adversarial Variational Autoencoder
    """

    def __init__(self, enc_layers, latent_layers, dec_layers, optimizer_params, beta=1, dropout=0, norm='none'):
        super(StructAutoEncoder, self).__init__()

        self.latent_scm = LateralLinearLayers(layers=latent_layers, dropout=dropout, norm=norm, activate_final=False)

        self.encoder = LinearLayers(layers=enc_layers[:-1], dropout=dropout, norm=norm)
        self.mu = LinearLayers(layers=[enc_layers[-2], enc_layers[-1]], activate_final=False)
        self.logvar = LinearLayers(layers=[enc_layers[-2], enc_layers[-1]], activate_final=False)

        self.decoder = LinearLayers(layers=dec_layers, dropout=dropout,
                                    norm=norm, activate_final=False)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.beta = beta

        self.rec_optimizer = configuration.setup_optimizer(optimizer_params['rec_optimizer'])(self.parameters())

    def embed(self, enc_input):
        h = self.encoder(enc_input)
        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps)

        return z

    def decod(self, z, fingerprint):
        c = self.latent_scm(z, fingerprint)
        x_prime = self.decoder(c)

        return c, x_prime

    def forward(self, x, fingerprint=0, compound=0, line=0):
        mu, logvar = self.embed(torch.cat((x, fingerprint), dim=1))
        z = self.reparameterize(mu, logvar)
        c, x_prime = self.decod(z, fingerprint)
        return {'z': z, 'c': c, 'x_prime': x_prime, 'x': x, 'mu': mu, 'logvar': logvar}

    def forward_backward_update(self, x, fingerprint=0, compound=0, line=0, device='cpu'):

        outputs = self.forward(x, fingerprint, compound, line)
        losses = self.loss(outputs)
        loss = sum(losses.values())

        self.rec_optimizer.zero_grad()
        loss.backward()
        self.rec_optimizer.step()

        return loss, losses

    def forward_loss(self, x, fingerprint=0, compound=0, line=0, device='cpu'):
        outputs = self.forward(x, fingerprint, compound, line)
        losses = self.loss(outputs)
        loss = sum(losses.values())

        return loss, losses

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        kl_div = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())

        # Apply beta scaling factor
        kl_div *= self.beta

        return {'recon_loss': recon_loss, 'kl_div': kl_div}