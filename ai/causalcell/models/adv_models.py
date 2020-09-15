import ai.causalcell.utils.register as register
from ai.causalcell.models.utils import *
from ai.causalcell.models.autoencoder import AutoEncoder
import ai.causalcell.utils.configuration as configuration
import torch
import torch.optim.adam


@register.setmodelname('adv_AE')
class AdversarialAutoEncoder(AutoEncoder):
    """
    Adversarial Autoencoder
    """

    def __init__(self, enc_layers, dec_layers, disc_layers, optimizer_params, dropout=0, norm='none'):
        super(AdversarialAutoEncoder, self).__init__(enc_layers, dec_layers, optimizer_params,
                                                     dropout=dropout, norm=norm)

        self.discriminator = LinearLayers(layers=disc_layers, dropout=dropout, norm=norm, activate_final=False)

        # Define two additional optimizers
        self.disc_optimizer = configuration.setup_optimizer(optimizer_params['disc_optimizer'])\
            (self.discriminator.parameters())
        self.gen_optimizer = configuration.setup_optimizer(optimizer_params['gen_optimizer'])\
            (self.encoder.parameters())

        self.disc_criterion = torch.nn.BCELoss()

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x, fingerprint=0, compound=0, line=0):
        z = self.embed(x)
        x_prime = self.decoder(z)
        return {'z': z, 'x_prime': x_prime, 'x': x}

    def sample(self, shape, device):
        return torch.randn(shape).to(device)

    def forward_backward_update(self, x, fingerprint=0, compound=0, line=0, device='cpu'):
        ################################################################################################################
        # Train AutoEncoder
        ################################################################################################################

        outputs = self.forward(x, fingerprint, compound, line)
        losses = self.loss(outputs)

        loss = sum(losses.values())

        self.rec_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.rec_optimizer.step()

        ################################################################################################################
        # Train Discriminator + Encoder to fool Discriminator
        ################################################################################################################

        fake_z = self.sample(outputs['z'].shape, device)
        disc_pred = torch.sigmoid(self.discriminator(torch.cat((outputs['z'], fake_z))))
        disc_losses = self.disc_loss(disc_pred, device)

        disc_loss = sum(disc_losses.values())

        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Invert gradients for encoder
        for group in self.gen_optimizer.param_groups:
            for p in group['params']:
                p.grad = -1 * p.grad

        self.gen_optimizer.step()

        losses.update(disc_losses)

        return loss + disc_loss, losses

    def forward_loss(self, x, fingerprint=0, compound=0, line=0, device='cpu'):
        outputs = self.forward(x, fingerprint, compound, line)
        losses = self.loss(outputs)
        loss = sum(losses.values())

        fake_z = self.sample(outputs['z'].shape, device)
        disc_pred = torch.sigmoid(self.discriminator(torch.cat((outputs['z'], fake_z))))
        disc_losses = self.disc_loss(disc_pred, device)
        disc_loss = sum(disc_losses.values())

        losses.update(disc_losses)

        return loss + disc_loss, losses

    def loss(self, outputs):
        recon_loss = self.criterion(outputs['x_prime'], outputs['x'])
        return {'recon_loss': recon_loss}

    def disc_loss(self, disc_pred, device):
        batch_size = int(disc_pred.shape[0]/2)
        labels = torch.Tensor([[1] * batch_size + [0] * batch_size]).T.to(device)
        disc_loss = self.disc_criterion(disc_pred, labels)

        return {'disc_loss': disc_loss}
