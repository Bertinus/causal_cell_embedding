import ai.causalcell.utils.configuration as configuration
import ai.causalcell.datasets.synthetic_dataset as sd
import logging
import numpy as np
import torch
import random
import os
import copy
import dill as pickle

# from ai.causalcell.datasets.synthetic_dataset import global_graph

_LOG = logging.getLogger(__name__)


def set_seed(seed, cuda=False):
    """
    Fix the seed for numpy, python random, and pytorch.
    """
    print('pytorch/random seed: {}'.format(seed))

    # Numpy, python, pytorch (cpu), pytorch (gpu).
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cuda:
        torch.cuda.manual_seed_all(seed)


def save_results(results, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save best model
    output_name = "best_model_{}.pth.tar".format(results["exp_id"])
    torch.save(results["best_model"].state_dict(), os.path.join(output_dir, output_name))

    # Save last model
    output_name = "last_model_{}.pth.tar".format(results["exp_id"])
    torch.save(results["last_model"].state_dict(), os.path.join(output_dir, output_name))

    # Save the rest of the results dictionary
    del results["best_model"]
    del results["last_model"]
    output_name = "results_{}.pkl".format(results["exp_id"])
    with open(os.path.join(output_dir, output_name), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def train_epoch(model, device, train_loader, optimizer, epoch):

    model.train()

    all_loss, all_losses = [], []

    for batch_idx, data in enumerate(train_loader):

        x, fingerprint, compound, line = data
        x = x.to(device)
        fingerprint = fingerprint.to(device)

        # Expected to return a dictionary of outputs.
        outputs = model.forward(x, fingerprint, compound, line)

        # Expected to return a dictionary of loss terms.
        losses = model.loss(outputs)
        all_losses.append(losses)

        # Optimization.
        optimizer.zero_grad()
        loss = sum(losses.values())
        all_loss.append(loss.detach())
        loss.backward()
        optimizer.step()

    all_loss = np.array(all_loss).mean()
    print('epoch {} Mean train loss: {:.4f}'.format(
        epoch, all_loss))

    return all_loss, all_losses


def evaluate_epoch(model, device, data_loader, epoch):
    """Evaluates a given model on given data."""
    model.eval()
    all_loss, all_losses = [], []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            x, fingerprint, compound, line = data
            x = x.to(device)
            fingerprint = fingerprint.to(device)

            # Expected to return a dictionary of outputs.
            outputs = model.forward(x, fingerprint, compound, line)
            losses = model.loss(outputs)
            all_losses.append(losses)

            # Sum up batch loss.
            loss = sum(losses.values())
            all_loss.append(loss)

    all_loss = np.array(all_loss).mean()
    print('epoch {} Mean valid loss: {:.4f}'.format(
        epoch, all_loss))

    return all_loss, all_losses


def train(cfg):
    """
    Trains a model on a dataset given the supplied configuration.
    save is by default True and will result in the model's performance being
    saved to a handy pickle file, as well as the best-performing model being
    saved. Set this to False when doing an outer loop of hyperparameter
    optimization.
    """
    exp_name = cfg['experiment_name']
    exp_id = cfg['exp_id']
    n_epochs = cfg['n_epochs']
    seed = cfg['seed']
    output_dir = os.path.join('results', cfg['experiment_name'])
    early_stopping = cfg['early_stopping']
    patience_max = cfg['patience_max']
    patience = 0

    set_seed(seed)

    # dataloader
    valid_loader = configuration.setup_dataloader(cfg, 'valid')
    train_loader = configuration.setup_dataloader(cfg, 'train')

    device = 'cuda' if cfg['cuda'] else 'cpu'
    model = configuration.setup_model(cfg).to(device)
    optim = configuration.setup_optimizer(cfg)(model.parameters())

    print('model: \n{}'.format(model))
    print('optimizer: {}'.format(optim))

    best_valid_loss = np.inf
    best_model, best_epoch = None, None
    all_train_losses, all_valid_losses = [], []

    for epoch in range(n_epochs):

        train_loss, train_losses = train_epoch(model=model, device=device, optimizer=optim, train_loader=train_loader,
                                               epoch=epoch)

        valid_loss, valid_losses = evaluate_epoch(model=model, device=device, data_loader=valid_loader, epoch=epoch)

        all_train_losses.append(train_losses)
        all_valid_losses.append(valid_losses)

        if valid_loss < best_valid_loss:
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_valid_loss = valid_loss
        else:
            patience += 1
            if early_stopping and patience > patience_max:
                break

        results = {"exp_name": exp_name,
                   "config": cfg,
                   "data_graph": sd.global_graph,
                   "seed": seed,
                   "exp_id": exp_id,
                   "n_envs_in_split": {"train": train_loader.batch_sampler.n_envs_in_split,
                                       "valid": valid_loader.batch_sampler.n_envs_in_split},
                   "n_samples_in_split": {"train": train_loader.batch_sampler.n_samples,
                                          "valid": valid_loader.batch_sampler.n_samples},
                   "losses": {"train": all_train_losses, "valid": all_valid_losses},
                   "best_epoch": best_epoch,
                   "best_model": best_model,
                   "last_model": model}

    save_results(results, output_dir)
