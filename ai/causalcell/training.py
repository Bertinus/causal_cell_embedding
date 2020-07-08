import ai.causalcell.utils.configuration as configuration
import ai.causalcell.datasets.synthetic_dataset as sd
import logging
import numpy as np
import torch
import random
import os
import copy
import dill as pickle
import skopt
from collections import OrderedDict

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
        all_losses.append({i: losses[i].detach().cpu().numpy() for i in losses.keys()})

        # Optimization.
        loss = sum(losses.values())
        all_loss.append(loss.detach())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    all_loss = float(torch.mean(torch.tensor(all_loss)).detach().numpy())
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
            all_losses.append({i: losses[i].detach().cpu().numpy() for i in losses.keys()})

            # Sum up batch loss.
            loss = sum(losses.values())
            all_loss.append(loss)

    all_loss = float(torch.mean(torch.tensor(all_loss)).detach().numpy())
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
    if len(list(model.parameters())) == 0:
        optim = None
    else:
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
                   "best_model": best_model.to('cpu'),
                   "last_model": model.to('cpu')}

    save_results(results, output_dir)

#
# def train_skopt(cfg, base_estimator, n_initial_points, random_state):
#     """
#     Do a Bayesian hyperparameter optimization
#     """
#
#     # Parse the parameters that we want to optimize, then flatten list.
#     hp_args = OrderedDict(configuration.parse_dict(cfg))
#     all_vals = []
#     for val in hp_args.values():
#         if isinstance(val, list):
#             all_vals.extend(val)
#         else:
#             all_vals.append(val)
#
#     hp_opt = skopt.Optimizer(dimensions=all_vals,
#                              base_estimator=base_estimator,
#                              n_initial_points=n_initial_points,
#                              random_state=random_state)
#
#     set_seed(cfg['seed'])
#
#     # best_valid and best_test score are used inside of train(), best_model
#     # score is only used in train_skopt() for final model selection.
#     state = {'base_iteration': 0,
#              'base_epoch': 0,
#              'best_valid_score': 0,
#              'best_test_score': 0,
#              'best_model_score': 0,
#              'best_epoch': 0,
#              'hp_opt': hp_opt,
#              'hp_args': hp_args,
#              'base_cfg': cfg,
#              'this_cfg': None,
#              'numpy_seed': None,
#              'optimizer': None,
#              'python_seed': None,
#              'scheduler': None,
#              'model': None,
#              'stats': None,
#              'metrics': [],
#              'torch_seed': None,
#              'suggestion': None,
#              'best_model': None}
#
#     # Do a bunch of loops.
#
#
#     for iteration in range(state['base_iteration'], n_iter + 1):
#
#         # Will not be true if training crashed for an iteration.
#         if isinstance(state['this_cfg'], type(None)):
#             suggestion = hp_opt.ask()
#             state['this_cfg'] = generate_config(cfg, hp_args, suggestion)
#             state['suggestion'] = suggestion
#         try:
#             this_valid_score, this_test_score, this_best_epoch, results, state = train(state['this_cfg'])
#
#             # Skopt tries to minimize the valid score, so it's inverted.
#             this_metric = this_valid_score * -1
#             hp_opt.tell(state['suggestion'], this_metric)
#         except RuntimeError as e:
#             # Something went wrong, (probably a CUDA error).
#             this_metric = 0.0
#             this_valid_score = 0.0
#             print("Experiment failed:\n{}\nAttempting next config.".format(e))
#             hp_opt.tell(suggestion, this_metric)
#
#         if this_valid_score > state['best_model_score']:
#             print("*** new best model found: score={}".format(this_valid_score))
#             state['best_model_score'] = this_valid_score
#             save_results(results, output_dir)