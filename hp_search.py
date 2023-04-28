import torch
import torch.nn as nn
import yaml
import logging
import optuna
import os
import sys
from utils import get_optimizer, accuracy
from core import get_perfromance
from models import get_model
from datasets import load_data, Metrics

logger = logging.getLogger('search')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
#logging.getLogger('core').setLevel(logging.WARNING)

def get_hp(trial, optim_name):

    if optim_name == 'SGD':
        batch_size = 2**trial.suggest_int('batch_size_base', 4, 14)
        return {
            'lr': trial.suggest_float('lr_base', 1e-2, 2, log=True)*batch_size/float(512),
            'momentum': trial.suggest_int('momentum_base', 0, 10)/float(10),
            'batch_size': batch_size,
            'epochs': 40
        }
    raise ValueError()

def objective(trial, args):

    hp = get_hp(trial, args['optim_name'])
    fixed_hps = set(hp.keys()).difference(set(trial.params.keys()))
    for hp_name in fixed_hps: 
        trial.set_user_attr(hp_name, hp[hp_name])

    data = load_data(args['data_name'])
    for key, val in data.items():
        data[key] = torch.utils.data.DataLoader(val, batch_size=hp['batch_size'], shuffle=True, num_workers=args['n_threads'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args['data_name'], device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args['optim_name'], model, hp)
    metrics = Metrics([loss_fn, accuracy], 1, hp['epochs'], device)

    get_perfromance(data, model, optimizer, loss_fn, 1, hp['epochs'], metrics, device)

    #_, val_loss, _ = metrics.get_best_val(loss_fn, loss_fn)
    #_, val_acc, _ = metrics.get_best_val(accuracy, loss_fn)
    #logger.info(f'val acc {val_acc:.5}')
    train_loss, _, _ = metrics.get_summary(loss_fn)
    train_acc, _, _ = metrics.get_summary(accuracy)
    logger.info(f'train acc {train_acc:.5}')

    return train_loss


if __name__ == '__main__':
    database_rep = './database/'
    with open('./config.yaml', 'r') as file:
        args = yaml.safe_load(file)

    # create/load study
    study_name = args['optim_name']+'_'+args['data_name']
    file = database_rep + study_name + '.db'
    storage = 'sqlite:///' + file
    if os.path.isfile(file): 
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage)

    logger.info(f'HP search for {args["optim_name"]} on {args["data_name"]}')

    # get objective function
    objective_aug = lambda trial: objective(trial, args)

    # carry out HP search
    study.optimize(objective_aug, n_trials=args['n_trials'], timeout=100000)

    # print study results
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])    
    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Value: {best_trial.value}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # save study results
    file_dump = yaml.dump({**best_trial.params, **best_trial.user_attrs})
    filename = f'./HP/{args["optim_name"]}/{args["data_name"]}.yaml'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(file_dump)