import torch
import torch.nn as nn
import yaml
import logging
import optuna
import os
import argparse
import warnings
from utils import get_optimizer, accuracy, make_dp, get_dataloader
from models import get_model
from datasets import load_data, Metrics
from core import train

warnings.simplefilter("ignore")
logger = logging.getLogger('search')
logger.setLevel(logging.INFO)
#logger.addHandler(logging.StreamHandler(sys.stdout))
#logging.getLogger('core').setLevel(logging.WARNING)

def get_hp(trial, optim_name):

    if optim_name == 'SGD':
        return {
            'lr': trial.suggest_float('lr_base', 1e-2, 1, log=True),
            'momentum': trial.suggest_int('momentum_base', 0, 10)/float(10),
            'batch_size': 2**trial.suggest_int('batch_size_base', 4, 14),
            'epochs': 40
        }
    
    if optim_name == 'ONC':
        return {
            'lr': trial.suggest_float('lr', 1e-2, 1, log=True),
            'batch_size': 2**trial.suggest_int('batch_size_base', 4, 14),
            'D': trial.suggest_float('D', 1e-6, 1e-2, log=True),
            'epochs': 40
        }
    
    if optim_name == 'PONC':
        return {
            'lr': trial.suggest_float('lr', 1e-2, 1, log=True),
            'batch_size': 2**trial.suggest_int('batch_size_base', 4, 14),
            'D': trial.suggest_float('D', 1e-6, 1e-2, log=True),
            'eps_frac_cp': trial.suggest_float('eps_frac_cp', 0.5, 0.99),
            'epochs': 40
        }
    raise ValueError()

def objective(trial, args):

    # load HP
    hp = get_hp(trial, args.optim_name)

    # print some info
    logger.info(f'evaluate {args.optim_name} on {args.dataset}')
    logger.info(f'hp setting {hp}')
    
    # load data. note: for now, data['val'] = data['test']
    data = get_dataloader(args, hp)

    # define learning objects shared over all runs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    metrics = Metrics([loss_fn, accuracy], args.runs, hp['epochs'], device)

    for run in range(args.runs):
        logger.info(f'##### RUN {run+1} / {args.runs} #####')
        # get model, optimizer and make DP
        model = get_model(args.dataset, device)
        optimizer = get_optimizer(args.optim_name, model, hp)
        privacy_engine, model, optimizer, data['train'] = make_dp(model, optimizer, data['train'], args, hp, logger)
        # train model, stats are saved in metrics object
        train(data, model, optimizer, loss_fn, 0, metrics, device, ponc=True)
        # increment run counter in metrics object
        metrics.increment_run()
        if args.optim_name in ['DPSGD']:
            logger.info(f'privacy budget spent {(privacy_engine.get_epsilon(args.delta), args.delta)}')
        if args.optim_name in ['PONC']:
            privacy_engine1, privacy_engine2 = privacy_engine
            #logger.info(f'privacy budget spent on checkpoint {(privacy_engine1.get_epsilon(args.delta), args.delta/2)}')
            #logger.info(f'privacy budget spent on ponc descent {(privacy_engine2.get_epsilon(args.delta), args.delta/2)}')

    train_loss, _, _ = metrics.get_best_train(loss_fn, loss_fn)
    #_, val_acc, _ = metrics.get_best_val(accuracy, loss_fn)
    #logger.info(f'val acc {val_acc:.5}')
    #train_loss, _, _ = metrics.get_summary(loss_fn)
    #train_acc, _, _ = metrics.get_summary(accuracy)
    #logger.info(f'train acc {train_acc:.5}')

    return train_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MNIST'])
    parser.add_argument('--optim_name', choices=['SGD', 'DPSGD', 'ONC', 'PONC'])
    parser.add_argument('--n_threads', default=0)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--n_trials', type=int)
    args = parser.parse_args()

    database_rep = './database/'

    # create/load study
    study_name = args.optim_name+'_'+args.dataset
    file = database_rep + study_name + '.db'
    storage = 'sqlite:///' + file
    if os.path.isfile(file): 
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage)

    logger.info(f'HP search for {args.optim_name} on {args.dataset}')

    # get objective function
    objective_aug = lambda trial: objective(trial, args)

    # carry out HP search
    study.optimize(objective_aug, n_trials=args.n_trials, timeout=100000)

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
    filename = f'./HP/{args.optim_name}/{args.dataset}.yaml'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(file_dump)